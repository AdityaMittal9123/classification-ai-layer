import os
import logging
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document  # MIT
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # MIT

import config
from llm_client import LLMClientWrapper

logger = logging.getLogger(__name__)

# Dimension and name of the fake embeddings used in mock mode (no Gemini API key)
MOCK_EMBEDDING_SIZE = 768
MOCK_EMBEDDING_MODEL = "mock-fake-embedding"


class FeedbackVectorStore:
    """
    RAG store of past human feedback, built with LangChain:
    - Chunking: RecursiveCharacterTextSplitter splits email body + attachment text into overlapping chunks.
    - Embeddings: GoogleGenerativeAIEmbeddings (default gemini-embedding-001), whatever the LLM provider.
    - Vector store: LangChain InMemoryVectorStore (cosine similarity), persisted to a JSON file.

    Every chunk carries the feedback metadata (verified classes, reviewer notes, ...), so a match on any
    chunk of a past email (e.g. one page of an attached PDF) retrieves that email's human decision.
    Every chunk is also tagged with the embedding model that created it; searches only compare chunks from
    the current model, because vectors from different models (or mock mode) are not comparable.
    """

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or config.VECTOR_STORE_PATH
        parent_dir = os.path.dirname(os.path.abspath(self.storage_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.embeddings, self.is_mock, self.embedding_model = self._create_embeddings()
        self.store = self._load()

    def _create_embeddings(self) -> Tuple[Any, bool, str]:
        """Gemini embeddings if a Gemini API key is set, otherwise deterministic fake embeddings (mock)."""
        if not LLMClientWrapper.is_placeholder_key(config.GEMINI_API_KEY):
            logger.info(f"[VectorStore] Using Gemini embeddings model '{config.GEMINI_EMBEDDING_MODEL}'")
            embeddings = GoogleGenerativeAIEmbeddings(
                model=config.GEMINI_EMBEDDING_MODEL,
                google_api_key=config.GEMINI_API_KEY,
            )
            return embeddings, False, config.GEMINI_EMBEDDING_MODEL

        logger.warning("[VectorStore] Gemini API key not set; using fake embeddings (Mock Mode).")
        return DeterministicFakeEmbedding(size=MOCK_EMBEDDING_SIZE), True, MOCK_EMBEDDING_MODEL

    def _load(self) -> InMemoryVectorStore:
        """Loads the saved vector store from disk if available."""
        if os.path.exists(self.storage_path):
            try:
                store = InMemoryVectorStore.load(self.storage_path, self.embeddings)
                logger.info(f"[VectorStore] Loaded {len(store.store)} feedback chunks from {self.storage_path}")
                return store
            except Exception as e:
                logger.error(f"[VectorStore] Error loading vector store from {self.storage_path}: {e}")
        return InMemoryVectorStore(self.embeddings)

    def _save(self):
        """Persists vectors and metadata to disk."""
        try:
            self.store.dump(self.storage_path)
        except Exception as e:
            logger.error(f"[VectorStore] Error saving vector store to {self.storage_path}: {e}")

    def _same_embedding_model(self, doc: Document) -> bool:
        """Only chunks embedded with the current model are comparable with the query vectors."""
        return doc.metadata.get("embedding_model") == self.embedding_model

    def split(self, documents: List[Document]) -> List[Document]:
        """Splits documents (email body, attachments) into overlapping chunks."""
        return [chunk for chunk in self.splitter.split_documents(documents) if chunk.page_content.strip()]

    def add_feedback(self, documents: List[Document], metadata: Dict[str, Any]) -> int:
        """
        Chunks the email + attachment documents, embeds every chunk and stores it with the feedback metadata.
        Returns the number of chunks stored.
        """
        chunks = self.split(documents)
        for chunk in chunks:
            chunk.metadata = {
                **metadata,
                "source": chunk.metadata.get("source", "unknown"),
                "embedding_model": self.embedding_model,
            }

        self.store.add_documents(chunks)
        self._save()
        logger.info(
            f"[VectorStore] Indexed {len(chunks)} feedback chunks for email_id='{metadata.get('email_id')}'. "
            f"Total chunks: {len(self.store.store)}"
        )
        return len(chunks)

    def search_similar_feedbacks(
        self,
        documents: List[Document],
        top_k: int = 3,
        min_similarity: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        Chunks the new email, searches every chunk against stored feedback chunks and returns the
        top-K most similar past feedback records (one entry per past email, best chunk score wins).
        """
        if not self.store.store:
            return []

        query_chunks = self.split(documents)[: config.MAX_QUERY_CHUNKS]
        if not query_chunks:
            return []
        query_vectors = self.embeddings.embed_documents([chunk.page_content for chunk in query_chunks])

        # Best matching chunk per past feedback record
        best_by_feedback: Dict[Any, Tuple[float, Document]] = {}
        for vector in query_vectors:
            for doc, score in self.store.similarity_search_with_score_by_vector(
                vector, k=top_k * 3, filter=self._same_embedding_model
            ):
                feedback_id = doc.metadata.get("feedback_db_id")
                if feedback_id not in best_by_feedback or score > best_by_feedback[feedback_id][0]:
                    best_by_feedback[feedback_id] = (score, doc)

        ranked = sorted(best_by_feedback.values(), key=lambda item: item[0], reverse=True)

        # Filter by min_similarity if matches exist; otherwise take top entries (for demo/mock compatibility)
        filtered = [item for item in ranked if item[0] >= min_similarity]
        final_pool = filtered if filtered else ranked

        return [
            {
                "similarity_score": round(float(score), 3),
                **doc.metadata,
                "matched_chunk": doc.page_content[:200],
            }
            for score, doc in final_pool[:top_k]
        ]
