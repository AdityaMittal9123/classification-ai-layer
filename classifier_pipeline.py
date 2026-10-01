import re
import difflib
import logging
from typing import List, Optional, Dict, Any

import config
from schemas import (
    EmailMessage,
    EmailClassificationResult,
    PredictedClass,
    UserFeedbackSubmission,
)
from langchain_core.documents import Document

from llm_client import LLMClientWrapper
from attachment_extractor import AttachmentExtractor
from feedback_db import FeedbackDatabase
from vector_store import FeedbackVectorStore

logger = logging.getLogger(__name__)


class EmailClassificationPipeline:
    """
    End-to-End LLM Email Classification Pipeline with:
    1. Multi-class multi-label classification across 7 target classes.
    2. Attachment awareness (full text extracted from PDF, Word, Excel, images and more).
    3. Confidence threshold verification.
    4. Feedback loop storing human corrections into DB table.
    5. In-context retrieval (RAG) using Vector DB embeddings of historical feedback.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        db_path: Optional[str] = None,
        vector_store_path: Optional[str] = None,
    ):
        self.threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.llm_client = LLMClientWrapper(provider=provider)
        self.feedback_db = FeedbackDatabase(db_path=db_path)
        self.vector_store = FeedbackVectorStore(storage_path=vector_store_path)
        self.target_classes = config.EMAIL_CLASSES
        self.attachment_extractor = AttachmentExtractor(self.llm_client)
        # Normalized name -> official name, used to correct class names returned by the LLM
        self._normalized_classes = {self._normalize_class_name(name): name for name in self.target_classes}

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        """Lowercases, turns '&' into 'and' and keeps only words, e.g. 'Billing & Invoicing' -> 'billing and invoicing'."""
        name = name.lower().replace("&", " and ")
        return " ".join(re.findall(r"[a-z0-9]+", name))

    def _match_class_name(self, name: str) -> Optional[str]:
        """
        Maps a class name returned by the LLM to an official class name, or None if there is no safe match:
        1. Exact match.
        2. Same words ignoring case / punctuation ('billing and invoicing' -> 'Billing & Invoicing').
        3. Whole-word partial match to exactly one class ('Billing' / 'Account Security' -> official name).
        4. Close spelling match ('Techincal Support / Bug Report' -> 'Technical Support / Bug Report').
        """
        if name in self.target_classes:
            return name

        normalized = self._normalize_class_name(name)
        if not normalized:
            return None
        if normalized in self._normalized_classes:
            return self._normalized_classes[normalized]

        # Compare word sets ('and' ignored): LLM name inside a class name, or class name inside LLM name
        words = set(normalized.split()) - {"and"}
        partial_matches = [
            official for norm, official in self._normalized_classes.items()
            if words and (words <= set(norm.split()) or set(norm.split()) - {"and"} <= words)
        ]
        if len(partial_matches) == 1:
            return partial_matches[0]

        close = difflib.get_close_matches(normalized, list(self._normalized_classes), n=1, cutoff=0.8)
        return self._normalized_classes[close[0]] if close else None

    def _validate_class_names(self, predictions: List[PredictedClass]) -> List[PredictedClass]:
        """
        Keeps only predictions whose class is one of the official classes (correcting near-miss names).
        Unknown classes are dropped. If two predictions map to the same class, the higher confidence wins.
        """
        validated: Dict[str, PredictedClass] = {}
        for pred in predictions:
            official = self._match_class_name(pred.class_name)
            if official is None:
                logger.warning(f"Dropping prediction with unknown class name '{pred.class_name}'")
                continue
            if official != pred.class_name:
                logger.info(f"Corrected class name '{pred.class_name}' -> '{official}'")
                pred = pred.model_copy(update={"class_name": official})

            existing = validated.get(official)
            if existing is None or pred.confidence > existing.confidence:
                validated[official] = pred

        if predictions and not validated:
            logger.warning("No valid class names returned by the LLM; email will require human review.")
        return list(validated.values())

    def _fill_missing_classes(self, predictions: List[PredictedClass]) -> List[PredictedClass]:
        """
        Returns one prediction per official class, in the official class order.
        Classes the LLM did not score get confidence 0.0.
        """
        by_class = {pred.class_name: pred for pred in predictions}
        missing = [name for name in self.target_classes if name not in by_class]
        if missing:
            logger.info(f"LLM did not score {len(missing)} classes, setting them to 0.0: {missing}")

        return [
            by_class.get(name) or PredictedClass(
                class_name=name,
                confidence=0.0,
                reasoning="Not scored by the model; defaulted to 0.0.",
            )
            for name in self.target_classes
        ]

    def prepare_email_for_llm(self, email: EmailMessage) -> EmailMessage:
        """
        Reads the email body and attachments and returns a copy that is ready for the LLM.
        Attachments without extracted_text are read in full from their file (content bytes or file_path).
        Nothing is summarized or shortened. The original email object is not modified.
        """
        prepared_attachments = []
        for att in email.attachments:
            text = att.extracted_text or self.attachment_extractor.extract_text(att)
            prepared_attachments.append(att.model_copy(update={"extracted_text": text}))

        return email.model_copy(update={"attachments": prepared_attachments})

    def build_llm_input(self, email: EmailMessage) -> str:
        """
        Returns the final text (email details, body and full attachment contents)
        that is sent to the LLM for classification.
        """
        return self._build_user_prompt(self.prepare_email_for_llm(email))

    def _build_system_prompt(self, similar_feedbacks: List[Dict[str, Any]]) -> str:
        """
        Constructs the system prompt with 7 class guidelines and dynamic few-shot feedback examples.
        """
        class_definitions = "\n".join([f"- **{name}**: {desc}" for name, desc in self.target_classes.items()])

        prompt = f"""You are an enterprise AI email triage specialist.
Your objective is to analyze customer emails (including any attachment contents) and accurately classify them into one or more of the 7 official classes.

### The 7 Official Target Classes:
{class_definitions}

### Classification Guidelines:
1. Score EVERY one of the 7 classes: return exactly one entry per class, with a confidence score between 0.0 and 1.0.
   Classes that do not apply must still be returned with a low score (e.g. 0.0 - 0.1).
2. Multi-label is allowed: An email may belong to multiple categories (e.g., Billing AND Complaint), so several classes can have high scores.
3. Provide a clear, concise justification for each score.
4. If an email does not clearly fit standard categories, give 'General Inquiry / Feedback' the highest score.
5. Use the class names exactly as written above. Do not shorten, rename or invent classes.
"""

        # Append historical feedback examples retrieved from Vector DB (RAG)
        if similar_feedbacks:
            prompt += "\n### Relevant Historical Feedback & Corrections (Learn from past human decisions):\n"
            for idx, fb in enumerate(similar_feedbacks, 1):
                prompt += (
                    f"Example {idx}:\n"
                    f"  - Subject: {fb.get('subject', 'N/A')}\n"
                    f"  - Snippet: {fb.get('body_snippet', 'N/A')}\n"
                    f"  - Verified Human Ground-Truth Classes: {', '.join(fb.get('verified_classes', []))}\n"
                    f"  - Reviewer Note: {fb.get('reviewer_notes', 'N/A')}\n"
                )

        return prompt

    def _build_user_prompt(self, email: EmailMessage) -> str:
        """Formats the incoming email details and attachment contents."""
        prompt = f"Email ID: {email.id}\n"
        prompt += f"Sender: {email.sender}\n"
        prompt += f"Subject: {email.subject}\n"
        prompt += f"Body:\n{email.body}\n"

        if email.attachments:
            prompt += "\n--- Email Attachments ---\n"
            for i, att in enumerate(email.attachments, 1):
                prompt += f"Attachment #{i}: {att.filename} (Type: {att.content_type})\n"
                if att.extracted_text:
                    prompt += f"Extracted Text Content:\n{att.extracted_text}\n"
                else:
                    prompt += "Content: (Binary/Empty or no text extracted)\n"

        return prompt

    def classify_email(self, email: EmailMessage) -> EmailClassificationResult:
        """
        Classifies an email by:
        1. Chunking + embedding the email (LangChain + Gemini embeddings) and querying the Vector DB for similar past feedback.
        2. Calling LLM (Gemini, OpenAI or Azure OpenAI).
        3. Correcting / dropping class names that are not official classes.
        4. Adding a 0.0 score for any of the 7 classes the LLM did not score.
        5. Comparing predicted confidence scores with threshold.
        Attachments are read in full first (see prepare_email_for_llm).
        """
        email = self.prepare_email_for_llm(email)

        # Step 1: Chunk + embed the email (body and attachments) and retrieve the most similar
        # past human feedback from the Vector DB (RAG). The LLM itself still gets the full content.
        logger.info(f"Retrieving similar past feedback for email '{email.id}'...")
        similar_feedbacks = self.vector_store.search_similar_feedbacks(
            documents=self._email_documents(email),
            top_k=config.TOP_K_FEEDBACK_EXAMPLES,
        )

        # Step 2: Build prompts
        system_prompt = self._build_system_prompt(similar_feedbacks)
        user_prompt = self._build_user_prompt(email)

        # Step 3: Call LLM
        logger.info(f"Invoking LLM ({self.llm_client.provider}) for classification...")
        llm_response = self.llm_client.classify_email(system_prompt, user_prompt)

        # Step 4: Make sure every class name is one of the official classes
        predictions = self._validate_class_names(llm_response.predicted_classes)

        # Step 5: Make sure all 7 classes have a score
        predictions = self._fill_missing_classes(predictions)

        # Step 6: Validate against threshold (all classes above it are returned, highest first)
        passed_classes = sorted(
            [pred for pred in predictions if pred.confidence >= self.threshold],
            key=lambda pred: pred.confidence,
            reverse=True,
        )
        is_confident = len(passed_classes) > 0

        result = EmailClassificationResult(
            email_id=email.id,
            predictions=predictions,
            passed_threshold_classes=passed_classes,
            is_confident=is_confident,
            threshold_used=self.threshold,
            primary_intent=llm_response.primary_intent,
            summary=llm_response.summary,
            few_shot_feedback_used=similar_feedbacks,
        )

        return result

    def record_user_feedback(
        self,
        email: EmailMessage,
        classification_result: EmailClassificationResult,
        submission: UserFeedbackSubmission,
    ) -> int:
        """
        Stores user feedback in the database table and indexes the email + attachments
        (chunked and embedded) with the feedback in the Vector DB.
        """
        # Read attachment contents (reuses extracted_text when already present)
        email = self.prepare_email_for_llm(email)

        # 1. Format attachment summary
        att_summary = ", ".join([f"{a.filename} ({a.content_type})" for a in email.attachments]) or "None"

        # 2. Store in Relational DB Table
        predictions_dict = [p.model_dump() for p in classification_result.predictions]
        record_id = self.feedback_db.insert_feedback(
            email_id=email.id,
            email_subject=email.subject,
            email_body=email.body,
            attachments_summary=att_summary,
            model_predictions=predictions_dict,
            is_approved=submission.is_approved,
            corrected_classes=submission.corrected_classes,
            feedback_notes=submission.feedback_notes,
        )
        logger.info(f"Saved feedback to database table (ID: {record_id}) for email '{email.id}'")

        # 3. Chunk + embed the email and attachments; every chunk carries the verified feedback
        metadata = {
            "feedback_db_id": record_id,
            "email_id": email.id,
            "subject": email.subject,
            "body_snippet": email.body[:250],
            "verified_classes": submission.corrected_classes,
            "reviewer_notes": submission.feedback_notes or "Verified by reviewer",
            "is_approved": submission.is_approved,
        }
        chunk_count = self.vector_store.add_feedback(self._email_documents(email), metadata)
        logger.info(f"Indexed {chunk_count} feedback chunks into Vector DB for email '{email.id}'")

        return record_id

    @staticmethod
    def _email_documents(email: EmailMessage) -> List[Document]:
        """
        Converts an email into LangChain Documents for chunking: one for subject + body,
        and one per attachment that has extracted text.
        """
        documents = [
            Document(
                page_content=f"Subject: {email.subject}\n{email.body}",
                metadata={"source": "email_body"},
            )
        ]
        for att in email.attachments:
            if att.extracted_text:
                documents.append(Document(page_content=att.extracted_text, metadata={"source": att.filename}))
        return documents
