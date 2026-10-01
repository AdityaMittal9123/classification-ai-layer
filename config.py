import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# LLM Provider Selection: 'gemini', 'openai' or 'azure'
# =========================================================
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

# Option 0 (default): Google Gemini Settings - get a key at https://aistudio.google.com/apikey
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Gemini's OpenAI-compatible endpoint (used for chat / image reading)
GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
# Embeddings (RAG over past feedback) ALWAYS use this Gemini model via LangChain, whatever LLM_PROVIDER is
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Option 1: OpenAI Settings
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# Option 2: Azure OpenAI Settings
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

# =========================================================
# Classification Configuration
# =========================================================
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
TOP_K_FEEDBACK_EXAMPLES: int = int(os.getenv("TOP_K_FEEDBACK_EXAMPLES", "3"))
# RAG chunking (LangChain RecursiveCharacterTextSplitter), sizes in characters.
# Chunks are only used to find similar past feedback; the LLM always receives the full content.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
# Max chunks of a new email used for the similarity search (limits embedding time for huge attachments)
MAX_QUERY_CHUNKS: int = int(os.getenv("MAX_QUERY_CHUNKS", "50"))

# Deterministic LLM output: temperature 0 and a fixed seed
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_SEED: int = int(os.getenv("LLM_SEED", "42"))

DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATABASE_PATH: str = os.getenv("DATABASE_PATH", os.path.join(DATA_DIR, "email_classification.db"))
VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", os.path.join(DATA_DIR, "feedback_vectors.json"))

# =========================================================
# 7 Predefined Target Classes with Detailed Guidelines
# =========================================================
EMAIL_CLASSES: Dict[str, str] = {
    "Billing & Invoicing": (
        "Inquiries, disputes, requests for receipts, invoice discrepancies, tax questions, "
        "payment failures, or billing cycle details."
    ),
    "Technical Support / Bug Report": (
        "Software glitches, error codes, service outages, system integration bugs, crashes, "
        "or technical troubleshooting requests."
    ),
    "Account & Security Management": (
        "Password resets, 2FA issues, unauthorized access alerts, permission changes, "
        "SSO configuration, or profile updates."
    ),
    "Product Inquiry & Sales": (
        "Requests for demo, pricing quotes, feature availability, enterprise licensing, "
        "partnerships, or purchasing new services."
    ),
    "Complaint & Escalation": (
        "Dissatisfaction with service quality, SLA breaches, unhelpful customer reps, "
        "urgent executive escalation, or threats of legal/regulatory action."
    ),
    "Cancellation & Refund Request": (
        "Requests to terminate subscriptions, downgrade tiers, close accounts, or seek refunds "
        "for charged transactions."
    ),
    "General Inquiry / Feedback": (
        "General questions about office hours, public webinars, compliments, suggestions, "
        "or general comments that do not fit into the other specific categories."
    ),
}

AVAILABLE_CLASS_NAMES: List[str] = list(EMAIL_CLASSES.keys())
