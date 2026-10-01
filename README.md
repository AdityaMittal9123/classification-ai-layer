# LLM Email Classification Layer with Multi-Label, Threshold Validation & Feedback RAG

This module implements an enterprise-ready LLM classification pipeline for emails fetched from a database. It features multi-label classification across 7 target categories, attachment handling, confidence scoring with configurable thresholds, a human-in-the-loop feedback database, and vector embeddings for dynamic few-shot in-context learning.

---

## 🌟 Key Features

1. **Dual LLM Provider Support**:
   - **Direct OpenAI**: Uses `OpenAI` client (`gpt-4o`, `text-embedding-3-small`).
   - **Azure OpenAI**: Uses `AzureOpenAI` client with enterprise endpoints and deployments.
   - Switchable via `LLM_PROVIDER=openai` or `LLM_PROVIDER=azure` in `.env` or CLI.

2. **Multi-Class & Multi-Label Classification**:
   - Emails can be assigned one or more of **7 predefined enterprise classes**:
     1. `Billing & Invoicing`
     2. `Technical Support / Bug Report`
     3. `Account & Security Management`
     4. `Product Inquiry & Sales`
     5. `Complaint & Escalation`
     6. `Cancellation & Refund Request`
     7. `General Inquiry / Feedback`
   - Each predicted class includes a **confidence score (0.0 to 1.0)** and justification.

3. **Attachment Support**:
   - Captures attachment metadata (file names, MIME types) and parsed text (OCR / PDF extraction) to inform the LLM's classification.

4. **Confidence Threshold & Review Gate**:
   - Compares predicted scores against a configurable `CONFIDENCE_THRESHOLD` (e.g. `0.75`).
   - Flags predictions as `CONFIDENT PREDICTION` (auto-triaged) or `REQUIRES HUMAN REVIEW`.

5. **Human Feedback Loop & Vector Database (RAG)**:
   - Stores user reviews and corrections into a relational database table (`email_feedbacks`).
   - Computes vector embeddings of the verified email + reviewer notes and indexes them in a **Vector DB**.
   - Subsequent emails automatically retrieve top-$k$ relevant past corrections to provide dynamic few-shot context to the LLM.

---

## 📂 Project Structure

```text
email_clasification/
├── .env.example             # Configuration template for OpenAI and Azure OpenAI
├── .gitignore               # Git ignore rules for cache and virtual environments
├── config.py                # Environment variables, classes, and threshold settings
├── schemas.py               # Pydantic data models for emails, attachments, and outputs
├── llm_client.py            # Unified client for OpenAI, Azure OpenAI, & mock mode
├── feedback_db.py           # SQLite relational database manager for feedback logs
├── vector_store.py          # Vector database with cosine similarity search
├── classifier_pipeline.py   # Core classification & feedback RAG pipeline
├── demo.py                  # End-to-end runnable demonstration script
├── requirements.txt         # Dependencies
├── README.md                # Documentation and guide
└── data/                    # Generated at runtime
    ├── email_classification.db  # Relational DB table storing human feedback
    └── vector_store.json        # Vector DB store of feedback embeddings
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

#### Option A: Direct OpenAI
```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CONFIDENCE_THRESHOLD=0.75
```

#### Option B: Azure OpenAI
```ini
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
CONFIDENCE_THRESHOLD=0.75
```

> **Note**: If API keys are not provided, the pipeline automatically runs in a simulated **Mock Mode** so you can test the entire workflow end-to-end immediately without cost.

---

## 🧪 Running the Demo

Execute the demo script to test all scenarios:
```bash
# Test with OpenAI
python demo.py openai

# Test with Azure OpenAI
python demo.py azure
```

### What the demo tests:
1. **Scenario 1**: Ingests an email with two attachments (`Invoice_INV-2026-88.pdf` and `error_log_timeout.txt`) and classifies it.
2. **Scenario 2**: Simulates a human reviewer reviewing and submitting verified ground-truth labels and notes, persisting to DB and Vector DB.
3. **Scenario 3**: Ingests a new incoming email with similar phrasing and demonstrates Vector DB RAG retrieving the prior human feedback as few-shot in-context learning.
4. **Scenario 4**: Displays the persisted SQLite database records.

---

## 🛠️ Code Usage Example

```python
from schemas import EmailMessage, EmailAttachment, UserFeedbackSubmission
from classifier_pipeline import EmailClassificationPipeline

# 1. Initialize Pipeline
pipeline = EmailClassificationPipeline(provider="openai", confidence_threshold=0.75)

# 2. Email fetched from DB
email = EmailMessage(
    id="EML-5001",
    sender="customer@company.com",
    subject="Overcharged on my subscription",
    body="I was charged twice this month. Please issue a refund.",
    attachments=[
        EmailAttachment(
            filename="receipt.pdf",
            content_type="application/pdf",
            extracted_text="Invoice #1234: Total $200.00 (Charged twice)"
        )
    ]
)

# 3. Classify with LLM (retrieves relevant Vector DB feedback automatically)
result = pipeline.classify_email(email)

print(f"Confident: {result.is_confident}")
for pred in result.passed_threshold_classes:
    print(f"Class: {pred.class_name}, Confidence: {pred.confidence}")

# 4. User provides feedback (stores in DB & Vector DB)
feedback = UserFeedbackSubmission(
    email_id=email.id,
    is_approved=True,
    corrected_classes=["Billing & Invoicing", "Cancellation & Refund Request"],
    feedback_notes="Disputed charge requiring billing reversal."
)
pipeline.record_user_feedback(email, result, feedback)
```
