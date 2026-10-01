"""
REST API for the email classification pipeline (FastAPI, open source).

Run from this folder:
    uvicorn api:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""
import uuid
import logging
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

import config
from schemas import EmailMessage, EmailAttachment, EmailClassificationResult
from classifier_pipeline import EmailClassificationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EmailAPI")

app = FastAPI(
    title="Email Classification API",
    description=(
        "Classifies an email (body + attachments) into the 7 official classes. "
        "Every class gets a confidence score; all classes at or above the threshold are returned."
    ),
)

# Created once at startup and reused for every request
pipeline = EmailClassificationPipeline()


@app.get("/", include_in_schema=False)
def root():
    """Opens the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Shows whether the API is up and which LLM provider is in use."""
    return {
        "status": "ok",
        "provider": pipeline.llm_client.provider,
        "mock_mode": pipeline.llm_client.is_mock,
        "threshold": pipeline.threshold,
        "classes": config.AVAILABLE_CLASS_NAMES,
    }


@app.post("/classify", response_model=EmailClassificationResult)
def classify(
    sender: str = Form(..., description="Sender email address"),
    subject: str = Form(..., description="Email subject"),
    body: str = Form(..., description="Email body text"),
    email_id: Optional[str] = Form(None, description="Email ID (generated if not provided)"),
    recipient: Optional[str] = Form(None),
    received_at: Optional[str] = Form(None),
    attachments: Optional[List[UploadFile]] = File(default=[], description="Attachment files (PDF, Office, images, text...)"),
):
    """
    Classifies one email. Send as multipart/form-data with the email fields and any number of attachment files.

    Returns scores for all 7 classes (`predictions`), the classes at or above the threshold
    (`passed_threshold_classes`, highest first), `is_confident`, the summary and the past feedback examples used.
    """
    email = EmailMessage(
        id=email_id or f"EML-{uuid.uuid4().hex[:8]}",
        sender=sender,
        recipient=recipient,
        subject=subject,
        body=body,
        received_at=received_at,
        attachments=[
            EmailAttachment(
                filename=upload.filename or "attachment",
                content_type=upload.content_type or "application/octet-stream",
                content=upload.file.read(),
            )
            for upload in attachments
        ],
    )

    try:
        return pipeline.classify_email(email)
    except Exception as e:
        logger.exception(f"Classification failed for email '{email.id}'")
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")
