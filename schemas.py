from typing import List, Optional
from pydantic import BaseModel, Field

import config


class EmailAttachment(BaseModel):
    """Represents an attachment associated with an email."""
    filename: str
    content_type: Optional[str] = "application/octet-stream"
    content: Optional[bytes] = None  # Raw file bytes (used when extracted_text is not provided)
    file_path: Optional[str] = None  # Path to the file on disk (used when content is not provided)
    extracted_text: Optional[str] = None  # Full text extracted from the file


class EmailMessage(BaseModel):
    """Email retrieved from the database."""
    id: str
    sender: str
    recipient: Optional[str] = None
    subject: str
    body: str
    attachments: List[EmailAttachment] = Field(default_factory=list)
    received_at: Optional[str] = None

    def get_full_text(self) -> str:
        """Combine email fields and attachment content into searchable text."""
        parts = [
            f"Subject: {self.subject}",
            f"Sender: {self.sender}",
            f"Body:\n{self.body}"
        ]
        if self.attachments:
            parts.append("\nAttachments:")
            for att in self.attachments:
                att_info = f"- File: {att.filename} ({att.content_type})"
                if att.extracted_text:
                    att_info += f"\n  Extracted Content: {att.extracted_text}"
                parts.append(att_info)
        return "\n".join(parts)


class PredictedClass(BaseModel):
    """A single predicted class with confidence and rationale."""
    class_name: str = Field(
        description="Name of the predicted class from the 7 target classes",
        # Restricts structured output (Gemini / OpenAI / Azure) to the official class names
        json_schema_extra={"enum": config.AVAILABLE_CLASS_NAMES},
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="Brief justification for assigning this class")


class LLMClassificationResponse(BaseModel):
    """Structured response schema returned by the LLM."""
    primary_intent: str = Field(description="Main user intention in 1-2 sentences")
    predicted_classes: List[PredictedClass] = Field(
        description="Exactly one entry for each of the 7 classes, each with its own confidence score (multi-label)"
    )
    summary: str = Field(description="Concise summary of the email content and attachments")


class EmailClassificationResult(BaseModel):
    """Pipeline classification output including threshold validation."""
    email_id: str
    predictions: List[PredictedClass]
    passed_threshold_classes: List[PredictedClass]
    is_confident: bool  # True if at least one predicted class >= threshold
    threshold_used: float
    primary_intent: str
    summary: str
    few_shot_feedback_used: List[dict] = Field(default_factory=list)


class UserFeedbackSubmission(BaseModel):
    """Ground truth feedback provided by human reviewer."""
    email_id: str
    is_approved: bool
    corrected_classes: List[str]
    feedback_notes: Optional[str] = None
