import sys
import json
import logging
from schemas import EmailMessage, EmailAttachment, UserFeedbackSubmission
from classifier_pipeline import EmailClassificationPipeline
import config

# Configure clean logging output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("EmailDemo")


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()} ")
    print("=" * 70)


def display_classification_result(result, threshold: float):
    print(f"\n[EMAIL ID]: {result.email_id}")
    print(f"[PRIMARY INTENT]: {result.primary_intent}")
    print(f"[SUMMARY]: {result.summary}")
    print(f"\n[CLASSIFICATION PREDICTIONS] (Threshold = {threshold}):")
    
    for p in result.predictions:
        is_pass = p.confidence >= threshold
        status = "[PASSED THRESHOLD]" if is_pass else "[BELOW THRESHOLD]"
        print(f"   * [{p.class_name}]: Score = {p.confidence:.2f} -> {status}")
        print(f"     Reasoning: {p.reasoning}")

    final_status = "CONFIDENT PREDICTION" if result.is_confident else "REQUIRES HUMAN REVIEW"
    print(f"\n[FINAL STATUS]: {final_status}")
    
    if result.few_shot_feedback_used:
        print(f"\n[IN-CONTEXT LEARNING] (Retrieved {len(result.few_shot_feedback_used)} relevant past human feedback from Vector DB):")
        for fb in result.few_shot_feedback_used:
            print(f"   - Match: \"{fb.get('subject')}\" (Cosine Sim: {fb.get('similarity_score')})")
            print(f"     Ground-truth labels: {fb.get('verified_classes')}")
            print(f"     Reviewer comment: {fb.get('reviewer_notes')}")


def main():
    # Allow overriding provider via CLI: python demo.py [gemini|openai|azure]
    selected_provider = sys.argv[1].lower() if len(sys.argv) > 1 else config.LLM_PROVIDER
    
    print_separator("LLM Email Classification Layer Initializing")
    print(f"Selected Provider : {selected_provider.upper()}")
    print(f"Target Classes    : {len(config.AVAILABLE_CLASS_NAMES)} classes defined")
    print(f"Threshold         : {config.CONFIDENCE_THRESHOLD}")

    # Initialize Pipeline
    pipeline = EmailClassificationPipeline(
        provider=selected_provider,
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )

    # -------------------------------------------------------------
    # SCENARIO 1: First Email Arrives (Multi-label + Attachment)
    # -------------------------------------------------------------
    print_separator("Scenario 1: Processing Email #1 (From Database)")
    
    email_1 = EmailMessage(
        id="EML-1001",
        sender="sarah.connor@cyberdyne.com",
        recipient="support@enterprise.com",
        subject="Dispute regarding Invoice #INV-2026-88 and lack of support",
        body=(
            "Hello,\n\n"
            "I noticed an unexpected charge of $1,450 on my latest monthly bill. "
            "I have attached the invoice for your reference. Furthermore, our team has "
            "reached out three times regarding the recurring API gateway timeout error and "
            "no one responded. This level of service is unacceptable. If this is not sorted out, "
            "we want a full refund and will terminate our subscription immediately.\n\n"
            "Regards,\nSarah"
        ),
        attachments=[
            EmailAttachment(
                filename="Invoice_INV-2026-88.pdf",
                content_type="application/pdf",
                extracted_text="INVOICE #INV-2026-88 | Date: 2026-09-15 | Item: Enterprise API Tier | Amount Due: $1,450.00 | Status: Past Due"
            ),
            EmailAttachment(
                filename="error_log_timeout.txt",
                content_type="text/plain",
                extracted_text="HTTP 504 Gateway Timeout: upstream request failed after 30000ms on /v2/data/sync"
            )
        ]
    )

    print(f"Incoming Email Subject: {email_1.subject}")
    print(f"Attachments detected  : {[a.filename for a in email_1.attachments]}")

    result_1 = pipeline.classify_email(email_1)
    display_classification_result(result_1, pipeline.threshold)

    # -------------------------------------------------------------
    # SCENARIO 2: Human Feedback Loop (Approved / Corrected by User)
    # -------------------------------------------------------------
    print_separator("Scenario 2: User / Reviewer Provides Ground-Truth Feedback")
    print("User reviews the classification result.")
    print("The reviewer confirms that Sarah is not just asking about billing, but also")
    print("submitting an escalation complaint and requesting cancellation/refund.\n")

    feedback_1 = UserFeedbackSubmission(
        email_id=email_1.id,
        is_approved=True,
        corrected_classes=[
            "Billing & Invoicing",
            "Complaint & Escalation",
            "Cancellation & Refund Request"
        ],
        feedback_notes=(
            "Customer had billing dispute, expressed strong dissatisfaction with support response, "
            "and explicitly threatened subscription cancellation and demanded refund."
        )
    )

    db_record_id = pipeline.record_user_feedback(
        email=email_1,
        classification_result=result_1,
        submission=feedback_1
    )
    print(f"[OK] Feedback saved to relational DB table 'email_feedbacks' with Record ID #{db_record_id}")
    print("[OK] Vector embedding generated and indexed into Vector DB.")

    # -------------------------------------------------------------
    # SCENARIO 3: Second Similar Email Arrives (In-Context RAG Triggered)
    # -------------------------------------------------------------
    print_separator("Scenario 3: Processing Email #2 (Vector DB RAG In Action)")
    
    email_2 = EmailMessage(
        id="EML-1002",
        sender="alex.murphy@omnicorp.com",
        recipient="support@enterprise.com",
        subject="Wrong charge on bill #9921 and no response from tech team",
        body=(
            "Hi Support,\n\n"
            "We were billed twice this month on statement #9921. Also, the sync server "
            "has been failing with error 504 since yesterday. This is completely unacceptable. "
            "Please reverse the charge or we will cancel our account.\n\n"
            "Alex"
        ),
        attachments=[
            EmailAttachment(
                filename="Statement_9921.pdf",
                content_type="application/pdf",
                extracted_text="Statement #9921 | Amount Charged: $800.00 x 2 (Duplicate)"
            )
        ]
    )

    print(f"Incoming Email Subject: {email_2.subject}")
    result_2 = pipeline.classify_email(email_2)
    display_classification_result(result_2, pipeline.threshold)

    # -------------------------------------------------------------
    # SCENARIO 4: Inspect Database Table Content
    # -------------------------------------------------------------
    print_separator("Scenario 4: Relational DB Table State")
    all_feedbacks = pipeline.feedback_db.get_feedbacks(limit=5)
    print(f"Total feedback records stored in DB table: {len(all_feedbacks)}")
    for row in all_feedbacks:
        print(f"- ID: {row['id']} | Email: {row['email_id']} | Verified Classes: {row['corrected_classes']}")
        print(f"  Notes: {row['feedback_notes']}")

    print_separator("Demo Run Completed Successfully")
    print("To test with live LLMs, set GEMINI_API_KEY (or OpenAI/Azure credentials) in .env.")


if __name__ == "__main__":
    main()
