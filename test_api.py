"""
Standalone FastAPI app for TESTING the email classification pipeline.

Run from this folder:
    .venv\\Scripts\\python.exe test_api.py

Then open http://127.0.0.1:8001 in your browser.

- Uses separate test data files (data/test_feedback.db, data/test_feedback_vectors.json),
  so testing never touches the real database.
- Runs on port 8001, so it does not clash with the main API (api.py on port 8000).
"""
import os
import uuid
import logging
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

import config
from schemas import EmailMessage, EmailAttachment, EmailClassificationResult
from classifier_pipeline import EmailClassificationPipeline

TEST_PORT = 8001
TEST_DB_PATH = os.path.join(config.DATA_DIR, "test_feedback.db")
TEST_VECTOR_STORE_PATH = os.path.join(config.DATA_DIR, "test_feedback_vectors.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EmailTestAPI")

app = FastAPI(
    title="Email Classification - TEST API",
    description="Test app for the email classification pipeline. Uses separate test data files.",
)

pipeline = EmailClassificationPipeline(db_path=TEST_DB_PATH, vector_store_path=TEST_VECTOR_STORE_PATH)


def _build_email(
    sender: str,
    subject: str,
    body: str,
    email_id: Optional[str],
    attachments: List[UploadFile],
) -> EmailMessage:
    """Builds an EmailMessage from form fields and uploaded files."""
    return EmailMessage(
        id=email_id or f"TEST-{uuid.uuid4().hex[:8]}",
        sender=sender,
        subject=subject,
        body=body,
        attachments=[
            EmailAttachment(
                filename=upload.filename,
                content_type=upload.content_type or "application/octet-stream",
                content=upload.file.read(),
            )
            for upload in attachments
            if upload.filename  # browsers send an empty file entry when nothing is selected
        ],
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def test_page():
    """Simple web form to test classification in the browser."""
    return TEST_PAGE_HTML


@app.get("/health")
def health():
    """Shows which LLM provider is used and whether it is in mock mode."""
    return {
        "status": "ok",
        "provider": pipeline.llm_client.provider,
        "mock_mode": pipeline.llm_client.is_mock,
        "chat_model": pipeline.llm_client.chat_model,
        "threshold": pipeline.threshold,
        "embedding_model": pipeline.vector_store.embedding_model,
        "embeddings_mock_mode": pipeline.vector_store.is_mock,
        "feedback_chunks_stored": len(pipeline.vector_store.store.store),
        "classes": config.AVAILABLE_CLASS_NAMES,
    }


@app.post("/classify", response_model=EmailClassificationResult)
def classify(
    sender: str = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    email_id: Optional[str] = Form(None),
    attachments: List[UploadFile] = File(default=[]),
):
    """Classifies an email: scores all 7 classes and returns every class at or above the threshold."""
    email = _build_email(sender, subject, body, email_id, attachments)
    try:
        return pipeline.classify_email(email)
    except Exception as e:
        logger.exception(f"Classification failed for email '{email.id}'")
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")


@app.post("/preview-llm-input")
def preview_llm_input(
    sender: str = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    email_id: Optional[str] = Form(None),
    attachments: List[UploadFile] = File(default=[]),
):
    """
    Shows the exact email text the LLM would receive (body + full attachment contents),
    without classifying. Useful to check attachment extraction.
    """
    email = _build_email(sender, subject, body, email_id, attachments)
    prepared = pipeline.prepare_email_for_llm(email)
    return {
        "email_id": email.id,
        "attachments": [
            {
                "filename": att.filename,
                "extracted_chars": len(att.extracted_text or ""),
                "text_found": bool(att.extracted_text),
            }
            for att in prepared.attachments
        ],
        "llm_input": pipeline._build_user_prompt(prepared),
    }


@app.get("/classify-sample", response_model=EmailClassificationResult)
def classify_sample():
    """Classifies a built-in sample email (no input needed)."""
    email = EmailMessage(
        id="TEST-SAMPLE",
        sender="alex.murphy@omnicorp.com",
        subject="Wrong charge on bill #9921 and no response from tech team",
        body=(
            "Hi Support,\n\nWe were billed twice this month on statement #9921. Also, the sync server "
            "has been failing with error 504 since yesterday. This is completely unacceptable. "
            "Please reverse the charge or we will cancel our account.\n\nAlex"
        ),
        attachments=[
            EmailAttachment(
                filename="Statement_9921.txt",
                content_type="text/plain",
                content=b"Statement #9921 | Amount Charged: $800.00 x 2 (Duplicate)",
            )
        ],
    )
    return pipeline.classify_email(email)


TEST_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Email Classification Test</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 22px; }
  label { display: block; margin-top: 12px; font-weight: 600; }
  input[type=text], textarea { width: 100%; padding: 8px; box-sizing: border-box; font: inherit; }
  textarea { height: 140px; }
  button { margin: 16px 8px 0 0; padding: 10px 16px; font: inherit; cursor: pointer; }
  #status { margin-top: 16px; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }
  .pass { background: #e3f6e3; }
  pre { background: #f4f4f4; padding: 12px; white-space: pre-wrap; word-break: break-word; }
  .banner { padding: 8px 12px; border-radius: 4px; margin-bottom: 8px; }
  .mock { background: #fff3cd; }
  .real { background: #e3f6e3; }
</style>
</head>
<body>
<h1>Email Classification &ndash; Test</h1>
<div id="mode" class="banner">Checking LLM mode...</div>

<form id="form">
  <label>Sender <input type="text" name="sender" value="alex@omnicorp.com" required></label>
  <label>Subject <input type="text" name="subject" value="Wrong charge and 504 errors" required></label>
  <label>Body
    <textarea name="body" required>We were billed twice this month and the sync server keeps failing with error 504. This is unacceptable. Please refund us or we will cancel.</textarea>
  </label>
  <label>Attachments (PDF, Word, Excel, PowerPoint, images, text...)
    <input type="file" name="attachments" multiple>
  </label>
  <button type="submit" data-action="/classify">Classify</button>
  <button type="submit" data-action="/preview-llm-input">Preview LLM input</button>
</form>

<div id="status"></div>
<div id="result"></div>

<script>
fetch('/health').then(r => r.json()).then(h => {
  const el = document.getElementById('mode');
  el.className = 'banner ' + (h.mock_mode ? 'mock' : 'real');
  el.textContent = h.mock_mode
    ? 'MOCK MODE (' + h.provider + ' not reachable) - results are simulated, not real.'
    : 'LIVE: ' + h.provider + ' / ' + h.chat_model + ' - threshold ' + h.threshold;
});

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const action = e.submitter.dataset.action;
  const status = document.getElementById('status');
  const result = document.getElementById('result');
  result.innerHTML = '';
  status.textContent = 'Working... (with a local LLM this can take a minute)';
  const started = Date.now();
  try {
    const resp = await fetch(action, { method: 'POST', body: new FormData(e.target) });
    const data = await resp.json();
    const secs = ((Date.now() - started) / 1000).toFixed(1);
    if (!resp.ok) { status.textContent = 'Error ' + resp.status + ' (' + secs + 's)'; result.innerHTML = '<pre>' + esc(JSON.stringify(data, null, 2)) + '</pre>'; return; }
    status.textContent = 'Done in ' + secs + 's';
    result.innerHTML = action === '/classify' ? renderClassification(data) : renderPreview(data);
  } catch (err) {
    status.textContent = 'Request failed: ' + err;
  }
});

function renderClassification(r) {
  const passed = new Set(r.passed_threshold_classes.map(p => p.class_name));
  let html = '<h2>' + (r.is_confident ? 'Confident' : 'Requires human review') + '</h2>';
  html += '<p><b>Selected categories:</b> ' + (r.passed_threshold_classes.map(p => esc(p.class_name) + ' (' + p.confidence.toFixed(2) + ')').join(', ') || 'none') + '</p>';
  html += '<p><b>Primary intent:</b> ' + esc(r.primary_intent) + '</p>';
  html += '<p><b>Summary:</b> ' + esc(r.summary) + '</p>';
  html += '<table><tr><th>Class</th><th>Confidence</th><th>Passed (&ge; ' + r.threshold_used + ')</th><th>Reasoning</th></tr>';
  for (const p of r.predictions) {
    const ok = passed.has(p.class_name);
    html += '<tr class="' + (ok ? 'pass' : '') + '"><td>' + esc(p.class_name) + '</td><td>' + p.confidence.toFixed(2) + '</td><td>' + (ok ? 'Yes' : 'No') + '</td><td>' + esc(p.reasoning) + '</td></tr>';
  }
  html += '</table>';
  if (r.few_shot_feedback_used.length) {
    html += '<h3>Past feedback examples used</h3><ul>';
    for (const f of r.few_shot_feedback_used) html += '<li>' + esc(f.subject) + ' (similarity ' + f.similarity_score + ') &rarr; ' + esc((f.verified_classes || []).join(', ')) + '</li>';
    html += '</ul>';
  }
  html += '<details><summary>Raw JSON</summary><pre>' + esc(JSON.stringify(r, null, 2)) + '</pre></details>';
  return html;
}

function renderPreview(p) {
  let html = '<h2>Attachments</h2><table><tr><th>File</th><th>Text found</th><th>Chars</th></tr>';
  for (const a of p.attachments) html += '<tr><td>' + esc(a.filename) + '</td><td>' + (a.text_found ? 'Yes' : 'No') + '</td><td>' + a.extracted_chars + '</td></tr>';
  html += '</table><h2>Text sent to the LLM</h2><pre>' + esc(p.llm_input) + '</pre>';
  return html;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"\nTest app running: open http://127.0.0.1:{TEST_PORT} in your browser")
    print(f"API docs:         http://127.0.0.1:{TEST_PORT}/docs")
    print("Stop with CTRL+C\n")
    uvicorn.run(app, host="127.0.0.1", port=TEST_PORT)
