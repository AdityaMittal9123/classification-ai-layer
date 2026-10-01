import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import config


class FeedbackDatabase:
    """
    Relational Database manager for tracking user feedback on email classifications.
    Uses SQLite for lightweight, production-ready relational storage.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DATABASE_PATH
        parent_dir = os.path.dirname(os.path.abspath(self.db_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.init_db()

    def init_db(self):
        """Create the user_feedback table if it doesn't already exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_feedbacks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id TEXT NOT NULL,
                    email_subject TEXT,
                    email_body TEXT,
                    attachments_summary TEXT,
                    model_predictions TEXT, -- JSON string of predicted classes
                    is_approved INTEGER NOT NULL, -- 1 for True, 0 for False
                    corrected_classes TEXT NOT NULL, -- JSON string of human-verified classes
                    feedback_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

    def insert_feedback(
        self,
        email_id: str,
        email_subject: str,
        email_body: str,
        attachments_summary: str,
        model_predictions: List[Dict[str, Any]],
        is_approved: bool,
        corrected_classes: List[str],
        feedback_notes: Optional[str] = None,
    ) -> int:
        """
        Record user feedback into the database table.
        Returns the inserted record ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO email_feedbacks (
                    email_id,
                    email_subject,
                    email_body,
                    attachments_summary,
                    model_predictions,
                    is_approved,
                    corrected_classes,
                    feedback_notes,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    email_id,
                    email_subject,
                    email_body,
                    attachments_summary,
                    json.dumps(model_predictions),
                    1 if is_approved else 0,
                    json.dumps(corrected_classes),
                    feedback_notes or "",
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_feedbacks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch historical feedback records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM email_feedbacks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
