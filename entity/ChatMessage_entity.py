from datetime import datetime
from db import db


class ChatMessage(db.Model):
    __tablename__ = "chat_logs"

    id          = db.Column(db.Integer,    primary_key=True, autoincrement=True)

    # Foreign key → projects
    project_id  = db.Column(db.String(20),
                            db.ForeignKey("floor_plan_projects.id", ondelete="CASCADE"),
                            nullable=False, index=True)

    # Chat content
    query       = db.Column(db.Text, nullable=False)   # client's question
    answer      = db.Column(db.Text, nullable=False)   # AI answer

    # Optional: language detection (for future multilingual support)
    language    = db.Column(db.String(10), nullable=True, default="en")

    # Response metadata
    model_used  = db.Column(db.String(50), nullable=True)   # "gpt-4o"
    tokens_used = db.Column(db.Integer,    nullable=True)

    created_at  = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":          self.id,
            "query":       self.query,
            "answer":      self.answer,
            "language":    self.language,
            "model_used":  self.model_used,
            "tokens_used": self.tokens_used,
            "created_at":  str(self.created_at),
        }

    def __repr__(self):
        return f"<ChatMessage id={self.id} project={self.project_id}>"
