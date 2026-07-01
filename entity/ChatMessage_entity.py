from datetime import datetime
from db import db


class ChatMessage(db.Model):
    __tablename__ = "chat_logs"

    id         = db.Column(db.Integer, primary_key=True, autoincrement=True)
    project_id = db.Column(
        db.String(20),
        db.ForeignKey("floor_plan_projects.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    query      = db.Column(db.Text, nullable=False)
    answer     = db.Column(db.Text, nullable=False)
    language   = db.Column(db.String(10), nullable=True, default="en")
    model_used = db.Column(db.String(50), nullable=True)
    tokens_used= db.Column(db.Integer,   nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":          self.id,
            "project_id":  self.project_id,
            "query":       self.query,
            "answer":      self.answer,
            "language":    self.language,
            "model_used":  self.model_used,
            "tokens_used": self.tokens_used,
            "created_at":  str(self.created_at),
        }

    def __repr__(self):
        return f"<ChatMessage id={self.id} project={self.project_id}>"