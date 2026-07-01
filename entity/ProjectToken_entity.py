"""
SmartArch — Share Token Entity
MySQL table: tokens
One row per project — the JWT token used to share
the floor plan chatbot link with the client.

Client URL: http://localhost:5173/chat/<token>
No login required to access the chat page.
"""
from datetime import datetime
from db import db


class ProjectToken(db.Model):
    __tablename__ = "tokens"

    id          = db.Column(db.Integer,    primary_key=True, autoincrement=True)

    # Foreign key → projects (one-to-one)
    project_id  = db.Column(db.String(20),
                            db.ForeignKey("floor_plan_projects.id", ondelete="CASCADE"),
                            nullable=False, unique=True)

    # JWT share token string
    token       = db.Column(db.Text,     nullable=False)
    expires_at  = db.Column(db.DateTime, nullable=True)
    created_at  = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self):
        return {
            "token":      self.token,
            "expires_at": str(self.expires_at) if self.expires_at else None,
            "created_at": str(self.created_at),
            "is_expired": self.is_expired(),
        }

    def __repr__(self):
        return f"<ProjectToken id={self.id} project_id={self.project_id}>"
