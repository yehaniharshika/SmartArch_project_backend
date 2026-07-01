from datetime import datetime
from db import db

class User(db.Model):
    __tablename__ = "users"

    id            = db.Column(db.Integer,     primary_key=True, autoincrement=True)
    full_name     = db.Column(db.String(150), nullable=False)
    email         = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role          = db.Column(db.String(50),  nullable=False, default="architect")  
    is_active     = db.Column(db.Boolean,     nullable=False, default=True)
    created_at    = db.Column(db.DateTime,    nullable=False, default=datetime.utcnow)
    updated_at    = db.Column(db.DateTime,    nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = db.Column(db.DateTime,    nullable=True)
    reset_token            = db.Column(db.String(512), nullable=True)
    reset_token_expires_at = db.Column(db.DateTime,    nullable=True)

    projects = db.relationship(
        "FloorPlan",
        backref="owner",
        lazy=True,
        cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id":           self.id,
            "full_name":    self.full_name,
            "email":        self.email,
            "role":         self.role, 
            "is_active":    self.is_active,
            "created_at":   str(self.created_at),
            "last_login_at":str(self.last_login_at) if self.last_login_at else None,
        }

    def __repr__(self):
        return f"<User id={self.id} email={self.email} role={self.role}>"