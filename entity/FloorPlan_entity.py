from datetime import datetime
from db import db
import uuid

def _gen_project_id():
    return "PRJ-" + uuid.uuid4().hex[:6].upper()


class FloorPlan(db.Model):
    __tablename__ = "floor_plan_projects"

    id      = db.Column(db.String(20), primary_key=True, default=_gen_project_id)
    user_id = db.Column(db.Integer,
                        db.ForeignKey("users.id", ondelete="CASCADE"),
                        nullable=False, index=True)

    project_name      = db.Column(db.String(255), nullable=False, default="Untitled Project")

    original_filename = db.Column(db.String(255), nullable=False)
    file_path         = db.Column(db.String(500), nullable=False)   # original upload
    image_path        = db.Column(db.String(500), nullable=True)    # PDF→PNG converted
    annotated_image   = db.Column(db.String(500), nullable=True)    # YOLO annotated

    status        = db.Column(
        db.Enum("pending", "processing", "ready", "error"),
        nullable=False, default="processing"
    )
    error_message = db.Column(db.Text, nullable=True)

    # Image dimensions 
    image_width_px  = db.Column(db.Integer, default=0)
    image_height_px = db.Column(db.Integer, default=0)

    pixels_per_meter = db.Column(db.Float,     default=50.0)
    scale_method     = db.Column(db.String(30), default="default")
    scale_confidence = db.Column(db.Float,     default=0.30)

    # Area results
    total_area_sqm  = db.Column(db.Float,   default=0.0)
    total_area_sqft = db.Column(db.Float,   default=0.0)

    # Detection counts
    wall_count       = db.Column(db.Integer, default=0)
    door_count       = db.Column(db.Integer, default=0)
    window_count     = db.Column(db.Integer, default=0)
    room_count       = db.Column(db.Integer, default=0)
    total_detections = db.Column(db.Integer, default=0)

    # GPT-4o analysis
    summary       = db.Column(db.Text, nullable=True)
    gpt_room_list = db.Column(db.JSON, nullable=True)

    # Processing time (seconds)
    processing_time = db.Column(db.Float, default=0.0)

    # Share token
    share_token      = db.Column(db.String(512), nullable=True)
    share_expires_at = db.Column(db.DateTime,   nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow,
                           onupdate=datetime.utcnow)

    # Relationships 
    detections   = db.relationship("Detection",   backref="project",
                                    lazy=True,    cascade="all, delete-orphan")
    ocr_result   = db.relationship("OCRResult",   backref="project",
                                    uselist=False, cascade="all, delete-orphan")
    chat_messages = db.relationship("ChatMessage", backref="project",
                                    lazy=True,    cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "project_id":        self.id,
            "project_name":      self.project_name,
            "user_id":           self.user_id,
            "original_filename": self.original_filename,
            "annotated_image":   self.annotated_image,
            "status":            self.status,
            "error_message":     self.error_message,
            "image_width_px":    self.image_width_px,
            "image_height_px":   self.image_height_px,
            "scale": {
                "pixels_per_meter": self.pixels_per_meter,
                "method":           self.scale_method,
                "confidence":       self.scale_confidence,
            },
            "total_area_sqm":    round(self.total_area_sqm  or 0, 2),
            "total_area_sqft":   round(self.total_area_sqft or 0, 2),
            "wall_count":        self.wall_count,
            "door_count":        self.door_count,
            "window_count":      self.window_count,
            "room_count":        self.room_count,
            "total_detections":  self.total_detections,
            "summary":           self.summary,
            "gpt_room_list":     self.gpt_room_list or [],
            "processing_time":   round(self.processing_time or 0, 2),
            "share_token":       self.share_token,
            "created_at":        str(self.created_at),
        }

    def __repr__(self):
        return f"<FloorPlan {self.id} '{self.project_name}' status={self.status}>"
