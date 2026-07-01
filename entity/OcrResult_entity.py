import json
from db import db


class OCRResult(db.Model):
    __tablename__ = "ocr_results"

    id          = db.Column(db.Integer,    primary_key=True, autoincrement=True)

    # Foreign key → projects (one-to-one)
    project_id  = db.Column(db.String(20),
                            db.ForeignKey("floor_plan_projects.id", ondelete="CASCADE"),
                            nullable=False, unique=True)

    # Scale detection result
    pixels_per_meter  = db.Column(db.Float,      default=50.0)
    scale_method      = db.Column(db.String(30),  default="default")
    scale_reference   = db.Column(db.String(100), nullable=True)  # e.g. "1:100"

    # Extracted text data (stored as JSON strings)
    room_labels = db.Column(db.Text, default="[]")
    # [{"text":"Bedroom 1","confidence":0.95,"center_x":120,"center_y":80}]

    dimensions  = db.Column(db.Text, default="[]")
    # [{"text":"3.5m","confidence":0.92,"parsed_meters":3.5}]

    raw_texts   = db.Column(db.Text, default="[]")
    # Full list of all OCR results

    def to_dict(self):
        return {
            "scale": {
                "pixels_per_meter": self.pixels_per_meter,
                "method":           self.scale_method,
                "reference":        self.scale_reference,
            },
            "room_labels": json.loads(self.room_labels or "[]"),
            "dimensions":  json.loads(self.dimensions  or "[]"),
            "raw_texts":   json.loads(self.raw_texts   or "[]"),
        }

    def set_room_labels(self, data: list):
        self.room_labels = json.dumps(data, ensure_ascii=False)

    def set_dimensions(self, data: list):
        self.dimensions = json.dumps(data, ensure_ascii=False)

    def set_raw_texts(self, data: list):
        self.raw_texts = json.dumps(data, ensure_ascii=False)

    def __repr__(self):
        return f"<OCRResult id={self.id} project_id={self.project_id}>"
