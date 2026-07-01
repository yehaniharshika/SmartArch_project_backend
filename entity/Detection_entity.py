"""
SmartArch — Detection Entity
MySQL table: detections
One row per detected element (wall / door / window / room)
from YOLOv8 inference on a floor plan.
"""
from db import db


class Detection(db.Model):
    __tablename__ = "detections"

    id          = db.Column(db.Integer,     primary_key=True, autoincrement=True)

    # Foreign key → projects
    project_id  = db.Column(db.String(20),
                            db.ForeignKey("floor_plan_projects.id", ondelete="CASCADE"),
                            nullable=False, index=True)

    # YOLO output
    label       = db.Column(db.String(80),  nullable=False)   # "wall" | "door" | "window" | room name
    confidence  = db.Column(db.Float,       nullable=False)   # 0.0 – 1.0

    # Bounding box (pixels)
    x1          = db.Column(db.Float, nullable=False)
    y1          = db.Column(db.Float, nullable=False)
    x2          = db.Column(db.Float, nullable=False)
    y2          = db.Column(db.Float, nullable=False)

    # Computed real-world dimensions
    width_m     = db.Column(db.Float, default=0.0)
    height_m    = db.Column(db.Float, default=0.0)
    area_sqm    = db.Column(db.Float, default=0.0)
    area_sqft   = db.Column(db.Float, default=0.0)
    perimeter_m = db.Column(db.Float, default=0.0)

    # OCR-matched text inside this bounding box
    ocr_label     = db.Column(db.String(255), nullable=True)  # e.g. "Bedroom 1"
    ocr_dimension = db.Column(db.String(100), nullable=True)  # e.g. "3.5m"

    def to_dict(self):
        return {
            "label":         self.label,
            "confidence":    round((self.confidence or 0) * 100, 1),
            "bbox": {
                "x1": self.x1, "y1": self.y1,
                "x2": self.x2, "y2": self.y2,
            },
            "width_m":       round(self.width_m    or 0, 3),
            "height_m":      round(self.height_m   or 0, 3),
            "area_sqm":      round(self.area_sqm   or 0, 3),
            "area_sqft":     round(self.area_sqft  or 0, 3),
            "perimeter_m":   round(self.perimeter_m or 0, 3),
            "ocr_label":     self.ocr_label,
            "ocr_dimension": self.ocr_dimension,
        }

    def __repr__(self):
        return f"<Detection id={self.id} label={self.label} conf={self.confidence:.2f}>"
