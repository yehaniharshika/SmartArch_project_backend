"""
SmartArch — Room Entity
MySQL table: rooms
One row per parsed room (from room_parser_service + area_service).
Mirrors the Detection entity's pattern — one project has many rooms.
"""
from db import db


class Room(db.Model):
    __tablename__ = "rooms"

    id         = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Foreign key → projects
    project_id = db.Column(db.String(20),
                           db.ForeignKey("floor_plan_projects.id", ondelete="CASCADE"),
                           nullable=False, index=True)

    name       = db.Column(db.String(100), nullable=False)
    room_type  = db.Column(db.String(50),  nullable=True)

    # Real-world dimensions
    width_ft     = db.Column(db.Float,  default=0.0)
    height_ft    = db.Column(db.Float,  default=0.0)
    width_ft_in  = db.Column(db.String(20), default="0' 0\"")
    height_ft_in = db.Column(db.String(20), default="0' 0\"")
    width_m      = db.Column(db.Float,  default=0.0)
    height_m     = db.Column(db.Float,  default=0.0)
    area_sqft    = db.Column(db.Float,  default=0.0)
    area_sqm     = db.Column(db.Float,  default=0.0)

    # Bounding box (pixels) — either a real wall-boundary bbox or a
    # zero-size placeholder, per room_parser_service's containment logic
    bbox_x1 = db.Column(db.Float, default=0.0)
    bbox_y1 = db.Column(db.Float, default=0.0)
    bbox_x2 = db.Column(db.Float, default=0.0)
    bbox_y2 = db.Column(db.Float, default=0.0)

    # How this room's dimensions were determined —
    # "ocr_exact_match" | "ocr_partial_match" | "ocr_partial_match_single_side_only"
    # | "wall_geometry_estimate" | "unmatched"
    dimension_source       = db.Column(db.String(50), nullable=True)
    label_match_confidence = db.Column(db.Float, default=0.0)
    notes                  = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "name":                   self.name,
            "room_type":              self.room_type,
            "width_ft":               round(self.width_ft or 0, 2),
            "height_ft":              round(self.height_ft or 0, 2),
            "width_ft_in":            self.width_ft_in,
            "height_ft_in":           self.height_ft_in,
            "width_m":                round(self.width_m or 0, 3),
            "height_m":               round(self.height_m or 0, 3),
            "area_sqft":              round(self.area_sqft or 0, 2),
            "area_sqm":               round(self.area_sqm or 0, 3),
            "bbox": {
                "x1": self.bbox_x1, "y1": self.bbox_y1,
                "x2": self.bbox_x2, "y2": self.bbox_y2,
            },
            "dimension_source":       self.dimension_source,
            "label_match_confidence": round(self.label_match_confidence or 0, 2),
            "notes":                  self.notes,
        }

    def __repr__(self):
        return f"<Room id={self.id} project={self.project_id} name='{self.name}'>"