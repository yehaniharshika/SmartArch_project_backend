"""
SmartArch — dto/DetectionDTO.py
One row = one YOLOv8 detection (a single wall segment, door, or window).
STRUCTURAL data only — does NOT represent a room.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionDTO:
    label: str          # "wall" | "door" | "window"
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    width_m: float = 0.0
    height_m: float = 0.0
    width_ft_in: str = "0' 0\""
    height_ft_in: str = "0' 0\""

    area_sqm: float = 0.0
    area_sqft: float = 0.0
    perimeter_m: float = 0.0

    ocr_label: Optional[str] = None
    ocr_dimension: Optional[str] = None

    def to_dict(self):
        return {
            "label": self.label,
            "confidence": round(self.confidence * 100, 1),
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "width_m": round(self.width_m, 3),
            "height_m": round(self.height_m, 3),
            "width_ft_in": self.width_ft_in,
            "height_ft_in": self.height_ft_in,
            "area_sqm": round(self.area_sqm, 3),
            "area_sqft": round(self.area_sqft, 3),
            "perimeter_m": round(self.perimeter_m, 3),
            "ocr_label": self.ocr_label,
            "ocr_dimension": self.ocr_dimension,
        }