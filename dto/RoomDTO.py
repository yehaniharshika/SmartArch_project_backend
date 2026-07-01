"""
SmartArch — dto/RoomDTO.py

A ROOM is not something YOLO detects directly. YOLO only gives us
walls/doors/windows. A "room" is a CONCEPT we build ourselves:

    Several walls that connect together and enclose an empty
    area = one room boundary.

This DTO represents ONE such room.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RoomDTO:
    name: str
    room_type: str = "unknown"

    boundary_points: List[Tuple[float, float]] = field(default_factory=list)
    boundary_area_px2: float = 0.0

    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0

    width_ft_in: str = "0' 0\""
    height_ft_in: str = "0' 0\""
    width_ft: float = 0.0
    height_ft: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0

    area_sqft: float = 0.0
    area_sqm: float = 0.0

    floor: str = "Ground"

    label_match_confidence: float = 0.0
    dimension_source: str = "unknown"
    # one of: "ocr_exact_match" | "ocr_partial_match" |
    #         "wall_geometry_estimate" | "unmatched"

    matched_dimension_texts: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "room_type": self.room_type,
            "width_ft_in": self.width_ft_in,
            "height_ft_in": self.height_ft_in,
            "width_ft": round(self.width_ft, 2),
            "height_ft": round(self.height_ft, 2),
            "width_m": round(self.width_m, 3),
            "height_m": round(self.height_m, 3),
            "area_sqft": round(self.area_sqft, 2),
            "area_sqm": round(self.area_sqm, 3),
            "floor": self.floor,
            "bbox": [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            "label_match_confidence": round(self.label_match_confidence, 2),
            "dimension_source": self.dimension_source,
            "matched_dimension_texts": self.matched_dimension_texts,
            "notes": self.notes,
        }