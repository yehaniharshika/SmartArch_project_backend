"""
SmartArch — dto/AnalysisResultDTO.py
The final "box" carrying everything from one full pipeline run.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from dto.DetectionDTO import DetectionDTO
from dto.OCRDataDTO import OCRDataDTO
from dto.RoomDTO import RoomDTO


@dataclass
class AnalysisResultDTO:
    project_id: str
    image_path: str
    annotated_image: str

    detections: List[DetectionDTO] = field(default_factory=list)
    ocr_data: Optional[OCRDataDTO] = None
    rooms: List[RoomDTO] = field(default_factory=list)

    total_area_sqm: float = 0.0
    total_area_sqft: float = 0.0

    room_count: int = 0
    door_count: int = 0
    window_count: int = 0
    wall_count: int = 0

    summary: str = ""

    image_width_px: int = 0
    image_height_px: int = 0
    pixels_per_meter: float = 50.0
    pixels_per_foot: float = 15.0
    scale_method: str = "default"
    scale_confidence: float = 0.30

    processing_time: float = 0.0

    # ── Diagnostics — explains WHAT WENT WRONG if room_count is 0 ──
    # Shown in the terminal report so problems are visible immediately,
    # not silently swallowed.
    pipeline_warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "project_id": self.project_id,
            "total_area_sqm": round(self.total_area_sqm, 2),
            "total_area_sqft": round(self.total_area_sqft, 2),
            "room_count": self.room_count,
            "door_count": self.door_count,
            "window_count": self.window_count,
            "wall_count": self.wall_count,
            "total_detections": len(self.detections),
            "summary": self.summary,
            "rooms": [r.to_dict() for r in self.rooms],
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "pixels_per_meter": self.pixels_per_meter,
            "pixels_per_foot": self.pixels_per_foot,
            "scale_method": self.scale_method,
            "scale_confidence": self.scale_confidence,
            "processing_time": round(self.processing_time, 2),
            "detections": [d.to_dict() for d in self.detections],
            "warnings": self.pipeline_warnings,
        }