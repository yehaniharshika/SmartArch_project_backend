"""
SmartArch — dto/OCRDataDTO.py
Holds the raw output from the OCR engine (EasyOCR).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class RawTextDTO:
    text: str
    confidence: float
    center_x: float
    center_y: float
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0

    def to_dict(self):
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "center_x": round(self.center_x, 1),
            "center_y": round(self.center_y, 1),
            "bbox": [self.x1, self.y1, self.x2, self.y2],
        }


@dataclass
class OCRDataDTO:
    room_labels: List[str] = field(default_factory=list)
    # e.g. ["BEDROOM", "MASTER BEDROOM", "KITCHEN", "DINING"]

    dimensions: List[str] = field(default_factory=list)
    # e.g. ["12' 6\"", "10' 4\"", "8'", "1:100"]

    raw_texts: List[dict] = field(default_factory=list)
    # Every text found, each item:
    # {"text": "...", "confidence": 0.9, "center_x": .., "center_y": ..,
    #  "bbox": [x1,y1,x2,y2]}