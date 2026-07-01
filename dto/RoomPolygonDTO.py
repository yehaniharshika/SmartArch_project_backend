from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RoomPolygonDTO:
    """
    A detected enclosed space (room) — the thing your pipeline was missing.

    This is NOT a YOLO detection box. It is a polygon computed by closing
    wall-segment gaps and finding the enclosed contour. One RoomPolygonDTO
    represents one physical room, regardless of how many wall/door/window
    boxes border it.
    """
    polygon_id:     int
    contour_px:     List[Tuple[float, float]]   # ordered boundary points in pixels
    area_px2:       float                        # raw pixel area of the polygon
    centroid_x:     float
    centroid_y:     float
    bbox:           Tuple[float, float, float, float]  # x1,y1,x2,y2 bounding box

    # ── Filled in by room_parser_service after OCR matching ──
    label:          Optional[str] = None         # e.g. "Bedroom 1" from OCR
    width_m:        float = 0.0
    height_m:       float = 0.0
    area_sqm:       float = 0.0
    area_sqft:      float = 0.0
    dimension_source: str = "estimated"           # "ocr_dimension" | "scale_calculated" | "estimated"

    def to_dict(self):
        return {
            "polygon_id":       self.polygon_id,
            "label":            self.label or f"Room {self.polygon_id}",
            "bbox":             list(self.bbox),
            "centroid":         [round(self.centroid_x, 1), round(self.centroid_y, 1)],
            "width_m":          round(self.width_m, 3),
            "height_m":         round(self.height_m, 3),
            "area_sqm":         round(self.area_sqm, 3),
            "area_sqft":        round(self.area_sqft, 3),
            "dimension_source": self.dimension_source,
        }