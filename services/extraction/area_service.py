"""
SmartArch — services/extraction/area_service.py

ONLY job: given a RoomDTO (boundary geometry + matched OCR dimension
texts already attached), compute the FINAL real-world width, height,
and area.

Strategy (priority order):
  1. If OCR found 2+ dimension texts near this room, USE THOSE DIRECTLY.
  2. If OCR found exactly 1, estimate the other side from the room's
     pixel-bbox aspect ratio (only valid if that bbox came from a real
     wall-boundary contour, not a guess).
  3. If none found AND the bbox is a real boundary (non-zero size),
     fall back to pixel-bbox + scale — flagged as an ESTIMATE.
  4. If none found AND there's no real bbox either (zero-size),
     report 0 and mark dimension_source="unmatched" — we do NOT
     invent a size. There is no rule that any room type has a fixed
     or typical width:height ratio, so guessing would be misleading.
"""
from dto.RoomDTO import RoomDTO
from services.extraction.gemini_ocr_service import parse_feet_inches


def decimal_feet_to_ft_in(decimal_feet: float) -> str:
    """12.5 -> "12' 6\"" """
    if decimal_feet <= 0:
        return "0' 0\""
    feet = int(decimal_feet)
    inches = round((decimal_feet - feet) * 12)
    if inches == 12:
        feet += 1
        inches = 0
    return f"{feet}' {inches}\""


def calculate_room_dimensions(room: RoomDTO, pixels_per_foot: float) -> RoomDTO:
    """Fills in room.width_ft_in, height_ft_in, area_sqft, etc. Mutates and returns the same RoomDTO."""

    feet_values = sorted(
        [v for v in (parse_feet_inches(d) for d in room.matched_dimension_texts) if v > 0],
        reverse=True
    )

    pixel_width = room.bbox_x2 - room.bbox_x1
    pixel_height = room.bbox_y2 - room.bbox_y1
    has_real_bbox = pixel_width > 0 and pixel_height > 0

    if len(feet_values) >= 2:
        room.height_ft = feet_values[0]
        room.width_ft = feet_values[1]
        room.dimension_source = "ocr_exact_match"
        room.label_match_confidence = max(room.label_match_confidence, 0.85)

    elif len(feet_values) == 1 and has_real_bbox:
        room.height_ft = feet_values[0]
        room.width_ft = _estimate_other_side(room, feet_values[0])
        room.dimension_source = "ocr_partial_match"

    elif len(feet_values) == 1 and not has_real_bbox:
        # We know one side for certain, but have no real geometry to
        # infer the other side's ratio from — report only what we know.
        room.height_ft = feet_values[0]
        room.width_ft = 0.0
        room.dimension_source = "ocr_partial_match_single_side_only"
        room.notes = (
            "Only one dimension was matched and no wall-boundary geometry "
            "was available to estimate the other side — width left as 0."
        )

    elif has_real_bbox:
        room.width_ft, room.height_ft = _estimate_from_pixels(room, pixels_per_foot)
        room.dimension_source = "wall_geometry_estimate"
        room.notes = "No dimension text matched nearby — size estimated from real wall geometry."

    else:
        # No OCR dimensions AND no real wall-boundary bbox.
        # Do not guess a size — there is no general rule for room
        # proportions that would make a guess meaningful.
        room.width_ft = 0.0
        room.height_ft = 0.0
        room.dimension_source = "unmatched"
        room.notes = (
            "No dimension text and no wall-boundary geometry were found "
            "for this room — size could not be determined."
        )

    room.width_ft_in = decimal_feet_to_ft_in(room.width_ft)
    room.height_ft_in = decimal_feet_to_ft_in(room.height_ft)
    room.width_m = round(room.width_ft * 0.3048, 3)
    room.height_m = round(room.height_ft * 0.3048, 3)

    room.area_sqft = round(room.width_ft * room.height_ft, 2)
    room.area_sqm = round(room.area_sqft * 0.092903, 3)

    return room


def _estimate_from_pixels(room: RoomDTO, pixels_per_foot: float) -> tuple:
    pixel_width = room.bbox_x2 - room.bbox_x1
    pixel_height = room.bbox_y2 - room.bbox_y1
    if not pixels_per_foot or pixels_per_foot <= 0:
        return 0.0, 0.0
    width_ft = round(pixel_width / pixels_per_foot, 2)
    height_ft = round(pixel_height / pixels_per_foot, 2)
    return width_ft, height_ft


def _estimate_other_side(room: RoomDTO, known_side_ft: float) -> float:
    pixel_width = room.bbox_x2 - room.bbox_x1
    pixel_height = room.bbox_y2 - room.bbox_y1
    if pixel_height == 0:
        return known_side_ft
    aspect_ratio = pixel_width / pixel_height
    return round(known_side_ft * aspect_ratio, 2)


def calculate_total_area(rooms: list) -> tuple:
    """Returns (total_sqft, total_sqm) summed over every room."""
    total_sqft = sum(r.area_sqft for r in rooms)
    total_sqm = sum(r.area_sqm for r in rooms)
    return round(total_sqft, 2), round(total_sqm, 3)