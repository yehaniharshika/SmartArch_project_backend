"""
SmartArch — services/extraction/room_parser_service.py

Two strategies, chosen automatically based on what the OCR engine gave us:

  1. GEMINI PAIRS (preferred) — when ocr_data.room_pairs is populated,
     each room's width/height dimension was ALREADY correctly associated
     by Gemini itself (it read the drawing and matched them, same as a
     human would). We trust this directly — no re-derivation, no radius
     search, no risk of picking up a neighbouring room's dimension.

  2. RADIUS SEARCH (fallback) — only used when ocr_data.room_pairs is
     empty (i.e. the EasyOCR fallback path, which has no such structured
     pairing, only a flat list of text regions). Here we must guess
     associations by proximity, which is inherently weaker and can
     mismatch compact/tightly-packed layouts.
"""
import re
import math
from dto.RoomDTO import RoomDTO
from dto.OCRDataDTO import OCRDataDTO


# ── Room type classification ────────────────────────────────────────────────
ROOM_TYPE_MAP = {
    "bedroom":   ["BED ROOM", "BEDROOM", "MASTER BED", "VISITOR BED", "GUEST BED"],
    "bathroom":  ["BATH ROOM", "BATHROOM", "TOILET", "WC", "WASH ROOM"],
    "kitchen":   ["KITCHEN", "PANTRY"],
    "living":    ["LIVING", "LIVING AREA", "LIVING ROOM", "DRAWING ROOM"],
    "dining":    ["DINING", "DINING AREA", "DINING ROOM"],
    "garage":    ["GARAGE", "CAR PORCH", "CARPORT"],
    "balcony":   ["BALCONY", "VERANDA", "VERANDAH", "TERRACE", "OPEN TERRACE", "BAL:"],
    "corridor":  ["CORRIDOR", "PASSAGE", "HALLWAY", "LOBBY", "STAIRCASE LOBBY", "T.V LOBBY"],
    "store":     ["STORE", "STORE ROOM", "STOREROOM", "UTILITY"],
    "office":    ["OFFICE", "STUDY", "OFFICE ROOM"],
    "shop":      ["SHOP"],
    "courtyard": ["COURTYARD", "COURTYARD AREA"],
}

# Dimension text pattern — matches "10'1''", "12'5''", "8'", "7'6''" etc.
_FEET_MARK   = r"[''`\u2019\u2018\u2032]"
_DIM_PATTERN = re.compile(
    r"(\d{1,3})" + _FEET_MARK + r"(?:(\d{1,2})" + _FEET_MARK + r"{1,2})?",
)

# Fallback-only constants (radius-search path)
SEARCH_RADIUS_PX = 220
DIM_DEDUP_PX = 18


def _parse_feet_inches(text: str) -> float:
    """Converts '10\'1\'\'' → 10.083, '12\'' → 12.0. Returns 0.0 if unreadable."""
    if not text:
        return 0.0
    m = _DIM_PATTERN.search(text)
    if not m:
        return 0.0
    feet   = float(m.group(1))
    inches = float(m.group(2) or 0)
    return round(feet + inches / 12.0, 4)


def _classify_room(name: str) -> str:
    """Returns a room_type string for a given room name."""
    name_upper = name.upper()
    for rtype, keywords in ROOM_TYPE_MAP.items():
        for kw in keywords:
            if kw in name_upper:
                return rtype
    return "room"


def _dist(x1, y1, x2, y2) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _is_dimension(text: str) -> bool:
    return bool(_DIM_PATTERN.search(text or ""))


def _is_label(text: str) -> bool:
    from services.extraction.ocr_service import is_label_text
    return is_label_text(text)


def _find_overlapping_boundary(lx: float, ly: float, room_boundaries: list):
    """Return the boundary dict whose bbox contains (lx, ly), if any."""
    for b in room_boundaries:
        x1, y1, x2, y2 = b["bbox"]
        if x1 <= lx <= x2 and y1 <= ly <= y2:
            return b
    return None


# ════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def build_room_objects(room_boundaries: list, ocr_data: OCRDataDTO) -> list:
    """
    Main entry point — called by FloorPlan_service.

    Chooses strategy based on what data is available:
      - ocr_data.room_pairs populated → trust Gemini's direct pairing
        (STRATEGY 1, preferred, accurate)
      - otherwise → fall back to radius-search over flat raw_texts
        (STRATEGY 2, EasyOCR path)
    """
    if ocr_data and getattr(ocr_data, "room_pairs", None):
        print(f"[ROOM-PARSER]"
              f"({len(ocr_data.room_pairs)} rooms) "
              f"label-to-dimension association, no radius re-derivation.")
        return _build_pairs(ocr_data.room_pairs, room_boundaries)

    print("[ROOM-PARSER] No direct pairs available — falling back to "
          "radius-search matching (EasyOCR path).")
    return _build_from_radius_search(room_boundaries, ocr_data)



def _build_pairs(room_pairs: list, room_boundaries: list) -> list:
    rooms = []

    for pair in room_pairs:
        name        = pair["name"]
        width_text  = pair.get("width_text")
        height_text = pair.get("height_text")
        x1, y1, x2, y2 = pair["x1"], pair["y1"], pair["x2"], pair["y2"]
        label_x = pair.get("label_x", (x1 + x2) / 2)
        label_y = pair.get("label_y", (y1 + y2) / 2)

        # Collect ONLY the dimensions Gemini itself paired with THIS
        # room — no proximity search, no risk of stealing a neighbour's
        # dimension text.
        matched_dims = [t for t in (width_text, height_text) if t]

        # Optional geometry refinement: if a real wall-boundary region
        # overlaps this room's label position, use its bbox (more
        # accurate real geometry estimated bbox). The
        # DIMENSION VALUES themselves still come only from matched_dims
        # above — this only affects the drawn bounding box, never which
        # numbers are assigned to this room.
        region = _find_overlapping_boundary(label_x, label_y, room_boundaries)
        if region is not None:
            bbox = region["bbox"]
            boundary_points = [tuple(p[0]) for p in region.get("contour", [])]
            boundary_area = region.get("area_px2", 0.0)
            source_note = "gemini_pair_with_boundary_bbox"
        else:
            bbox = (x1, y1, x2, y2)
            boundary_points = []
            boundary_area = 0.0
            source_note = "g_pair_estimated_bbox"

        if len(matched_dims) == 2:
            dimension_source = "ocr_exact_match"
        elif len(matched_dims) == 1:
            dimension_source = "ocr_partial_match_single_side_only"
        else:
            dimension_source = "unmatched"

        room = RoomDTO(
            name=name,
            room_type=_classify_room(name),
            boundary_points=boundary_points,
            boundary_area_px2=boundary_area,
            bbox_x1=float(bbox[0]), bbox_y1=float(bbox[1]),
            bbox_x2=float(bbox[2]), bbox_y2=float(bbox[3]),
            matched_dimension_texts=matched_dims,
            dimension_source=dimension_source,
            label_match_confidence=pair.get("confidence", 0.95),
            notes=source_note,
        )

        print(f"[ROOM-PARSER] '{name}' ({room.room_type}) — "
              f"width={width_text} height={height_text} "
              f"source={dimension_source}")
        rooms.append(room)

    return rooms


# STRATEGY 2 — Radius-search fallback (EasyOCR path only)

def _build_from_radius_search(room_boundaries: list, ocr_data: OCRDataDTO) -> list:
    rooms = []

    if not ocr_data or not ocr_data.room_labels:
        print("[ROOM-PARSER] No room labels found in OCR data")
        return rooms

    raw_texts      = ocr_data.raw_texts or []
    room_labels    = ocr_data.room_labels or []
    all_dimensions = [t for t in raw_texts if _is_dimension(t.get("text", ""))]

    label_map = {}
    for rt in raw_texts:
        txt = rt.get("text", "").strip().upper()
        if _is_label(txt) and txt in room_labels:
            label_map[txt] = rt

    print(f"[ROOM-PARSER] Processing {len(room_labels)} labels | "
          f"{len(all_dimensions)} dimension regions | "
          f"{len(room_boundaries)} boundary regions")

    used_dim_indices = set()

    for label in room_labels:
        label_entry = label_map.get(label)

        if label_entry:
            cx = label_entry.get("center_x", 0)
            cy = label_entry.get("center_y", 0)
            bx1 = label_entry.get("x1", 0)
            by1 = label_entry.get("y1", 0)
            bx2 = label_entry.get("x2", 0)
            by2 = label_entry.get("y2", 0)
        else:
            cx = cy = bx1 = by1 = bx2 = by2 = 0

        room_w = max(bx2 - bx1, 100)
        room_h = max(by2 - by1, 100)
        search_radius = max(room_w, room_h) * 1.5 + 80

        nearby_dims = []
        for idx, dt in enumerate(all_dimensions):
            if idx in used_dim_indices:
                continue
            dcx, dcy = dt.get("center_x", 0), dt.get("center_y", 0)
            d = _dist(cx, cy, dcx, dcy)
            if d <= search_radius:
                nearby_dims.append((d, idx, dt.get("text", "")))
        nearby_dims.sort(key=lambda x: x[0])

        valid_dims = []
        seen = set()
        for _, idx, t in nearby_dims:
            val = _parse_feet_inches(t)
            if val > 0.5 and t not in seen:
                valid_dims.append(t)
                seen.add(t)
                used_dim_indices.add(idx)
            if len(valid_dims) >= 2:
                break

        source = "ocr_exact_match" if len(valid_dims) == 2 else (
            "ocr_partial_match_single_side_only" if len(valid_dims) == 1 else "unmatched"
        )

        region = _find_overlapping_boundary(cx, cy, room_boundaries)
        if region:
            bbox = region["bbox"]
            boundary_points = [tuple(p[0]) for p in region.get("contour", [])]
            boundary_area = region.get("area_px2", 0.0)
        else:
            bbox = (bx1, by1, bx2, by2)
            boundary_points = []
            boundary_area = 0.0

        room = RoomDTO(
            name=label,
            room_type=_classify_room(label),
            boundary_points=boundary_points,
            boundary_area_px2=boundary_area,
            bbox_x1=float(bbox[0]), bbox_y1=float(bbox[1]),
            bbox_x2=float(bbox[2]), bbox_y2=float(bbox[3]),
            matched_dimension_texts=valid_dims,
            dimension_source=source,
            label_match_confidence=label_entry.get("confidence", 0.0) if label_entry else 0.0,
            notes="radius_search_fallback",
        )

        print(f"[ROOM-PARSER] '{label}' ({room.room_type}) — "
              f"dims={valid_dims} source={source} (radius fallback)")
        rooms.append(room)

    return rooms