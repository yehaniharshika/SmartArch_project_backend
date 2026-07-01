"""
SmartArch — services/extraction/room_parser_service.py

LABEL-FIRST approach, using WALL-BOUNDARY CONTAINMENT (not a fixed pixel
radius) to decide which dimension texts belong to which room.

WHY CONTAINMENT INSTEAD OF RADIUS:
  A fixed search radius around a label can accidentally pick up a
  dimension that belongs to a NEIGHBOURING room, if that room is close
  enough on screen — even though a wall separates them. On a compact
  floor plan (rooms packed tightly together), this happens often and
  silently assigns the wrong number to a room.

  The fix: a dimension text only "belongs" to a room if it falls INSIDE
  that room's wall-boundary contour (the enclosed area room_boundary_
  service extracts from the YOLO wall/door/window detections). A wall
  physically separates rooms, so containment inside the same enclosed
  region is a much stronger guarantee of correctness than raw distance.

FALLBACK: if a label has no matching boundary region at all (boundary
detection found nothing usable there), we fall back to a radius search
— better an imperfect guess than nothing — but this path is flagged
separately (match_method) so it's visible in the logs which rooms used
the weaker method.

BBOX SOURCE for area_service's pixel-ratio fallback:
  - If a boundary region was found, its bbox IS the room's bbox — real
    wall geometry, not a guess.
  - If no boundary region was found, we do NOT invent a bbox. We use a
    zero-size bbox; area_service will see that and report
    dimension_source="unmatched" instead of fabricating a size. There is
    no general rule that any room type has a fixed or typical
    width:height ratio, so guessing would be misleading.
"""
import math
from dto.RoomDTO import RoomDTO
from services.extraction.gemini_ocr_service import is_label_text, is_dimension_text

# Fallback-only: used when NO boundary region contains the label at all.
SEARCH_RADIUS_PX = 220

# If two dimension texts are within this many pixels of each other we treat
# them as duplicates and keep only the higher-confidence one.
DIM_DEDUP_PX = 18

# How far outside a boundary's bbox a dimension can still sit and count as
# "belonging" to that room. Dimension numbers are usually drawn just
# outside the wall line, not inside it — so we need a small margin.
BOUNDARY_MARGIN_PX = 35


def build_room_objects(room_boundaries: list, ocr_data) -> list:
    """
    Primary entry point called by FloorPlan_service.

    Strategy:
      1. Build rooms from OCR labels. For each label, find which
         wall-boundary region (if any) contains it, then ONLY consider
         dimension texts that fall inside that same region (+ margin).
      2. For any wall-boundary room whose centroid does NOT overlap any
         label-first room, add it as an unnamed fallback room.
    """
    label_rooms = _build_from_labels(ocr_data, room_boundaries)
    boundary_rooms = _build_from_boundaries(room_boundaries, ocr_data)

    margin = 30
    merged = list(label_rooms)
    for br in boundary_rooms:
        br_cx = (br.bbox_x1 + br.bbox_x2) / 2
        br_cy = (br.bbox_y1 + br.bbox_y2) / 2
        overlaps = any(
            (r.bbox_x1 - margin) <= br_cx <= (r.bbox_x2 + margin) and
            (r.bbox_y1 - margin) <= br_cy <= (r.bbox_y2 + margin)
            for r in label_rooms
        )
        if not overlaps:
            merged.append(br)

    return merged


# ────────────────────────────────────────────────────────────────
# LABEL-FIRST builder — containment-based dimension matching
# ────────────────────────────────────────────────────────────────

def _build_from_labels(ocr_data, room_boundaries: list = None) -> list:
    room_boundaries = room_boundaries or []
    label_items = [t for t in ocr_data.raw_texts if is_label_text(t["text"])]
    dim_items = [t for t in ocr_data.raw_texts if is_dimension_text(t["text"])]
    dim_items = _dedup_dims(dim_items)

    rooms = []
    used_dim_indices = set()

    for label in label_items:
        lx, ly = label["center_x"], label["center_y"]
        label_name = label["text"].upper()
        room_type = classify_room_type(label_name)

        # ── STEP A: which boundary region (if any) contains this label? ──
        region = _find_overlapping_boundary(lx, ly, room_boundaries)

        if region is not None:
            # ── STEP B: only dimension texts INSIDE that region's bbox
            #            (+ margin) are even considered candidates.
            #            This is what stops a neighbouring room's
            #            dimension from being stolen. ────────────────────
            rx1, ry1, rx2, ry2 = region["bbox"]
            rx1 -= BOUNDARY_MARGIN_PX; ry1 -= BOUNDARY_MARGIN_PX
            rx2 += BOUNDARY_MARGIN_PX; ry2 += BOUNDARY_MARGIN_PX

            candidates = []
            for i, dim in enumerate(dim_items):
                dx, dy = dim["center_x"], dim["center_y"]
                if rx1 <= dx <= rx2 and ry1 <= dy <= ry2:
                    dist = math.hypot(dx - lx, dy - ly)
                    candidates.append((dist, i, dim))
            candidates.sort(key=lambda c: c[0])
            match_method = "boundary_containment"
            bbox = (rx1 + BOUNDARY_MARGIN_PX, ry1 + BOUNDARY_MARGIN_PX,
                    rx2 - BOUNDARY_MARGIN_PX, ry2 - BOUNDARY_MARGIN_PX)
        else:
            # No boundary region found for this label at all — fall back
            # to a radius search. Weaker guarantee, but still useful when
            # wall detection missed this particular room's contour.
            candidates = []
            for i, dim in enumerate(dim_items):
                dx, dy = dim["center_x"], dim["center_y"]
                dist = math.hypot(dx - lx, dy - ly)
                if dist <= SEARCH_RADIUS_PX:
                    candidates.append((dist, i, dim))
            candidates.sort(key=lambda c: c[0])
            match_method = "radius_fallback"
            bbox = (lx, ly, lx, ly)  # no real geometry — area_service treats as unmatched

        # ── Split candidates into horizontal/vertical relative to label,
        #    operating on a pre-filtered, room-correct candidate set. ────
        h_dims, v_dims = [], []
        for dist, idx, dim in candidates:
            dx, dy = dim["center_x"], dim["center_y"]
            delta_x = abs(dx - lx)
            delta_y = abs(dy - ly)
            if delta_y >= delta_x:
                h_dims.append((dist, idx, dim))
            else:
                v_dims.append((dist, idx, dim))

        matched_dims = []
        for pool in (h_dims, v_dims):
            for dist, idx, dim in pool:
                if idx not in used_dim_indices:
                    matched_dims.append(dim["text"])
                    used_dim_indices.add(idx)
                    break
            if len(matched_dims) == 2:
                break

        if len(matched_dims) < 2:
            for dist, idx, dim in candidates:
                if idx not in used_dim_indices:
                    matched_dims.append(dim["text"])
                    used_dim_indices.add(idx)
                    if len(matched_dims) == 2:
                        break

        room = RoomDTO(
            name=label_name,
            room_type=room_type,
            boundary_points=[],
            boundary_area_px2=0.0,
            bbox_x1=float(bbox[0]), bbox_y1=float(bbox[1]),
            bbox_x2=float(bbox[2]), bbox_y2=float(bbox[3]),
            label_match_confidence=round(label["confidence"], 2),
            matched_dimension_texts=matched_dims,
        )
        rooms.append(room)

        print(f"[ROOM-PARSER] '{label_name}' ({room_type}) — "
              f"label_confidence={label['confidence']:.2f}, "
              f"dimensions_found={matched_dims}, match_method={match_method}")

    return rooms


# ────────────────────────────────────────────────────────────────
# BOUNDARY FALLBACK builder (unchanged)
# ────────────────────────────────────────────────────────────────

def _build_from_boundaries(room_boundaries: list, ocr_data) -> list:
    rooms = []
    for i, boundary in enumerate(room_boundaries):
        label, label_confidence = _match_label_to_boundary(boundary, ocr_data)
        dimension_texts = _match_dims_to_boundary(boundary, ocr_data)

        room_name = label if label else f"Room {i + 1}"
        room_type = classify_room_type(label)

        x1, y1, x2, y2 = boundary["bbox"]
        room = RoomDTO(
            name=room_name,
            room_type=room_type,
            boundary_points=[tuple(p[0]) for p in boundary["contour"]],
            boundary_area_px2=boundary["area_px2"],
            bbox_x1=float(x1), bbox_y1=float(y1),
            bbox_x2=float(x2), bbox_y2=float(y2),
            label_match_confidence=label_confidence,
            matched_dimension_texts=dimension_texts,
        )
        rooms.append(room)
    return rooms


def _match_label_to_boundary(boundary: dict, ocr_data) -> tuple:
    x1, y1, x2, y2 = boundary["bbox"]
    best_label, best_conf = None, 0.0
    for t in ocr_data.raw_texts:
        if not is_label_text(t["text"]):
            continue
        cx, cy = t["center_x"], t["center_y"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            if t["confidence"] > best_conf:
                best_label = t["text"].upper()
                best_conf = t["confidence"]
    return best_label, best_conf


def _match_dims_to_boundary(boundary: dict, ocr_data, margin=50) -> list:
    x1, y1, x2, y2 = boundary["bbox"]
    ex1, ey1, ex2, ey2 = x1 - margin, y1 - margin, x2 + margin, y2 + margin
    return [
        t["text"] for t in ocr_data.raw_texts
        if is_dimension_text(t["text"])
        and ex1 <= t["center_x"] <= ex2
        and ey1 <= t["center_y"] <= ey2
    ]


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _find_overlapping_boundary(lx: float, ly: float, room_boundaries: list):
    """Return the boundary dict (not just bbox) whose bbox contains (lx, ly)."""
    for b in room_boundaries:
        x1, y1, x2, y2 = b["bbox"]
        if x1 <= lx <= x2 and y1 <= ly <= y2:
            return b
    return None


def _dedup_dims(dim_items: list) -> list:
    kept = []
    for item in dim_items:
        cx, cy = item["center_x"], item["center_y"]
        too_close = any(
            math.hypot(cx - k["center_x"], cy - k["center_y"]) < DIM_DEDUP_PX
            for k in kept
        )
        if not too_close:
            kept.append(item)
    return kept


def classify_room_type(room_name) -> str:
    if not room_name:
        return "unknown"
    n = room_name.upper()
    if "BED" in n:           return "bedroom"
    if "BATH" in n or "TOILET" in n or "WC" in n: return "bathroom"
    if "KITCHEN" in n:       return "kitchen"
    if "LIVING" in n or "HALL" in n: return "living"
    if "DINING" in n:        return "dining"
    if "GARAGE" in n or "PORCH" in n or "POACH" in n or "CAR" in n: return "garage"
    if "BALCONY" in n or "VERANDA" in n: return "balcony"
    if "STORE" in n or "UTILITY" in n or "PANTRY" in n: return "store"
    if "VISITOR" in n:       return "bedroom"
    if "GARDEN" in n or "YARD" in n: return "garden"
    if "STUDY" in n or "OFFICE" in n: return "study"
    if "LAUNDRY" in n:       return "laundry"
    return "unknown"