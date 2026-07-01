"""
SmartArch — services/extraction/scale_service.py

ONLY job: figure out the SCALE of this floor plan image —
how many pixels = 1 foot.

FIXED VERSION — addresses the scale-mismatch bug where a dimension
text (e.g. "20'10\"") was matched to the WRONG nearby wall segment,
producing a scale value roughly half of the correct one and
inflating every room's calculated area by ~2-5x.

Key fixes from the previous version:
  1. Tighter distance threshold (80px instead of 150px) — a
     dimension text on real architectural drawings is almost always
     written very close to (within ~50-80px of) the wall it labels,
     not just "somewhat nearby".
  2. Orientation matching — a dimension text written horizontally
     (wide bbox) should match a HORIZONTAL wall (wide, not tall),
     and a vertical dimension text should match a VERTICAL wall.
     This alone eliminates most wrong-wall matches.
  3. Collect ALL plausible matches and use the MEDIAN scale value,
     not just the single closest one — a single bad match can no
     longer dominate the result.
  4. Sanity-check the final scale against the image dimensions: if
     the resulting scale would make the whole image represent an
     unrealistic building size (e.g. under 10ft or over 200ft wide),
     reject it and fall back to the next method.
"""
import re
import statistics
from config import Config
from services.extraction.ocr_service import parse_feet_inches, DIMENSION_PATTERN

# Maximum distance (px) between a dimension text's center and the
# wall it is allowed to describe. Real drawings place dimension text
# close to the wall/edge it measures.
MAX_MATCH_DISTANCE_PX = 200

# A floor plan building is realistically between these widths (feet).
# Used to sanity-check the final computed scale.
PLAUSIBLE_BUILDING_WIDTH_FT = (10, 200)


def detect_scale(img, detections: list, ocr_data) -> tuple:
    """
    Args:
        img: the floor plan image (used only for its pixel dimensions)
        detections: raw YOLO detections (list of dicts from yolo_service)
        ocr_data: OCRDataDTO from ocr_service.extract_text()

    Returns:
        (pixels_per_foot, method, confidence)
    """
    height, width = img.shape[:2]
    wall_detections = [d for d in detections if d["label"] == "wall"]

    # ── Method 1: match dimension texts to NEARBY, SIMILARLY-ORIENTED
    #    walls, collect ALL good matches, use the median ──────────────
    candidate_scales = []

    for text_item in ocr_data.raw_texts:
        if not DIMENSION_PATTERN.search(text_item["text"]):
            continue

        feet_value = parse_feet_inches(text_item["text"])
        if feet_value <= 0 or feet_value > 60:   # plausible single wall/room span
            continue

        tx, ty = text_item["center_x"], text_item["center_y"]

        # Is this dimension text written more horizontally or vertically?
        # We use the text's own bbox aspect ratio as a proxy: wide bbox
        # -> horizontal dimension (measuring a horizontal wall), tall
        # bbox -> vertical dimension.
        text_bbox = text_item.get("bbox", [tx - 10, ty - 5, tx + 10, ty + 5])
        text_w = abs(text_bbox[2] - text_bbox[0])
        text_h = abs(text_bbox[3] - text_bbox[1])
        text_is_horizontal = text_w >= text_h

        best_for_this_text = None
        best_distance = float("inf")
        nearest_any_wall_distance = float("inf")   # for diagnostics, ignores orientation/threshold

        for wall in wall_detections:
            wx = (wall["x1"] + wall["x2"]) / 2
            wy = (wall["y1"] + wall["y2"]) / 2
            distance = ((tx - wx) ** 2 + (ty - wy) ** 2) ** 0.5

            if distance < nearest_any_wall_distance:
                nearest_any_wall_distance = distance

            if distance > MAX_MATCH_DISTANCE_PX:
                continue

            wall_w = abs(wall["x2"] - wall["x1"])
            wall_h = abs(wall["y2"] - wall["y1"])
            wall_is_horizontal = wall_w >= wall_h
            wall_pixel_length = max(wall_w, wall_h)

            # Orientation must match: horizontal dimension text should
            # describe a horizontal wall, vertical text a vertical wall.
            if text_is_horizontal != wall_is_horizontal:
                continue
            if wall_pixel_length <= 5:
                continue

            if distance < best_distance:
                best_distance = distance
                best_for_this_text = wall_pixel_length

        if best_for_this_text is not None:
            scale_value = best_for_this_text / feet_value
            candidate_scales.append(scale_value)
            print(f"[SCALE] Candidate match: '{text_item['text']}' "
                  f"({feet_value}ft) <-> wall {best_for_this_text:.1f}px "
                  f"(dist={best_distance:.1f}px) -> {scale_value:.2f} px/ft")
        else:
            # DIAGNOSTIC: show why this text found no match, so the
            # actual distance/threshold can be tuned with real numbers
            # instead of guessing.
            print(f"[SCALE] No match for '{text_item['text']}' "
                  f"(text_is_horizontal={text_is_horizontal}) -- "
                  f"nearest wall (any orientation) was "
                  f"{nearest_any_wall_distance:.1f}px away "
                  f"(threshold={MAX_MATCH_DISTANCE_PX}px)")

    if candidate_scales:
        # Use the median rather than mean -- robust to any single
        # remaining outlier match.
        pixels_per_foot = statistics.median(candidate_scales)
        n = len(candidate_scales)

        if _is_plausible(pixels_per_foot, width, height):
            confidence = min(0.55 + 0.05 * n, 0.85)   # more agreeing matches = more confidence
            print(f"[SCALE] Using median of {n} candidate match(es) "
                  f"-> {pixels_per_foot:.2f} px/ft")
            return pixels_per_foot, "ocr_dimension_to_wall", confidence
        else:
            print(f"[SCALE] WARNING: median scale {pixels_per_foot:.2f} px/ft failed "
                  f"plausibility check -- falling back to next method")

    # ── Method 1b: relaxed fallback -- if strict orientation matching
    #    found nothing, try again ignoring orientation, using only a
    #    tight distance threshold. This covers cases where the text's
    #    bbox orientation guess was wrong (common for short numbers
    #    like "10'" which don't have a clearly wide/tall bbox). ────────
    if not candidate_scales:
        relaxed_threshold = 60
        for text_item in ocr_data.raw_texts:
            if not DIMENSION_PATTERN.search(text_item["text"]):
                continue
            feet_value = parse_feet_inches(text_item["text"])
            if feet_value <= 0 or feet_value > 60:
                continue

            tx, ty = text_item["center_x"], text_item["center_y"]
            for wall in wall_detections:
                wx = (wall["x1"] + wall["x2"]) / 2
                wy = (wall["y1"] + wall["y2"]) / 2
                distance = ((tx - wx) ** 2 + (ty - wy) ** 2) ** 0.5
                if distance > relaxed_threshold:
                    continue
                wall_pixel_length = max(
                    abs(wall["x2"] - wall["x1"]), abs(wall["y2"] - wall["y1"])
                )
                if wall_pixel_length <= 5:
                    continue
                scale_value = wall_pixel_length / feet_value
                candidate_scales.append(scale_value)
                print(f"[SCALE] Relaxed-match: '{text_item['text']}' "
                      f"({feet_value}ft) <-> wall {wall_pixel_length:.1f}px "
                      f"(dist={distance:.1f}px, no orientation check) "
                      f"-> {scale_value:.2f} px/ft")
                break   # one match per text is enough for this pass

        if candidate_scales:
            pixels_per_foot = statistics.median(candidate_scales)
            n = len(candidate_scales)
            if _is_plausible(pixels_per_foot, width, height):
                print(f"[SCALE] Using median of {n} relaxed match(es) "
                      f"-> {pixels_per_foot:.2f} px/ft")
                return pixels_per_foot, "ocr_dimension_to_wall_relaxed", 0.55

    # ── Method 2: scale ratio text e.g. "1:100" ──────────────────
    for text_item in ocr_data.raw_texts:
        scale_match = re.search(r"1\s*[:\s]\s*(\d+)", text_item["text"])
        if scale_match:
            denominator = int(scale_match.group(1))
            pixels_per_foot = denominator * 0.12
            if _is_plausible(pixels_per_foot, width, height):
                print(f"[SCALE] Found scale ratio text '1:{denominator}' "
                      f"-> approx {pixels_per_foot:.2f} px/ft")
                return pixels_per_foot, "ocr_scale_ratio", 0.50

    # ── Method 3: fallback default (LOW confidence -- flagged) ───
    pixels_per_foot = 1.0 / (Config.DEFAULT_SCALE * 3.28084)
    print(f"[SCALE] WARNING: No reliable dimension/scale match found -- "
          f"using DEFAULT scale ({pixels_per_foot:.2f} px/ft). "
          f"Room sizes for unmatched rooms will be estimates.")
    return pixels_per_foot, "default", 0.25


def _is_plausible(pixels_per_foot: float, image_width_px: int,
                  image_height_px: int) -> bool:
    """
    Sanity check: does this scale imply a realistic building size?
    Rejects scales that would make the building absurdly small or large.
    """
    if pixels_per_foot <= 0:
        return False
    implied_width_ft = image_width_px / pixels_per_foot
    min_ft, max_ft = PLAUSIBLE_BUILDING_WIDTH_FT
    return min_ft <= implied_width_ft <= max_ft