import os
import json
import re

from dto.OCRDataDTO import OCRDataDTO


# Dimension parsing (public — used by area_service.py)
# Matches "10'1''", "12'5''", "8'", "7'6''" etc.
_FEET_MARK   = r"[''`\u2019\u2018\u2032]"
_DIM_PATTERN = re.compile(
    r"(\d{1,3})" + _FEET_MARK + r"(?:(\d{1,2})" + _FEET_MARK + r"{1,2})?",
)


def parse_feet_inches(text: str) -> float:
    """
    Converts a dimension string like "10'1''" -> 10.083 (decimal feet).
    Returns 0.0 if the text doesn't contain a recognizable feet mark.
    """
    if not text:
        return 0.0
    m = _DIM_PATTERN.search(text)
    if not m:
        return 0.0
    feet   = float(m.group(1))
    inches = float(m.group(2) or 0)
    return round(feet + inches / 12.0, 4)


# PROMPT 
_EXTRACTION_PROMPT = """You are an expert at reading 2D architectural floor plan drawings.

The image is {width}x{height} pixels. Analyse the entire floor plan carefully.

Your task has TWO parts:

PART 1 — ROOM LIST:
For every room or space visible in the floor plan, identify:
  a) "name": The room label exactly as written (e.g. "BED ROOM 02", "KITCHEN", "BATH ROOM 01").
     IMPORTANT: If a label is split across multiple lines (e.g. "BED" on one line and "ROOM 02"
     below it), combine them into ONE label with a space: "BED ROOM 02".
     If underlined (AutoCAD style), ignore the underline, just read the text.
     Include EVERY labelled space, even abbreviations like "BAL:" (balcony),
     "T.V LOBBY", terraces, courtyards, and shops — not just bedrooms/kitchens.
  b) "width_text": The dimension annotation for the room's WIDTH (horizontal measurement),
     exactly as written including the ' and " symbols (e.g. "10'1''", "12'5''", "8'8''").
     Look for dimension lines with arrows/tick marks along the top or bottom walls
     OF THIS SPECIFIC ROOM.
     If not found, use null.
  c) "height_text": The dimension annotation for the room's HEIGHT/LENGTH (vertical measurement),
     exactly as written. Look along the left or right walls OF THIS SPECIFIC ROOM.
     If not found, use null.
  d) "x1", "y1", "x2", "y2": The approximate pixel bounding box of the room's interior space.
  e) "label_x", "label_y": The pixel center of where the label text sits.

CRITICAL: width_text and height_text MUST belong to THIS room, not a neighbouring
room. Double-check each dimension is on a wall that actually borders this room's
own interior space before assigning it.

PART 2 — ALL DIMENSION TEXTS:
List every dimension annotation visible anywhere in the plan (even ones not yet matched
to a specific room). Format: the text exactly as written, e.g. "10'1''", "15'2''", "28'".

IMPORTANT RULES:
- Read dimension text EXACTLY — preserve feet (') and inches ('') marks.
- Do NOT invent or guess dimensions. Only report what is actually visible.
- Do NOT skip small rooms (bathrooms, store rooms, lobbies, terraces, balconies, shops, courtyards).
- Dimension lines in AutoCAD plans have arrow tips or tick marks (small diagonal lines)
  at each end — these mark the measured distance.
- The number between two tick marks is the dimension for THAT wall segment.
- For rooms labelled across multiple lines, ALWAYS combine into one name.

Return ONLY valid JSON with NO markdown, NO backticks, in exactly this structure:

{{
  "rooms": [
    {{
      "name": "BED ROOM 02",
      "width_text": "10'1''",
      "height_text": "12'5''",
      "x1": 45, "y1": 80, "x2": 280, "y2": 350,
      "label_x": 160, "label_y": 200
    }}
  ],
  "all_dimensions": ["10'1''", "12'5''", "7'6''", "28'", "15'2''"]
}}
"""


def extract_text_gemini(img) -> OCRDataDTO:
    """
    Main entry point — same contract as ocr_service.extract_text(img).
    Takes numpy BGR image, returns OCRDataDTO.
    """
    from google import genai
    from google.genai import types
    import cv2

    api_key    = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    height, width = img.shape[:2]

    success, encoded = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image for Gemini Vision")
    image_bytes = encoded.tobytes()

    prompt = _EXTRACTION_PROMPT.format(width=width, height=height)

    client   = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {raw[:500]}")
        return OCRDataDTO()

    return _build_ocr_dto(parsed)


def _clean_label(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[\n\r\t]+", " ", text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip().upper()


def _clean_dimension(text: str) -> str:
    if not text:
        return ""
    feet_chars  = "'`\u2019\u2018\u2032"
    inch_chars  = '"\u201c\u201d\u2033'
    kept = []
    for ch in text:
        if ch.isdigit():
            kept.append(ch)
        elif ch in feet_chars:
            kept.append("'")
        elif ch in inch_chars:
            kept.append('"')
        elif ch in ("-", " "):
            kept.append(ch)
    return "".join(kept).strip()


def _is_label(text: str) -> bool:
    from services.extraction.ocr_service import is_label_text
    return is_label_text(text)


def _is_dimension(text: str) -> bool:
    cleaned = _clean_dimension(text)
    return bool(re.search(r"\d{1,2}'", cleaned))


def _build_ocr_dto(parsed: dict) -> OCRDataDTO:
    room_labels = []
    dimensions  = []
    raw_texts   = []
    room_pairs  = []

    rooms = parsed.get("rooms", [])

    for i, room in enumerate(rooms):
        raw_name = room.get("name", "")
        name     = _clean_label(raw_name)
        if not name:
            continue

        x1 = float(room.get("x1", 0))
        y1 = float(room.get("y1", 0))
        x2 = float(room.get("x2", 0))
        y2 = float(room.get("y2", 0))

        label_x = float(room.get("label_x", (x1 + x2) / 2))
        label_y = float(room.get("label_y", (y1 + y2) / 2))

        width_text  = _clean_dimension(room.get("width_text")  or "")
        height_text = _clean_dimension(room.get("height_text") or "")

        room_labels.append(name)

        # Preserve Gemini's own direct pairing 
        room_pairs.append({
            "name": name,
            "width_text": width_text if width_text and _is_dimension(width_text) else None,
            "height_text": height_text if height_text and _is_dimension(height_text) else None,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "label_x": label_x, "label_y": label_y,
            "confidence": 0.95,
        })

        # Room label entry (used by _draw_annotations / boundary lookup)
        raw_texts.append({
            "text":       name,
            "confidence": 0.95,
            "center_x":   round(label_x, 1),
            "center_y":   round(label_y, 1),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "bbox": [x1, y1, x2, y2],
        })

        if width_text and _is_dimension(width_text):
            dimensions.append(width_text)
        if height_text and _is_dimension(height_text):
            dimensions.append(height_text)

    # Also process any standalone dimensions from PART 2
    all_dims = parsed.get("all_dimensions", [])
    seen_dims = set(dimensions)
    for d in all_dims:
        cleaned = _clean_dimension(d)
        if cleaned and _is_dimension(cleaned) and cleaned not in seen_dims:
            dimensions.append(cleaned)
            seen_dims.add(cleaned)

    # Deduplicate room_labels
    seen_labels = []
    seen_set = set()
    for lbl in room_labels:
        if lbl not in seen_set:
            seen_labels.append(lbl)
            seen_set.add(lbl)

    print(f"Extracted {len(seen_labels)} rooms | "
          f"{len(dimensions)} dimensions")
    print(f"Room Labels: {seen_labels}")
    print(f"Dimensions: {dimensions}")

    return OCRDataDTO(
        room_labels=seen_labels,
        dimensions=dimensions,
        raw_texts=raw_texts,
        room_pairs=room_pairs,
    )