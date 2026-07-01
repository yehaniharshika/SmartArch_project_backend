"""
SmartArch — services/extraction/gemini_ocr_service.py

DROP-IN REPLACEMENT for the text-reading step of ocr_service.py.

WHY THIS FILE EXISTS:

  This file does NOT change the system's architecture. It only swaps
  the TEXT-READING component. Every downstream component —
  room_boundary_service.py, room_parser_service.py, area_service.py —
  is completely untouched, because this function returns the exact
  same OCRDataDTO shape that ocr_service.extract_text() already
  returns.
"""
import os
import json
import re
from dto.OCRDataDTO import OCRDataDTO


_EXTRACTION_PROMPT = """You are analysing a 2D architectural floor plan image.

The image is exactly {width}x{height} pixels. For every piece of text
you can identify in the image, return ONE entry per text region:

1. "text": the exact text as written (e.g. "BED ROOM 02", "11'", "12'6\\"")
2. "type": either "label" (a room name) or "dimension" (a measurement
   with a ' or " mark) or "other" (anything else, e.g. "FLOOR LAYOUT" title)
3. "confidence": your confidence this reading is correct, 0.0 to 1.0
4. "x1", "y1", "x2", "y2": the PIXEL bounding box of this text in the
   image, using the image's actual {width}x{height} pixel coordinate
   system (top-left = 0,0).

IMPORTANT RULES:
- Read dimension text EXACTLY as drawn — do not normalise or convert units.
- If a room label is split across two lines (e.g. "BED ROOM" then "02"),
  report it as ONE combined entry: "BED ROOM 02", with a bounding box
  that spans both lines.
- Do not guess or invent text that isn't actually visible in the image.
- Report EVERY dimension annotation you can see, even short ones like "9'" or "6'".

Return ONLY valid JSON, no markdown, in exactly this shape:

{{
  "text_regions": [
    {{"text": "BED ROOM 02", "type": "label", "confidence": 0.95,
      "x1": 40, "y1": 90, "x2": 180, "y2": 130}},
    {{"text": "11'", "type": "dimension", "confidence": 0.9,
      "x1": 150, "y1": 40, "x2": 175, "y2": 55}}
  ]
}}
"""


# Keep the same text-classification behavior the pipeline expects,
# without depending on ocr_service.py.
DIMENSION_PATTERN = re.compile(
    r'(\d{1,2})'
    r"\s*[''`′\u2019]\s*"
    r'-?'
    r'(\d{1,2})?'
    r'\s*["""″\u201d]?'
    r'=?'
)

LABEL_PATTERN = re.compile(
    r'(BED\s*ROOM|MASTER\s*BED|BATH\s*ROOM|BATHROOM|KITCHEN|LIVING|'
    r'DINING|GARAGE|CAR\s*PORCH|CORRIDOR|BALCONY|STAIRCASE|TOILET|'
    r'STORE\s*ROOM|STORE|HALL|PANTRY|UTILITY|VERANDA|LOBBY|VISITOR|'
    r'FAMILY|STUDY|LAUNDRY|OFFICE|PRAYER|DRAWING)',
    re.IGNORECASE
)


def is_dimension_text(text: str) -> bool:
    return bool(DIMENSION_PATTERN.search(text or ""))


def is_label_text(text: str) -> bool:
    return bool(LABEL_PATTERN.search(text or ""))


def parse_feet_inches(text: str) -> float:
    match = DIMENSION_PATTERN.search(text or "")
    if not match:
        return 0.0
    feet = float(match.group(1) or 0)
    inches = float(match.group(2) or 0)
    return round(feet + (inches / 12.0), 3)


def extract_text_gemini(img) -> OCRDataDTO:
    """
    Same contract as ocr_service.extract_text(img): takes a numpy BGR
    image (as loaded by cv2.imread), returns an OCRDataDTO with
    room_labels, dimensions, and raw_texts populated — identical shape
    to the EasyOCR version, so nothing downstream needs to change.
    """
    from google import genai
    from google.genai import types
    import cv2

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment/.env")

    height, width = img.shape[:2]

    success, encoded = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image for Gemini Vision call")
    image_bytes = encoded.tobytes()

    prompt = _EXTRACTION_PROMPT.format(width=width, height=height)

    client = genai.Client(api_key=api_key)
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

    raw_text = response.text.strip()
    parsed = json.loads(raw_text)
    regions = parsed.get("text_regions", [])

    return _to_ocr_data_dto(regions)


def _to_ocr_data_dto(regions: list) -> OCRDataDTO:
    """
    Converts Gemini's text_regions list into the EXACT same OCRDataDTO
    shape that ocr_service.extract_text() produces, re-using the SAME
    is_label_text / is_dimension_text classifiers from ocr_service so
    label/dimension classification logic isn't duplicated or out of
    sync between the two engines.
    """
    room_labels, dimensions, raw_texts = [], [], []

    for region in regions:
        text = (region.get("text") or "").strip()
        if not text:
            continue

        x1 = float(region.get("x1", 0))
        y1 = float(region.get("y1", 0))
        x2 = float(region.get("x2", 0))
        y2 = float(region.get("y2", 0))
        confidence = float(region.get("confidence", 0.8))
        center_x = round((x1 + x2) / 2, 1)
        center_y = round((y1 + y2) / 2, 1)

        raw_texts.append({
            "text": text,
            "confidence": round(confidence, 3),
            "center_x": center_x,
            "center_y": center_y,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "bbox": [x1, y1, x2, y2],
        })

        if is_label_text(text):
            room_labels.append(text.upper())
        if is_dimension_text(text):
            dimensions.append(text)

    print(f"[GEMINI-OCR] Parsed {len(raw_texts)} text regions | "
          f"{len(room_labels)} room labels | {len(dimensions)} dimensions")
    if room_labels:
        print(f"[GEMINI-OCR] Labels found: {room_labels}")
    if dimensions:
        print(f"[GEMINI-OCR] Dimensions found: {dimensions}")

    return OCRDataDTO(room_labels=room_labels, dimensions=dimensions, raw_texts=raw_texts)