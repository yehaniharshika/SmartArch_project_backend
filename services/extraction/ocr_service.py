"""
SmartArch — services/extraction/ocr_service.py
"""
import re
import cv2
import numpy as np
from config import Config
from dto.OCRDataDTO import OCRDataDTO, RawTextDTO

_ocr_reader = None

DEDUP_RADIUS_PX = 25

# Dimension pattern — catches: 10'  12'6"  12' 6"  14'10"  20'10"=  7'7"
DIMENSION_PATTERN = re.compile(
    r'(\d{1,2})'
    r"\s*[''`′\u2019]\s*"
    r'-?'
    r'(\d{1,2})?'
    r'\s*["""″\u201d]?'
    r'=?'
)

# Room label pattern
LABEL_PATTERN = re.compile(
    r'(BED\s*ROOM|MASTER\s*BED|BATH\s*ROOM|BATHROOM|KITCHEN|LIVING|'
    r'DINING|GARAGE|CAR\s*PORCH|CORRIDOR|BALCONY|STAIRCASE|TOILET|'
    r'STORE\s*ROOM|STORE|HALL|PANTRY|UTILITY|VERANDA|LOBBY|VISITOR|'
    r'FAMILY|STUDY|LAUNDRY|OFFICE|PRAYER|DRAWING)',
    re.IGNORECASE
)

def is_dimension_text(text: str) -> bool:
    return bool(DIMENSION_PATTERN.search(text))

def is_label_text(text: str) -> bool:
    return bool(LABEL_PATTERN.search(text))

def parse_feet_inches(text: str) -> float:
    match = DIMENSION_PATTERN.search(text)
    if not match:
        return 0.0
    feet = float(match.group(1) or 0)
    inches = float(match.group(2) or 0)
    return round(feet + (inches / 12.0), 3)

def extract_text(img: np.ndarray) -> OCRDataDTO:
    reader = _get_reader()

    # Pass A: original image brightened — best for ROOM LABELS (large text)
    img_for_labels = _preprocess_for_labels(img)

    # Pass B: wall-suppressed — best for DIMENSION annotations (small text near walls)
    img_for_dims = _preprocess_for_dimensions(img)

    results_labels = reader.readtext(
        img_for_labels, detail=1, paragraph=False,
        width_ths=0.4, height_ths=0.4,
    )
    results_dims = reader.readtext(
        img_for_dims, detail=1, paragraph=False,
        width_ths=0.15, height_ths=0.15,
        min_size=4,
    )

    merged = _merge_ocr_results(results_labels, results_dims)

    room_labels, dimensions, raw_texts = [], [], []

    for (bbox, text, confidence) in merged:
        text = text.strip()
        if not text or confidence < 0.22:
            continue

        points = np.array(bbox, dtype=np.float32)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        center_x = float((x1 + x2) / 2)
        center_y = float((y1 + y2) / 2)

        raw_dict = {
            "text": text,
            "confidence": round(confidence, 3),
            "center_x": round(center_x, 1),
            "center_y": round(center_y, 1),
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
        }
        raw_texts.append(raw_dict)

        if is_label_text(text):
            room_labels.append(text.upper())
        if is_dimension_text(text):
            dimensions.append(text)

    print(f"[OCR] Scanned {len(raw_texts)} text regions | "
          f"{len(room_labels)} room labels | {len(dimensions)} dimensions")
    if room_labels:
        print(f"[OCR] Labels found: {room_labels}")
    if dimensions:
        print(f"[OCR] Dimensions found: {dimensions}")

    return OCRDataDTO(room_labels=room_labels, dimensions=dimensions, raw_texts=raw_texts)


def _preprocess_for_labels(img: np.ndarray) -> np.ndarray:
    """
    High-contrast grayscale for large room label text.
    No wall suppression — labels are far from walls and survive fine.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Mild sharpening so thin font strokes are cleaner
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(enhanced, -1, kernel)


def _preprocess_for_dimensions(img: np.ndarray) -> np.ndarray:
    """
    Wall-suppressed image for small dimension annotations.
    Key insight: dimension numbers (e.g. 11', 10') sit right beside
    wall lines. Removing wall lines exposes them to OCR.

    We use threshold=200 (lighter than Otsu) because dimension text
    on CAD exports is often thin/light and Otsu clips it.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light threshold — catches thin dimension text
    _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY_INV)

    # Suppress ONLY long wall lines (≥60px).
    # Dimension tick marks and arrowheads are shorter → preserved.
    wall_len = 60
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (wall_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, wall_len))
    wall_mask = cv2.add(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel),
    )

    text_only = cv2.subtract(binary, wall_mask)

    # Reconnect broken strokes after subtraction
    d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.dilate(text_only, d_kernel, iterations=1)

    return cv2.bitwise_not(cleaned)  # white bg, black text for EasyOCR


def _merge_ocr_results(primary: list, secondary: list) -> list:
    """
    primary (label pass) takes priority.
    secondary adds any text whose centre is > DEDUP_RADIUS_PX from all primary detections.
    """
    merged = list(primary)
    centres = [_bbox_centre(b) for (b, _, _) in primary]

    for (bbox_b, text_b, conf_b) in secondary:
        cx_b, cy_b = _bbox_centre(bbox_b)
        if not any(
            ((cx_b - cx)**2 + (cy_b - cy)**2) ** 0.5 < DEDUP_RADIUS_PX
            for (cx, cy) in centres
        ):
            merged.append((bbox_b, text_b, conf_b))
            centres.append((cx_b, cy_b))

    return merged


def _bbox_centre(bbox) -> tuple:
    pts = np.array(bbox, dtype=np.float32)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _get_reader():
    global _ocr_reader
    if _ocr_reader is not None:
        return _ocr_reader
    import easyocr
    print("[OCR] Initialising EasyOCR reader ...")
    _ocr_reader = easyocr.Reader(Config.OCR_LANG, gpu=Config.OCR_GPU, verbose=False)
    print("[OCR] ✅ Reader ready")
    return _ocr_reader