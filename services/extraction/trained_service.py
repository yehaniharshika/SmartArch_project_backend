import cv2
import numpy as np
from config import Config
from dto.OCRDataDTO import OCRDataDTO
from services.extraction.ocr_service import (
    is_label_text, is_dimension_text, parse_both_dimensions,
    LABEL_PATTERN, DIMENSION_PATTERN, ROOM_DIM_PATTERN
)
from services.extraction.text_cleaner import clean_ocr_text, clean_label_text
from services.extraction.ocr_fallback_service import read_crop_with_gemini

_text_yolo      = None
_easyocr_reader = None

# Lowered from 5 — small trained model may detect fewer boxes
MIN_REGIONS_BEFORE_FALLBACK = 3

# Upscale factor for each crop before EasyOCR
CROP_UPSCALE = 2.5

# Load the Text region detection model
def _get_text_detector():
    global _text_yolo
    if _text_yolo is not None:
        return _text_yolo

    from ultralytics import YOLO
    import torch, os

    orig = torch.load
    def _patched(*a, **kw):
        kw["weights_only"] = False
        return orig(*a, **kw)
    torch.load = _patched

    weights = str(Config.TEXT_WEIGHTS_PATH)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Text detector weights not found: {weights}")

    print(f"[TRAINED-OCR] Loading text detector from {weights} ...")
    _text_yolo = YOLO(weights)
    print("[TRAINED-OCR] ✅ Text detector ready")
    return _text_yolo

# Initialises EasyOCR, which is used as the primary OCR method for recognising text.
def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    import easyocr
    print("[TRAINED-OCR] Initialising EasyOCR reader ...")
    _easyocr_reader = easyocr.Reader(
        Config.OCR_LANG, gpu=Config.OCR_GPU, verbose=False
    )
    print("[TRAINED-OCR] ✅ EasyOCR ready")
    return _easyocr_reader

# Crops the text region, preprocesses the crop, and passes it to EasyOCR.”
def _crop_and_read(img, x1, y1, x2, y2, reader, det_class):
    h_img, w_img = img.shape[:2]
    pad = 8
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w_img, x2 + pad)
    cy2 = min(h_img, y2 + pad)

    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return []

    # Upscale — small crops need more pixels for reliable recognition
    crop_h, crop_w = crop.shape[:2]
    scale = min(max(CROP_UPSCALE, 40.0 / max(crop_h, 1)), 4.0)
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # Mild contrast enhance
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    crop_ocr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # EasyOCR — paragraph=False so we get individual word/line bboxes
    # text_threshold lowered for thin CAD fonts
    ocr_results = reader.readtext(
        crop_ocr,
        detail=1,
        paragraph=False,
        text_threshold=0.3,
        low_text=0.2,
    )

    items = []
    for (bbox, raw_text, conf) in ocr_results:
        if conf < 0.20 or not raw_text.strip():
            continue

        if det_class == "dimension_text":
            text_for_dim = clean_ocr_text(raw_text)
            text_upper   = raw_text.strip().upper()
        else:
            text_upper   = clean_label_text(raw_text)
            text_for_dim = text_upper

        is_label = is_label_text(text_upper)
        is_dim   = is_dimension_text(text_for_dim)

        # Noise check — reject titles/watermarks
        from services.extraction.ocr_service import is_noise_text
        if is_label and is_noise_text(text_upper):
            print(f"  [NOISE] Rejected '{text_upper}' (title/watermark)")
            is_label = False

        if not is_label and not is_dim:
            if det_class == "room_label":
                is_label = True
            elif det_class == "dimension_text":
                is_dim = True
            else:
                continue

        final_text = text_upper if is_label else text_for_dim
        cx = float(x1 + (x2 - x1) / 2)
        cy = float(y1 + (y2 - y1) / 2)

        items.append({
            "text":      final_text,
            "text_dim":  text_for_dim,
            "confidence": round(conf, 3),
            "center_x":  round(cx, 1),
            "center_y":  round(cy, 1),
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "is_label":  is_label,
            "is_dim":    is_dim,
            "det_class": det_class,
            "source":    "easyocr",
        })

    cx = float(x1 + (x2 - x1) / 2)
    cy = float(y1 + (y2 - y1) / 2)

    # ROOM LABEL fallback — unchanged: fires only when EasyOCR found
    # For extracted room labels incomplete,unclear or difficult to read due to complexity, Gemini is used only as a fallback when EasyOCR does not produce a usable result.
    # give the cropped region to Gemini to read the text.
    if not items and det_class == "room_label":
        gemini_text = read_crop_with_gemini(crop_ocr, det_class)
        if gemini_text:
            gemini_text = clean_label_text(gemini_text)
            if is_label_text(gemini_text):
                items.append({
                    "text": gemini_text, "text_dim": gemini_text,
                    "confidence": 0.60,
                    "center_x": round(cx, 1), "center_y": round(cy, 1),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "is_label": True, "is_dim": False,
                    "det_class": det_class,
                    "source": "gemini_fallback",
                })
                print(f"  [GEMINI-FALLBACK] recovered '{gemini_text}' "
                      f"(room_label) where EasyOCR found nothing")

    # DIMENSION TEXT cross-check — fires on EVERY dimension crop,
    # not just empty ones, because dimension misreads are the main accuracy problem even when EasyOCR produces a result (e.g."8'X10'" misread as "8XI0'"). Gemini's reading REPLACES the
    # Gemini is used as a validation and correction mechanism. It checks every detected dimension crop
    # we always double-check it with Gemini, because dimensions are the most error-prone part. 
    # EasyOCR often misreads or mismatch the dimention. Gemini's answer replaces EasyOCR's answer only if it passes our own validation rules
    elif det_class == "dimension_text":
        gemini_raw = read_crop_with_gemini(crop_ocr, det_class)
        if gemini_raw:
            gemini_clean = clean_ocr_text(gemini_raw)
            gw, gh = parse_both_dimensions(gemini_clean)
            gemini_valid = is_dimension_text(gemini_clean) and (gw > 0 or gh > 0)

            if gemini_valid:
                items = [{
                    "text": gemini_clean, "text_dim": gemini_clean,
                    "confidence": 0.65,
                    "center_x": round(cx, 1), "center_y": round(cy, 1),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "is_label": False, "is_dim": True,
                    "det_class": det_class,
                    "source": "gemini_verified",
                }]
                print(f"  [GEMINI-VERIFY] '{gemini_clean}' used for this "
                      f"dimension crop (EasyOCR raw text discarded)")
            elif not items:
                print(f"  [GEMINI-VERIFY] Gemini reading '{gemini_raw}' "
                      f"invalid too — crop stays unresolved")

    return items

# Extract all room label and dimmension text
def extract_text(img: np.ndarray) -> OCRDataDTO:
    detector = _get_text_detector()
    reader   = _get_easyocr()

    results = detector(
        img,
        conf   = Config.TEXT_YOLO_CONF,
        iou    = Config.TEXT_YOLO_IOU,
        imgsz  = Config.TEXT_YOLO_IMG_SIZE,
        verbose= False,
    )
    names = results[0].names
    print(f"[TRAINED-OCR] Model classes: {names}")

    room_labels, dimensions, raw_texts = [], [], []
    gemini_fallback_count = 0
    gemini_verify_count   = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        det_class  = names[int(box.cls[0])].lower()
        det_conf   = float(box.conf[0])

        items = _crop_and_read(img, x1, y1, x2, y2, reader, det_class)

        for item in items:
            if item.get("source") == "gemini_fallback":
                gemini_fallback_count += 1
            elif item.get("source") == "gemini_verified":
                gemini_verify_count += 1

            # Store in raw_texts — use label-safe text as the primary key
            raw_entry = {k: v for k, v in item.items()
                         if k not in ("text_dim", "is_label", "is_dim", "det_class")}
            raw_texts.append(raw_entry)

            if item["is_label"]:
                room_labels.append(item["text"])
                print(f"  [LABEL] '{item['text']}' "
                      f"@ ({item['center_x']:.0f},{item['center_y']:.0f}) "
                      f"[{item['source']}]")

            if item["is_dim"]:
                # Store the cleaned dimension text for downstream parsers
                dim_text = item["text_dim"] if item["text_dim"] else item["text"]
                dimensions.append(dim_text)
                # Also add dim-version to raw_texts so room_parser can find it
                dim_entry = dict(raw_entry)
                dim_entry["text"] = dim_text
                raw_texts.append(dim_entry)
                print(f"  [DIM]   '{dim_text}' "
                      f"@ ({item['center_x']:.0f},{item['center_y']:.0f}) "
                      f"[{item['source']}]")

    # De-duplicate raw_texts by (center_x, center_y, text)
    seen = set()
    deduped = []
    for rt in raw_texts:
        key = (rt["center_x"], rt["center_y"], rt["text"])
        if key not in seen:
            seen.add(key)
            deduped.append(rt)
    raw_texts = deduped

    print(f"[TRAINED-OCR] Scanned {len(raw_texts)} regions | "
          f"{len(room_labels)} room labels | {len(dimensions)} dimensions | "
          f"{gemini_fallback_count} label(s) recovered via Gemini | "
          f"{gemini_verify_count} dimension(s) corrected via Gemini")

    if len(raw_texts) < MIN_REGIONS_BEFORE_FALLBACK:
        print(f"[TRAINED-OCR] Only {len(raw_texts)} regions "
              f"(threshold {MIN_REGIONS_BEFORE_FALLBACK}) "
              f"— falling back to EasyOCR full-scan.")
        from services.extraction import ocr_service as blind_ocr
        return blind_ocr.extract_text(img)

    return OCRDataDTO(
        room_labels=room_labels,
        dimensions=dimensions,
        raw_texts=raw_texts,
    )