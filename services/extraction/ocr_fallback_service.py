
import warnings
import cv2
from config import Config

warnings.filterwarnings(
    "ignore",
    message="All support for the `google.generativeai` package has ended.*",
    category=FutureWarning,
)

_gemini_model = None


def _get_gemini():
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    import google.generativeai as genai
    genai.configure(api_key=Config.GEMINI_API_KEY)
    _gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)
    return _gemini_model


_LABEL_PROMPT = (
    "This image is a small cropped region from an architectural floor "
    "plan. It should contain a ROOM LABEL (e.g. KITCHEN, BED ROOM, HALL). "
    "Return ONLY the exact text visible, uppercase, no punctuation, no "
    "explanation. If nothing readable is present, return exactly: NONE"
)

_DIM_PROMPT = (
    "This image is a small cropped region from an architectural floor "
    "plan. It should contain a DIMENSION measurement in feet/inches "
    "(e.g. 8'X10', 11'X10', 16'9\"X10'). Return ONLY the exact dimension "
    "text as printed, preserving the ' and \" marks and the X separator. "
    "If nothing readable is present, return exactly: NONE"
)


def _extract_text(response) -> str | None:
    if not getattr(response, "candidates", None):
        return None

    candidate = response.candidates[0]

    finish_reason = getattr(candidate, "finish_reason", None)
    if finish_reason is not None and str(finish_reason) not in ("1", "STOP"):
        print(f"  [GEMINI-FALLBACK] non-STOP finish_reason: {finish_reason}")

    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return None

    text = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))
    return text.strip() or None


def read_crop_with_gemini(crop_bgr, det_class: str) -> str | None:
   
    if not Config.GEMINI_API_KEY:
        # No key configured — fail closed and quietly, not with a crash
        return None

    try:
        model = _get_gemini()
        ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return None
        image_bytes = buf.tobytes()

        prompt = _LABEL_PROMPT if det_class == "room_label" else _DIM_PROMPT

        response = model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 200,
            },
        )
        text = _extract_text(response)
        if not text:
            print("  [GEMINI-FALLBACK] no usable text in response "
                  "(empty/blocked/truncated)")
            return None

        text = text.strip().upper()
        if not text or text == "NONE":
            return None
        return text

    except Exception as e:
        print(f"  [GEMINI-FALLBACK] error: {e}")
        return None