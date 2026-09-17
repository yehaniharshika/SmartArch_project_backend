"""
SmartArch — services/extraction/text_cleaner.py

clean_ocr_text() is for DIMENSION text only.
NEVER apply it to room label text — O→0 / I→1 replacement corrupts
labels (e.g. "TOILET"→"T0ILET", "LOBBY"→"1OBBY").

clean_label_text() is the ROOM LABEL equivalent — it strips stray
punctuation/bracket noise picked up from nearby plan annotations
(e.g. "KITCHEN]" → "KITCHEN") without ever touching letters or
digits, so it is always safe to use on label text.
"""
import re


def clean_ocr_text(text: str) -> str:
    """
    Normalise DIMENSION text only (not labels).
    Fixes common EasyOCR misreads in dimension strings.
    """
    if not text:
        return ""

    t = text.upper().strip()

    # Digit-lookalike replacements — safe for dimension text only,
    # since a dimension string should never legitimately contain
    # letters. I/L are added because EasyOCR frequently misreads the
    # digit '1' as a capital I or L on thin/small CAD-style fonts
    # (e.g. "8'X10'" → "8'XI0'").
    digit_fixes = {
        "O": "0",   # letter O → zero
        "Q": "0",
        "I": "1",   # capital I → one
        "L": "1",   # capital L → one
        "×": "X",   # multiplication sign → X
        "÷": "X",
        "\u2019": "'",  # right single quote → apostrophe
        "\u201c": '"',  # left double quote → double quote
        "\u201d": '"',  # right double quote → double quote
    }

    result = []
    for ch in t:
        if ch in digit_fixes:
            result.append(digit_fixes[ch])
        else:
            result.append(ch)

    t = "".join(result)

    # Remove spaces inside dimension expressions ("13' 4\"" → "13'4\"")
    t = re.sub(r"(\d+)'\s+(\d)", r"\1'\2", t)

    # Normalise X separator
    t = re.sub(r"\s*[xX×]\s*", "X", t)

    # Fix "13'4"X10'" format consistency
    t = re.sub(r"(\d+)'(\d+)[\"']?X(\d+)'?(\d*)[\"']?",
               lambda m: (f"{m.group(1)}'{m.group(2)}\"X{m.group(3)}'"
                          f"{m.group(4)}\"" if m.group(4)
                          else f"{m.group(1)}'{m.group(2)}\"X{m.group(3)}'"),
               t)

    # Remove trailing noise characters
    t = re.sub(r"[^0-9'\"X\.\-]$", "", t).strip()

    return t


def clean_label_text(text: str) -> str:
    
    if not text:
        return ""

    t = text.strip().upper()

    # Strip leading/trailing characters that aren't letters, digits,
    # or spaces (brackets, stray punctuation, noise marks)
    t = re.sub(r"^[^A-Z0-9]+", "", t)
    t = re.sub(r"[^A-Z0-9]+$", "", t)

    # Collapse repeated internal whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t


def is_dimension_crop(det_class: str, text: str) -> bool:
    """True if this crop should be treated as dimension text."""
    return det_class == "dimension_text"


def is_label_crop(det_class: str) -> bool:
    """True if this crop should be treated as a room label."""
    return det_class == "room_label"