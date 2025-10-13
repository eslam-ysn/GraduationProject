import os
import re
from typing import Literal


AR_MIN_MIXED = float(os.getenv("MIXED_AR_MIN", "0.12"))  # نسبة أحرف عربية ≥ 12%
EN_MIN_MIXED = float(os.getenv("MIXED_EN_MIN", "0.12"))  # نسبة أحرف لاتينية ≥ 12%

LD_MIN_AR = float(os.getenv("LD_MIN_AR", "0.20"))        # arabic prob ≥ 20%
LD_MIN_EN = float(os.getenv("LD_MIN_EN", "0.20"))        # english prob ≥ 20%
LD_MAX_DIFF = float(os.getenv("LD_MAX_DIFF", "0.25"))    # الفرق ≤ 25% لنعتبره مختلط

SHORT_LEN = int(os.getenv("LD_SHORT_LEN", "8"))



AR_RE = re.compile(r'[\u0600-\u06FF]')
EN_RE = re.compile(r'[A-Za-z]')

def _fallback_detect_by_chars(text: str) -> str:
    text = text.strip()
    if not text:
        return "en"

    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')

    total = arabic_chars + latin_chars
    if total == 0:
        return "en"

    ar_ratio = arabic_chars / total
    en_ratio = latin_chars / total

    # mixed if both above threshold
    if ar_ratio >= 0.05 and en_ratio >= 0.05:
        return "mixed"
    return "ar" if ar_ratio > en_ratio else "en"



# =======================
def detect_lang(text: str) -> Literal["ar", "en", "mixed"]:


    if not text or len(text.strip()) < SHORT_LEN:
        return _fallback_detect_by_chars(text or "")

    try:
        from langdetect import detect_langs
    except Exception:
        return _fallback_detect_by_chars(text)

    try:
        probs = detect_langs(text)
        p_map = {str(p.lang): float(p.prob) for p in probs}
        ar_p = p_map.get("ar", 0.0)
        en_p = p_map.get("en", 0.0)

        if (ar_p >= LD_MIN_AR) and (en_p >= LD_MIN_EN) and (abs(ar_p - en_p) <= LD_MAX_DIFF):
            return "mixed"

        return "ar" if ar_p >= en_p else "en"

    except Exception:
        return _fallback_detect_by_chars(text)
