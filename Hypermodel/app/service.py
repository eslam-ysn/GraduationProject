# app/service.py
import os
import math
import itertools
import re
from typing import Dict, List, Tuple
import torch

from .utils.langdet import detect_lang
from .loaders.ar_model import AR_BUNDLE
from .loaders.en_model import EN_BUNDLE
from .loaders.common import DEVICE


# Compute the softmax probabilities for a list of logits
def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    s = sum(e)
    return [v / s for v in e]


@torch.inference_mode()
# Generate model logits for the entire text without truncation
def logits_for_text(text: str, bundle) -> List[float]:
    tok = bundle["tokenizer"]
    mdl = bundle["model"]
    inputs = tok(text, return_tensors="pt", truncation=False, padding=False)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    return mdl(**inputs).logits.squeeze(0).tolist()


@torch.inference_mode()
# Predict sentiment for the input text without length limits
def predict_one(text: str, bundle, id2label: Dict[int, str]) -> Tuple[str, float, str]:
    tok = bundle["tokenizer"]
    mdl = bundle["model"]
    inputs = tok(text, return_tensors="pt", truncation=False, padding=False)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = mdl(**inputs).logits.squeeze(0).tolist()
    ps = softmax(logits)
    idx = int(max(range(len(ps)), key=lambda i: ps[i]))
    return id2label.get(idx, str(idx)).lower(), float(ps[idx]), bundle["version"]


# Seed examples for score calibration
SEEDS_AR = {
    "positive": ["الخدمة ممتازة والطعام لذيذ جدًا", "التجربة رائعة وسأعود مرة أخرى", "جودة ممتازة وسعر مناسب"],
    "negative": ["سيء جدًا والخدمة بطيئة", "التجربة كانت مزعجة والطعام بارد", "لا أنصح به إطلاقًا"],
    "neutral": ["الطعام عادي والخدمة مقبولة", "مكان متوسط لا أكثر ولا أقل", "التجربة كانت عادية"],
}
SEEDS_EN = {
    "positive": ["Amazing food and great service", "Excellent experience, I will come again",
                 "Very tasty and well priced"],
    "negative": ["Terrible service and cold food", "Very bad experience, not recommended", "I will never come back"],
    "neutral": ["Average food and acceptable service", "It was okay, nothing special", "The experience was fine"],
}


# Compute the average probability distribution over seed texts
def _avg_probs(bundle, texts):
    acc = [0.0, 0.0, 0.0]
    for t in texts:
        lg = logits_for_text(t, bundle)
        m = max(lg)
        e = [math.exp(x - m) for x in lg]
        s = sum(e)
        p = [v / s for v in e]
        acc = [a + b for a, b in zip(acc, p)]
    return [v / len(texts) for v in acc]


# Re-map sentiment labels based on seed performance for improved accuracy
def auto_calibrate_labels(bundle, seeds):
    avg = {
        "positive": _avg_probs(bundle, seeds["positive"]),
        "neutral": _avg_probs(bundle, seeds["neutral"]),
        "negative": _avg_probs(bundle, seeds["negative"]),
    }
    best_perm, best_score = None, -1.0
    for perm in itertools.permutations([0, 1, 2], 3):
        score = avg["positive"][perm[0]] + avg["neutral"][perm[1]] + avg["negative"][perm[2]]
        if score > best_score:
            best_perm, best_score = perm, score
    return {best_perm[0]: "positive", best_perm[1]: "neutral", best_perm[2]: "negative"}


try:
    AR_ID2LABEL = auto_calibrate_labels(AR_BUNDLE, SEEDS_AR)
    EN_ID2LABEL = auto_calibrate_labels(EN_BUNDLE, SEEDS_EN)
    print("[AR mapping]", AR_ID2LABEL)
    print("[EN mapping]", EN_ID2LABEL)
except Exception as e:
    print("Calibration failed, fallback 0:neg,1:neu,2:pos", e)
    AR_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
    EN_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


# Check if a Unicode character belongs to Arabic script
def is_arabic(char: str) -> bool:
    return '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' or '\u08A0' <= char <= '\u08FF'


# Check if a character belongs to English alphabet
def is_english(char: str) -> bool:
    return ('a' <= char.lower() <= 'z')


# Separate text into Arabic and English segments
def split_text_by_language(text: str) -> Dict[str, str]:
    ar_chars = []
    en_chars = []

    for char in text:
        if is_arabic(char):
            ar_chars.append(char)
        elif is_english(char):
            en_chars.append(char)
        else:
            ar_chars.append(char)
            en_chars.append(char)

    ar_text = ''.join(ar_chars).strip()
    en_text = ''.join(en_chars).strip()

    return {"ar": ar_text, "en": en_text}


# Check if input contains both Arabic and English characters
def detect_mixed_language(text: str) -> bool:
    has_arabic = any(is_arabic(c) for c in text)
    has_english = any(is_english(c) for c in text)
    return has_arabic and has_english


# Select the best prediction when two models are evaluated
def fuse_predictions(ar_label: str, ar_score: float, en_label: str, en_score: float) -> Tuple[str, float]:
    if ar_label == "neutral" and en_label != "neutral":
        return en_label, en_score
    if en_label == "neutral" and ar_label != "neutral":
        return ar_label, ar_score
    if ar_label == "neutral" and en_label == "neutral":
        return (ar_label, ar_score) if ar_score >= en_score else (en_label, en_score)
    return (ar_label, ar_score) if ar_score >= en_score else (en_label, en_score)


# Main routing and prediction entry point with mixed-language support
def route_and_predict(text: str, lang_hint: str | None = None) -> dict:
    text = text.strip()
    if not text:
        return {"label": "neutral", "score": 0.0, "model_version": "none", "ensemble": False, "lang": "unknown"}

    is_mixed = detect_mixed_language(text)

    if is_mixed:
        segments = split_text_by_language(text)
        ar_text = segments["ar"]
        en_text = segments["en"]

        ar_label, ar_score, ar_version = "neutral", 0.0, AR_BUNDLE["version"]
        if ar_text:
            ar_label, ar_score, ar_version = predict_one(ar_text, AR_BUNDLE, AR_ID2LABEL)

        en_label, en_score, en_version = "neutral", 0.0, EN_BUNDLE["version"]
        if en_text:
            en_label, en_score, en_version = predict_one(en_text, EN_BUNDLE, EN_ID2LABEL)

        final_label, final_score = fuse_predictions(ar_label, ar_score, en_label, en_score)
        winning_model = ar_version if (final_label == ar_label and final_score == ar_score) else en_version

        return {
            "label": final_label,
            "score": final_score,
            "model_version": winning_model,
            "ensemble": True,
            "lang": "mixed",
            "ar_prediction": {"label": ar_label, "score": ar_score, "model": ar_version},
            "en_prediction": {"label": en_label, "score": en_score, "model": en_version}
        }

    lang = (lang_hint or detect_lang(text)).lower()

    if lang == "ar":
        label, score, version = predict_one(text, AR_BUNDLE, AR_ID2LABEL)
        return {
            "label": label,
            "score": score,
            "model_version": version,
            "ensemble": False,
            "lang": "ar"
        }

    if lang == "en":
        label, score, version = predict_one(text, EN_BUNDLE, EN_ID2LABEL)
        return {
            "label": label,
            "score": score,
            "model_version": version,
            "ensemble": False,
            "lang": "en"
        }

    ar_label, ar_score, ar_version = predict_one(text, AR_BUNDLE, AR_ID2LABEL)
    en_label, en_score, en_version = predict_one(text, EN_BUNDLE, EN_ID2LABEL)

    final_label, final_score = fuse_predictions(ar_label, ar_score, en_label, en_score)
    winning_model = ar_version if (final_label == ar_label and final_score == ar_score) else en_version

    return {
        "label": final_label,
        "score": final_score,
        "model_version": winning_model,
        "ensemble": True,
        "lang": "unknown",
        "ar_prediction": {"label": ar_label, "score": ar_score, "model": ar_version},
        "en_prediction": {"label": en_label, "score": en_score, "model": en_version}
    }
