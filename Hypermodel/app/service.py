# app/service.py
import os
import math
import itertools
from typing import Dict, List, Tuple
import torch

from .utils.langdet import detect_lang
from .loaders.ar_model import AR_BUNDLE
from .loaders.en_model import EN_BUNDLE
from .loaders.common import DEVICE

def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    s = sum(e)
    return [v / s for v in e]

@torch.inference_mode()
def logits_for_text(text: str, bundle, max_length: int = 256) -> List[float]:
    tok = bundle["tokenizer"]; mdl = bundle["model"]
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    return mdl(**inputs).logits.squeeze(0).tolist()

@torch.inference_mode()
def predict_one(text: str, bundle, id2label: Dict[int, str], max_length: int = 256) -> Tuple[str, float, str]:
    tok = bundle["tokenizer"]; mdl = bundle["model"]
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = mdl(**inputs).logits.squeeze(0).tolist()
    ps = softmax(logits)
    idx = int(max(range(len(ps)), key=lambda i: ps[i]))
    return id2label.get(idx, str(idx)).lower(), float(ps[idx]), bundle["version"]

SEEDS_AR = {
    "positive": ["الخدمة ممتازة والطعام لذيذ جدًا","التجربة رائعة وسأعود مرة أخرى","جودة ممتازة وسعر مناسب"],
    "negative": ["سيء جدًا والخدمة بطيئة","التجربة كانت مزعجة والطعام بارد","لا أنصح به إطلاقًا"],
    "neutral":  ["الطعام عادي والخدمة مقبولة","مكان متوسط لا أكثر ولا أقل","التجربة كانت عادية"],
}
SEEDS_EN = {
    "positive": ["Amazing food and great service","Excellent experience, I will come again","Very tasty and well priced"],
    "negative": ["Terrible service and cold food","Very bad experience, not recommended","I will never come back"],
    "neutral":  ["Average food and acceptable service","It was okay, nothing special","The experience was fine"],
}

def _avg_probs(bundle, texts):
    acc = [0.0, 0.0, 0.0]
    for t in texts:
        lg = logits_for_text(t, bundle)
        m = max(lg)
        e = [math.exp(x - m) for x in lg]
        s = sum(e)
        p = [v/s for v in e]
        acc = [a+b for a,b in zip(acc, p)]
    return [v/len(texts) for v in acc]

def auto_calibrate_labels(bundle, seeds):
    avg = {
        "positive": _avg_probs(bundle, seeds["positive"]),
        "neutral":  _avg_probs(bundle, seeds["neutral"]),
        "negative": _avg_probs(bundle, seeds["negative"]),
    }
    best_perm, best_score = None, -1.0
    for perm in itertools.permutations([0,1,2], 3):
        score = avg["positive"][perm[0]] + avg["neutral"][perm[1]] + avg["negative"][perm[2]]
        if score > best_score:
            best_perm, best_score = perm, score
    return {best_perm[0]:"positive", best_perm[1]:"neutral", best_perm[2]:"negative"}

try:
    AR_ID2LABEL = auto_calibrate_labels(AR_BUNDLE, SEEDS_AR)
    EN_ID2LABEL = auto_calibrate_labels(EN_BUNDLE, SEEDS_EN)
    print("[AR mapping]", AR_ID2LABEL)
    print("[EN mapping]", EN_ID2LABEL)
except Exception as e:
    print("Calibration failed, fallback 0:neg,1:neu,2:pos", e)
    AR_ID2LABEL = {0:"negative",1:"neutral",2:"positive"}
    EN_ID2LABEL = {0:"negative",1:"neutral",2:"positive"}

ALWAYS_ENSEMBLE_WHEN_LANG_MISSING = os.getenv("ALWAYS_ENSEMBLE_WHEN_LANG_MISSING", "false").lower() in ("1","true","yes")

def route_and_predict(text: str, lang_hint: str | None = None) -> dict:
    if (lang_hint is None) and ALWAYS_ENSEMBLE_WHEN_LANG_MISSING:
        la, sa, va = predict_one(text, AR_BUNDLE, AR_ID2LABEL)
        le, se, ve = predict_one(text, EN_BUNDLE, EN_ID2LABEL)
        if sa >= se:
            return {"label": la, "score": sa, "model_version": va, "ensemble": True, "lang": "mixed"}
        return {"label": le, "score": se, "model_version": ve, "ensemble": True, "lang": "mixed"}

    lang = (lang_hint or detect_lang(text)).lower()

    if lang == "ar":
        label, score, version = predict_one(text, AR_BUNDLE, AR_ID2LABEL)
        return {"label": label, "score": score, "model_version": version, "ensemble": False, "lang": "ar"}

    if lang == "en":
        label, score, version = predict_one(text, EN_BUNDLE, EN_ID2LABEL)
        return {"label": label, "score": score, "model_version": version, "ensemble": False, "lang": "en"}

    la, sa, va = predict_one(text, AR_BUNDLE, AR_ID2LABEL)
    le, se, ve = predict_one(text, EN_BUNDLE, EN_ID2LABEL)
    if sa >= se:
        return {"label": la, "score": sa, "model_version": va, "ensemble": True, "lang": "mixed"}
    return {"label": le, "score": se, "model_version": ve, "ensemble": True, "lang": "mixed"}
