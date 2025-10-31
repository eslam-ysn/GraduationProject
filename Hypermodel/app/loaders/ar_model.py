import os
from .common import load_hf_bundle

print("DEBUG AR_MODEL_DIR =", os.getenv("AR_MODEL_DIR"))

AR_MODEL_DIR = os.getenv("AR_MODEL_DIR")
if not AR_MODEL_DIR:
    raise RuntimeError("AR_MODEL_DIR is not set in environment (.env)")

AR_BUNDLE = load_hf_bundle(AR_MODEL_DIR)
