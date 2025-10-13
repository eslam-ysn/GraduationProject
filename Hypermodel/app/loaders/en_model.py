import os
from .common import load_hf_bundle

EN_MODEL_DIR = os.getenv("EN_MODEL_DIR")
if not EN_MODEL_DIR:
    raise RuntimeError("EN_MODEL_DIR is not set in environment (.env)")

EN_BUNDLE = load_hf_bundle(EN_MODEL_DIR)
