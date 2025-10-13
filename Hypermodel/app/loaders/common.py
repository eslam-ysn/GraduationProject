import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():  # لأجهزة Mac
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

DEVICE = get_device()

def load_hf_bundle(path: str):
    """
    يتوقع وجود: config.json, tokenizer_config.json, vocab.json (+ merges لروبيرتا)
    وملف weights: model.safetensors أو pytorch_model.bin داخل المسار.
    """
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    mdl.to(DEVICE); mdl.eval()
    # مؤقتًا: نسميها LABEL_0..2، وسيتم استبدالها بالمعايرة لاحقًا
    id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    label2id = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
    version = getattr(cfg, "_name_or_path", os.path.basename(path))
    return {"config": cfg, "tokenizer": tok, "model": mdl,
            "id2label": id2label, "label2id": label2id, "version": version}
