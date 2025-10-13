import os
from pathlib import Path
from dotenv import load_dotenv

# احسب جذر المشروع انطلاقاً من هذا الملف: app/__init__.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# حمّل .env بشكل صريح
loaded = load_dotenv(dotenv_path=ENV_PATH, override=True)

# تشخيص مفيد
print(f"[DEBUG] .env path = {ENV_PATH} | exists = {ENV_PATH.exists()} | loaded = {loaded}")
print("AR_MODEL_DIR =", os.getenv("AR_MODEL_DIR"))
print("EN_MODEL_DIR =", os.getenv("EN_MODEL_DIR"))
