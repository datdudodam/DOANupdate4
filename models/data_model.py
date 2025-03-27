import json
import os
from config.config import DB_PATH, FORM_HISTORY_PATH

def load_db():
    """
    Tải dữ liệu từ file JSON
    """
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_db(data):
    """
    Lưu dữ liệu vào file JSON
    """
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_form_history():
    """
    Tải lịch sử biểu mẫu từ file JSON
    """
    if os.path.exists(FORM_HISTORY_PATH):
        with open(FORM_HISTORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_form_history(data):
    """
    Lưu lịch sử biểu mẫu vào file JSON
    """
    with open(FORM_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)