import os

# Đường dẫn gốc của ứng dụng
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cấu hình đường dẫn
DB_PATH = os.path.join(BASE_DIR, "form_data.json")
FORM_HISTORY_PATH = os.path.join(BASE_DIR, "form_history.json")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# Đảm bảo thư mục uploads tồn tại
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# Cấu hình OpenAI
OPENAI_API_KEY = 'your-api-key-here'  # Thay thế bằng API key thực tế

# Cấu hình ứng dụng
DEBUG = True