import re
from docx import Document
import os
import uuid
from werkzeug.utils import secure_filename
from config.config import UPLOADS_DIR

# Biến toàn cục để lưu đường dẫn tài liệu hiện tại
doc_path = None

def load_document(doc_path):
    """
    Tải nội dung từ tài liệu docx
    """
    doc = Document(doc_path)
    return "\n".join([para.text.strip() for para in doc.paragraphs])

def extract_fields(text):
    """
    Trích xuất các trường từ văn bản
    """
    matches = re.finditer(r"\[_\d+_\]", text)
    fields = []
    special_words = {"ngày", "tháng", "năm"}
    
    for match in matches:
        end_index = match.start()
        field_name = ""
        i = end_index - 1
        while i >= 0 and text[i] == " ":
            i -= 1
        
        for word in special_words:
            if text[max(0, i - len(word) + 1):end_index].strip().lower() == word:
                field_name = word
                break
        
        if not field_name:
            for j in range(i, -1, -1):
                if text[j].isupper():
                    field_name = text[j:end_index].strip()
                    break
        
        if field_name:
            fields.append({"field_name": field_name, "field_code": match.group()})
    return fields

def upload_document(file):
    """
    Xử lý tải lên tài liệu
    """
    global doc_path
    
    if file.filename == '':
        return {"error": "No selected file"}, 400
    
    if not file.filename.endswith('.docx'):
        return {"error": "Only DOCX files are allowed"}, 400
    
    filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
    filepath = os.path.join(UPLOADS_DIR, filename)
    file.save(filepath)
    
    doc_path = filepath
    
    return {"message": "File uploaded successfully", "filename": filename}, 200

def get_doc_path():
    """
    Trả về đường dẫn tài liệu hiện tại
    """
    return doc_path

def set_doc_path(path):
    """
    Thiết lập đường dẫn tài liệu hiện tại
    """
    global doc_path
    doc_path = path