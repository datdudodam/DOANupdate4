from flask import request, jsonify, make_response
from utils.docx_generator import generate_docx
from utils.document_utils import get_doc_path
from models.data_model import load_db
import os

def register_docx_routes(app):
    """
    Đăng ký các route cho việc tạo và xuất tài liệu docx
    """
    @app.route('/generate-docx', methods=['POST'])
    def generate_docx_route():
        try:
            # Kiểm tra và xử lý dữ liệu từ request
            request_data = None
            form_data = None
            
            if request.is_json:
                try:
                    request_data = request.get_json()
                    if request_data and isinstance(request_data, dict):
                        form_data = {k: v for k, v in request_data.items() if k != 'filename'}
                except Exception as e:
                    print(f"Error parsing JSON: {str(e)}")
                    return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400
            else:
                try:
                    form_data = request.form.to_dict()
                except Exception as e:
                    print(f"Error parsing form data: {str(e)}")
                    return jsonify({'error': 'Không thể đọc dữ liệu form'}), 400

            # Nếu không có dữ liệu form, sử dụng dữ liệu mới nhất từ DB
            if not form_data or not any(form_data.values()):
                db_data = load_db()
                if not db_data:
                    return jsonify({'error': 'Không có dữ liệu để tạo tài liệu'}), 400
                form_data = db_data[-1]['data']

            custom_filename = request_data.get('filename') if request_data else None
            doc_path = get_doc_path()
            
            # Gọi hàm tạo tài liệu
            result, status_code = generate_docx(form_data, doc_path, custom_filename)
            
            if status_code != 200:
                return jsonify(result), status_code
                
            # Xử lý kết quả thành công
            doc_data = result.get("doc_data")
            ascii_filename = result.get("ascii_filename")
            utf8_filename = result.get("utf8_filename")
            temp_doc_path = result.get("temp_doc_path")
            
            # Tạo response
            response = make_response(doc_data)
            response.headers.set('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
            response.headers.set('Content-Disposition', f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{utf8_filename}')
            response.headers.set('Content-Length', str(len(doc_data)))
            response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate')
            response.headers.set('Pragma', 'no-cache')
            response.headers.set('Expires', '0')
            response.headers.set('Access-Control-Expose-Headers', 'Content-Disposition')
            response.headers.set('X-Content-Type-Options', 'nosniff')
            
            # Xóa file tạm sau khi gửi
            try:
                if os.path.exists(temp_doc_path):
                    os.remove(temp_doc_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not remove temporary file: {str(cleanup_error)}")
                
            return response

        except Exception as e:
            print(f"Error generating document: {str(e)}")
            return jsonify({'error': 'Có lỗi xảy ra khi tạo tài liệu. Vui lòng thử lại sau.'}), 500