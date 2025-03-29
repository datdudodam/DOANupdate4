from flask import request, jsonify
from utils.field_matcher import get_field_matcher, auto_fill_form
from utils.ml_recommender import get_ml_suggestions, get_recommender
import json

def register_enhanced_routes(app):
    """
    Đăng ký các route cho tính năng nâng cao
    """
    @app.route('/auto_fill_form', methods=['POST'])
    def auto_fill_form_route():
        """
        API endpoint để tự động điền biểu mẫu dựa trên lịch sử
        
        Request JSON:
        {
            "target_fields": ["[_001_]", "[_002_]", ...]
        }
        
        Response:
        {
            "auto_fill_data": {"[_001_]": "giá trị", ...},
            "error_details": "Chi tiết lỗi nếu có"
        }
        """
        try:
            data = request.json
            if not data or 'target_fields' not in data:
                return jsonify({
                    'error': 'Target fields are required',
                    'error_details': 'Thiếu danh sách trường cần điền'
                }), 400
                
            target_fields = data.get('target_fields', [])
            
            # Lấy user_id từ session (nếu đã đăng nhập)
            user_id = None
            # TODO: Lấy user_id từ Flask-Login nếu đã triển khai
            # from flask_login import current_user
            # if current_user.is_authenticated:
            #     user_id = current_user.id
            
            # Lấy dữ liệu tự động điền
            auto_fill_data = auto_fill_form(user_id, target_fields)
            
            return jsonify({
                'auto_fill_data': auto_fill_data
            }), 200
            
        except Exception as e:
            print(f"Error in auto_fill_form: {str(e)}")
            return jsonify({
                'error': 'Failed to auto-fill form',
                'error_details': str(e)
            }), 500
    
    @app.route('/get_enhanced_suggestions', methods=['POST'])
    def get_enhanced_suggestions_route():
        """
        API endpoint để lấy gợi ý nâng cao cho một trường cụ thể
        
        Request JSON:
        {
            "field_code": "[_XXX_]",  # Mã trường cần gợi ý
            "partial_form": {},      # Dữ liệu đã điền trong form (tùy chọn)
            "context_text": ""       # Văn bản ngữ cảnh (tùy chọn)
        }
        
        Response:
        {
            "field_code": "[_XXX_]",
            "suggestions": ["gợi ý 1", "gợi ý 2", ...],
            "error_details": "Chi tiết lỗi nếu có"
        }
        """
        try:
            data = request.json
            if not data or 'field_code' not in data:
                return jsonify({
                    'error': 'Field code is required',
                    'error_details': 'Thiếu mã trường cần gợi ý'
                }), 400
                
            field_code = data.get('field_code')
            partial_form = data.get('partial_form', {})
            context_text = data.get('context_text', '')
            
            # Kiểm tra field_code có hợp lệ không
            if not field_code or not isinstance(field_code, str):
                return jsonify({
                    'field_code': field_code,
                    'suggestions': [],
                    'error_details': 'Mã trường không hợp lệ'
                }), 200
            
            # Lấy gợi ý từ ML nâng cao
            suggestions = get_ml_suggestions(field_code, partial_form, context_text)
            
            # Nếu không có gợi ý, thử sử dụng field matcher để tìm trường tương tự
            if not suggestions:
                matcher = get_field_matcher()
                
                # Tìm các trường tương tự trong lịch sử
                similar_fields = []
                for form in matcher.form_history:
                    if 'form_data' in form:
                        form_data = form['form_data']
                        for f_code in form_data.keys():
                            if f_code not in similar_fields and matcher.extract_field_name(f_code) == matcher.extract_field_name(field_code):
                                similar_fields.append(f_code)
                
                # Lấy giá trị từ các trường tương tự
                for similar_field in similar_fields:
                    field_values = []
                    for form in matcher.form_history:
                        if 'form_data' in form and similar_field in form['form_data']:
                            value = form['form_data'][similar_field]
                            if value and value not in field_values:
                                field_values.append(value)
                    
                    if field_values:
                        return jsonify({
                            'field_code': field_code,
                            'suggestions': field_values[:5],
                            'info': f'Gợi ý từ trường tương tự: {similar_field}'
                        }), 200
            
            return jsonify({
                'field_code': field_code,
                'suggestions': suggestions
            }), 200
            
        except Exception as e:
            print(f"Error in get_enhanced_suggestions: {str(e)}")
            return jsonify({
                'field_code': data.get('field_code', ''),
                'suggestions': [],
                'error_details': str(e)
            }), 200  # Trả về 200 ngay cả khi có lỗi để không làm gián đoạn trải nghiệm người dùng