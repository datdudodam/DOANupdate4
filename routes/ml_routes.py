from flask import request, jsonify
from utils.ml_recommender import get_ml_suggestions

def register_ml_routes(app):
    """
    Đăng ký các route cho ML recommendations
    """
    @app.route('/get_ml_suggestions', methods=['POST'])
    def get_ml_suggestions_route():
        """
        API endpoint để lấy gợi ý từ ML cho một trường cụ thể
        
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
            
            # Lấy gợi ý từ ML
            suggestions = get_ml_suggestions(field_code, partial_form, context_text)
            
            # Nếu không có gợi ý, thêm thông tin chi tiết về lỗi
            if not suggestions:
                # Kiểm tra xem có dữ liệu lịch sử form không
                from utils.ml_recommender import get_recommender
                recommender = get_recommender()
                
                error_details = ''
                if not recommender.form_data:
                    error_details = 'Chưa có dữ liệu lịch sử form để đưa ra gợi ý'
                else:
                    # Kiểm tra xem field_code có xuất hiện trong bất kỳ form nào không
                    field_exists = False
                    field_has_value = False
                    for form in recommender.form_data:
                        if field_code in form:
                            field_exists = True
                            if form[field_code]:  # Kiểm tra xem trường có giá trị không
                                field_has_value = True
                                break
                    
                    if not field_exists:
                        error_details = f'Trường {field_code} chưa có dữ liệu trong lịch sử form'
                    elif not field_has_value:
                        error_details = f'Trường {field_code} chưa có giá trị trong lịch sử form'
                    else:
                        # Kiểm tra xem có trường nào đã được điền trong form hiện tại không
                        filled_fields = {k: v for k, v in partial_form.items() if v and k != field_code}
                        if not filled_fields:
                            error_details = 'Vui lòng điền thêm thông tin vào các trường khác để nhận gợi ý chính xác hơn'
                        else:
                            error_details = 'Không đủ dữ liệu liên quan để đưa ra gợi ý cho trường này. Hệ thống đang học thêm.'
                
                # Thử lấy các giá trị phổ biến nhất cho trường này
                common_values = recommender._get_most_common_values(field_code)
                if common_values:
                    return jsonify({
                        'field_code': field_code,
                        'suggestions': common_values,
                        'warning': error_details  # Đổi error_details thành warning
                    }), 200
                else:
                    return jsonify({
                        'field_code': field_code,
                        'suggestions': [],
                        'error_details': error_details
                    }), 200
            
            return jsonify({
                'field_code': field_code,
                'suggestions': suggestions
            }), 200
            
        except Exception as e:
            print(f"Error getting ML suggestions: {str(e)}")
            return jsonify({
                'error': 'Failed to get ML suggestions',
                'error_details': str(e)
            }), 500