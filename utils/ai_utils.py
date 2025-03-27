from collections import defaultdict, Counter
from openai import OpenAI
from config.config import OPENAI_API_KEY

# Khởi tạo OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_suggestions(db_data, field_code=None):
    """
    Tạo gợi ý cho các trường dựa trên dữ liệu lịch sử và OpenAI API
    """
    suggestions = defaultdict(Counter)  # Dùng Counter để đếm tần suất các giá trị

    # Tổng hợp dữ liệu từ cơ sở dữ liệu
    for entry in db_data:
        for key, value in entry["data"].items():
            if field_code is None or key == field_code:
                suggestions[key][value] += 1  # Đếm số lần xuất hiện

    result = {}
    for key, counter in suggestions.items():
        # Lấy 3 giá trị phổ biến nhất từ lịch sử nhập liệu
        most_common_values = [val for val, _ in counter.most_common(3)]

        # Gọi OpenAI API để lấy thêm gợi ý thông minh
        gpt_suggestions = []
        if len(counter) > 1:  # Chỉ gọi GPT nếu có đủ dữ liệu để phân tích
            try:
                prompt = (
                    f"Dựa trên các giá trị lịch sử cho trường '{key}': {list(counter.keys())}, "
                    "hãy gợi ý 5 giá trị phù hợp nhất cho trường này. Chỉ trả về danh sách, không kèm theo giải thích."
                )
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,
                    max_tokens=100
                )

                # Xử lý kết quả GPT trả về
                gpt_suggestions = [s.strip() for s in response.choices[0].message.content.split(",") if s.strip()]
                gpt_suggestions = gpt_suggestions[:5]  # Giới hạn 5 giá trị

            except Exception as e:
                print(f"Lỗi OpenAI API: {e}")

        # Hợp nhất giá trị lịch sử và gợi ý từ GPT, loại bỏ trùng lặp
        combined_suggestions = list(dict.fromkeys(most_common_values + gpt_suggestions))[:5]
        result[key] = combined_suggestions

    return result if field_code is None else result.get(field_code, [])

def get_gpt_suggestions(field_code):
    """
    Lấy gợi ý từ GPT cho một trường cụ thể
    """
    if not client.api_key or client.api_key == 'your-api-key-here':
        return {"error": "OpenAI API key is not configured"}, 500

    suggestions = []

    try:
        prompt = f"Hãy gợi ý 5 giá trị phù hợp nhất cho trường '{field_code}'. Chỉ trả về danh sách, không kèm theo giải thích."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI giúp đề xuất giá trị phù hợp cho các trường trong biểu mẫu."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )

        suggestions = [s.strip() for s in response.choices[0].message.content.split(",") if s.strip()]
        suggestions = suggestions[:5]  # Giới hạn 5 giá trị

    except Exception as e:
        print(f"Lỗi OpenAI API: {e}")
        return {"error": "Có lỗi xảy ra khi gọi API"}, 500

    return {"field_code": field_code, "suggestions": suggestions}, 200