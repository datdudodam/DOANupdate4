import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Removed surprise import that was causing issues

from collections import defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import json
from config.config import FORM_HISTORY_PATH


# Đảm bảo các tài nguyên NLTK được tải xuống
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class MLRecommender:
    def __init__(self):
        self.form_data = []
        self.field_vectors = {}
        self.user_item_matrix = None
        self.svd_model = None
        self.field_similarities = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Thêm stopwords tiếng Việt
        self.vietnamese_stopwords = {'của', 'và', 'các', 'có', 'được', 'trong', 'là', 'cho', 'những', 'với', 'không', 'này', 'đến', 'khi', 'về', 'như', 'từ', 'một', 'người', 'năm', 'bị', 'đã', 'sẽ', 'cũng', 'vào', 'ra', 'nếu', 'để', 'tại', 'theo', 'sau', 'trên', 'hoặc'}
        self.stop_words.update(self.vietnamese_stopwords)
        self.load_data()
        
    def load_data(self):
        """Tải dữ liệu từ form_history.json"""
        if os.path.exists(FORM_HISTORY_PATH):
            with open(FORM_HISTORY_PATH, 'r', encoding='utf-8') as f:
                form_history = json.load(f)
                
            # Chuyển đổi dữ liệu lịch sử thành định dạng phù hợp cho ML
            for form in form_history:
                if 'form_data' in form:
                    self.form_data.append(form['form_data'])
    
    def preprocess_text(self, text):
        """Tiền xử lý văn bản cho NLP"""
        if not text or not isinstance(text, str):
            return ""
            
        # Chuyển về chữ thường và loại bỏ ký tự đặc biệt
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize và loại bỏ stopwords
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return " ".join(tokens)
    
    def build_content_based_model(self):
        """Xây dựng mô hình Content-Based Filtering"""
        if not self.form_data:
            return
            
        # Tạo từ điển chứa tất cả các giá trị cho mỗi trường
        field_values = defaultdict(list)
        
        for form in self.form_data:
            for field, value in form.items():
                if isinstance(value, str) and value.strip():
                    processed_value = self.preprocess_text(value)
                    if processed_value:  # Chỉ thêm nếu không rỗng sau khi xử lý
                        field_values[field].append(processed_value)
        
        # Tạo vector TF-IDF cho mỗi trường
        for field, values in field_values.items():
            if len(values) > 1:  # Cần ít nhất 2 giá trị để tính toán similarity
                vectorizer = TfidfVectorizer()
                try:
                    tfidf_matrix = vectorizer.fit_transform(values)
                    self.field_vectors[field] = {
                        'vectorizer': vectorizer,
                        'matrix': tfidf_matrix,
                        'values': values
                    }
                except Exception as e:
                    print(f"Lỗi khi xây dựng vector TF-IDF cho trường {field}: {e}")
    
    def build_collaborative_model(self):
        """Xây dựng mô hình Collaborative Filtering sử dụng BiasedMF (SVD)"""
        if len(self.form_data) < 2:  # Cần ít nhất 2 form để so sánh
            return
            
        # Tạo ma trận user-item (form-field)
        # Mỗi form là một user, mỗi field là một item
        ratings_data = []
        
        for user_id, form in enumerate(self.form_data):
            for field, value in form.items():
                if value:  # Chỉ xét các trường đã có giá trị
                    # Đánh giá là 1 nếu có giá trị, có thể mở rộng để tính rating phức tạp hơn
                    ratings_data.append((user_id, field, 1))
        
        if not ratings_data:
            return
            
        # Chuyển đổi dữ liệu sang định dạng Surprise
        df = pd.DataFrame(ratings_data, columns=['user', 'item', 'rating'])
        reader = Reader(rating_scale=(0, 1))
        
        try:
            data = Dataset.load_from_df(df, reader)
            trainset, testset = train_test_split(data, test_size=0.2)
            
            # Sử dụng SVD (BiasedMF)
            self.svd_model = SVD(n_factors=10, n_epochs=20, lr_all=0.005, reg_all=0.02)
            self.svd_model.fit(trainset)
            
            # Đánh giá mô hình
            predictions = self.svd_model.test(testset)
            rmse = accuracy.rmse(predictions)
            print(f"RMSE của mô hình BiasedMF: {rmse}")
            
            self.user_item_matrix = df
        except Exception as e:
            print(f"Lỗi khi xây dựng mô hình Collaborative Filtering: {e}")
    
    def get_content_based_recommendations(self, partial_form, field_code):
        """Lấy gợi ý dựa trên nội dung đã nhập"""
        if not self.field_vectors or field_code not in self.field_vectors:
            return []
            
        # Tìm các trường đã có giá trị trong form hiện tại
        filled_fields = {k: v for k, v in partial_form.items() if v and k != field_code}
        
        # Nếu chưa có trường nào được điền, trả về các giá trị phổ biến nhất cho field_code
        if not filled_fields:
            return self._get_most_common_values(field_code)
            
        # Tính toán similarity giữa các trường
        if not self.field_similarities:
            self._calculate_field_similarities()
            
        # Tìm các trường có liên quan đến field_code
        related_fields = self._get_related_fields(field_code, threshold=0.2)  # Giảm ngưỡng xuống 0.2 (từ 0.3)
        
        # Lọc ra các trường đã điền và có liên quan
        relevant_filled_fields = {k: v for k, v in filled_fields.items() if k in related_fields}
        
        # Nếu không có trường liên quan nào được điền, sử dụng tất cả các trường đã điền
        if not relevant_filled_fields:
            # Sử dụng tất cả các trường đã điền thay vì chỉ các trường liên quan
            similar_forms = self._find_similar_forms(filled_fields)
        else:
            # Tìm các form có giá trị tương tự cho các trường đã điền và có liên quan
            similar_forms = self._find_similar_forms(relevant_filled_fields)
        
        # Lấy giá trị của field_code từ các form tương tự
        recommendations = []
        for form_idx, _ in similar_forms:
            if form_idx < len(self.form_data) and field_code in self.form_data[form_idx]:
                value = self.form_data[form_idx][field_code]
                if value and value not in recommendations:
                    recommendations.append(value)
                    if len(recommendations) >= 5:  # Giới hạn 5 gợi ý
                        break
        
        # Nếu không tìm thấy đủ gợi ý, bổ sung bằng các giá trị phổ biến nhất
        if len(recommendations) < 2:
            common_values = self._get_most_common_values(field_code)
            for value in common_values:
                if value not in recommendations:
                    recommendations.append(value)
                    if len(recommendations) >= 5:
                        break
                        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, field_code):
        """Lấy gợi ý dựa trên Collaborative Filtering nâng cao"""
        if not self.svd_model or not self.user_item_matrix.shape[0] > 0:
            return []
            
        try:
            # Dự đoán rating cho field_code
            # Nếu user_id mới, sử dụng user_id cuối cùng trong tập dữ liệu
            if user_id >= len(self.form_data):
                user_id = len(self.form_data) - 1
                
            prediction = self.svd_model.predict(user_id, field_code)
            
            # Cải tiến 1: Sử dụng ngưỡng rating động dựa trên dữ liệu
            # Tính toán ngưỡng dựa trên trung bình của các dự đoán
            all_predictions = []
            for u_id in range(min(5, len(self.form_data))):
                try:
                    pred = self.svd_model.predict(u_id, field_code)
                    all_predictions.append(pred.est)
                except:
                    pass
                    
            # Tính ngưỡng động nếu có đủ dữ liệu, nếu không sử dụng ngưỡng mặc định
            if all_predictions:
                dynamic_threshold = max(0.4, np.mean(all_predictions) * 0.8)
            else:
                dynamic_threshold = 0.5
                
            # Nếu dự đoán rating cao hơn ngưỡng, tìm các giá trị phổ biến cho field_code
            if prediction.est > dynamic_threshold:
                # Cải tiến 2: Tìm các form tương tự với form hiện tại
                similar_users = []
                
                # Tìm các user có hành vi tương tự
                if self.user_item_matrix is not None and len(self.form_data) > 1:
                    # Lấy các trường đã điền của user hiện tại
                    current_user_fields = set()
                    if user_id < len(self.form_data):
                        current_user_fields = set(self.form_data[user_id].keys())
                    
                    # Tìm các user có nhiều trường chung nhất
                    for u_id, form in enumerate(self.form_data):
                        if u_id != user_id:
                            common_fields = len(set(form.keys()).intersection(current_user_fields))
                            if common_fields > 0:
                                similar_users.append((u_id, common_fields))
                    
                    # Sắp xếp theo số lượng trường chung giảm dần
                    similar_users.sort(key=lambda x: x[1], reverse=True)
                    similar_users = [u[0] for u in similar_users[:5]]  # Lấy top 5 user tương tự
                
                # Tìm các form có điền field_code
                field_values = []
                
                # Ưu tiên lấy giá trị từ các user tương tự
                for u_id in similar_users:
                    if u_id < len(self.form_data) and field_code in self.form_data[u_id] and self.form_data[u_id][field_code]:
                        field_values.append(self.form_data[u_id][field_code])
                
                # Nếu chưa đủ, lấy thêm từ các form khác
                if len(field_values) < 5:
                    for form in self.form_data:
                        if field_code in form and form[field_code] and form[field_code] not in field_values:
                            field_values.append(form[field_code])
                            if len(field_values) >= 5:
                                break
                        
                # Đếm tần suất và lấy top 5 giá trị phổ biến nhất
                if field_values:
                    value_counts = pd.Series(field_values).value_counts()
                    return value_counts.index.tolist()[:5]
        except Exception as e:
            print(f"Lỗi khi lấy gợi ý collaborative: {e}")
            
        return []
    
    def get_nlp_recommendations(self, field_code, context_text=""):
        """Lấy gợi ý dựa trên phân tích NLP nâng cao"""
        if not self.field_vectors or field_code not in self.field_vectors:
            return []
            
        if not context_text:  # Nếu không có văn bản ngữ cảnh
            return []
            
        # Tiền xử lý văn bản ngữ cảnh
        processed_context = self.preprocess_text(context_text)
        
        # Chuyển đổi văn bản ngữ cảnh thành vector TF-IDF
        vectorizer = self.field_vectors[field_code]['vectorizer']
        values = self.field_vectors[field_code]['values']
        matrix = self.field_vectors[field_code]['matrix']
        
        try:
            context_vector = vectorizer.transform([processed_context])
            
            # Tính toán similarity giữa văn bản ngữ cảnh và các giá trị của field_code
            similarities = cosine_similarity(context_vector, matrix).flatten()
            
            # Cải tiến 1: Áp dụng ngưỡng động dựa trên phân phối similarity
            # Tính ngưỡng dựa trên trung bình và độ lệch chuẩn
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            dynamic_threshold = max(0.1, mean_sim + 0.5 * std_sim)  # Ngưỡng tối thiểu 0.1
            
            # Lấy top 5 giá trị tương tự nhất vượt qua ngưỡng
            top_indices = similarities.argsort()[-10:][::-1]  # Lấy top 10 để có nhiều lựa chọn hơn
            recommendations = []
            
            # Cải tiến 2: Phân tích ngữ cảnh sâu hơn bằng cách tìm từ khóa chung
            for i in top_indices:
                if similarities[i] > dynamic_threshold:
                    # Kiểm tra sự xuất hiện của các từ khóa chung
                    value_tokens = set(values[i].split())
                    context_tokens = set(processed_context.split())
                    common_tokens = value_tokens.intersection(context_tokens)
                    
                    # Tính điểm dựa trên similarity và số từ khóa chung
                    score = similarities[i] * (1 + 0.2 * len(common_tokens))
                    
                    recommendations.append((values[i], score))
            
            # Sắp xếp theo điểm và lấy top 5
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return [rec[0] for rec in recommendations[:5]]
            
        except Exception as e:
            print(f"Lỗi khi lấy gợi ý NLP: {e}")
            
        return []
    
    def _calculate_field_similarities(self):
        """Tính toán similarity giữa các trường dựa trên dữ liệu form"""
        # Tạo ma trận field-form: mỗi hàng là một field, mỗi cột là một form
        all_fields = set()
        for form in self.form_data:
            all_fields.update(form.keys())
            
        field_form_matrix = {}
        for field in all_fields:
            field_form_matrix[field] = [1 if field in form and form[field] else 0 for form in self.form_data]
            
        # Tính toán cosine similarity giữa các trường
        for field1 in all_fields:
            self.field_similarities[field1] = {}
            vec1 = np.array(field_form_matrix[field1]).reshape(1, -1)
            for field2 in all_fields:
                if field1 != field2:
                    vec2 = np.array(field_form_matrix[field2]).reshape(1, -1)
                    similarity = cosine_similarity(vec1, vec2)[0][0]
                    self.field_similarities[field1][field2] = similarity
    
    def _get_related_fields(self, field_code, threshold=0.3):
        """Lấy các trường có liên quan đến field_code"""
        if field_code not in self.field_similarities:
            return []
            
        related = [(field, sim) for field, sim in self.field_similarities[field_code].items() if sim >= threshold]
        related.sort(key=lambda x: x[1], reverse=True)
        
        return [field for field, _ in related]
    
    def _find_similar_forms(self, filled_fields):
        """Tìm các form có giá trị tương tự cho các trường đã điền"""
        form_scores = []
        
        for i, form in enumerate(self.form_data):
            score = 0
            for field, value in filled_fields.items():
                if field in form and form[field]:
                    # So sánh giá trị của trường
                    if field in self.field_vectors:
                        # Sử dụng TF-IDF similarity nếu có
                        vectorizer = self.field_vectors[field]['vectorizer']
                        matrix = self.field_vectors[field]['matrix']
                        values = self.field_vectors[field]['values']
                        
                        try:
                            # Tìm index của giá trị trong form hiện tại
                            form_value_idx = values.index(self.preprocess_text(form[field]))
                            # Tìm vector TF-IDF của giá trị đã nhập
                            input_vector = vectorizer.transform([self.preprocess_text(value)])
                            # Tính similarity
                            sim = cosine_similarity(input_vector, matrix[form_value_idx].reshape(1, -1))[0][0]
                            score += sim
                        except (ValueError, IndexError) as e:
                            # Nếu không tìm thấy, so sánh chuỗi đơn giản
                            if value.lower() == form[field].lower():
                                score += 1
                    else:
                        # So sánh chuỗi đơn giản
                        if value.lower() == form[field].lower():
                            score += 1
            
            if score > 0:  # Chỉ xét các form có điểm > 0
                form_scores.append((i, score))
        
        # Sắp xếp theo điểm giảm dần
        form_scores.sort(key=lambda x: x[1], reverse=True)
        
        return form_scores
        
    def _get_most_common_values(self, field_code):
        """Lấy các giá trị phổ biến nhất cho một trường"""
        if not self.form_data:
            return []
            
        # Thu thập tất cả các giá trị của field_code
        values = []
        for form in self.form_data:
            if field_code in form and form[field_code]:
                values.append(form[field_code])
                
        if not values:
            return []
            
        # Đếm tần suất xuất hiện của mỗi giá trị
        value_counts = {}
        for value in values:
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1
                
        # Sắp xếp theo tần suất giảm dần
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Trả về top 5 giá trị phổ biến nhất
        return [value for value, _ in sorted_values[:5]]
    
    def get_combined_recommendations(self, partial_form, field_code, context_text=""):
        """Kết hợp các phương pháp gợi ý để có kết quả tốt nhất với trọng số thông minh"""
        # Lấy gợi ý từ các phương pháp khác nhau
        content_recs = self.get_content_based_recommendations(partial_form, field_code)
        collab_recs = self.get_collaborative_recommendations(len(self.form_data), field_code)  # Coi form hiện tại là user mới
        nlp_recs = self.get_nlp_recommendations(field_code, context_text)
        
        # Cải tiến 1: Sử dụng hệ thống trọng số thông minh
        # Xác định trọng số dựa trên số lượng trường đã điền và độ dài context_text
        filled_fields_count = sum(1 for _, v in partial_form.items() if v)
        
        # Tính trọng số cho từng phương pháp
        content_weight = 1.0
        collab_weight = 1.0
        nlp_weight = 1.0
        
        # Điều chỉnh trọng số dựa trên dữ liệu hiện có
        if filled_fields_count > 2:  # Nếu đã điền nhiều trường
            content_weight = 1.5  # Tăng trọng số cho content-based
        else:
            collab_weight = 1.3  # Tăng trọng số cho collaborative
            
        if context_text and len(context_text) > 10:  # Nếu có ngữ cảnh văn bản đủ dài
            nlp_weight = 1.4  # Tăng trọng số cho NLP
        
        # Cải tiến 2: Tạo từ điển điểm số cho mỗi gợi ý
        rec_scores = {}
        
        # Tính điểm cho các gợi ý từ content-based
        for i, rec in enumerate(content_recs):
            # Điểm giảm dần theo thứ tự xuất hiện
            score = content_weight * (1.0 - i * 0.1)  # 1.0, 0.9, 0.8, ...
            if rec in rec_scores:
                rec_scores[rec] += score
            else:
                rec_scores[rec] = score
        
        # Tính điểm cho các gợi ý từ collaborative
        for i, rec in enumerate(collab_recs):
            score = collab_weight * (1.0 - i * 0.1)
            if rec in rec_scores:
                rec_scores[rec] += score
            else:
                rec_scores[rec] = score
        
        # Tính điểm cho các gợi ý từ NLP
        for i, rec in enumerate(nlp_recs):
            score = nlp_weight * (1.0 - i * 0.1)
            if rec in rec_scores:
                rec_scores[rec] += score
            else:
                rec_scores[rec] = score
        
        # Cải tiến 3: Sắp xếp gợi ý theo điểm số và trả về top 5
        sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
        combined_recs = [rec for rec, _ in sorted_recs[:5]]
        
        # Nếu không có đủ gợi ý, thêm các gợi ý từ các phương pháp theo thứ tự ưu tiên
        if len(combined_recs) < 5:
            # Thêm gợi ý từ content-based (ưu tiên cao nhất)
            for rec in content_recs:
                if rec not in combined_recs:
                    combined_recs.append(rec)
                    if len(combined_recs) >= 5:
                        break
            
            # Thêm gợi ý từ collaborative
            if len(combined_recs) < 5:
                for rec in collab_recs:
                    if rec not in combined_recs:
                        combined_recs.append(rec)
                        if len(combined_recs) >= 5:
                            break
            
            # Thêm gợi ý từ NLP
            if len(combined_recs) < 5:
                for rec in nlp_recs:
                    if rec not in combined_recs:
                        combined_recs.append(rec)
                        if len(combined_recs) >= 5:
                            break
        
        return combined_recs[:5]

# Singleton instance
_recommender_instance = None

def get_recommender():
    """Trả về instance của MLRecommender (Singleton pattern)"""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = MLRecommender()
        # Xây dựng các mô hình
        _recommender_instance.build_content_based_model()
        _recommender_instance.build_collaborative_model()
    return _recommender_instance

def get_ml_suggestions(field_code, partial_form=None, context_text=""):
    """Hàm tiện ích để lấy gợi ý từ ML"""
    if partial_form is None:
        partial_form = {}
    
    # Kiểm tra xem field_code có hợp lệ không
    if not field_code or not isinstance(field_code, str):
        print(f"Lỗi: field_code không hợp lệ: {field_code}")
        return []
        
    recommender = get_recommender()
    
    # Kiểm tra xem có dữ liệu form nào không
    if not recommender.form_data:
        print("Lỗi: Không có dữ liệu form nào để đưa ra gợi ý")
        return []
    
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
        print(f"Lỗi: Trường {field_code} không tồn tại trong bất kỳ form nào")
        return []
        
    if not field_has_value:
        print(f"Lỗi: Trường {field_code} chưa có giá trị trong bất kỳ form nào")
        return []
    
    # Lấy các giá trị phổ biến nhất cho trường này
    common_values = recommender._get_most_common_values(field_code)
    
    # Nếu có ít nhất một giá trị phổ biến, trả về ngay cả khi không có đủ dữ liệu liên quan
    if common_values:
        # Thử lấy gợi ý từ các phương pháp kết hợp
        suggestions = recommender.get_combined_recommendations(partial_form, field_code, context_text)
        
        # Nếu không có gợi ý từ phương pháp kết hợp, sử dụng các giá trị phổ biến
        if not suggestions:
            suggestions = common_values
    else:
        # Nếu không có giá trị phổ biến nào, thử lấy gợi ý từ phương pháp kết hợp
        suggestions = recommender.get_combined_recommendations(partial_form, field_code, context_text)
    
    # Ghi log kết quả để debug
    print(f"Gợi ý cho trường {field_code}: {suggestions}")
    
    return suggestions