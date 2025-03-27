import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
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
        
        if not filled_fields:  # Nếu chưa có trường nào được điền
            return []
            
        # Tính toán similarity giữa các trường
        if not self.field_similarities:
            self._calculate_field_similarities()
            
        # Tìm các trường có liên quan đến field_code
        related_fields = self._get_related_fields(field_code)
        
        # Lọc ra các trường đã điền và có liên quan
        relevant_filled_fields = {k: v for k, v in filled_fields.items() if k in related_fields}
        
        if not relevant_filled_fields:  # Nếu không có trường liên quan nào được điền
            return []
            
        # Tìm các form có giá trị tương tự cho các trường đã điền
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
                        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, field_code):
        """Lấy gợi ý dựa trên Collaborative Filtering"""
        if not self.svd_model or not self.user_item_matrix.shape[0] > 0:
            return []
            
        try:
            # Dự đoán rating cho field_code
            # Nếu user_id mới, sử dụng user_id cuối cùng trong tập dữ liệu
            if user_id >= len(self.form_data):
                user_id = len(self.form_data) - 1
                
            prediction = self.svd_model.predict(user_id, field_code)
            
            # Nếu dự đoán rating cao, tìm các giá trị phổ biến cho field_code
            if prediction.est > 0.5:  # Ngưỡng rating
                # Tìm các form có điền field_code
                field_values = []
                for form in self.form_data:
                    if field_code in form and form[field_code]:
                        field_values.append(form[field_code])
                        
                # Đếm tần suất và lấy top 5 giá trị phổ biến nhất
                if field_values:
                    value_counts = pd.Series(field_values).value_counts()
                    return value_counts.index.tolist()[:5]
        except Exception as e:
            print(f"Lỗi khi lấy gợi ý collaborative: {e}")
            
        return []
    
    def get_nlp_recommendations(self, field_code, context_text=""):
        """Lấy gợi ý dựa trên phân tích NLP"""
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
            
            # Lấy top 5 giá trị tương tự nhất
            top_indices = similarities.argsort()[-5:][::-1]
            recommendations = [values[i] for i in top_indices if similarities[i] > 0]
            
            return recommendations
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
    
    def get_combined_recommendations(self, partial_form, field_code, context_text=""):
        """Kết hợp các phương pháp gợi ý để có kết quả tốt nhất"""
        # Lấy gợi ý từ các phương pháp khác nhau
        content_recs = self.get_content_based_recommendations(partial_form, field_code)
        collab_recs = self.get_collaborative_recommendations(len(self.form_data), field_code)  # Coi form hiện tại là user mới
        nlp_recs = self.get_nlp_recommendations(field_code, context_text)
        
        # Kết hợp các gợi ý, ưu tiên theo thứ tự: content > collaborative > nlp
        combined_recs = []
        
        # Thêm gợi ý từ content-based (ưu tiên cao nhất)
        combined_recs.extend([rec for rec in content_recs if rec not in combined_recs])
        
        # Thêm gợi ý từ collaborative
        combined_recs.extend([rec for rec in collab_recs if rec not in combined_recs])
        
        # Thêm gợi ý từ NLP
        combined_recs.extend([rec for rec in nlp_recs if rec not in combined_recs])
        
        # Giới hạn số lượng gợi ý
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
        
    recommender = get_recommender()
    suggestions = recommender.get_combined_recommendations(partial_form, field_code, context_text)
    
    return suggestions