from flask import Flask, redirect, url_for
from config.config import DEBUG  # Đảm bảo DEBUG tồn tại trong config.py
from routes.home_routes import register_home_routes
from routes.form_routes import register_form_routes
from routes.docx_routes import register_docx_routes
from routes.auth import dangNhap_Dangky
from flask_login import LoginManager
from models.user import User, db

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Cấu hình cơ sở dữ liệu (Thay thế bằng URI thật của bạn)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'  # Hoặc PostgreSQL/MySQL URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your-secret-key-here'  # Thêm secret key cho session

# Khởi tạo database với Flask app
db.init_app(app)

# Đảm bảo có một app context khi truy vấn DB
with app.app_context():
    db.create_all()

# Cấu hình Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Đặt view đăng nhập mặc định

@login_manager.user_loader
def load_user(user_id):
    with app.app_context():
        return User.query.get(int(user_id))

# Đăng ký các route
register_home_routes(app)
register_form_routes(app)
register_docx_routes(app)
dangNhap_Dangky(app)

if __name__ == '__main__':
    app.run(debug=DEBUG)
