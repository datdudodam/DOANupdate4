from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, login_required, logout_user, current_user
from models.user import db, User
from utils.validators import validate_email, validate_password

def dangNhap_Dangky(app):
    """ Đăng ký và Đăng nhập
    """

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('index'))

        if request.method == 'POST':
            email = request.form.get('email')
            password = request.form.get('password')
            remember = request.form.get('remember') == 'on'

            user = User.query.filter_by(email=email).first()
            
            if user and user.check_password(password):
                login_user(user, remember=remember)
                flash('Đăng nhập thành công!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Email hoặc mật khẩu không chính xác', 'error')

        return render_template('DangNhap.html')

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if current_user.is_authenticated:
            return redirect(url_for('index'))

        if request.method == 'POST':
            fullname = request.form.get('fullname')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')

            if not all([fullname, email, password, confirm_password]):
                flash('Vui lòng điền đầy đủ thông tin', 'error')
                return redirect(url_for('signup'))

            if not validate_email(email):
                flash('Email không hợp lệ', 'error')
                return redirect(url_for('signup'))

            if User.query.filter_by(email=email).first():
                flash('Email đã được đăng ký', 'error')
                return redirect(url_for('signup'))

            if password != confirm_password:
                flash('Mật khẩu xác nhận không khớp', 'error')
                return redirect(url_for('signup'))

            is_valid_password, msg = validate_password(password)
            if not is_valid_password:
                flash(msg, 'error')
                return redirect(url_for('signup'))

            user = User(fullname=fullname, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
            return render_template('DangNhap.html')

        return render_template('DangKy.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('Đã đăng xuất thành công.', 'success')
        return redirect(url_for('login'))
