from flask import render_template, request, jsonify
from utils.document_utils import upload_document

def register_home_routes(app):
    """
    Đăng ký các route cho trang chủ và tải lên tài liệu
    """
    @app.route('/')
    def index():
        from flask import redirect, url_for
        from flask_login import current_user
        
        if not current_user.is_authenticated:
            return redirect('/login')
        return render_template("TrangChu.html")
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        result, status_code = upload_document(file)
        return jsonify(result), status_code
    
    @app.route('/get-recent-forms')
    def get_recent_forms():
        from models.data_model import load_form_history
        import datetime
        import os
        
        try:
            form_history = load_form_history()
            query = request.args.get('query', '')
            
            # Format the forms for display
            formatted_forms = []
            for form in form_history:
                # Format the date for display
                timestamp = datetime.datetime.fromisoformat(form['timestamp'])
                formatted_date = timestamp.strftime('%d/%m/%Y %H:%M')
                
                # Create a formatted form object
                formatted_form = {
                    'id': os.path.basename(form['path']).split('.')[0],
                    'name': form['name'],
                    'date': formatted_date,
                    'path': form['path']
                }
                
                # Add form data if available
                if 'form_data' in form:
                    formatted_form['form_data'] = form['form_data']
                
                # Filter by search query if provided
                if query.lower() in form['name'].lower():
                    formatted_forms.append(formatted_form)
            
            # If no query, return all forms
            if not query:
                formatted_forms = formatted_forms
                
            # Sort by date (newest first)
            formatted_forms.sort(key=lambda x: x['date'], reverse=True)
            
            return jsonify({'forms': formatted_forms})
        except Exception as e:
            print(f"Error loading recent forms: {str(e)}")
            return jsonify({'error': 'Failed to load recent forms'}), 500