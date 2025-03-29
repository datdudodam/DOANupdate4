from .home_routes import register_home_routes
from .form_routes import register_form_routes
from .docx_routes import register_docx_routes
from .auth import register_auth_routes
from .ml_routes import register_ml_routes
from .enhanced_routes import register_enhanced_routes

def register_routes(app):
    register_home_routes(app)
    register_form_routes(app)
    register_docx_routes(app)
    register_auth_routes(app)
    register_ml_routes(app)
    register_enhanced_routes(app)