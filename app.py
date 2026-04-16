"""
Data Pipeline Web Application with JWT Authentication
======================================================
Flask web app with JWT token-based authentication and per-user dataset storage.
Now refactored into production-grade Blueprint architecture.
"""

import os
import logging
from dotenv import load_dotenv
from flask import Flask

# Load .env file so SECRET_KEY and other vars are available
load_dotenv()

from extensions import db
from routes.auth import auth_bp
from routes.ml_pipeline import ml_bp
from routes.ml_models import ml_models_bp
from routes.llm_pipeline import llm_bp
from routes.firecrawl import firecrawl_bp


def create_app():
    app = Flask(__name__)
    # SECURITY: SECRET_KEY must come from environment — never use a hardcoded default.
    secret = os.environ.get("SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "FATAL: SECRET_KEY environment variable is not set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    app.config["SECRET_KEY"] = secret
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL", "sqlite:///pipeline_users.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # Max upload limit
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max upload

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(ml_models_bp)
    app.register_blueprint(llm_bp)
    app.register_blueprint(firecrawl_bp)

    # Init database
    with app.app_context():
        db.create_all()

    # ─── Security Headers (OWASP) ─────────────────────────────────
    @app.after_request
    def set_security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "frame-ancestors 'none';"
        )
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        return response

    # ─── Structured Logging ───────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    app.logger.setLevel(logging.INFO)

    return app


app = create_app()

# ASGI wrapped application for uvicorn (run with: uvicorn app:asgi_app --reload)
try:
    from starlette.middleware.wsgi import WSGIMiddleware
    asgi_app = WSGIMiddleware(app)
except ImportError:
    pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    use_reloader = os.environ.get("FLASK_RELOADER", "0") == "1"
    print("=" * 60)
    print("DATA PIPELINE WEB APP (JWT Authentication)")
    print("=" * 60)
    print(f"\n🌐 Open in browser: http://127.0.0.1:{port}")
    print("🔐 Auth: JWT tokens in HttpOnly cookies")
    print("🧬 LLM Pipeline: /llm\n")
    app.run(debug=debug, use_reloader=use_reloader, host="127.0.0.1", port=port)
