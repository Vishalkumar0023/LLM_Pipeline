"""
Data Pipeline Web Application with JWT Authentication
======================================================
Flask web app with JWT token-based authentication and per-user dataset storage.
Now refactored into production-grade Blueprint architecture.
"""

import os
from flask import Flask

from extensions import db
from routes.auth import auth_bp
from routes.ml_pipeline import ml_bp
from routes.ml_models import ml_models_bp
from routes.llm_pipeline import llm_bp


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", "dev-secret-key-change-in-production"
    )
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

    # Init database
    with app.app_context():
        db.create_all()

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("=" * 60)
    print("DATA PIPELINE WEB APP (JWT Authentication)")
    print("=" * 60)
    print(f"\n🌐 Open in browser: http://127.0.0.1:{port}")
    print("🔐 Auth: JWT tokens in HttpOnly cookies")
    print("🧬 LLM Pipeline: /llm\n")
    app.run(debug=True, host="0.0.0.0", port=port)
