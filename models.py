from datetime import datetime
from extensions import db


class User(db.Model):
    """User model for authentication."""

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Relationship with datasets
    datasets = db.relationship("Dataset", backref="owner", lazy=True)
    llm_runs = db.relationship(
        "LLMRun", backref="owner", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, password):
        from werkzeug.security import generate_password_hash

        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        from werkzeug.security import check_password_hash

        return check_password_hash(self.password_hash, password)

    @property
    def is_authenticated(self):
        return True


class Dataset(db.Model):
    """Model to store user datasets."""

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)
    cleaned_path = db.Column(db.String(500))
    final_path = db.Column(db.String(500))
    original_rows = db.Column(db.Integer)
    original_cols = db.Column(db.Integer)
    cleaned_rows = db.Column(db.Integer)
    cleaned_cols = db.Column(db.Integer)
    final_rows = db.Column(db.Integer)
    final_cols = db.Column(db.Integer)
    target_column = db.Column(db.String(100))
    problem_type = db.Column(db.String(50))
    processing_log = db.Column(db.Text)
    model_path = db.Column(db.String(500))
    model_results = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


class LLMRun(db.Model):
    """Model to store per-user LLM export runs for DB-backed management."""

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(120), nullable=False, index=True)
    session_id = db.Column(db.String(64), nullable=False, index=True)
    sample_count = db.Column(db.Integer, default=0)
    avg_quality = db.Column(db.Float, default=0.0)
    template = db.Column(db.String(50))
    model = db.Column(db.String(255))
    method = db.Column(db.String(50))
    version = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=datetime.now, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)

    __table_args__ = (
        db.UniqueConstraint("user_id", "run_id", name="uq_user_llm_run"),
    )
