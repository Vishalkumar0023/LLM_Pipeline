"""
Data Pipeline Web Application with JWT Authentication
======================================================
Flask web app with JWT token-based authentication and per-user dataset storage.
"""

import os
import json
import io
import gc
import base64
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, g, make_response, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import jwt as pyjwt

# Heavy libraries are imported lazily to save memory on 512MB hosting
# Split into groups so each route only loads what it needs
pd = None
np = None
plt = None
DataPipeline = None
ModelTrainer = None
DataCleaner = None

# Max rows to prevent OOM on large datasets
MAX_ROWS = 50000

def _load_data_libs():
    """Load only pandas + numpy (for upload/download routes)."""
    global pd, np
    if pd is None:
        import pandas
        pd = pandas
    if np is None:
        import numpy
        np = numpy

def _load_plot_libs():
    """Load matplotlib only (no seaborn — saves ~170MB)."""
    global plt
    _load_data_libs()
    if plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot
        plt = matplotlib.pyplot

def _load_pipeline_libs():
    """Load data_pipeline (scikit-learn based - for processing/training)."""
    global DataPipeline, ModelTrainer, DataCleaner
    _load_data_libs()
    if DataPipeline is None:
        from data_pipeline import DataPipeline as _DP, ModelTrainer as _MT, DataCleaner as _DC
        DataPipeline = _DP
        ModelTrainer = _MT
        DataCleaner = _DC

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///pipeline_users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload

# ── JWT Configuration ──
JWT_SECRET_KEY = app.config['SECRET_KEY']
JWT_ALGORITHM = 'HS256'
JWT_ACCESS_EXPIRY = timedelta(minutes=30)
JWT_REFRESH_EXPIRY = timedelta(days=7)

# Create base folders
BASE_UPLOAD_FOLDER = 'user_data'
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)


# ==============================================================================
# DATABASE MODELS
# ==============================================================================

class User(db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with datasets
    datasets = db.relationship('Dataset', backref='owner', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


# ==============================================================================
# JWT HELPER FUNCTIONS
# ==============================================================================

def create_access_token(user_id):
    """Create a short-lived JWT access token."""
    payload = {
        'sub': str(user_id),
        'type': 'access',
        'iat': datetime.now(timezone.utc),
        'exp': datetime.now(timezone.utc) + JWT_ACCESS_EXPIRY
    }
    return pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id):
    """Create a long-lived JWT refresh token."""
    payload = {
        'sub': str(user_id),
        'type': 'refresh',
        'iat': datetime.now(timezone.utc),
        'exp': datetime.now(timezone.utc) + JWT_REFRESH_EXPIRY
    }
    return pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token):
    """Decode and validate a JWT token. Returns payload dict or None."""
    try:
        payload = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except pyjwt.ExpiredSignatureError:
        return None
    except pyjwt.InvalidTokenError:
        return None


def get_current_user():
    """
    Extract the current user from:
      1. HttpOnly cookie 'access_token'  (browser flow)
      2. Authorization: Bearer <token>   (API flow)
    Returns User object or None.
    """
    token = None
    
    # Try cookie first (browser flow)
    token = request.cookies.get('access_token')
    
    # Try Authorization header (API flow)
    if not token:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
    
    if not token:
        return None
    
    payload = decode_token(token)
    if not payload or payload.get('type') != 'access':
        return None
    
    user = User.query.get(int(payload['sub']))
    return user


def jwt_required(f):
    """Decorator to protect routes – replaces @login_required."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            # For API requests return 401
            if request.is_json or request.headers.get('Authorization'):
                return jsonify({'error': 'Authentication required', 'code': 'TOKEN_EXPIRED'}), 401
            # For browser requests redirect to login
            return redirect(url_for('login'))
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def set_auth_cookies(response, user_id):
    """Set access and refresh token cookies on a response."""
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id)
    
    response.set_cookie(
        'access_token', access_token,
        httponly=True, samesite='Lax',
        max_age=int(JWT_ACCESS_EXPIRY.total_seconds()),
        path='/'
    )
    response.set_cookie(
        'refresh_token', refresh_token,
        httponly=True, samesite='Lax',
        max_age=int(JWT_REFRESH_EXPIRY.total_seconds()),
        path='/'
    )
    return response


def clear_auth_cookies(response):
    """Clear authentication cookies."""
    response.delete_cookie('access_token', path='/')
    response.delete_cookie('refresh_token', path='/')
    return response


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_user_folder(user_id):
    """Get or create user-specific folder."""
    folder = os.path.join(BASE_UPLOAD_FOLDER, str(user_id))
    os.makedirs(folder, exist_ok=True)
    return folder


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def generate_plots(df, target_col=None):
    """Generate EDA plots and return as base64 images."""
    _load_plot_libs()
    plots = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Correlation Heatmap
    if len(numeric_cols) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            masked_corr = np.ma.masked_where(mask, corr.values)
            cax = ax.imshow(masked_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            fig.colorbar(cax, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(corr.columns, fontsize=8)
            ax.set_title('Feature Correlation Heatmap')
            plt.tight_layout()
            plots['correlation'] = fig_to_base64(fig)
            plt.close(fig)
        except:
            pass
    
    # Distribution plots
    if numeric_cols:
        try:
            cols_to_plot = numeric_cols[:6]
            n_cols = min(3, len(cols_to_plot))
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
            axes = np.atleast_2d(axes).flatten()
            
            for idx, col in enumerate(cols_to_plot):
                ax = axes[idx]
                data = df[col].dropna()
                ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                ax.axvline(data.mean(), color='red', linestyle='--', label='Mean')
                ax.axvline(data.median(), color='green', linestyle='--', label='Median')
                ax.set_title(col)
                ax.legend(fontsize=8)
            
            for idx in range(len(cols_to_plot), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Feature Distributions', fontsize=14)
            plt.tight_layout()
            plots['distributions'] = fig_to_base64(fig)
            plt.close(fig)
        except:
            pass
    
    return plots


# ==============================================================================
# AUTH ROUTES
# ==============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    user = get_current_user()
    if user:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if request.is_json:
                resp = jsonify({
                    'success': True,
                    'redirect': url_for('dashboard'),
                    'access_token': create_access_token(user.id),
                    'refresh_token': create_refresh_token(user.id)
                })
            else:
                resp = make_response(redirect(url_for('dashboard')))
            
            set_auth_cookies(resp, user.id)
            return resp
        
        if request.is_json:
            return jsonify({'error': 'Invalid username or password'}), 401
    
    return render_template('auth.html', mode='login')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page."""
    user = get_current_user()
    if user:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Validate
        if User.query.filter_by(username=username).first():
            if request.is_json:
                return jsonify({'error': 'Username already exists'}), 400
            return render_template('auth.html', mode='signup')
        
        if User.query.filter_by(email=email).first():
            if request.is_json:
                return jsonify({'error': 'Email already registered'}), 400
            return render_template('auth.html', mode='signup')
        
        # Create user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        # Create user folder
        get_user_folder(user.id)
        
        if request.is_json:
            resp = jsonify({
                'success': True,
                'redirect': url_for('dashboard'),
                'access_token': create_access_token(user.id),
                'refresh_token': create_refresh_token(user.id)
            })
        else:
            resp = make_response(redirect(url_for('dashboard')))
        
        set_auth_cookies(resp, user.id)
        return resp
    
    return render_template('auth.html', mode='signup')


@app.route('/logout')
def logout():
    """Logout user — clear JWT cookies."""
    resp = make_response(redirect(url_for('login')))
    clear_auth_cookies(resp)
    return resp


@app.route('/api/refresh', methods=['POST'])
def refresh_token():
    """
    Refresh the access token using the refresh token.
    Reads from cookie or JSON body.
    """
    token = request.cookies.get('refresh_token')
    
    if not token and request.is_json:
        token = request.get_json().get('refresh_token')
    
    if not token:
        return jsonify({'error': 'Refresh token required'}), 401
    
    payload = decode_token(token)
    if not payload or payload.get('type') != 'refresh':
        return jsonify({'error': 'Invalid or expired refresh token', 'code': 'REFRESH_EXPIRED'}), 401
    
    user = User.query.get(int(payload['sub']))
    if not user:
        return jsonify({'error': 'User not found'}), 401
    
    new_access = create_access_token(user.id)
    
    resp = jsonify({
        'success': True,
        'access_token': new_access
    })
    resp.set_cookie(
        'access_token', new_access,
        httponly=True, samesite='Lax',
        max_age=int(JWT_ACCESS_EXPIRY.total_seconds()),
        path='/'
    )
    return resp


# ==============================================================================
# MAIN ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Home page - landing page for visitors, dashboard for logged-in users."""
    user = get_current_user()
    if user:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/dashboard')
@jwt_required
def dashboard():
    """User dashboard with their datasets."""
    datasets = Dataset.query.filter_by(user_id=g.current_user.id).order_by(Dataset.created_at.desc()).all()
    return render_template('dashboard.html', datasets=datasets, user=g.current_user)


@app.route('/upload', methods=['POST'])
@jwt_required
def upload_file():
    """Handle file upload."""
    _load_data_libs()
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        return jsonify({'error': 'Only CSV and Excel files are supported'}), 400
    
    try:
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Convert pandas extension types to standard numpy types
        for col in df.columns:
            if hasattr(df[col].dtype, 'name') and df[col].dtype.name in ('string', 'String'):
                df[col] = df[col].astype('object')
            elif hasattr(df[col].dtype, 'numpy_dtype'):
                df[col] = df[col].astype(df[col].dtype.numpy_dtype)
        
        # Save to user folder
        user_folder = get_user_folder(g.current_user.id)
        temp_path = os.path.join(user_folder, 'temp_upload.csv')
        df.to_csv(temp_path, index=False)
        
        # Get column info
        columns = df.columns.tolist()
        dtypes = {col: str(df[col].dtype) for col in columns}
        missing = {col: int(df[col].isnull().sum()) for col in columns}
        
        # Get sample data - handle NaN
        sample_df = df.head(5).replace({np.nan: None})
        sample = sample_df.to_dict('records')
        for row in sample:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif hasattr(value, 'item'):
                    row[key] = value.item()
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'shape': list(df.shape),
            'columns': columns,
            'dtypes': dtypes,
            'missing': missing,
            'sample': sample
        })
    
    except Exception as e:
        import traceback
        print(f"Upload error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
@jwt_required
def process_data():
    """Process uploaded data through the pipeline."""
    _load_pipeline_libs()
    try:
        data = request.json
        target_col = data.get('target_column')
        problem_type = data.get('problem_type', 'regression')
        dataset_name = data.get('dataset_name', 'Untitled Dataset')
        original_filename = data.get('original_filename', 'unknown.csv')
        
        # Load from user's temp file
        user_folder = get_user_folder(g.current_user.id)
        temp_path = os.path.join(user_folder, 'temp_upload.csv')
        
        if not os.path.exists(temp_path):
            return jsonify({'error': 'No file uploaded. Please upload a file first.'}), 400
        
        # Run pipeline
        pipeline = DataPipeline()
        pipeline.load(temp_path)
        
        # Validate
        validation = pipeline.validate()
        
        # Clean
        # Calculate Initial Quality & Suggestions
        raw_cleaner = DataCleaner(pipeline.raw_df)
        initial_quality = raw_cleaner.validate_quality()
        suggestions = raw_cleaner.generate_suggestions()
        
        pipeline.clean()
        cleaning_summary = pipeline.cleaner.get_cleaning_summary()
        
        # Calculate Final Quality
        final_quality = pipeline.cleaner.validate_quality()
        
        # Feature engineering
        if target_col and target_col in pipeline.cleaned_df.columns:
            pipeline.engineer_features(target_col=target_col, problem_type=problem_type)
            feature_summary = pipeline.engineer.get_summary()
        else:
            pipeline.engineer_features()
            feature_summary = pipeline.engineer.get_summary() if pipeline.engineer else {}
        
        # Save results with unique names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cleaned_filename = f'cleaned_{timestamp}.csv'
        final_filename = f'final_{timestamp}.csv'
        
        cleaned_path = os.path.join(user_folder, cleaned_filename)
        final_path = os.path.join(user_folder, final_filename)
        
        pipeline.cleaned_df.to_csv(cleaned_path, index=False)
        pipeline.final_df.to_csv(final_path, index=False)
        
        # Save to database
        dataset = Dataset(
            name=dataset_name,
            original_filename=original_filename,
            cleaned_path=cleaned_path,
            final_path=final_path,
            original_rows=validation['shape'][0],
            original_cols=validation['shape'][1],
            cleaned_rows=pipeline.cleaned_df.shape[0],
            cleaned_cols=pipeline.cleaned_df.shape[1],
            final_rows=pipeline.final_df.shape[0],
            final_cols=pipeline.final_df.shape[1],
            target_column=target_col,
            problem_type=problem_type,
            processing_log=json.dumps({
                'cleaning': cleaning_summary['operations'],
                'feature_engineering': feature_summary.get('transformations', []),
                'quality_impact': {
                    'before': initial_quality,
                    'after': final_quality
                },
                'suggestions': suggestions,
                'row_changes': cleaning_summary.get('row_changes', [])
            }),
            user_id=g.current_user.id
        )
        db.session.add(dataset)
        db.session.commit()
        
        # Save raw data for model comparison
        raw_filename = f"raw_{dataset.id}.csv"
        raw_path = os.path.join(user_folder, raw_filename)
        print(f"DEBUG: Saving raw data to {raw_path}, matches id {dataset.id}")
        pipeline.raw_df.to_csv(raw_path, index=False)
        if os.path.exists(raw_path):
             print(f"DEBUG: Raw file created successfully: {os.path.getsize(raw_path)} bytes")
        else:
             print("DEBUG: Raw file creation FAILED")
        
        # Save row changes to CSV for download
        row_changes_df = pd.DataFrame(cleaning_summary.get('row_changes', []))
        row_changes_filename = f"row_changes_{dataset.id}.csv"
        row_changes_path = os.path.join(user_folder, row_changes_filename)
        
        if not row_changes_df.empty:
            row_changes_df.to_csv(row_changes_path, index=False)
        else:
            # Create empty CSV with headers if no changes
            pd.DataFrame(columns=['index', 'column', 'old_value', 'new_value', 'operation', 'reason']).to_csv(row_changes_path, index=False)

        # Generate plots
        plots = generate_plots(pipeline.cleaned_df, target_col)
        
        # Remove temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass
        
        return jsonify({
            'success': True,
            'dataset_id': dataset.id,
            'row_changes_csv': row_changes_filename,
            'validation': {
                'original_shape': validation['shape'],
                'missing_count': validation['missing_values']['total_missing_cells'],
                'duplicate_count': validation['duplicates']['count']
            },
            'cleaning': {
                'final_shape': list(pipeline.cleaned_df.shape),
                'operations': cleaning_summary['operations'],
                'row_changes': cleaning_summary.get('row_changes', [])
            },
            'feature_engineering': {
                'final_shape': list(pipeline.final_df.shape),
                'transformations': feature_summary.get('transformations', [])
            },
            'quality_impact': {
                'before': initial_quality,
                'after': final_quality
            },
            'suggestions': suggestions,
            'plots': plots
        })
    
    except Exception as e:
        import traceback
        print(f"Process error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/<int:dataset_id>')
@jwt_required
def view_dataset(dataset_id):
    """View a specific dataset."""
    _load_data_libs()
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Ensure user owns this dataset
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    # Load samples for preview
    cleaned_sample = []
    cleaned_columns = []
    final_sample = []
    final_columns = []
    
    try:
        if os.path.exists(dataset.cleaned_path):
            df_clean = pd.read_csv(dataset.cleaned_path)
            cleaned_columns = df_clean.columns.tolist()
            # Replace NaN with None for Jinja
            cleaned_sample = df_clean.head(10).replace({np.nan: None}).to_dict('records')
            
        if os.path.exists(dataset.final_path):
            df_final = pd.read_csv(dataset.final_path)
            final_columns = df_final.columns.tolist()
            final_sample = df_final.head(10).replace({np.nan: None}).to_dict('records')
    except Exception as e:
        print(f"Error loading dataset samples: {e}")
        
    # Get other datasets for drift comparison
    other_datasets = Dataset.query.filter(
        Dataset.user_id == g.current_user.id,
        Dataset.id != dataset_id
    ).all()
        
    return render_template('view_dataset.html', 
        dataset=dataset,
        cleaned_columns=cleaned_columns,
        cleaned_sample=cleaned_sample,
        final_columns=final_columns,
        final_sample=final_sample,
        processing_log=json.loads(dataset.processing_log) if dataset.processing_log else {},
        other_datasets=other_datasets
    )


@app.route('/api/dataset/<int:dataset_id>/transform', methods=['POST'])
@jwt_required
def transform_dataset(dataset_id):
    """Apply data transformations."""
    _load_pipeline_libs()
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    data = request.json
    operation = data.get('operation')
    params = data.get('params', {})
    
    # Load pipeline (just to use loader/cleaner logic)
    pipeline = DataPipeline()
    # Use cleaned path as starting point
    if os.path.exists(dataset.cleaned_path):
        pipeline.load(dataset.cleaned_path)
    else:
        return jsonify({'error': 'Dataset file not found'}), 404
        
    # Initialize cleaner with current data
    cleaner = DataCleaner(pipeline.raw_df)
    
    try:
        if operation == 'clean_numeric_text':
            cleaner.clean_numeric_text(**params)
        elif operation == 'rename_columns':
            cleaner.rename_columns(**params)
        elif operation == 'extract_regex':
            cleaner.extract_regex_feature(**params)
        elif operation == 'remove_duplicates':
            cleaner.remove_duplicates(**params)
        elif operation == 'drop_columns':
            cleaner.drop_columns(**params)
        else:
            return jsonify({'error': 'Invalid operation'}), 400
            
        # Get updated dataframe
        new_df = cleaner.get_cleaned_data()
        
        # Save updated file (overwriting cleaned path for now)
        cleaned_path = dataset.cleaned_path
        new_df.to_csv(cleaned_path, index=False)
        
        # Also update final path if we are treating them similarly, 
        # or just let future feature engineering handle it. 
        # For this demo, let's keep them in sync if no detailed FE has been done yet.
        if os.path.exists(dataset.final_path):
            new_df.to_csv(dataset.final_path, index=False)
            dataset.final_rows = new_df.shape[0]
            dataset.final_cols = new_df.shape[1]
        
        # Update metadata
        dataset.cleaned_rows = new_df.shape[0]
        dataset.cleaned_cols = new_df.shape[1]
        
        # Add to log
        try:
            log = json.loads(dataset.processing_log) if dataset.processing_log else {}
        except:
            log = {}
            
        if 'cleaning' not in log: log['cleaning'] = []
        
        # Get last operation from cleaner log
        if cleaner.cleaning_log:
            log['cleaning'].extend(cleaner.cleaning_log[-1:])
            
        dataset.processing_log = json.dumps(log)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'preview': new_df.head().replace({np.nan: None}).to_dict(orient='records'),
            'columns': new_df.columns.tolist(),
            'stats': {
                'rows': new_df.shape[0],
                'cols': new_df.shape[1]
            },
            'message': cleaner.cleaning_log[-1] if cleaner.cleaning_log else "Transformation applied"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    # Load cleaned data preview
    cleaned_df = pd.read_csv(dataset.cleaned_path)
    final_df = pd.read_csv(dataset.final_path)
    
    # Get sample data
    cleaned_sample = cleaned_df.head(10).replace({np.nan: None}).to_dict('records')
    final_sample = final_df.head(10).replace({np.nan: None}).to_dict('records')
    
    # Parse processing log
    processing_log = json.loads(dataset.processing_log) if dataset.processing_log else {}
    
    return render_template('view_dataset.html', 
                          dataset=dataset,
                          cleaned_columns=cleaned_df.columns.tolist(),
                          final_columns=final_df.columns.tolist(),
                          cleaned_sample=cleaned_sample,
                          final_sample=final_sample,
                          processing_log=processing_log)


@app.route('/dataset/<int:dataset_id>/download/<file_type>')
@jwt_required  
def download_dataset(dataset_id, file_type):
    """Download dataset file."""
    _load_data_libs()
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    # Check for format query parameter (csv or xlsx)
    file_format = request.args.get('format', 'csv')
    
    if file_type == 'cleaned':
        path = dataset.cleaned_path
        base_filename = f'{dataset.name}_cleaned'
    elif file_type == 'final':
        path = dataset.final_path
        base_filename = f'{dataset.name}_model_ready'
    elif file_type == 'model':
        path = dataset.model_path
        # Models are always .pkl downloads
        filename = f'{dataset.name}_model.pkl'
        if not path or not os.path.exists(path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(path, as_attachment=True, download_name=filename)
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
    if not path or not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
        
    # Handle CSV download (default)
    if file_format == 'csv':
        filename = f'{base_filename}.csv'
        return send_file(path, as_attachment=True, download_name=filename)
        
    # Handle Excel download
    elif file_format == 'xlsx':
        try:
            # Read CSV and convert to Excel in memory
            df = pd.read_csv(path)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            
            filename = f'{base_filename}.xlsx'
            return send_file(
                output,
                as_attachment=True,
                download_name=filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            return jsonify({'error': f'Error converting to Excel: {str(e)}'}), 500
            
    else:
        return jsonify({'error': 'Invalid format requested'}), 400


@app.route('/dataset/<int:dataset_id>/delete', methods=['POST'])
@jwt_required
def delete_dataset(dataset_id):
    """Delete a dataset and associated files."""
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    try:
        # Delete files
        for path in [dataset.cleaned_path, dataset.final_path, dataset.model_path]:
            if path and os.path.exists(path):
                os.remove(path)
                
        # Also delete raw and report if exist
        user_folder = get_user_folder(g.current_user.id)
        raw_path = os.path.join(user_folder, f"raw_{dataset.id}.csv")
        report_path = os.path.join(user_folder, f"model_report_{dataset.id}.md")
        
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(report_path): os.remove(report_path)
        
        # HTML report
        html_report = os.path.join(user_folder, f"model_report_{dataset.id}.html")
        if os.path.exists(html_report): os.remove(html_report)

        db.session.delete(dataset)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets')
@jwt_required
def api_datasets():
    """API endpoint to get user's datasets."""
    datasets = Dataset.query.filter_by(user_id=g.current_user.id).order_by(Dataset.created_at.desc()).all()
    
    return jsonify([{
        'id': d.id,
        'name': d.name,
        'original_filename': d.original_filename,
        'original_rows': d.original_rows,
        'original_cols': d.original_cols,
        'final_rows': d.final_rows,
        'final_cols': d.final_cols,
        'target_column': d.target_column,
        'problem_type': d.problem_type,
        'created_at': d.created_at.isoformat()
    } for d in datasets])


# ==============================================================================
# MODEL TRAINING ROUTE
# ==============================================================================

@app.route('/dataset/<int:dataset_id>/report')
@jwt_required
def view_model_report(dataset_id):
    """View the detailed HTML report in browser."""
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    user_folder = get_user_folder(g.current_user.id)
    report_filename = f"model_report_{dataset.id}.html"
    report_path = os.path.join(user_folder, report_filename)
    
    if not os.path.exists(report_path):
        return jsonify({'error': 'Report not found. Please train a new model to generate it.'}), 404
        
    return send_file(report_path)

@app.route('/dataset/<int:dataset_id>/download/report')
@jwt_required
def download_model_report(dataset_id):
    """Download the auto-generated model report (HTML)."""
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    user_folder = get_user_folder(g.current_user.id)
    # Prefer HTML report if available (it allows print to PDF)
    html_filename = f"model_report_{dataset.id}.html"
    html_path = os.path.join(user_folder, html_filename)
    
    if os.path.exists(html_path):
         return send_file(html_path, as_attachment=True, download_name=html_filename)
    
    # Fallback to Markdown
    report_filename = f"model_report_{dataset.id}.md"
    report_path = os.path.join(user_folder, report_filename)
    
    if not os.path.exists(report_path):
        return jsonify({'error': 'Report not found. Please train a new model to generate it.'}), 404
        
    return send_file(report_path, as_attachment=True, download_name=report_filename)

@app.route('/dataset/<int:dataset_id>/train', methods=['POST'])
@jwt_required
def train_model(dataset_id):
    """Train ML models on a dataset."""
    _load_pipeline_libs()
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # Load final (or cleaned) data
        data_path = dataset.final_path or dataset.cleaned_path
        if not data_path or not os.path.exists(data_path):
            return jsonify({'error': 'Dataset file not found'}), 404
        
        df = pd.read_csv(data_path)
        
        # Get parameters
        data = request.json or {}
        target_col = data.get('target_column', dataset.target_column)
        problem_type = data.get('problem_type', dataset.problem_type)
        
        # Run model trainer
        user_folder = get_user_folder(g.current_user.id)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'model_{dataset_id}_{timestamp}.pkl'
        model_path = os.path.join(user_folder, model_filename)
        
        # Load raw data for comparison
        raw_filename = f"raw_{dataset_id}.csv"
        raw_path = os.path.join(user_folder, raw_filename)
        raw_df = None
        if os.path.exists(raw_path):
             try:
                raw_df = pd.read_csv(raw_path)
             except:
                pass

        trainer = ModelTrainer(df, target_col=target_col, problem_type=problem_type, raw_df=raw_df)
        # Use new comparison method
        results = trainer.run_full_comparison() 
        trainer.export_model(model_path)
        
        # Save results to database
        dataset.model_path = model_path
        dataset.model_results = json.dumps(results, default=str)
        if trainer.target_col:
            dataset.target_column = trainer.target_col
        if trainer.problem_type:
            dataset.problem_type = trainer.problem_type
        db.session.commit()
        
        # Generate Markdown Report
        try:
             # Rehydrate a pipeline object for reporting
             # We iterate cleaning/engineering logs from DB
             processed_log = json.loads(dataset.processing_log) if dataset.processing_log else {}
             
             dummy = DataPipeline()
             dummy.target_col = trainer.target_col
             dummy.problem_type = trainer.problem_type
             dummy.raw_df = raw_df
             dummy.cleaned_df = df # accessible as final/clean
             dummy.model_results = results
             
             # improving list format
             steps = []
             if 'cleaning' in processed_log: steps.extend([f"Cleaning: {x}" for x in processed_log['cleaning']])
             if 'feature_engineering' in processed_log: steps.extend([f"Feature Eng: {x}" for x in processed_log['feature_engineering']])
             
             dummy.pipeline_report = {
                 'preprocessing': {
                     'steps_executed': steps
                 }
             }
             
             report_filename = f'model_report_{dataset_id}.md'
             report_path = os.path.join(user_folder, report_filename)
             dummy.generate_markdown_report(report_path)
             
             # Also generate HTML detailed report
             html_report_filename = f'model_report_{dataset_id}.html'
             html_report_path = os.path.join(user_folder, html_report_filename)
             dummy.generate_html_report(html_report_path)
             
             print(f"DEBUG: Generated reports at {report_path} and {html_report_path}")

        except Exception as e:
            print(f"Warning: Report generation failed: {e}")
            import traceback
            traceback.print_exc()

        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        import traceback
        print(f"Training error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# FILE MANAGEMENT ROUTES
# ==============================================================================

@app.route('/files')
@jwt_required
def file_manager():
    """Render the file manager page."""
    return render_template('files.html')

@app.route('/api/user_files')
@jwt_required
def get_user_files():
    """Get list of files in user's directory."""
    user_folder = get_user_folder(g.current_user.id)
    files = []
    
    if os.path.exists(user_folder):
        for entry in os.scandir(user_folder):
            if entry.is_file() and not entry.name.startswith('.'):
                try:
                    stat = entry.stat()
                    file_type = 'Unknown'
                    if entry.name.endswith('.pkl'):
                        file_type = 'Model (.pkl)'
                    elif entry.name.endswith('.csv'):
                        if 'cleaned' in entry.name:
                            file_type = 'Cleaned Data (.csv)'
                        elif 'final' in entry.name:
                            file_type = 'Model-Ready Data (.csv)'
                        else:
                            file_type = 'Raw Data (.csv)'
                    
                    files.append({
                        'name': entry.name,
                        'size': stat.st_size,
                        'date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': file_type
                    })
                except Exception as e:
                    print(f"Error reading file {entry.name}: {e}")

    # Sort by date desc
    files.sort(key=lambda x: x['date'], reverse=True)
    return jsonify(files)

@app.route('/api/delete_files', methods=['POST'])
@jwt_required
def delete_user_files():
    """Bulk delete files."""
    data = request.json
    filenames = data.get('filenames', [])
    user_folder = get_user_folder(g.current_user.id)
    deleted = []
    errors = []
    
    for filename in filenames:
        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            errors.append(f"Invalid filename: {filename}")
            continue
            
        path = os.path.join(user_folder, filename)
        try:
            if os.path.exists(path):
                os.remove(path)
                deleted.append(filename)
            else:
                errors.append(f"File not found: {filename}")
        except Exception as e:
            errors.append(f"Error deleting {filename}: {str(e)}")
            
    return jsonify({'deleted': deleted, 'errors': errors})


# ==============================================================================
# INIT DATABASE
# ==============================================================================

with app.app_context():
    db.create_all()


# ==============================================================================
# DEMO & PREDICTION ROUTE
# ==============================================================================

DEMO_MODELS_DIR = 'demo_models'

def load_demo_model(model_type):
    """Load a demo model (cached if possible)."""
    try:
        if model_type not in ['sales', 'student']:
            return None
        
        filename = f'{model_type}_model.pkl'
        path = os.path.join(DEMO_MODELS_DIR, filename)
        
        if not os.path.exists(path):
            return None
            
        import joblib
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading demo model: {e}")
        return None

@app.route('/api/predict', methods=['POST'])
def predict_demo():
    """
    Public API endpoint for demo predictions.
    Does NOT require JWT auth to allow easy testing.
    Payload: { "model_type": "sales"|"student", "features": {...} }
    """
    _load_data_libs()
    try:
        data = request.json
        model_type = data.get('model_type')
        features = data.get('features')
        
        if not model_type or not features:
            return jsonify({'error': 'Missing model_type or features'}), 400
            
        model = load_demo_model(model_type)
        if not model:
            return jsonify({'error': 'Model not found or could not be loaded'}), 404
            
        # Prepare input data
        # Expecting features to be a dict, convertible to DataFrame for sklearn
        # e.g. {"TV_Ad_Budget": 100, ...}
        
        # Ensure correct feature order/names based on training
        if model_type == 'sales':
            feature_names = ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']
        elif model_type == 'student':
            feature_names = ['Study_Hours', 'Attendance_Percentage', 'Previous_Score']
        else:
            return jsonify({'error': 'Unknown model type'}), 400
            
        # Create DataFrame
        try:
            input_df = pd.DataFrame([features])
            # Select/Reorder columns
            input_df = input_df[feature_names]
        except KeyError as e:
            return jsonify({'error': f'Missing feature: {str(e)}'}), 400
            
        # Predict
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'prediction': float(prediction)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo')
def demo_page():
    """Render the public demo page."""
    user = get_current_user()
    user_models = []
    untrained_datasets = []
    
    if user:
        # Fetch all user datasets
        datasets = Dataset.query.filter_by(user_id=user.id).all()
        for d in datasets:
            info = {
                'id': d.id,
                'name': d.name,
                'target': d.target_column,
                'type': d.problem_type,
                'created_at': d.created_at.strftime('%Y-%m-%d')
            }
            
            if d.model_path and os.path.exists(d.model_path):
                user_models.append(info)
            else:
                untrained_datasets.append(info)
                
    return render_template('demo.html', user=user, user_models=user_models, untrained_datasets=untrained_datasets)


@app.route('/api/user_model_info/<int:dataset_id>')
@jwt_required
def get_user_model_info(dataset_id):
    """Get metadata for a user's trained model."""
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    if not dataset.model_path or not os.path.exists(dataset.model_path):
        return jsonify({'error': 'Model not found'}), 404
        
    try:
        import joblib
        model_data = joblib.load(dataset.model_path)
        
        # Extract metadata
        # model_data is a dict with keys: scalar, label_encoder, feature_names, etc.
        return jsonify({
            'success': True,
            'name': dataset.name,
            'features': model_data.get('feature_names', []),
            'target': dataset.target_column,
            'problem_type': dataset.problem_type,
            'metrics': model_data.get('metrics', {})
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_user_model', methods=['POST'])
@jwt_required
def predict_user_model():
    """Predict using a user's trained model."""
    _load_data_libs()
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        features = data.get('features')
        
        if not dataset_id or not features:
            return jsonify({'error': 'Missing dataset_id or features'}), 400
            
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != g.current_user.id:
            return jsonify({'error': 'Access denied'}), 403
            
        if not dataset.model_path or not os.path.exists(dataset.model_path):
            return jsonify({'error': 'Model not found'}), 404
            
        import joblib
        model_data = joblib.load(dataset.model_path)
        
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names', [])
        
        # Prepare input
        input_df = pd.DataFrame([features])
        
        # Ensure columns match training data
        # Fill missing with 0 or mean? 0 for now
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder
        input_df = input_df[feature_names]
        
        # Scale
        if scaler:
            input_df = scaler.transform(input_df)
            
        # Predict
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'target': dataset.target_column
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# RUN APP
# ==============================================================================

@app.route('/dataset/<int:dataset_id>/synthesize', methods=['POST'])
@jwt_required
def generate_synthetic(dataset_id):
    """Generate synthetic data based on the dataset."""
    _load_data_libs()
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    try:
        df = pd.read_csv(dataset.cleaned_path)
        
        # Generator
        from data_pipeline.synthetic_generator import SyntheticGenerator
        gen = SyntheticGenerator(df)
        gen.fit()
        
        # Get count from request or default to len(df)
        data = request.get_json() or {}
        n_rows = int(data.get('n_rows', len(df)))
        
        synthetic_df = gen.generate(n_rows=n_rows)
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'synthetic_{dataset.id}_{timestamp}.csv'
        user_folder = get_user_folder(g.current_user.id)
        path = os.path.join(user_folder, filename)
        synthetic_df.to_csv(path, index=False)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'preview': synthetic_df.head(5).replace({np.nan: None}).to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
@jwt_required
def download_file(filename):
    """Generic download for user files (synthetic, row changes, etc)."""
    user_folder = get_user_folder(g.current_user.id)
    return send_from_directory(user_folder, filename, as_attachment=True)


@app.route('/download_changes/<filename>')
@jwt_required
def download_changes(filename):
    """Legacy route for row-level changes (aliased to download_file)."""
    return download_file(filename)



@app.route('/dataset/<int:dataset_id>/evolve', methods=['POST'])
@jwt_required
def evolve_features(dataset_id):
    """Automatically evolve features using interaction terms."""
    _load_data_libs()
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    try:
        # Load cleaned data (pre-encoding)
        df = pd.read_csv(dataset.cleaned_path)
        
        # Initialize Feature Engineer
        from data_pipeline.feature_engineer import FeatureEngineer
        engineer = FeatureEngineer(
            df, 
            target_col=dataset.target_column, 
            problem_type=dataset.problem_type
        )
        
        # Standard steps + Evolve
        engineer.create_datetime_features()
        engineer.encode_categorical()
        engineer.auto_evolve(max_new_features=5)
        # engineer.scale_features() # Maybe skip scaling here to keep it readable? 
        # But 'final' usually implies scaled. Let's scale.
        engineer.scale_features()
        
        # Save new final data
        final_df = engineer.get_transformed_data()
        final_df.to_csv(dataset.final_path, index=False)
        
        # Update metadata
        dataset.final_rows = final_df.shape[0]
        dataset.final_cols = final_df.shape[1]
        
        # Update log
        try:
            log = json.loads(dataset.processing_log)
        except:
            log = {}
            
        log['feature_engineering'] = engineer.get_summary()['transformations']
        dataset.processing_log = json.dumps(log)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'features_count': final_df.shape[1],
            'transformations': log['feature_engineering']
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/<int:dataset_id>/drift', methods=['POST'])
@jwt_required
def check_drift(dataset_id):
    """Check for data drift against a baseline."""
    current_ds = Dataset.query.get_or_404(dataset_id)
    if current_ds.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    data = request.get_json()
    baseline_id = data.get('baseline_id')
    
    if not baseline_id:
        return jsonify({'error': 'Baseline dataset ID required'}), 400
        
    baseline_ds = Dataset.query.get(baseline_id)
    if not baseline_ds or baseline_ds.user_id != g.current_user.id:
        return jsonify({'error': 'Invalid baseline dataset'}), 400
        
    try:
        # Load both datasets (cleaned for fair comparison)
        cur_df = pd.read_csv(current_ds.cleaned_path)
        base_df = pd.read_csv(baseline_ds.cleaned_path)
        
        from data_pipeline.drift_detector import DriftDetector
        # Note: DriftDetector(baseline, current)
        detector = DriftDetector(base_df, cur_df)
        detector.run()
        
        return jsonify({
            'success': True,
            'report': detector.report
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# LLM PIPELINE ROUTES
# ==============================================================================

def get_llm_session_path(user_id):
    return os.path.join(get_user_folder(user_id), 'llm_sessions.json')

def get_llm_runs_path(user_id):
    return os.path.join(get_user_folder(user_id), 'llm_runs.json')

def load_llm_sessions(user_id):
    path = get_llm_session_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_llm_sessions(user_id, sessions):
    path = get_llm_session_path(user_id)
    with open(path, 'w') as f:
        json.dump(sessions, f)

def load_llm_runs(user_id):
    path = get_llm_runs_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

def save_llm_runs(user_id, runs):
    path = get_llm_runs_path(user_id)
    with open(path, 'w') as f:
        json.dump(runs, f)

@app.route('/llm')
@jwt_required
def llm_page():
    """Render the LLM Pipeline page."""
    return render_template('llm.html', user=g.current_user)


@app.route('/api/llm/ingest', methods=['POST'])
@jwt_required
def llm_ingest():
    """Ingest documents from uploaded files and URLs."""
    import tempfile, uuid
    from data_pipeline.document_ingestor import DocumentIngestor

    session_id = str(uuid.uuid4())[:8]
    user_folder = get_user_folder(g.current_user.id)
    llm_folder = os.path.join(user_folder, 'llm_temp', session_id)
    os.makedirs(llm_folder, exist_ok=True)

    sources = []

    # Save uploaded files
    files = request.files.getlist('files')
    for f in files:
        if f.filename:
            path = os.path.join(llm_folder, f.filename)
            f.save(path)
            sources.append(path)

    # URLs
    urls_json = request.form.get('urls', '[]')
    try:
        url_list = json.loads(urls_json)
        sources.extend(url_list)
    except:
        pass

    if not sources:
        return jsonify({'error': 'No files or URLs provided'}), 400

    try:
        ingestor = DocumentIngestor()
        docs = ingestor.ingest(sources)
        stats = ingestor.get_stats()

        # Store in session (persisted)
        user_id_str = str(g.current_user.id)
        sessions = load_llm_sessions(g.current_user.id)
        sessions[session_id] = {
            'documents': docs,
            'user_id': g.current_user.id,
            'folder': llm_folder
        }
        save_llm_sessions(g.current_user.id, sessions)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'total_docs': len(docs),
            'total_chars': stats.get('total_chars', 0),
            'total_words': stats.get('total_words', 0),
            'source_types': list(stats.get('by_type', {}).keys())
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/process', methods=['POST'])
@jwt_required
def llm_process():
    """Run chunking, formatting, and quality scoring."""
    from data_pipeline.text_chunker import TextChunker
    from data_pipeline.instruct_formatter import InstructFormatter
    from data_pipeline.quality_scorer import QualityScorer

    data = request.json
    session_id = data.get('session_id')

    sessions = load_llm_sessions(g.current_user.id)
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session. Run ingestion first.'}), 400

    session = sessions[session_id]
    if session['user_id'] != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403

    docs = session['documents']

    try:
        # Chunk
        chunker = TextChunker(
            method=data.get('chunk_method', 'sliding_window'),
            chunk_size=data.get('chunk_size', 512),
            overlap=64
        )
        chunks = chunker.chunk_documents(docs)

        # Format
        formatter = InstructFormatter(template=data.get('template', 'alpaca'))
        pairs = formatter.format_chunks(
            chunks,
            domain=data.get('domain', 'general'),
            generate_qa=True,
            pairs_per_chunk=2
        )

        # Score
        min_quality = data.get('min_quality', 0.4)
        scorer = QualityScorer(min_quality_score=min_quality)
        scored = scorer.score(pairs)
        filtered = scorer.filter(scored, min_score=min_quality)

        # Calculate average quality
        scores = [p.get('quality', {}).get('overall_score', 0) for p in filtered]
        avg_quality = sum(scores) / len(scores) if scores else 0

        # Store in session (persisted)
        session['chunks'] = chunks
        session['pairs'] = pairs
        session['filtered_pairs'] = filtered
        session['template'] = data.get('template', 'alpaca')
        session['avg_quality'] = avg_quality
        save_llm_sessions(g.current_user.id, sessions)

        # Return sample pairs (first 10)
        sample = []
        for p in filtered[:10]:
            s = {k: v for k, v in p.items() if k in ('instruction', 'output', 'input', 'messages', 'quality')}
            sample.append(s)

        return jsonify({
            'success': True,
            'total_chunks': len(chunks),
            'total_pairs': len(pairs),
            'filtered_pairs': len(filtered),
            'avg_quality': avg_quality,
            'sample_pairs': sample
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/export', methods=['POST'])
@jwt_required
def llm_export():
    """Version dataset and generate training config."""
    from data_pipeline.dataset_registry import DatasetRegistry
    from data_pipeline.finetune_config import FineTuneConfig
    from data_pipeline.instruct_formatter import InstructFormatter

    data = request.json
    session_id = data.get('session_id')

    sessions = load_llm_sessions(g.current_user.id)
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session. Run the pipeline first.'}), 400

    session = sessions[session_id]
    if session['user_id'] != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403

    filtered = session.get('filtered_pairs', [])
    if not filtered:
        return jsonify({'error': 'No processed data. Run process first.'}), 400

    try:
        user_folder = get_user_folder(g.current_user.id)
        export_dir = os.path.join(user_folder, 'llm_exports', session_id)
        os.makedirs(export_dir, exist_ok=True)

        # Export JSONL
        data_path = os.path.join(export_dir, 'training_data.jsonl')
        formatter = InstructFormatter(template=session.get('template', 'alpaca'))
        formatter.export_jsonl(filtered, data_path, include_metadata=False)

        # Version
        version = data.get('version', 'v1.0.0')
        description = data.get('description', '')
        registry_dir = os.path.join(user_folder, 'llm_registry')
        registry = DatasetRegistry(registry_dir)
        try:
            registry.register(filtered, version=version, description=description)
        except Exception:
            pass  # Version may already exist

        # Training config
        model = data.get('model', 'meta-llama/Meta-Llama-3-8B')
        method = data.get('method', 'lora')
        config = FineTuneConfig(model_name=model, method=method, backend='trl')
        config_files = config.export(export_dir, dataset_path='./training_data.jsonl')

        # Log run (persisted)
        runs = load_llm_runs(g.current_user.id)
        runs.append({
            'run_id': f"{version}-{session_id}",
            'session_id': session_id,
            'user_id': g.current_user.id,
            'sample_count': len(filtered),
            'avg_quality': session.get('avg_quality', 0),
            'template': session.get('template', 'alpaca'),
            'model': model,
            'method': method,
            'version': version,
            'timestamp': datetime.now().isoformat()
        })
        save_llm_runs(g.current_user.id, runs)

        # Model short name
        model_short = model.split('/')[-1] if '/' in model else model
        if len(model_short) > 15:
            model_short = model_short[:15] + '…'

        files = [
            {'name': 'training_data.jsonl', 'label': 'Training Data (JSONL)'},
            {'name': 'training_config.json', 'label': 'Config (JSON)'},
            {'name': 'train.py', 'label': 'Train Script'},
            {'name': 'requirements_training.txt', 'label': 'Requirements'},
        ]

        return jsonify({
            'success': True,
            'sample_count': len(filtered),
            'version': version,
            'model_short': model_short,
            'method': method,
            'files': files
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/download/<session_id>/<filename>')
@jwt_required
def llm_download(session_id, filename):
    """Download a generated LLM pipeline file."""
    user_folder = get_user_folder(g.current_user.id)
    export_dir = os.path.join(user_folder, 'llm_exports', session_id)
    file_path = os.path.join(export_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(file_path, as_attachment=True, download_name=filename)


@app.route('/api/llm/runs')
@jwt_required
def llm_runs():
    """List LLM pipeline runs for current user."""
    runs = load_llm_runs(g.current_user.id)
    return jsonify({'runs': runs})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print("=" * 60)
    print("DATA PIPELINE WEB APP (JWT Authentication)")
    print("=" * 60)
    print(f"\n🌐 Open in browser: http://127.0.0.1:{port}")
    print("🔐 Auth: JWT tokens in HttpOnly cookies")
    print("🧬 LLM Pipeline: /llm\n")
    app.run(debug=True, host='0.0.0.0', port=port)
