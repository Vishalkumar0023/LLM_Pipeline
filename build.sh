#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit  # exit on error

pip install --upgrade pip

# Install heavy scientific packages as binary-only (no source compilation)
pip install --only-binary=:all: numpy pandas scipy scikit-learn matplotlib

# Install remaining lightweight packages
pip install -r requirements.txt

# Initialize the database
python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('Database initialized successfully!')
"
