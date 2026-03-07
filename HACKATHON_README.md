# 🔬 DataPipe — Automated Data Pipeline & ML Platform
## One-Click Data Cleaning, ML Training & LLM Fine-Tuning

---

## 📋 Problem Statement

### The Challenge
Data scientists spend **60-80% of their time** on data preprocessing. This repetitive work:
- Delays model development by hours/days
- Introduces human errors and inconsistencies
- Lacks standardization across projects
- Requires significant technical expertise

### Our Solution
An **automated, web-based data pipeline** that transforms raw messy data into ML-ready datasets with:
- ✅ One-click data cleaning (missing values, duplicates, outliers)
- ✅ Automated ML model training with accuracy comparison
- ✅ Explainable AI (SHAP) for model transparency
- ✅ Synthetic data generation (privacy-safe)
- ✅ **HuggingFace Integration:** Instant 1-click cloud model deployment
- ✅ **LLM Generative Pipeline:** Ingest raw web URLs → Auto-chunking → LoRA Instruct formatting
- ✅ Premium dark UI with modern glassmorphism design

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python 3.10+, Flask, SQLAlchemy, JWT Auth |
| **ML Engine** | Scikit-learn, XGBoost, SHAP, Pandas, NumPy |
| **Generative AI & LLMOps** | HuggingFace `huggingface_hub`, BeautifulSoup4 for web scraping |
| **LLM Pipeline** | Sliding-window Overlap Chunking, `Alpaca` Instruct formatting |
| **Frontend** | HTML5, CSS3 (Clarid Dark Premium), JavaScript ES6+ |
| **Database** | SQLite (users, datasets, model metadata) |
| **Auth** | JWT tokens in HttpOnly cookies, bcrypt hashing |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Web Browser (Clarid Dark Premium UI)        │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/REST + JWT
┌─────────────────────▼───────────────────────────────────┐
│                  Flask Web Server                        │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌───────────┐   │
│  │ Auth/JWT │ │ Upload   │ │ ML API  │ │ LLM API   │   │
│  └──────────┘ └──────────┘ └─────────┘ └───────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                Data Pipeline Engine                      │
│  ┌────────────┐ ┌────────────┐ ┌──────────────────┐     │
│  │DataCleaner │ │ModelTrainer│ │ LLM Gen-AI       │     │
│  │• Missing   │ │• AutoML    │ │• Web Scraping    │     │
│  │• Outliers  │ │• HuggingFace │ │• Sliding Context │     │
│  │• Encoding  │ │• Deploy APIs │ │• Instruct Format │     │
│  └────────────┘ └────────────┘ └──────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Core Features

### 🧹 Automated Data Cleaning
- Missing value imputation (mean/median/mode — auto-selected)
- Duplicate detection & removal
- Outlier detection via IQR method
- Data type optimization & column standardization
- Inconsistent category normalization

### 🧠 AutoML Training & Cloud Deployment
- Trains 5+ models automatically (Random Forest, XGBoost, SVM, etc.)
- **Raw vs Cleaned accuracy comparison** — proves the pipeline works!
- Feature importance with animated bars
- Reliability grading (A/B/C/D)
- Downloadable `.pkl` model + printable PDF report
- 🚀 **One-Click HuggingFace Deployment:** Automatically wrap the trained model and push it securely to the HuggingFace Model Hub via API tokens for instant global inference scaling!

### 🔮 Explainable AI (SHAP)
- Feature impact magnitude plots
- Feature value distribution analysis
- Per-prediction explanations for model transparency

### ⚡ Feature Evolution
- AI-discovers polynomial & interaction features
- Auto-selects statistically significant "super features"
- Updates model-ready dataset automatically

### 🧬 Synthetic Data Generator
- Privacy-safe statistical data generation
- Configurable row count
- Preserves original distributions
- Downloadable CSV output

### 🤖 Unstructured Generative AI (LLM) Pipeline
- **Ingest**: Paste raw Web URLs and Wikipedia pages; system auto-scrapes via `BeautifulSoup4`.
- **Advanced Context Processing**: Utilizes a Sliding-Window Overlapping Chunking algorithm mapping to preserve semantic context across Transformer memory bounds.
- **Instruct Schema**: Formats generic arrays natively into behavioral JSON pairs (`system`, `instruction`, `input`, `output`) required for LoRA (Low-Rank Adaptation).
- **Export**: `training_data.jsonl` + `training_config.json` (Mistral/Llama 3 ready).

### 🔐 Security
- JWT token authentication (HttpOnly cookies)
- Password hashing (bcrypt via Werkzeug)
- Per-user data isolation
- Auto token refresh

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python3 app.py

# Open in browser
open http://127.0.0.1:8080
```

---

## 🎯 Demo Guide (for Judges)

### Sample Dataset Included
A pre-built **`demo_student_performance.csv`** is included with intentional data quality issues:

| Issue | Count |
|-------|-------|
| Missing values | 148 across 10+ columns |
| Duplicate rows | 7 |
| Outliers | Study_Hours=80/95, Attendance=120/-5 |
| Inconsistent categories | Male/M, Yes/yes/YES, highschool/High School |
| Negative values | Where impossible (scores, attendance) |

**Target column**: `Final_Grade` (A/B/C/D/F — Classification)

### Recommended Demo Flow

1. **Landing Page** → Show premium dark UI design
2. **Sign Up** → Demonstrate JWT authentication
3. **Upload `demo_student_performance.csv`** → Select target: `Final_Grade`, type: `Classification`
4. **View Cleaned Results** → Show stats (before/after), cleaning log
5. **Click "Train Model"** → Watch Raw vs Cleaned comparison, SHAP plots
6. **Boost Features** → Run Feature Evolution
7. **Generate Synthetic Data** → Create 500 privacy-safe rows
8. **Live Demo** (`/demo`) → Make real-time predictions
9. **LLM Pipeline** (`/llm`) → Paste any URL → Generate training data

### Key Metrics to Highlight

| Metric | Before (Manual) | After (DataPipe) |
|--------|-----------------|-------------------|
| Data prep time | 4-8 hours | < 2 minutes |
| Lines of code | 200+ | 0 |
| Error rate | Variable | Standardized |
| Reproducibility | Low | 100% |

---

## 📁 Project Structure

```
dataClean1/
├── app.py                          # Flask web application (JWT + all routes)
├── data_pipeline/                  # Core pipeline engine
│   ├── data_loader.py              # File loading utilities
│   ├── data_cleaner.py             # Cleaning operations
│   ├── eda.py                      # Analysis & visualization
│   ├── feature_engineer.py         # Feature transformations
│   ├── pipeline.py                 # Pipeline orchestrator
│   ├── model_trainer.py            # AutoML training + SHAP
│   ├── dataset_registry.py         # LLM dataset registry
│   └── instruct_formatter.py       # LLM instruction formatting
├── templates/                      # HTML templates (Clarid Dark Premium)
│   ├── index.html                  # Landing page
│   ├── auth.html                   # Login/Signup
│   ├── dashboard.html              # User dashboard
│   ├── view_dataset.html           # Dataset analysis & ML training
│   ├── files.html                  # File manager
│   ├── demo.html                   # Live prediction demo
│   └── llm.html                    # LLM fine-tuning pipeline
├── demo_student_performance.csv    # Sample messy dataset for demos
├── requirements.txt                # Python dependencies
└── pipeline_users.db               # SQLite database
```

---

## 🎨 Design System

**Clarid Dark Premium** — A custom dark theme with:
- Deep background (`#060911`) with ambient teal/purple glow
- Glassmorphism cards with subtle borders
- Gradient typography (`#38bdf8 → #818cf8 → #c084fc`)
- Smooth micro-animations and hover effects
- Fully responsive layout
- Inter font family

---

## 👥 Team

**Project**: DataPipe — Automated Data Pipeline & ML Platform  
**Built with**: Python · Flask · Scikit-learn · SHAP · JWT · Clarid Dark UI

---

*Built for Hackathon 2026* 🏆
