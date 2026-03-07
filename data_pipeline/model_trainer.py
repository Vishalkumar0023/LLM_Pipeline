"""
Model Trainer Module
====================
Advanced ML training with auto-detection, multi-model comparison,
cross-validation, hyperparameter tuning, and model export.
Memory-optimized for 512MB hosting (no SHAP, reduced models).
"""

import pandas as pd
import numpy as np
import time
import joblib
import re
import gc
import warnings
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from sklearn.model_selection import (
    cross_validate, StratifiedKFold, KFold, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error,
    classification_report, make_scorer
)

HAS_XGBOOST = False  # Disabled for memory optimization

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

warnings.filterwarnings('ignore')


# ─── Column pattern filters ─────────────────────────────────────────────────
ID_PATTERNS = re.compile(
    r'^(id|_id|index|row_?num|serial|sr_?no|unnamed)', re.IGNORECASE
)
URL_PATTERNS = re.compile(
    r'(url|link|href|src|image|img|photo|avatar|thumbnail|path|file)', re.IGNORECASE
)
TARGET_HINTS = [
    'target', 'label', 'class', 'outcome', 'result', 'y',
    'is_', 'has_', 'survived', 'churn', 'fraud', 'default',
    'price', 'salary', 'revenue', 'cost', 'amount', 'score'
]


class ModelTrainer:
    """
    Advanced ML model trainer with auto-detection, multi-model comparison,
    cross-validation, hyperparameter tuning, and explainability.
    
    Example
    -------
    >>> trainer = ModelTrainer(df, target_col='price', problem_type='regression')
    >>> results = trainer.run()
    >>> print(results['best_model'])
    >>> print(results['metrics_dashboard'])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        problem_type: Optional[str] = None,
        raw_df: Optional[pd.DataFrame] = None
    ):
        self.original_df = raw_df.copy() if raw_df is not None else df.copy()
        self.df = df.copy()
        self.target_col = target_col
        self.problem_type = problem_type  # 'classification' or 'regression'
        self.warnings: List[str] = []
        self.log: List[str] = []

        # Will be set during prepare
        self.X = None
        self.y = None
        self.feature_names: List[str] = []
        self.scaler = None
        self.label_encoder = None

        # Results
        self.models: Dict[str, Any] = {}
        self.cv_results: Dict[str, Dict] = {}
        self.best_model_name: str = ''
        self.best_model = None
        self.best_metrics: Dict[str, float] = {}
        self.importances: Dict[str, float] = {}

    # ─── 1. Target Detection ────────────────────────────────────────────
    def detect_target(self) -> str:
        """Auto-detect or validate the target column."""
        cols = self.df.columns.tolist()

        # If already provided, validate it
        if self.target_col:
            if self.target_col in cols:
                self.log.append(f"Target column confirmed: '{self.target_col}'")
                return self.target_col
            else:
                self.warnings.append(
                    f"Specified target '{self.target_col}' not found in data."
                )
                self.target_col = None

        # Try heuristic search
        for hint in TARGET_HINTS:
            for col in cols:
                if hint.lower() == col.lower() or col.lower().startswith(hint.lower()):
                    self.target_col = col
                    self.log.append(f"Auto-detected target column: '{col}' (matched hint '{hint}')")
                    return col

        # Fallback: last column
        self.target_col = cols[-1]
        self.warnings.append(
            f"No obvious target found — using last column '{self.target_col}'. "
            "You can specify a target column explicitly."
        )
        self.log.append(f"Fallback target column: '{self.target_col}'")
        return self.target_col

    # ─── 2. Problem Type Detection ──────────────────────────────────────
    def detect_problem_type(self) -> str:
        """Detect whether this is a classification or regression problem."""
        if self.target_col is None:
            self.detect_target()
            
        y = self.df[self.target_col]
        
        # User override with validation
        if self.problem_type:
            if self.problem_type == 'regression':
                # Validate that target is numeric
                if y.dtype == 'object' or y.dtype.name == 'category':
                    try:
                        pd.to_numeric(y, errors='raise')
                    except Exception:
                        self.warnings.append(
                            f"Regression requested but target '{self.target_col}' is non-numeric. "
                            "Switching to classification."
                        )
                        self.log.append(f"Auto-corrected problem type to 'classification' (target is string)")
                        self.problem_type = 'classification'
                        return 'classification'
            
            self.log.append(f"Problem type specified: {self.problem_type}")
            return self.problem_type

        # Auto-detection
        # Object/category → classification
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.problem_type = 'classification'
        # Few unique values → classification
        elif y.nunique() <= 15 and y.nunique() / len(y) < 0.05:
            self.problem_type = 'classification'
        else:
            self.problem_type = 'regression'

        self.log.append(f"Auto-detected problem type: {self.problem_type}")
        return self.problem_type

    # ─── 3. Feature Preparation ─────────────────────────────────────────
    def _drop_junk_columns(self):
        """Remove ID, URL, image, and metadata columns."""
        cols_to_drop = []
        for col in self.df.columns:
            if col == self.target_col:
                continue
            # ID-like
            if ID_PATTERNS.match(col):
                cols_to_drop.append(col)
                continue
            # URL/image-like
            if URL_PATTERNS.search(col):
                cols_to_drop.append(col)
                continue
            # Columns that are all unique strings (likely IDs)
            if self.df[col].dtype == 'object':
                if self.df[col].nunique() > 0.9 * len(self.df) and len(self.df) > 20:
                    cols_to_drop.append(col)
                    continue

        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.log.append(f"Dropped {len(cols_to_drop)} junk columns: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")

    def _drop_high_correlation(self, threshold: float = 0.9):
        """Remove one of each pair of features correlated above threshold."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        if len(numeric_cols) < 2:
            return

        corr = self.df[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > threshold)]

        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.log.append(f"Dropped {len(to_drop)} highly correlated features (r > {threshold})")

    def _encode_categoricals(self):
        """Label-encode remaining categorical columns."""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_col in cat_cols:
            cat_cols.remove(self.target_col)

        for col in cat_cols:
            le = LabelEncoder()
            mask = self.df[col].notna()
            self.df.loc[mask, col] = le.fit_transform(self.df.loc[mask, col].astype(str))
            self.df[col] = self.df[col].astype(float)

        if cat_cols:
            self.log.append(f"Label-encoded {len(cat_cols)} categorical features")

    def _extract_datetime_parts(self):
        """Extract date parts from datetime columns."""
        dt_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        for col in dt_cols:
            if col == self.target_col:
                continue
            self.df[f'{col}_year'] = self.df[col].dt.year
            self.df[f'{col}_month'] = self.df[col].dt.month
            self.df[f'{col}_day'] = self.df[col].dt.day
            self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek
            self.df.drop(columns=[col], inplace=True)
        if dt_cols:
            self.log.append(f"Extracted date parts from {len(dt_cols)} datetime columns")

    def _create_ratio_features(self, max_ratios: int = 5):
        """Generate ratio features from top numeric pairs."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        if len(numeric_cols) < 2:
            return

        # Use top columns by variance
        variances = self.df[numeric_cols].var().sort_values(ascending=False)
        top_cols = variances.head(min(4, len(numeric_cols))).index.tolist()

        created = 0
        for i in range(len(top_cols)):
            for j in range(i + 1, len(top_cols)):
                if created >= max_ratios:
                    break
                a, b = top_cols[i], top_cols[j]
                denom = self.df[b].replace(0, np.nan)
                ratio = self.df[a] / denom
                if ratio.notna().sum() > 0.5 * len(self.df):
                    name = f'{a}_div_{b}'
                    self.df[name] = ratio.fillna(0)
                    created += 1
            if created >= max_ratios:
                break

        if created:
            self.log.append(f"Created {created} ratio features")

    def _create_bins(self, n_bins: int = 5):
        """Bin the top numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        cols = numeric_cols[:5]
        created = 0
        for col in cols:
            try:
                self.df[f'{col}_bin'] = pd.qcut(
                    self.df[col], q=n_bins, labels=False, duplicates='drop'
                )
                created += 1
            except Exception:
                pass
        if created:
            self.log.append(f"Created binned features for {created} columns")

    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Full feature preparation pipeline. Returns (X, y)."""
        self.detect_target()
        self.detect_problem_type()

        # Dataset size warning
        if len(self.df) < 100:
            self.warnings.append(
                f"⚠ Small dataset ({len(self.df)} rows). "
                "Results may be unreliable. Consider collecting more data."
            )

        # Feature engineering steps
        self._drop_junk_columns()
        self._extract_datetime_parts()
        self._encode_categoricals()
        self._create_ratio_features()
        self._create_bins()
        self._drop_high_correlation(threshold=0.9)

        # Separate target
        y = self.df[self.target_col].copy()
        X = self.df.drop(columns=[self.target_col]).copy()

        # Drop any remaining non-numeric
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            X.drop(columns=non_numeric, inplace=True)
            self.log.append(f"Dropped {len(non_numeric)} non-numeric columns before training")

        # Fill remaining NaN
        X = X.fillna(X.median())

        # Encode target for classification if needed
        if self.problem_type == 'classification' and y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y), name=self.target_col)
            self.log.append("Label-encoded target column")

        y = y.fillna(y.mode()[0] if self.problem_type == 'classification' else y.median())

        # Scale features
        self.feature_names = X.columns.tolist()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.log.append(f"Scaled {len(self.feature_names)} features with StandardScaler")

        self.X = X_scaled
        self.y = y.values if hasattr(y, 'values') else np.array(y)

        self.log.append(f"Final training shape: X={self.X.shape}, y={self.y.shape}")
        return self.X, self.y

    # ─── 4. Imbalance Detection ─────────────────────────────────────────
    def detect_imbalance(self) -> Dict[str, Any]:
        """Detect class imbalance for classification problems."""
        if self.problem_type != 'classification':
            return {'imbalanced': False, 'reason': 'Regression problem'}

        counts = pd.Series(self.y).value_counts(normalize=True)
        majority_pct = counts.iloc[0]
        minority_pct = counts.iloc[-1]

        imbalanced = majority_pct > 0.70
        info = {
            'imbalanced': imbalanced,
            'majority_class': str(counts.index[0]),
            'majority_pct': round(majority_pct * 100, 1),
            'minority_class': str(counts.index[-1]),
            'minority_pct': round(minority_pct * 100, 1),
            'class_distribution': {str(k): round(v * 100, 1) for k, v in counts.items()}
        }

        if imbalanced:
            self.warnings.append(
                f"Class imbalance detected: majority {info['majority_pct']}% / "
                f"minority {info['minority_pct']}%. Using class weights or SMOTE."
            )
            self.log.append(f"Imbalance detected: {info['class_distribution']}")

        return info

    # ─── 5. Model Definitions ──────────────────────────────────────────
    def _get_models_and_params(self) -> Dict[str, Dict]:
        """Return model instances and hyperparameter grids."""
        imbalance = self.detect_imbalance()
        use_balanced = imbalance.get('imbalanced', False)
        n_classes = len(np.unique(self.y)) if self.problem_type == 'classification' else 0

        if self.problem_type == 'classification':
            models = {
                'Logistic Regression': {
                    'model': LogisticRegression(
                        max_iter=1000, random_state=42,
                        class_weight='balanced' if use_balanced else None
                    ),
                    'params': {
                        'C': [0.1, 1, 10],
                        'solver': ['lbfgs']
                    }
                },
                'Random Forest': {
                    'model': RandomForestClassifier(
                        random_state=42, n_jobs=1,
                        class_weight='balanced' if use_balanced else None
                    ),
                    'params': {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10],
                        'min_samples_split': [2, 5]
                    }
                }
            }
        else:  # regression
            models = {
                'Ridge Regression': {
                    'model': Ridge(random_state=42),
                    'params': {
                        'alpha': [0.1, 1, 10]
                    }
                },
                'Random Forest': {
                    'model': RandomForestRegressor(random_state=42, n_jobs=1),
                    'params': {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10],
                        'min_samples_split': [2, 5]
                    }
                }
            }

        return models

    # ─── 6. Training ───────────────────────────────────────────────────
    def train_all_models(self) -> Dict[str, Dict]:
        """Train all models with cross-validation and hyperparameter tuning."""
        if self.X is None:
            self.prepare_features()

        model_defs = self._get_models_and_params()

        # Handle SMOTE for classification imbalance
        X_train, y_train = self.X, self.y
        imbalance = self.detect_imbalance()
        if imbalance.get('imbalanced') and HAS_SMOTE and self.problem_type == 'classification':
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                self.log.append(f"Applied SMOTE: {len(self.y)} → {len(y_train)} samples")
            except Exception as e:
                self.log.append(f"SMOTE failed, using original data: {e}")

        # CV strategy
        n_folds = 3
        if self.problem_type == 'classification':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            scoring_primary = 'f1_weighted'
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            try:
                if len(np.unique(y_train)) == 2:
                    scoring.append('roc_auc')
            except Exception:
                pass
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            scoring_primary = 'r2'
            scoring = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']

        # Limit search iterations for speed
        n_iter = min(10, max(3, 100 // len(model_defs)))

        for name, config in model_defs.items():
            self.log.append(f"Training {name}...")
            try:
                # Hyperparameter tuning
                search = RandomizedSearchCV(
                    config['model'],
                    config['params'],
                    n_iter=min(n_iter, self._param_combinations(config['params'])),
                    cv=cv,
                    scoring=scoring_primary,
                    random_state=42,
                    n_jobs=1,
                    error_score='raise'
                )
                search.fit(X_train, y_train)
                best_estimator = search.best_estimator_

                # Cross-validate with all metrics
                cv_scores = cross_validate(
                    best_estimator, X_train, y_train,
                    cv=cv, scoring=scoring, n_jobs=1
                )

                # Store results
                metrics = {}
                for metric_name in scoring:
                    key = f'test_{metric_name}'
                    if key in cv_scores:
                        scores = cv_scores[key]
                        # Convert negative metrics
                        if metric_name.startswith('neg_'):
                            scores = -scores
                            clean_name = metric_name.replace('neg_', '')
                        else:
                            clean_name = metric_name
                        metrics[clean_name] = {
                            'mean': round(float(np.mean(scores)), 4),
                            'std': round(float(np.std(scores)), 4)
                        }

                self.models[name] = best_estimator
                self.cv_results[name] = {
                    'metrics': metrics,
                    'best_params': search.best_params_,
                    'best_cv_score': round(float(search.best_score_), 4)
                }
                self.log.append(f"  {name} → best CV score: {search.best_score_:.4f}")

            except Exception as e:
                self.log.append(f"  {name} failed: {str(e)[:100]}")
                self.cv_results[name] = {'error': str(e)}
            finally:
                gc.collect()

        return self.cv_results

    def _param_combinations(self, params: Dict) -> int:
        """Calculate total parameter combinations."""
        total = 1
        for v in params.values():
            total *= len(v)
        return total

    # ─── 7. Best Model Selection ───────────────────────────────────────
    def get_best_model(self) -> Dict[str, Any]:
        """Select the best model based on primary metric."""
        if not self.cv_results:
            self.train_all_models()

        primary_metric = 'f1_weighted' if self.problem_type == 'classification' else 'r2'

        best_score = -np.inf
        for name, result in self.cv_results.items():
            if 'error' in result:
                continue
            score = result['metrics'].get(primary_metric, {}).get('mean', -np.inf)
            if score > best_score:
                best_score = score
                self.best_model_name = name

        if self.best_model_name:
            self.best_model = self.models[self.best_model_name]
            self.best_metrics = self.cv_results[self.best_model_name]['metrics']
            self.log.append(f"Best model: {self.best_model_name} ({primary_metric}={best_score:.4f})")

        return {
            'name': self.best_model_name,
            'metrics': self.best_metrics,
            'params': self.cv_results.get(self.best_model_name, {}).get('best_params', {})
        }

    # ─── 8. Feature Importance ─────────────────────────────────────────
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Extract top feature importances from the best model."""
        if self.best_model is None:
            self.get_best_model()
        if self.best_model is None:
            return {}

        importances = None
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                coef = self.best_model.coef_
                if coef.ndim > 1:
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    importances = np.abs(coef)
        except Exception:
            pass

        if importances is None or len(importances) != len(self.feature_names):
            return {}

        imp_dict = dict(zip(self.feature_names, importances.tolist()))
        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))

        self.importances = {k: round(v, 6) for k, v in list(imp_dict.items())[:top_n]}
        return self.importances

    # ─── 9. Metrics Dashboard ──────────────────────────────────────────
    def get_metrics_dashboard(self) -> Dict[str, Any]:
        """Build a complete metrics dashboard."""
        if not self.cv_results:
            self.train_all_models()

        dashboard = {
            'problem_type': self.problem_type,
            'target_column': self.target_col,
            'dataset_shape': {
                'rows': int(self.X.shape[0]),
                'features': int(self.X.shape[1])
            },
            'models_compared': {},
            'best_model': self.get_best_model(),
            'feature_importance': self.get_feature_importance(),
            'reliability': self.get_reliability_score(),
            'warnings': self.warnings,
            'log': self.log,
            'explanation': self.explain_model()  # Add explanation
        }

        # Per-model summary
        for name, result in self.cv_results.items():
            if 'error' in result:
                dashboard['models_compared'][name] = {'error': result['error']}
            else:
                dashboard['models_compared'][name] = {
                    'metrics': result['metrics'],
                    'best_params': result['best_params']
                }

        return dashboard

    # ─── 15. Explainability (disabled for memory) ─────────────────────
    def explain_model(self) -> Dict[str, Any]:
        """SHAP disabled for memory optimization on 512MB hosting."""
        return {'available': False, 'error': 'Disabled for memory optimization'}

    # ─── 14. Full Pipeline Run ─────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        """
        Run the full ML training pipeline:
        1. Detect target & problem type
        2. Prepare features
        3. Train all models with CV + tuning
        4. Select best model
        5. Get feature importance
        6. Build metrics dashboard
        """
        self.prepare_features()
        self.train_all_models()
        self.get_best_model()
        dashboard = self.get_metrics_dashboard()
        gc.collect()
        return dashboard

    # ─── 10. Reliability Score ─────────────────────────────────────────
    def _compute_reliability(self, n_rows: int, cv_std: float, metric_val: float) -> Dict[str, Any]:
        """
        Compute a 0–100 reliability score based on:
        - Dataset size (bigger = better)
        - CV score variance (lower = better)
        - Primary metric quality
        """
        score = 50  # baseline
        reasons = []

        # Dataset size factor (0–25 pts)
        if n_rows >= 10000:
            score += 25
            reasons.append("Large dataset (+25)")
        elif n_rows >= 1000:
            pts = int(15 + (n_rows - 1000) / 900 * 10)
            score += pts
            reasons.append(f"Medium dataset (+{pts})")
        elif n_rows >= 100:
            pts = int(5 + (n_rows - 100) / 900 * 10)
            score += pts
            reasons.append(f"Small dataset (+{pts})")
        else:
            score -= 15
            reasons.append("Very small dataset (-15)")

        # CV variance factor (0–15 pts)
        # Use provided std dev
        if cv_std < 0.02:
            score += 15
            reasons.append("Very stable CV scores (+15)")
        elif cv_std < 0.05:
            score += 10
            reasons.append("Stable CV scores (+10)")
        elif cv_std < 0.1:
            score += 5
            reasons.append("Moderate CV variance (+5)")
        else:
            score -= 5
            reasons.append("High CV variance (-5)")

        # Metric quality factor (0–10 pts)
        if metric_val >= 0.9:
            score += 10
            reasons.append("Excellent primary metric (+10)")
        elif metric_val >= 0.75:
            score += 5
            reasons.append("Good primary metric (+5)")
        elif metric_val < 0.5:
            score -= 10
            reasons.append("Poor primary metric (-10)")

        score = max(0, min(100, score))
        return {
            'score': score,
            'grade': 'A' if score >= 85 else 'B' if score >= 70 else 'C' if score >= 50 else 'D',
            'reasons': reasons
        }

    def get_reliability_score(self) -> Dict[str, Any]:
        """Wrapper for current model reliability."""
        n = self.X.shape[0] if self.X is not None else len(self.df)
        
        primary = 'f1_weighted' if self.problem_type == 'classification' else 'r2'
        std = 0.1
        val = 0
        
        if self.best_metrics:
            std = self.best_metrics.get(primary, {}).get('std', 0.1)
            val = self.best_metrics.get(primary, {}).get('mean', 0)
            
        return self._compute_reliability(n, std, val)

    # ─── 11. Export ────────────────────────────────────────────────────
    def export_model(self, path: str) -> str:
        """Export the best model and metadata to a .pkl file."""
        if self.best_model is None:
            self.get_best_model()

        export_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'target_col': self.target_col,
            'problem_type': self.problem_type,
            'metrics': self.best_metrics,
            'feature_importance': self.importances,
            'exported_at': datetime.now().isoformat(),
            'reliability': self.get_reliability_score()
        }

        joblib.dump(export_data, path)
        self.log.append(f"Model exported to '{path}'")
        return path

    # ─── 13. Comparison Logic ──────────────────────────────────────────
    def train_naive_baseline(self) -> Dict[str, Any]:
        """
        Train a 'Naive' model on raw data for comparison.
        Strategy: 'Just Encoding' + Minimal Imputation (to prevent crash).
        """
        try:
            if self.original_df is None:
                msg = "Raw data not available for comparison"
                self.log.append(msg)
                return {"score": 0.0, "metrics": {}, "error": msg, "time_taken": 0, "memory_mb": 0}

            # work on a copy of original_df
            df = self.original_df.copy()
            
            # 1. Drop IDs (essential) but keep almost everything else
            for col in df.columns:
                if col == self.target_col: continue
                if ID_PATTERNS.match(col) or URL_PATTERNS.search(col):
                    df.drop(columns=[col], inplace=True)
            
            # 2. Handle Target (and potential cleaning-renaming mismatch)
            if self.target_col not in df.columns:
                # Try case-insensitive match
                col_map = {c.lower(): c for c in df.columns}
                if self.target_col.lower() in col_map:
                     found_col = col_map[self.target_col.lower()]
                     df.rename(columns={found_col: self.target_col}, inplace=True)
                     self.log.append(f"Mapped raw target '{found_col}' to '{self.target_col}'")
                else:
                    # Fail gracefully
                    msg = f"Target '{self.target_col}' not found in raw data"
                    self.log.append(msg)
                    return {"score": 0.0, "metrics": {}, "error": msg, "time_taken": 0, "memory_mb": 0}
            
            y = df[self.target_col]
            X = df.drop(columns=[self.target_col])
            
            # 3. Naive Preprocessing (The "Lazy" Way - using Pipeline to avoid leakage)
            
            # Identify columns
            num_cols = X.select_dtypes(include=[np.number]).columns
            cat_cols = X.select_dtypes(exclude=[np.number]).columns
            
            # Preprocessing for numeric data: simple mean imputation
            num_transformer = SimpleImputer(strategy='mean')
            
            # Preprocessing for categorical data: constant fill + ordinal encoding
            # OrdinalEncoder is used because it handles 2D arrays (unlike LabelEncoder)
            # and we can handle unknown values by encoding them as -1
            cat_transformer = make_pipeline(
                SimpleImputer(strategy='constant', fill_value='missing'),
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            )
            
            preprocessor = make_column_transformer(
                (num_transformer, num_cols),
                (cat_transformer, cat_cols),
                remainder='passthrough'
            )
            
            # Encode target if necessary
            if self.problem_type == 'classification' and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
            
            # 4. Train Naive Model (Simple Decision Tree)
            if self.problem_type == 'classification':
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(random_state=42)
                scoring = 'accuracy'
            else:
                from sklearn.tree import DecisionTreeRegressor
                model = DecisionTreeRegressor(random_state=42)
                scoring = 'r2'
                
            # Create full pipeline
            # Note: We don't scale or do anything fancy. Just impute -> encode -> tree.
            pipeline = make_pipeline(preprocessor, model)
            
            # Quick CV
            n_folds = 3
            metrics = {}
            if self.problem_type == 'classification':
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                scoring_dict = {
                    'accuracy': 'accuracy',
                    'precision': 'precision_weighted',
                    'f1': 'f1_weighted'
                }
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                scoring_dict = {
                    'r2': 'r2',
                    'mae': 'neg_mean_absolute_error',
                    'mse': 'neg_mean_squared_error'
                }
                
            start_time = time.time()
            try:
                cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_dict)
                for metric_name, scorer_name in scoring_dict.items():
                    key = f"test_{metric_name}"
                    if key in cv_results:
                         score = np.mean(cv_results[key])
                         if 'neg_' in scorer_name: score = -score
                         metrics[metric_name] = round(score, 4)
                
                # Main score for comparison
                main_metric = 'accuracy' if self.problem_type == 'classification' else 'r2'
                mean_score = metrics.get(main_metric, 0)
                
            except ValueError as ve:
                # Fallback for very small datasets
                model.fit(preprocessor.fit_transform(X, y), y)
                mean_score = model.score(preprocessor.transform(X), y)
                metrics[main_metric] = round(mean_score, 4)
            
            end_time = time.time()
            time_taken = round(end_time - start_time, 4)
            memory_mb = round(self.original_df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            
            # Calculate Reliability for Raw
            # Dataset size
            n_raw = len(df)
            
            # Std dev of primary metric
            try:
                if 'results' in locals() and results is not None:
                    if self.problem_type == 'classification':
                        raw_std = results['test_f1'].std()
                        raw_val = results['test_f1'].mean()
                    else:
                        raw_std = results['test_r2'].std()
                        raw_val = results['test_r2'].mean()
                else:
                     raise ValueError("Results not available")
            except Exception:
                # Fallback if CV failed or results missing
                raw_std = 0.2  # Assume high variance
                raw_val = mean_score

            reliability = self._compute_reliability(n_raw, raw_std, raw_val)

            return {
                "score": round(mean_score, 4),
                "metrics": metrics,
                "model_type": "Decision Tree (Baseline)",
                "time_taken": time_taken,
                "memory_mb": memory_mb,
                "reliability": reliability
            }
            
        except Exception as e:
            print(f"DEBUG: Naive baseline failed: {e}")
            print(f"DEBUG: Target: {self.target_col}")
            if self.original_df is not None:
                print(f"DEBUG: Columns: {self.original_df.columns.tolist()}")
            self.log.append(f"Naive baseline failed: {e}")
            return {"score": 0.0, "metrics": {}, "error": str(e), "time_taken": 0, "memory_mb": 0}

    def run_full_comparison(self) -> Dict[str, Any]:
        """
        Run both Naive (Raw) and Advanced (Cleaned) pipelines and compare.
        """
        # 1. Train Naive
        naive_results = self.train_naive_baseline()
        
        # 2. Run Advanced Pipeline (with timing)
        start_adv = time.time()
        advanced_dashboard = self.run()
        end_adv = time.time()
        advanced_dashboard['time_taken'] = round(end_adv - start_adv, 4)
        
        # 3. Compare
        advanced_score = 0
        advanced_metrics = {}
        
        # Calculate Advanced Stats
        start_time_adv = time.time()
        # (The 'run' method was already called above, so we can't truly measure it here unless we wrap 'self.run()'. 
        # But 'self.run()' includes 'train_all_models' which logs time. 
        # For this demo, let's assume Advanced takes slightly longer than Baseline + overhead due to search)
        # We'll use a heuristic based on complexity or just measure overhead of this function call as proxy if 'run' isn't wrapped.
        # BETTER: Let's assume 'run' took some time. We can't retroactively measure it easily without changing 'run'.
        # Let's use a placeholder heuristic: Advanced Time = Baseline Time * 1.5 (Simulated for Demo effect as "Optimization" usually implies *better* result but *more* compute? 
        # Actually user asked for "optimization like cpu time... by cleaned data".
        # Let's calculate memory for cleaned df.
        advanced_memory_mb = round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        
        # In a real app we'd time 'self.run()' properly. 
        # Let's wrap 'self.run()' in the caller next time. For now, let's estimate advanced_time based on log entries or just use a random factor of baseline to show data.
        # Wait, I can just measure 'time.time()' before and after 'self.run()' call in this method!
        # Ah, 'self.run()' is called at line 958. I should have wrapped it there.
        # I will replace the whole function to wrap 'self.run()' with timing.
        
        if 'best_model' in advanced_dashboard and 'metrics' in advanced_dashboard['best_model']:
            all_metrics = advanced_dashboard['best_model']['metrics']
            print(f"DEBUG: Available metrics keys: {list(all_metrics.keys())}")
            for k, v in all_metrics.items():
                if isinstance(v, dict) and 'mean' in v:
                    # Normalize keys: backend might use 'f1_weighted', frontend wants 'f1'
                    metric_key = k
                    if k == 'f1_weighted': metric_key = 'f1'
                    if k == 'precision_weighted': metric_key = 'precision'
                    advanced_metrics[metric_key] = round(v['mean'], 4)
            
            if self.problem_type == 'classification':
                advanced_score = advanced_metrics.get('accuracy', 0)
            else:
                advanced_score = advanced_metrics.get('r2', 0)
        
        naive_score = naive_results.get('score', 0)
        naive_metrics = naive_results.get('metrics', {})
        
        # Calculate Improvement
        improvement = 0
        if naive_score != 0:
            improvement = ((advanced_score - naive_score) / abs(naive_score)) * 100
        elif advanced_score > 0:
            improvement = 100 
            
        # Add comparison to dashboard
        advanced_dashboard['comparison'] = {
            'raw_score': naive_score,
            'cleaned_score': advanced_score,
            'raw_metrics': naive_metrics,
            'cleaned_metrics': advanced_metrics,
            'improvement_pct': round(improvement, 1),
            'metric': 'Accuracy' if self.problem_type == 'classification' else 'R² Score',
            'raw_error': naive_results.get('error'),
            'raw_reliability': naive_results.get('reliability'),
            'optimization': {
                'raw_time': naive_results.get('time_taken', 0),
                'cleaned_time': advanced_dashboard.get('time_taken', 0), # Will be set by wrapper
                'raw_space': naive_results.get('memory_mb', 0),
                'cleaned_space': advanced_memory_mb
            }
        }
        
        return advanced_dashboard

    # ─── 14. Full Pipeline Run ─────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        """
        Run the full ML training pipeline:
        1. Detect target & problem type
        2. Prepare features
        3. Train all models with CV + tuning
        4. Select best model
        5. Get feature importance
        6. Build metrics dashboard
        """
        self.prepare_features()
        self.train_all_models()
        dashboard = self.get_metrics_dashboard()
        return dashboard
