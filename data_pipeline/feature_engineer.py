"""
Feature Engineering Module
==========================
Handles feature transformation, encoding, scaling, and creation.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, 
    OneHotEncoder, PolynomialFeatures
)
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering and transformation toolkit."""
    
    def __init__(
        self, 
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        problem_type: Optional[str] = None
    ):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str, optional
            Target column name
        problem_type : str, optional
            'classification', 'regression', or None
        """
        self.df = df.copy()
        self.target_col = target_col
        self.problem_type = problem_type
        self.transformations: List[str] = []
        self.encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}
    
    def encode_categorical(
        self,
        method: str = 'auto',
        columns: Optional[List[str]] = None,
        max_categories: int = 10,
        drop_first: bool = True
    ) -> 'FeatureEngineer':
        """
        Encode categorical variables.
        
        Parameters:
        -----------
        method : str
            'auto', 'onehot', 'label', or 'ordinal'
        columns : list, optional
            Specific columns to encode
        max_categories : int
            Max categories for one-hot encoding
        drop_first : bool
            Drop first category in one-hot encoding
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Exclude target column
            if self.target_col in columns:
                columns.remove(self.target_col)
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            n_unique = self.df[col].nunique()
            
            # Determine encoding method
            if method == 'auto':
                if n_unique == 2:
                    use_method = 'label'
                elif n_unique <= max_categories:
                    use_method = 'onehot'
                else:
                    use_method = 'label'
            else:
                use_method = method
            
            if use_method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(
                    self.df[col], 
                    prefix=col,
                    drop_first=drop_first,
                    dtype=int
                )
                self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                self.transformations.append(f"One-hot encoded '{col}' → {len(dummies.columns)} columns")
                
            elif use_method == 'label':
                # Label encoding
                le = LabelEncoder()
                # Handle NaN values
                mask = self.df[col].notna()
                self.df.loc[mask, col] = le.fit_transform(self.df.loc[mask, col].astype(str))
                self.df[col] = self.df[col].astype(float)
                self.encoders[col] = le
                self.transformations.append(f"Label encoded '{col}'")
        
        return self
    
    def scale_features(
        self,
        method: str = 'standard',
        columns: Optional[List[str]] = None
    ) -> 'FeatureEngineer':
        """
        Scale numeric features.
        
        Parameters:
        -----------
        method : str
            'standard' (z-score) or 'minmax' (0-1 range)
        columns : list, optional
            Specific columns to scale
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude target column
            if self.target_col in columns:
                columns.remove(self.target_col)
        
        if len(columns) == 0:
            return self
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Handle missing values for scaling
        cols_to_scale = [c for c in columns if c in self.df.columns]
        
        if cols_to_scale:
            self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
            self.scalers['main'] = scaler
            self.transformations.append(f"{method.capitalize()} scaled {len(cols_to_scale)} numeric features")
        
        return self
    
    def create_datetime_features(
        self,
        columns: Optional[List[str]] = None,
        features: List[str] = None
    ) -> 'FeatureEngineer':
        """
        Extract features from datetime columns.
        
        Parameters:
        -----------
        columns : list, optional
            Datetime columns to process
        features : list
            Features to extract: 'year', 'month', 'day', 'dayofweek', 
            'hour', 'minute', 'quarter', 'is_weekend'
        """
        if features is None:
            features = ['year', 'month', 'day', 'dayofweek', 'is_weekend']
        
        if columns is None:
            columns = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            dt = self.df[col]
            
            if 'year' in features:
                self.df[f'{col}_year'] = dt.dt.year
            if 'month' in features:
                self.df[f'{col}_month'] = dt.dt.month
            if 'day' in features:
                self.df[f'{col}_day'] = dt.dt.day
            if 'dayofweek' in features:
                self.df[f'{col}_dayofweek'] = dt.dt.dayofweek
            if 'quarter' in features:
                self.df[f'{col}_quarter'] = dt.dt.quarter
            if 'hour' in features and hasattr(dt.dt, 'hour'):
                self.df[f'{col}_hour'] = dt.dt.hour
            if 'minute' in features and hasattr(dt.dt, 'minute'):
                self.df[f'{col}_minute'] = dt.dt.minute
            if 'is_weekend' in features:
                self.df[f'{col}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
            
            # Drop original datetime column
            self.df = self.df.drop(columns=[col])
            self.transformations.append(f"Extracted {len(features)} features from '{col}'")
        
        return self
    
    def create_polynomial_features(
        self,
        columns: Optional[List[str]] = None,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False
    ) -> 'FeatureEngineer':
        """
        Create polynomial and interaction features.
        
        Parameters:
        -----------
        columns : list, optional
            Columns to use for polynomial features
        degree : int
            Polynomial degree
        interaction_only : bool
            If True, only interaction features
        include_bias : bool
            Include bias column
        """
        if columns is None:
            # Use top numeric columns (limit to avoid explosion)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_col in numeric_cols:
                numeric_cols.remove(self.target_col)
            columns = numeric_cols[:5]  # Limit to 5 columns
        
        if len(columns) < 2:
            return self
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        # Create polynomial features
        poly_data = poly.fit_transform(self.df[columns])
        poly_features = poly.get_feature_names_out(columns)
        
        # Add new features (excluding original columns)
        new_features = poly_features[len(columns):]
        new_data = poly_data[:, len(columns):]
        
        for i, feat_name in enumerate(new_features):
            self.df[feat_name] = new_data[:, i]
        
        self.transformations.append(f"Created {len(new_features)} polynomial features (degree={degree})")
        
        return self
    
    def create_binned_features(
        self,
        columns: Optional[List[str]] = None,
        n_bins: int = 5,
        strategy: str = 'quantile'
    ) -> 'FeatureEngineer':
        """
        Create binned versions of numeric features.
        
        Parameters:
        -----------
        columns : list, optional
            Columns to bin
        n_bins : int
            Number of bins
        strategy : str
            'quantile' or 'uniform'
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()[:5]
            if self.target_col in columns:
                columns.remove(self.target_col)
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if strategy == 'quantile':
                self.df[f'{col}_binned'] = pd.qcut(
                    self.df[col], q=n_bins, labels=False, duplicates='drop'
                )
            else:
                self.df[f'{col}_binned'] = pd.cut(
                    self.df[col], bins=n_bins, labels=False
                )
            
        self.transformations.append(f"Created binned features for {len(columns)} columns")
        
        return self
    
    def compute_feature_importance(
        self,
        n_features: int = 20
    ) -> Dict[str, float]:
        """
        Compute feature importance using mutual information.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to return
        """
        if self.target_col is None or self.target_col not in self.df.columns:
            return {}
        
        # Get numeric features
        feature_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns
                       if c != self.target_col]
        
        if len(feature_cols) == 0:
            return {}
        
        X = self.df[feature_cols].fillna(0)
        y = self.df[self.target_col]
        
        # Compute mutual information
        if self.problem_type == 'classification' or y.dtype == 'object':
            mi = mutual_info_classif(X, y, random_state=42)
        else:
            mi = mutual_info_regression(X, y, random_state=42)
        
        # Create importance dictionary
        importance = dict(zip(feature_cols, mi))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        # Keep top n
        self.feature_importance = dict(list(importance.items())[:n_features])
        
        return self.feature_importance
    
    def drop_low_importance_features(
        self,
        threshold: float = 0.01,
        keep_n: Optional[int] = None
    ) -> 'FeatureEngineer':
        """
        Drop features with low importance.
        
        Parameters:
        -----------
        threshold : float
            Minimum importance threshold
        keep_n : int, optional
            Keep top n features regardless of threshold
        """
        if not self.feature_importance:
            self.compute_feature_importance()
        
        if not self.feature_importance:
            return self
        
        # Determine features to keep
        if keep_n:
            features_to_keep = list(self.feature_importance.keys())[:keep_n]
        else:
            features_to_keep = [f for f, imp in self.feature_importance.items() 
                               if imp >= threshold]
        
        # Add target column
        if self.target_col:
            features_to_keep.append(self.target_col)
        
        # Get current columns
        cols_to_drop = [c for c in self.df.columns if c not in features_to_keep]
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.transformations.append(f"Dropped {len(cols_to_drop)} low-importance features")
        
        return self
    
    def drop_low_variance_features(
        self,
        threshold: float = 0.01
    ) -> 'FeatureEngineer':
        """
        Drop features with low variance.
        
        Parameters:
        -----------
        threshold : float
            Variance threshold
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        
        if len(numeric_cols) == 0:
            return self
        
        selector = VarianceThreshold(threshold=threshold)
        
        try:
            selector.fit(self.df[numeric_cols])
            mask = selector.get_support()
            cols_to_drop = [c for c, keep in zip(numeric_cols, mask) if not keep]
            
            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
                self.transformations.append(f"Dropped {len(cols_to_drop)} low-variance features")
        except:
            pass
        
        return self
    
    def drop_highly_correlated(
        self,
        threshold: float = 0.95
    ) -> 'FeatureEngineer':
        """
        Drop one of each pair of highly correlated features.
        
        Parameters:
        -----------
        threshold : float
            Correlation threshold
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        
        if len(numeric_cols) < 2:
            return self
        
        corr_matrix = self.df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        cols_to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.transformations.append(f"Dropped {len(cols_to_drop)} highly correlated features (r > {threshold})")
        
        return self
    
    def handle_class_imbalance(
        self,
        method: str = 'smote',
        sampling_strategy: Union[str, float] = 'auto'
    ) -> 'FeatureEngineer':
        """
        Handle class imbalance for classification problems.
        
        Parameters:
        -----------
        method : str
            'smote', 'oversample', or 'undersample'
        sampling_strategy : str or float
            Sampling strategy
        """
        if self.target_col is None or self.problem_type != 'classification':
            return self
        
        try:
            if method == 'smote':
                from imblearn.over_sampling import SMOTE
                
                X = self.df.drop(columns=[self.target_col])
                y = self.df[self.target_col]
                
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                self.df = pd.concat([X_resampled, y_resampled], axis=1)
                self.transformations.append(f"Applied SMOTE: {len(y)} → {len(y_resampled)} samples")
                
            elif method == 'oversample':
                # Simple oversampling
                max_count = self.df[self.target_col].value_counts().max()
                dfs = []
                for val in self.df[self.target_col].unique():
                    df_class = self.df[self.df[self.target_col] == val]
                    df_upsampled = df_class.sample(max_count, replace=True, random_state=42)
                    dfs.append(df_upsampled)
                self.df = pd.concat(dfs).reset_index(drop=True)
                self.transformations.append(f"Oversampled minority classes")
                
            elif method == 'undersample':
                # Simple undersampling
                min_count = self.df[self.target_col].value_counts().min()
                dfs = []
                for val in self.df[self.target_col].unique():
                    df_class = self.df[self.df[self.target_col] == val]
                    df_downsampled = df_class.sample(min_count, random_state=42)
                    dfs.append(df_downsampled)
                self.df = pd.concat(dfs).reset_index(drop=True)
                self.transformations.append(f"Undersampled majority classes")
                
        except ImportError:
            self.transformations.append("SMOTE not available (install imbalanced-learn)")
        except Exception as e:
            self.transformations.append(f"Could not handle imbalance: {e}")
        
        except Exception as e:
            self.transformations.append(f"Could not handle imbalance: {e}")
        
        return self

    def auto_evolve(self, max_new_features: int = 10) -> 'FeatureEngineer':
        """
        Automatically generate and select best interaction features ("Super Features").
        """
        if self.target_col is None:
            return self

        # 1. Identify top numeric features
        if not self.feature_importance:
            self.compute_feature_importance()
        
        # Filter for numeric only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        # Get top features from importance
        top_features = [f for f in self.feature_importance.keys() if f in numeric_cols][:5]
        
        if len(top_features) < 2:
            return self
            
        initial_features = set(self.df.columns)

        # 2. Generate interactions
        # We use PolynomialFeatures with interaction_only=True to get A*B terms
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        try:
            poly_data = poly.fit_transform(self.df[top_features])
            new_feature_names = poly.get_feature_names_out(top_features)
            
            # Add new features to df
            # Skip the first len(top_features) as they are the original ones
            new_feats = new_feature_names[len(top_features):]
            new_data = poly_data[:, len(top_features):]
            
            created_features = []
            for i, feat_name in enumerate(new_feats):
                # Clean name (replace spaces with _)
                clean_name = feat_name.replace(' ', '_x_')
                self.df[clean_name] = new_data[:, i]
                created_features.append(clean_name)
                
            # 3. Re-evaluate importance
            self.compute_feature_importance()
            
            # 4. Filter: Keep only features that have decent importance
            # Threshold: median importance of original features? or just > 0.01?
            # Let's simple keep top N new features
            new_feat_importance = {f: self.feature_importance.get(f, 0) for f in created_features}
            
            # Sort by importance
            sorted_new = sorted(new_feat_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Keep top max_new_features
            keep_features = [f for f, imp in sorted_new[:max_new_features] if imp > 0.001]
            drop_features = [f for f in created_features if f not in keep_features]
            
            if drop_features:
                self.df = self.df.drop(columns=drop_features)
            
            if keep_features:
                self.transformations.append(f"Auto-evolved {len(keep_features)} new interaction features: {', '.join(keep_features[:3])}...")
        
        except Exception as e:
            self.transformations.append(f"Auto-evolution failed: {e}")
            
        return self
    
    def get_transformed_data(self) -> pd.DataFrame:
        """Return the transformed DataFrame."""
        return self.df
    
    def get_summary(self) -> Dict[str, Any]:
        """Return summary of all transformations."""
        return {
            "final_shape": self.df.shape,
            "transformations": self.transformations,
            "encoders": list(self.encoders.keys()),
            "feature_importance": self.feature_importance
        }
    
    def print_summary(self) -> None:
        """Print transformation summary."""
        summary = self.get_summary()
        
        print("=" * 60)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 60)
        print(f"\n📊 Final shape: {summary['final_shape']}")
        
        print(f"\n🔧 Transformations applied:")
        for t in summary['transformations']:
            print(f"   • {t}")
        
        if summary['feature_importance']:
            print(f"\n📈 Top Feature Importances:")
            for feat, imp in list(summary['feature_importance'].items())[:10]:
                print(f"   • {feat}: {imp:.4f}")
        
        print("\n" + "=" * 60)
