"""
Exploratory Data Analysis (EDA) Module
======================================
Performs comprehensive EDA with visualizations and insights.
Memory-optimized: uses only matplotlib (no seaborn/scipy).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import warnings
import gc

warnings.filterwarnings('ignore')


class EDAAnalyzer:
    """Perform exploratory data analysis on datasets."""
    
    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df = df.copy()
        self.target_col = target_col
        self.insights: List[str] = []
        self.stats: Dict[str, Any] = {}
    
    def summary_statistics(self) -> pd.DataFrame:
        """Generate comprehensive summary statistics."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return pd.DataFrame()
        
        stats = numeric_df.describe().T
        stats['missing'] = self.df[numeric_df.columns].isnull().sum()
        stats['missing_pct'] = (stats['missing'] / len(self.df) * 100).round(2)
        stats['skewness'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurtosis()
        
        self.stats['numeric_summary'] = stats
        
        skewed = stats[stats['skewness'].abs() > 1].index.tolist()
        if skewed:
            self.insights.append(f"Highly skewed features detected: {skewed}")
        
        return stats
    
    def categorical_summary(self) -> Dict[str, pd.DataFrame]:
        """Summarize categorical columns."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        summaries = {}
        
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            pct = (value_counts / len(self.df) * 100).round(2)
            
            summary = pd.DataFrame({
                'count': value_counts,
                'percentage': pct
            })
            summaries[col] = summary
            
            if len(value_counts) > 50:
                self.insights.append(f"High cardinality in '{col}': {len(value_counts)} unique values")
        
        self.stats['categorical_summary'] = summaries
        return summaries
    
    def correlation_analysis(self, method: str = 'pearson', threshold: float = 0.7) -> pd.DataFrame:
        """Compute correlation matrix and identify highly correlated features."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return pd.DataFrame()
        
        corr_matrix = numeric_df.corr(method=method)
        self.stats['correlation_matrix'] = corr_matrix
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': round(corr_matrix.iloc[i, j], 3)
                    })
        
        if high_corr_pairs:
            self.stats['high_correlations'] = high_corr_pairs
            self.insights.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > {threshold})")
        
        return corr_matrix
    
    def distribution_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Analyze distributions of numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        distributions = {}
        
        for col in numeric_cols:
            series = self.df[col].dropna()
            distributions[col] = {
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'q1': series.quantile(0.25),
                'q3': series.quantile(0.75),
                'iqr': series.quantile(0.75) - series.quantile(0.25),
                'range': series.max() - series.min()
            }
        
        self.stats['distributions'] = distributions
        return distributions
    
    def target_analysis(self) -> Dict[str, Any]:
        """Analyze target variable if specified."""
        if self.target_col is None or self.target_col not in self.df.columns:
            return {}
        
        target = self.df[self.target_col]
        analysis = {'column': self.target_col}
        
        if target.dtype in [np.number, 'int64', 'float64']:
            analysis['type'] = 'continuous'
            analysis['stats'] = {
                'mean': target.mean(),
                'median': target.median(),
                'std': target.std(),
                'min': target.min(),
                'max': target.max()
            }
        else:
            analysis['type'] = 'categorical'
            value_counts = target.value_counts()
            analysis['class_distribution'] = value_counts.to_dict()
            analysis['class_balance'] = round(value_counts.min() / value_counts.max(), 3)
            
            if analysis['class_balance'] < 0.5:
                self.insights.append(f"Class imbalance detected in target: ratio = {analysis['class_balance']}")
        
        self.stats['target_analysis'] = analysis
        return analysis
    
    def plot_distributions(self, columns=None, figsize=(15, 10), save_path=None):
        """Plot distribution histograms for numeric columns."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) == 0:
            return
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes).flatten()
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            data = self.df[col].dropna()
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='#4C72B0')
            ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
            ax.axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.2f}')
            ax.set_title(f'{col}\n(skew: {data.skew():.2f})')
            ax.legend(fontsize=8)
        
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.close(fig)
        gc.collect()
    
    def plot_boxplots(self, columns=None, figsize=(15, 10), save_path=None):
        """Plot boxplots for numeric columns to visualize outliers."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) == 0:
            return
        
        n_cols = min(4, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes).flatten()
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            data = self.df[col].dropna()
            bp = ax.boxplot(data, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            ax.set_title(col)
            ax.set_xticks([])
        
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Boxplots (Outlier Detection)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.close(fig)
        gc.collect()
    
    def plot_correlation_heatmap(self, figsize=(12, 10), save_path=None):
        """Plot correlation heatmap using pure matplotlib (no seaborn)."""
        if 'correlation_matrix' not in self.stats:
            self.correlation_analysis()
        
        corr_matrix = self.stats.get('correlation_matrix')
        
        if corr_matrix is None or corr_matrix.empty:
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Pure matplotlib heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        masked_corr = np.ma.masked_where(mask, corr_matrix.values)
        
        cax = ax.imshow(masked_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        fig.colorbar(cax, ax=ax, shrink=0.8)
        
        # Labels
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(corr_matrix.columns, fontsize=8)
        
        # Annotate if small enough
        if len(corr_matrix.columns) <= 15:
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    if not mask[i, j]:
                        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                ha='center', va='center', fontsize=7)
        
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.close(fig)
        gc.collect()
    
    def plot_categorical(self, columns=None, figsize=(15, 10), save_path=None):
        """Plot count plots for categorical columns."""
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(columns) == 0:
            return
        
        columns = [c for c in columns if self.df[c].nunique() <= 20]
        
        if len(columns) == 0:
            return
        
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes).flatten()
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            value_counts = self.df[col].value_counts()
            ax.bar(range(len(value_counts)), value_counts.values, color='#4C72B0')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_title(f'{col} ({len(value_counts)} categories)')
            ax.set_ylabel('Count')
        
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Categorical Feature Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.close(fig)
        gc.collect()
    
    def plot_target_relationships(self, figsize=(15, 10), save_path=None):
        """Plot relationships between features and target."""
        if self.target_col is None:
            return
        
        numeric_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns 
                       if c != self.target_col][:6]
        
        if len(numeric_cols) == 0:
            return
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes).flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            ax.scatter(self.df[col], self.df[self.target_col], alpha=0.5, s=10, color='#4C72B0')
            ax.set_xlabel(col)
            ax.set_ylabel(self.target_col)
            ax.set_title(f'{col} vs {self.target_col}')
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature-Target Relationships', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.close(fig)
        gc.collect()
    
    def run_full_analysis(self, show_plots=True, save_plots=False, output_dir=None):
        """Run complete EDA pipeline."""
        print("=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        print("\n📊 Summary Statistics:")
        stats = self.summary_statistics()
        if not stats.empty:
            print(stats.round(2).to_string())
        
        print("\n📋 Categorical Columns:")
        cat_summary = self.categorical_summary()
        for col, summary in cat_summary.items():
            print(f"\n{col}:")
            print(summary.head(5).to_string())
        
        print("\n🔗 Correlation Analysis:")
        corr = self.correlation_analysis()
        if 'high_correlations' in self.stats:
            for pair in self.stats['high_correlations'][:10]:
                print(f"   {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']}")
        
        if self.target_col:
            print(f"\n🎯 Target Variable Analysis ({self.target_col}):")
            target_analysis = self.target_analysis()
            if target_analysis:
                if target_analysis['type'] == 'continuous':
                    for k, v in target_analysis['stats'].items():
                        print(f"   {k}: {v:.3f}")
                else:
                    for cls, count in target_analysis['class_distribution'].items():
                        print(f"   {cls}: {count}")
        
        if self.insights:
            print("\n💡 Key Insights:")
            for insight in self.insights:
                print(f"   ⚠️  {insight}")
        
        if show_plots:
            save_path = lambda name: f"{output_dir}/{name}.png" if save_plots and output_dir else None
            
            print("\n📈 Generating visualizations...")
            self.plot_distributions(save_path=save_path('distributions'))
            self.plot_boxplots(save_path=save_path('boxplots'))
            self.plot_correlation_heatmap(save_path=save_path('correlation'))
            self.plot_categorical(save_path=save_path('categorical'))
            
            if self.target_col:
                self.plot_target_relationships(save_path=save_path('target_relationships'))
        
        print("\n" + "=" * 60)
        
        gc.collect()
        
        return {
            'stats': self.stats,
            'insights': self.insights
        }
    
    def get_insights(self) -> List[str]:
        """Return list of insights discovered during analysis."""
        return self.insights
