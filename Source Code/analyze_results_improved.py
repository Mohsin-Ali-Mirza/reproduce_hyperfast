"""
Comprehensive results analyzer with statistical tests and visualizations.

This module provides advanced analysis of benchmark results including
rankings, statistical comparisons, and detailed visualizations.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import ExperimentConfig

# Configure plotting
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """
    Custom encoder for numpy data types.
    Fixes the "Object of type int64 is not JSON serializable" error.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class ResultsAnalyzer:
    """Analyzes and visualizes benchmark results."""
    
    def __init__(self, results_path: Path, config: Optional[ExperimentConfig] = None):
        self.results_path = Path(results_path)
        self.config = config or ExperimentConfig()
        
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        self.df = pd.DataFrame(results)
        
        # Filter out failed experiments
        if 'error' in self.df.columns:
            n_errors = self.df['error'].notna().sum()
            if n_errors > 0:
                logger.warning(f"Filtering {n_errors} failed experiments")
                self.df = self.df[self.df['error'].isna()].copy()
        
        logger.info(f"Loaded {len(self.df)} successful experiments")
        
        # Compute rankings
        self.rankings_df = self._compute_rankings()
    
    def _compute_rankings(self) -> pd.DataFrame:
        """Compute rankings for each dataset-budget combination."""
        rankings = []
        
        for dataset in self.df['dataset'].unique():
            for budget in self.df['budget'].unique():
                # Get results for this combination
                subset = self.df[
                    (self.df['dataset'] == dataset) & 
                    (self.df['budget'] == budget)
                ].copy()
                
                if len(subset) == 0:
                    continue
                
                # Rank by balanced accuracy (higher is better)
                subset = subset.sort_values('balanced_accuracy', ascending=False)
                subset['rank'] = range(1, len(subset) + 1)
                
                for _, row in subset.iterrows():
                    rankings.append({
                        'dataset': dataset,
                        'budget': budget,
                        'model': row['model'],
                        'rank': row['rank'],
                        'balanced_accuracy': row['balanced_accuracy'],
                        'total_time': row['total_time']
                    })
        
        return pd.DataFrame(rankings)
    
    def generate_summary_statistics(self) -> Dict:
        """Generate comprehensive summary statistics."""
        summary = {
            'overall_performance': self._overall_performance(),
            'performance_by_budget': self._performance_by_budget(),
            'average_ranks': self._average_ranks(),
            'runtime_statistics': self._runtime_statistics(),
            'budget_compliance': self._budget_compliance(),
            'statistical_tests': self._statistical_tests()
        }
        
        return summary
    
    def _overall_performance(self) -> pd.DataFrame:
        """Compute overall performance across all budgets."""
        stats = self.df.groupby('model').agg({
            'balanced_accuracy': ['mean', 'std', 'min', 'max', 'count'],
            'accuracy': ['mean', 'std'],
            'f1_score': ['mean', 'std']
        }).round(4)
        
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        return stats.sort_values('balanced_accuracy_mean', ascending=False)
    
    def _performance_by_budget(self) -> pd.DataFrame:
        """Compute performance by budget."""
        pivot = self.df.pivot_table(
            values='balanced_accuracy',
            index='budget',
            columns='model',
            aggfunc='mean'
        ).round(4)
        
        return pivot
    
    def _average_ranks(self) -> Dict[str, pd.DataFrame]:
        """Compute average ranks overall and by budget."""
        # Overall average rank
        overall = self.rankings_df.groupby('model')['rank'].agg([
            'mean', 'std', 'min', 'max'
        ]).sort_values('mean').round(2)
        
        # Average rank by budget
        by_budget = self.rankings_df.pivot_table(
            values='rank',
            index='budget',
            columns='model',
            aggfunc='mean'
        ).round(2)
        
        return {
            'overall': overall,
            'by_budget': by_budget
        }
    
    def _runtime_statistics(self) -> pd.DataFrame:
        """Compute runtime statistics."""
        stats = self.df.groupby('model').agg({
            'total_time': ['mean', 'median', 'std', 'min', 'max'],
            'fit_time': ['mean', 'median'],
            'predict_time': ['mean', 'median']
        }).round(3)
        
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        return stats.sort_values('total_time_mean')
    
    def _budget_compliance(self) -> pd.DataFrame:
        """Compute budget compliance statistics."""
        compliance = self.df.groupby('model').agg({
            'budget_exceeded': ['sum', 'count']
        })
        
        compliance['exceeded_pct'] = (
            compliance[('budget_exceeded', 'sum')] / 
            compliance[('budget_exceeded', 'count')] * 100
        ).round(1)
        
        compliance.columns = ['exceeded_count', 'total_runs', 'exceeded_pct']
        return compliance.sort_values('exceeded_pct')
    
    def _statistical_tests(self) -> Dict:
        """Perform statistical significance tests."""
        tests = {}
        
        # Friedman test (non-parametric repeated measures)
        models = self.df['model'].unique()
        if len(models) > 2:
            # Prepare data for Friedman test
            pivot = self.df.pivot_table(
                values='balanced_accuracy',
                index=['dataset', 'budget'],
                columns='model',
                aggfunc='mean'
            )
            
            if not pivot.empty and len(pivot.columns) > 2:
                # Remove rows with missing values
                pivot_clean = pivot.dropna()
                
                if len(pivot_clean) > 0:
                    try:
                        statistic, pvalue = stats.friedmanchisquare(
                            *[pivot_clean[col].values for col in pivot_clean.columns]
                        )
                        tests['friedman'] = {
                            'statistic': float(statistic),
                            'pvalue': float(pvalue),
                            'significant': pvalue < 0.05
                        }
                    except Exception as e:
                        logger.warning(f"Could not perform Friedman test: {e}")
        
        return tests
    
    def create_visualizations(self, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Create all visualizations.
        
        Args:
            output_dir: Directory to save figures (default: config.paths.figures_dir)
            
        Returns:
            List of paths to saved figures
        """
        if output_dir is None:
            output_dir = self.config.paths.figures_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        saved_figures = []
        
        # 1. Runtime vs Performance (Figure 2a style)
        fig_path = self._plot_runtime_vs_performance(output_dir)
        saved_figures.append(fig_path)
        
        # 2. Average Rank by Budget (Figure 2b style)
        fig_path = self._plot_rank_by_budget(output_dir)
        saved_figures.append(fig_path)
        
        # 3. Combined Figure 2
        fig_path = self._plot_figure2_combined(output_dir)
        saved_figures.append(fig_path)
        
        # 4. Detailed performance heatmap
        fig_path = self._plot_performance_heatmap(output_dir)
        saved_figures.append(fig_path)
        
        # 5. Budget compliance
        fig_path = self._plot_budget_compliance(output_dir)
        saved_figures.append(fig_path)
        
        logger.info(f"Created {len(saved_figures)} visualizations")
        
        return saved_figures
    
    def _plot_runtime_vs_performance(self, output_dir: Path) -> Path:
        """Create runtime vs performance plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Aggregate data
        agg_data = self.df.groupby(['model', 'budget']).agg({
            'balanced_accuracy': 'mean',
            'total_time': 'mean'
        }).reset_index()
        
        # Plot each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.df['model'].unique())))
        for idx, model in enumerate(sorted(self.df['model'].unique())):
            model_data = agg_data[agg_data['model'] == model]
            ax.plot(
                model_data['total_time'],
                model_data['balanced_accuracy'],
                marker='o',
                label=model,
                linewidth=2.5,
                markersize=10,
                color=colors[idx],
                alpha=0.8
            )
        
        ax.set_xlabel('Runtime (seconds, log scale)', fontsize=13)
        ax.set_ylabel('Balanced Accuracy', fontsize=13)
        ax.set_xscale('log')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_title(
            'Runtime vs Performance on Mini-Test Datasets',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        
        # Add reference lines for key budgets
        for budget in [1, 10, 60, 300]:
            ax.axvline(x=budget, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        filepath = output_dir / 'runtime_vs_performance.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filepath}")
        return filepath
    
    def _plot_rank_by_budget(self, output_dir: Path) -> Path:
        """Create average rank by budget plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get rank data
        avg_ranks_by_budget = self.rankings_df.pivot_table(
            values='rank',
            index='budget',
            columns='model',
            aggfunc='mean'
        )
        
        # Budget labels
        budget_labels = [f"{int(b)}s" if b < 60 else f"{int(b/60)}m" 
                        for b in avg_ranks_by_budget.index]
        x_pos = np.arange(len(avg_ranks_by_budget))
        
        # Plot each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.df['model'].unique())))
        for idx, model in enumerate(sorted(self.df['model'].unique())):
            if model in avg_ranks_by_budget.columns:
                ax.plot(
                    x_pos,
                    avg_ranks_by_budget[model],
                    marker='o',
                    label=model,
                    linewidth=2.5,
                    markersize=10,
                    color=colors[idx],
                    alpha=0.8
                )
        
        ax.set_xlabel('Runtime Budget (fit + predict)', fontsize=13)
        ax.set_ylabel('Avg. Accuracy Rank', fontsize=13)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(budget_labels)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Lower rank is better
        ax.set_title(
            'Average Rank by Budget',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        
        filepath = output_dir / 'rank_by_budget.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filepath}")
        return filepath
    
    def _plot_figure2_combined(self, output_dir: Path) -> Path:
        """Create combined Figure 2 (matching paper style)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Runtime vs Performance
        agg_data = self.df.groupby(['model', 'budget']).agg({
            'balanced_accuracy': 'mean',
            'total_time': 'mean'
        }).reset_index()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.df['model'].unique())))
        for idx, model in enumerate(sorted(self.df['model'].unique())):
            model_data = agg_data[agg_data['model'] == model]
            ax1.plot(
                model_data['total_time'],
                model_data['balanced_accuracy'],
                marker='o',
                label=model,
                linewidth=2,
                markersize=8,
                color=colors[idx]
            )
        
        ax1.set_xlabel('Runtime (fit + predict) (s)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', fontsize=12, fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Runtime vs Performance', fontsize=13, fontweight='bold')
        
        # Right: Average Rank by Budget
        avg_ranks_by_budget = self.rankings_df.pivot_table(
            values='rank',
            index='budget',
            columns='model',
            aggfunc='mean'
        )
        
        budget_labels = [f"{int(b)}s" if b < 60 else f"{int(b/60)}m"
                        for b in avg_ranks_by_budget.index]
        x_pos = np.arange(len(avg_ranks_by_budget))
        
        for idx, model in enumerate(sorted(self.df['model'].unique())):
            if model in avg_ranks_by_budget.columns:
                ax2.plot(
                    x_pos,
                    avg_ranks_by_budget[model],
                    marker='o',
                    label=model,
                    linewidth=2,
                    markersize=8,
                    color=colors[idx]
                )
        
        ax2.set_xlabel('Runtime Budget (fit + predict)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Avg. Acc. Rank', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(budget_labels)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        ax2.set_title('Average Rank by Budget', fontsize=13, fontweight='bold')
        
        filepath = output_dir / 'figure2_reproduction.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filepath}")
        return filepath
    
    def _plot_performance_heatmap(self, output_dir: Path) -> Path:
        """Create performance heatmap."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot = self.df.pivot_table(
            values='balanced_accuracy',
            index='model',
            columns='budget',
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=pivot.values.mean(),
            ax=ax,
            cbar_kws={'label': 'Balanced Accuracy'}
        )
        
        ax.set_xlabel('Time Budget (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title(
            'Performance Heatmap by Model and Budget',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        
        filepath = output_dir / 'performance_heatmap.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filepath}")
        return filepath
    
    def _plot_budget_compliance(self, output_dir: Path) -> Path:
        """Create budget compliance visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        compliance = self._budget_compliance()
        
        x = np.arange(len(compliance))
        ax.bar(x, compliance['exceeded_pct'], alpha=0.7, color='coral')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Budget Exceeded (%)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(compliance.index, rotation=45, ha='right')
        ax.set_title(
            'Budget Compliance by Model',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for i, v in enumerate(compliance['exceeded_pct']):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        filepath = output_dir / 'budget_compliance.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filepath}")
        return filepath
    
    def save_analysis(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save all analysis results.
        
        Returns:
            Dictionary mapping output type to file path
        """
        if output_dir is None:
            output_dir = self.config.paths.results_dir
        
        output_dir = Path(output_dir)
        saved_files = {}
        
        # Generate summary
        summary = self.generate_summary_statistics()
        
        # Save summary JSON
        summary_path = output_dir / 'analysis_summary.json'
        with open(summary_path, 'w') as f:
            # Convert DataFrames to dicts for JSON serialization
            json_summary = {}
            for key, value in summary.items():
                if isinstance(value, pd.DataFrame):
                    json_summary[key] = value.to_dict()
                elif isinstance(value, dict) and any(isinstance(v, pd.DataFrame) for v in value.values()):
                    json_summary[key] = {k: v.to_dict() if isinstance(v, pd.DataFrame) else v 
                                        for k, v in value.items()}
                else:
                    json_summary[key] = value
            
            # Use the custom NumpyEncoder here
            json.dump(json_summary, f, indent=2, cls=NumpyEncoder)
            
        saved_files['summary_json'] = summary_path
        
        # Save rankings CSV
        rankings_path = output_dir / 'rankings.csv'
        self.rankings_df.to_csv(rankings_path, index=False)
        saved_files['rankings_csv'] = rankings_path
        
        # Save detailed stats CSVs
        for name, df in [
            ('overall_performance', summary['overall_performance']),
            ('performance_by_budget', summary['performance_by_budget']),
            ('runtime_statistics', summary['runtime_statistics']),
            ('budget_compliance', summary['budget_compliance'])
        ]:
            csv_path = output_dir / f'{name}.csv'
            df.to_csv(csv_path)
            saved_files[f'{name}_csv'] = csv_path
        
        logger.info(f"Saved {len(saved_files)} analysis files")
        
        return saved_files
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.generate_summary_statistics()
        
        print("\n" + "="*70)
        print("BENCHMARK ANALYSIS SUMMARY")
        print("="*70)
        
        print("\n1. Overall Performance (Balanced Accuracy)")
        print("-"*70)
        print(summary['overall_performance']['balanced_accuracy_mean'].to_string())
        
        print("\n2. Average Ranks (Overall)")
        print("-"*70)
        print(summary['average_ranks']['overall']['mean'].to_string())
        
        print("\n3. Runtime Statistics (seconds)")
        print("-"*70)
        print(summary['runtime_statistics'][['total_time_mean', 'total_time_median']].to_string())
        
        print("\n4. Budget Compliance")
        print("-"*70)
        print(summary['budget_compliance'].to_string())
        
        if 'statistical_tests' in summary and summary['statistical_tests']:
            print("\n5. Statistical Tests")
            print("-"*70)
            if 'friedman' in summary['statistical_tests']:
                friedman = summary['statistical_tests']['friedman']
                print(f"Friedman Test: χ² = {friedman['statistic']:.4f}, "
                      f"p = {friedman['pvalue']:.4f}, "
                      f"Significant: {friedman['significant']}")
        
        print("\n" + "="*70)


def main():
    """Main entry point for results analysis."""
    config = ExperimentConfig()
    results_path = config.paths.results_dir / 'benchmark_results.json'
    
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return
    
    # Create analyzer
    analyzer = ResultsAnalyzer(results_path, config)
    
    # Print summary
    analyzer.print_summary()
    
    # Save analysis
    saved_files = analyzer.save_analysis()
    print(f"\n✓ Analysis files saved: {len(saved_files)}")
    
    # Create visualizations
    figures = analyzer.create_visualizations()
    print(f"✓ Figures created: {len(figures)}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()