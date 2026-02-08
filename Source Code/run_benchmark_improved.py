"""
Improved benchmark runner with progress tracking and error handling.

This module runs the mini-test benchmark with better organization,
error handling, and progress feedback.
"""
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from tqdm import tqdm

from time_budget_models_improved import ModelFactory, get_available_models
from config import ExperimentConfig

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Container for a single experiment result."""
    dataset: str
    model: str
    budget: float
    balanced_accuracy: float
    accuracy: float
    f1_score: float
    fit_time: float
    predict_time: float
    total_time: float
    budget_exceeded: bool
    n_train: int
    n_test: int
    n_features: int
    n_classes: int
    timestamp: str
    error: Optional[str] = None


class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.scalers = {}
        self.label_encoders = {}
        self.imputers = {}
    
    def load_and_preprocess(
        self,
        dataset_name: str,
        cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess a dataset.
        
        Args:
            dataset_name: Name of the dataset
            cache: Whether to cache preprocessing objects
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load data
        X_train = pd.read_csv(self.data_dir / f'{dataset_name}_X_train.csv')
        X_test = pd.read_csv(self.data_dir / f'{dataset_name}_X_test.csv')
        y_train = pd.read_csv(self.data_dir / f'{dataset_name}_y_train.csv').values.ravel()
        y_test = pd.read_csv(self.data_dir / f'{dataset_name}_y_test.csv').values.ravel()
        
        # Handle categorical features
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
            X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)
            
            # Align columns
            X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
        
        # Convert to numpy arrays
        X_train = X_train.values.astype(np.float64)
        X_test = X_test.values.astype(np.float64)
        
        # Impute missing values
        if cache and dataset_name in self.imputers:
            imputer = self.imputers[dataset_name]
            X_train = imputer.transform(X_train)
            X_test = imputer.transform(X_test)
        else:
            imputer = SimpleImputer(strategy='mean')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            if cache:
                self.imputers[dataset_name] = imputer
        
        # Standardize
        if cache and dataset_name in self.scalers:
            scaler = self.scalers[dataset_name]
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            if cache:
                self.scalers[dataset_name] = scaler
        
        # Encode labels
        if cache and dataset_name in self.label_encoders:
            le = self.label_encoders[dataset_name]
            y_train = le.transform(y_train)
            y_test = le.transform(y_test)
        else:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            if cache:
                self.label_encoders[dataset_name] = le
        
        return X_train, X_test, y_train, y_test


class BenchmarkRunner:
    """Runs benchmark experiments with progress tracking."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_processor = DataProcessor(config.paths.data_dir)
        self.results: List[ExperimentResult] = []
        
        # Create directories
        config.paths.create_directories()
        
        # Get available datasets
        self.datasets = self._discover_datasets()
        
        # Get available models
        available = get_available_models()
        self.models = [m for m in config.benchmark.models if available.get(m, False)]
        
        if config.benchmark.include_hyperfast and available.get('HyperFast', False):
            self.models.append('HyperFast')
        
        logger.info(f"Found {len(self.datasets)} datasets")
        logger.info(f"Testing {len(self.models)} models: {self.models}")
        logger.info(f"Using {len(config.benchmark.budgets)} time budgets: {config.benchmark.budgets}")
    
    def _discover_datasets(self) -> List[str]:
        """Discover available datasets in data directory."""
        datasets = []
        for f in self.config.paths.data_dir.glob('*_X_train.csv'):
            name = f.stem.replace('_X_train', '')
            datasets.append(name)
        return sorted(datasets)
    
    def run_single_experiment(
        self,
        dataset_name: str,
        model_name: str,
        time_budget: float
    ) -> Optional[ExperimentResult]:
        """
        Run a single experiment.
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            time_budget: Time budget in seconds
            
        Returns:
            ExperimentResult or None if experiment failed
        """
        try:
            # Load data
            X_train, X_test, y_train, y_test = self.data_processor.load_and_preprocess(dataset_name)
            
            # Create model
            model = ModelFactory.create_model(
                model_name=model_name,
                time_budget=time_budget,
                n_features=X_train.shape[1],
                n_classes=len(np.unique(y_train)),
                random_state=self.config.benchmark.random_state,
                n_jobs=self.config.benchmark.n_jobs
            )
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Compute metrics
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Create result
            result = ExperimentResult(
                dataset=dataset_name,
                model=model_name,
                budget=time_budget,
                balanced_accuracy=float(bal_acc),
                accuracy=float(acc),
                f1_score=float(f1),
                fit_time=float(model.metrics.fit_time),
                predict_time=float(model.metrics.predict_time),
                total_time=float(model.metrics.total_time),
                budget_exceeded=model.metrics.budget_exceeded,
                n_train=len(y_train),
                n_test=len(y_test),
                n_features=X_train.shape[1],
                n_classes=len(np.unique(y_train)),
                timestamp=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Error in experiment [{dataset_name}/{model_name}/{time_budget}s]: {str(e)}"
            )
            return ExperimentResult(
                dataset=dataset_name,
                model=model_name,
                budget=time_budget,
                balanced_accuracy=0.0,
                accuracy=0.0,
                f1_score=0.0,
                fit_time=0.0,
                predict_time=0.0,
                total_time=0.0,
                budget_exceeded=True,
                n_train=0,
                n_test=0,
                n_features=0,
                n_classes=0,
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    def run_all_experiments(self, save_interval: int = 10) -> pd.DataFrame:
        """
        Run all experiments with progress tracking.
        
        Args:
            save_interval: Save results every N experiments
            
        Returns:
            DataFrame with all results
        """
        total_experiments = len(self.datasets) * len(self.models) * len(self.config.benchmark.budgets)
        
        logger.info(f"Starting {total_experiments} experiments...")
        
        # Create progress bar
        with tqdm(total=total_experiments, desc="Running experiments") as pbar:
            experiment_count = 0
            
            for dataset_name in self.datasets:
                for time_budget in self.config.benchmark.budgets:
                    for model_name in self.models:
                        # Update progress
                        pbar.set_description(
                            f"{dataset_name[:20]:20s} | {model_name:12s} | {time_budget:4.0f}s"
                        )
                        
                        # Run experiment
                        result = self.run_single_experiment(
                            dataset_name, model_name, time_budget
                        )
                        
                        if result:
                            self.results.append(result)
                            
                            # Log result
                            status = '✓' if not result.budget_exceeded else '⚠'
                            logger.debug(
                                f"{status} {model_name:12s} | "
                                f"Budget: {time_budget:4.0f}s | "
                                f"Actual: {result.total_time:6.2f}s | "
                                f"Bal.Acc: {result.balanced_accuracy:.4f}"
                            )
                        
                        experiment_count += 1
                        pbar.update(1)
                        
                        # Periodic save
                        if experiment_count % save_interval == 0:
                            self.save_results(checkpoint=True)
        
        # Final save
        self.save_results(checkpoint=False)
        
        logger.info(f"✓ Completed {len(self.results)}/{total_experiments} experiments")
        
        return pd.DataFrame([asdict(r) for r in self.results])
    
    def save_results(self, checkpoint: bool = False) -> Path:
        """
        Save results to JSON file.
        
        Args:
            checkpoint: Whether this is a checkpoint save
            
        Returns:
            Path to saved file
        """
        suffix = "_checkpoint" if checkpoint else ""
        filepath = self.config.paths.results_dir / f'benchmark_results{suffix}.json'
        
        results_dict = [asdict(r) for r in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        if not checkpoint:
            logger.info(f"Results saved to: {filepath}")
        
        return filepath
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics from results."""
        if not self.results:
            return {}
        
        df = pd.DataFrame([asdict(r) for r in self.results if r.error is None])
        
        summary = {
            'total_experiments': len(self.results),
            'successful_experiments': len(df),
            'failed_experiments': sum(1 for r in self.results if r.error is not None),
            'datasets': sorted(df['dataset'].unique().tolist()),
            'models': sorted(df['model'].unique().tolist()),
            'budgets': sorted(df['budget'].unique().tolist()),
            'avg_balanced_accuracy_by_model': df.groupby('model')['balanced_accuracy'].mean().to_dict(),
            'avg_runtime_by_model': df.groupby('model')['total_time'].mean().to_dict(),
            'budget_exceeded_count': df.groupby('model')['budget_exceeded'].sum().to_dict(),
        }
        
        return summary


def main():
    """Main entry point for benchmark runner."""
    # Load or create config
    config = ExperimentConfig()
    
    # Save config
    config.save()
    logger.info(f"Configuration saved to: {config.paths.results_dir / 'config.json'}")
    
    # Create and run benchmark
    runner = BenchmarkRunner(config)
    
    # Run all experiments
    start_time = time.time()
    df_results = runner.run_all_experiments()
    total_time = time.time() - start_time
    
    # Generate and save summary
    summary = runner.generate_summary()
    summary['total_runtime_seconds'] = total_time
    summary['total_runtime_hours'] = total_time / 3600
    
    summary_path = config.paths.results_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Total runtime: {total_time/60:.1f} minutes")
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful_experiments']}")
    print(f"Failed: {summary['failed_experiments']}")
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print("="*70)
    
    return df_results, summary


if __name__ == "__main__":
    main()