"""
Configuration management for HyperFast benchmark reproduction.

This module centralizes all configuration parameters for better maintainability
and easier experimentation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    max_samples: int = 1000
    max_features: int = 100
    max_classes: int = 10
    test_size: float = 0.2
    random_state: int = 42
    
    # OpenML dataset specifications
    datasets: List[dict] = field(default_factory=lambda: [
        {'name': 'blood-transfusion-service-center', 'openml_id': 1464},
        {'name': 'ilpd', 'openml_id': 1480},
        {'name': 'diabetes', 'openml_id': 37},
        {'name': 'kc2', 'openml_id': 1063},
    ])


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    # Runtime budgets in seconds (matching Figure 2 from paper)
    budgets: List[int] = field(default_factory=lambda: [1, 3, 10, 30, 60, 300])
    
    # Models to test
    models: List[str] = field(default_factory=lambda: ['KNN', 'LogReg', 'XGBoost', 'LightGBM', 'CatBoost'])
    
    # Add HyperFast if available
    include_hyperfast: bool = True
    
    # Resource limits
    max_memory_gb: int = 32
    timeout_multiplier: float = 1.1  # Allow 10% buffer on time budgets
    
    # Random seed for reproducibility
    random_state: int = 42
    
    # Batch processing
    batch_size: int = 2048
    
    # Number of jobs for parallel processing (-1 = all cores)
    n_jobs: int = -1


@dataclass
class HyperFastConfig:
    """Configuration for HyperFast model."""
    # Ensemble configuration by budget
    ensemble_by_budget: dict = field(default_factory=lambda: {
        1: {'n_ensemble': 1, 'optimization': None, 'optimize_steps': 0},
        3: {'n_ensemble': 4, 'optimization': None, 'optimize_steps': 0},
        10: {'n_ensemble': 8, 'optimization': None, 'optimize_steps': 0},
        30: {'n_ensemble': 16, 'optimization': None, 'optimize_steps': 0},
        60: {'n_ensemble': 16, 'optimization': None, 'optimize_steps': 0},
        300: {'n_ensemble': 16, 'optimization': 'ensemble_optimize', 'optimize_steps': 64},
    })
    
    batch_size: int = 2048
    nn_bias: bool = False
    seed: int = 42


@dataclass
class PathsConfig:
    """Configuration for file paths."""
    data_dir: Path = field(default_factory=lambda: Path('data_mini'))
    results_dir: Path = field(default_factory=lambda: Path('results_mini'))
    figures_dir: Path = field(default_factory=lambda: Path('results_mini/figures'))
    
    def __post_init__(self):
        """Ensure all directories are Path objects."""
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.figures_dir = Path(self.figures_dir)
    
    def create_directories(self):
        """Create all necessary directories."""
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.figures_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class ExperimentConfig:
    """Master configuration combining all sub-configurations."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    hyperfast: HyperFastConfig = field(default_factory=HyperFastConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    # Experiment metadata
    experiment_name: str = "hyperfast_mini_benchmark"
    description: str = "Mini-test benchmark reproduction"
    
    def save(self, filepath: Optional[Path] = None):
        """Save configuration to JSON file."""
        if filepath is None:
            filepath = self.paths.results_dir / "config.json"
        
        config_dict = {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'dataset': self.dataset.__dict__,
            'benchmark': self.benchmark.__dict__,
            'hyperfast': self.hyperfast.__dict__,
            'paths': {k: str(v) for k, v in self.paths.__dict__.items()},
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.experiment_name = config_dict.get('experiment_name', config.experiment_name)
        config.description = config_dict.get('description', config.description)
        
        # Load sub-configs
        if 'dataset' in config_dict:
            config.dataset = DatasetConfig(**config_dict['dataset'])
        if 'benchmark' in config_dict:
            config.benchmark = BenchmarkConfig(**config_dict['benchmark'])
        if 'hyperfast' in config_dict:
            config.hyperfast = HyperFastConfig(**config_dict['hyperfast'])
        if 'paths' in config_dict:
            config.paths = PathsConfig(**config_dict['paths'])
        
        return config


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    config.paths.create_directories()
    print("Configuration created:")
    print(f"  Data directory: {config.paths.data_dir}")
    print(f"  Results directory: {config.paths.results_dir}")
    print(f"  Budgets: {config.benchmark.budgets}")
    print(f"  Models: {config.benchmark.models}")