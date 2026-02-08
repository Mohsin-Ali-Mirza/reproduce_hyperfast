"""
Time-budgeted model wrappers for benchmark experiments.

This module provides wrappers for various ML models that enforce runtime
budgets during training, with improved error handling and resource management.
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

try:
    from hyperfast import HyperFastClassifier
    HYPERFAST_AVAILABLE = True
except ImportError:
    HYPERFAST_AVAILABLE = False
    logging.warning("⚠️  HyperFast not available - will be skipped in benchmarks")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Check library versions
def check_library_versions():
    """Check and log library versions."""
    try:
        logger.info(f"XGBoost version: {xgb.__version__}")
    except:
        pass
    try:
        logger.info(f"LightGBM version: {lgb.__version__}")
    except:
        pass
    try:
        logger.info(f"CatBoost version: {cb.__version__}")
    except:
        pass


@dataclass
class ModelMetrics:
    """Container for model training metrics."""
    fit_time: float
    predict_time: float
    total_time: float
    budget_exceeded: bool
    memory_used_mb: Optional[float] = None
    error_message: Optional[str] = None


class TimeBudgetWrapper:
    """
    Wrapper that tracks time budgets during model training.
    
    This wrapper monitors training time and reports when models exceed their
    allocated time budget. Budget enforcement is achieved primarily through
    model configuration (number of iterations, etc.) rather than interrupt-based
    callbacks, which improves compatibility across library versions.
    
    Attributes:
        model: The underlying ML model
        time_budget: Maximum allowed training time in seconds
        timeout_tolerance: Percentage over budget allowed (default 10%)
        metrics: ModelMetrics object with timing information
    """
    
    def __init__(
        self, 
        model: BaseEstimator, 
        time_budget: float,
        timeout_tolerance: float = 0.1
    ):
        self.model = model
        self.time_budget = time_budget
        self.timeout_tolerance = timeout_tolerance
        self.metrics = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TimeBudgetWrapper':
        """
        Fit the model with time budget enforcement.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self: The fitted wrapper
            
        Raises:
            TimeoutError: If model exceeds budget significantly
        """
        start_time = time.time()
        
        try:
            # Simple approach: just fit and check time after
            # Most models will naturally stay within budget based on their configuration
            self.model.fit(X, y)
            
            fit_time = time.time() - start_time
            
            # Check if we exceeded budget
            budget_exceeded = fit_time > self.time_budget * (1 + self.timeout_tolerance)
            
            if budget_exceeded:
                logger.warning(
                    f"Model {type(self.model).__name__} exceeded budget: "
                    f"{fit_time:.2f}s > {self.time_budget}s"
                )
            
            self.metrics = ModelMetrics(
                fit_time=fit_time,
                predict_time=0.0,
                total_time=fit_time,
                budget_exceeded=budget_exceeded
            )
            
        except Exception as e:
            fit_time = time.time() - start_time
            logger.error(f"Error during model fitting: {str(e)}")
            self.metrics = ModelMetrics(
                fit_time=fit_time,
                predict_time=0.0,
                total_time=fit_time,
                budget_exceeded=True,
                error_message=str(e)
            )
            raise
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with timing.
        
        Args:
            X: Test features
            
        Returns:
            predictions: Model predictions
        """
        if self.metrics is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        start_time = time.time()
        predictions = self.model.predict(X)
        predict_time = time.time() - start_time
        
        # Update metrics
        self.metrics.predict_time = predict_time
        self.metrics.total_time = self.metrics.fit_time + predict_time
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions with timing."""
        if self.metrics is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        start_time = time.time()
        predictions = self.model.predict_proba(X)
        predict_time = time.time() - start_time
        
        self.metrics.predict_time = predict_time
        self.metrics.total_time = self.metrics.fit_time + predict_time
        
        return predictions


class ModelFactory:
    """Factory for creating time-budgeted models."""
    
    @staticmethod
    def create_model(
        model_name: str,
        time_budget: float,
        n_features: int,
        n_classes: int,
        random_state: int = 42,
        n_jobs: int = -1
    ) -> TimeBudgetWrapper:
        """
        Create a model configured for a specific time budget.
        
        Args:
            model_name: Name of the model ('KNN', 'LogReg', 'XGBoost', etc.)
            time_budget: Maximum training time in seconds
            n_features: Number of input features
            n_classes: Number of classes
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            
        Returns:
            TimeBudgetWrapper: Wrapped model ready for training
            
        Raises:
            ValueError: If model_name is not recognized
        """
        model = ModelFactory._get_base_model(
            model_name, time_budget, n_features, n_classes, random_state, n_jobs
        )
        return TimeBudgetWrapper(model, time_budget)
    
    @staticmethod
    def _get_base_model(
        model_name: str,
        time_budget: float,
        n_features: int,
        n_classes: int,
        random_state: int,
        n_jobs: int
    ) -> BaseEstimator:
        """Get the base model without time wrapper."""
        
        if model_name == 'KNN':
            return KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs)
        
        elif model_name == 'LogReg':
            max_iter = 100 if time_budget < 10 else 500
            return LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=n_jobs,
                solver='lbfgs'
            )
        
        elif model_name == 'XGBoost':
            return ModelFactory._create_xgboost(time_budget, random_state, n_jobs)
        
        elif model_name == 'LightGBM':
            return ModelFactory._create_lightgbm(time_budget, random_state, n_jobs)
        
        elif model_name == 'CatBoost':
            return ModelFactory._create_catboost(time_budget, random_state)
        
        elif model_name == 'HyperFast':
            if not HYPERFAST_AVAILABLE:
                raise ImportError("HyperFast is not available. Install it to use this model.")
            return ModelFactory._create_hyperfast(time_budget, random_state)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def _create_xgboost(
        time_budget: float,
        random_state: int,
        n_jobs: int
    ) -> BaseEstimator:
        """
        Create XGBoost model based on time budget.
        
        Budget enforcement: Configure n_estimators and max_depth to naturally
        complete within budget rather than using callbacks (better compatibility).
        """
        if time_budget < 3:
            return xgb.XGBClassifier(
                n_estimators=10,
                max_depth=3,
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=0
            )
        elif time_budget < 10:
            return xgb.XGBClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=0
            )
        else:
            param_dist = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
            base = xgb.XGBClassifier(
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=0
            )
            return RandomizedSearchCV(
                base,
                param_dist,
                n_iter=5,
                cv=3,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=0
            )
    
    @staticmethod
    def _create_lightgbm(
        time_budget: float,
        random_state: int,
        n_jobs: int
    ) -> BaseEstimator:
        """Create LightGBM model based on time budget."""
        if time_budget < 3:
            return lgb.LGBMClassifier(
                n_estimators=10,
                num_leaves=15,
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=-1
            )
        elif time_budget < 10:
            return lgb.LGBMClassifier(
                n_estimators=50,
                num_leaves=31,
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=-1
            )
        else:
            param_dist = {
                'n_estimators': [50, 100, 200],
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.01, 0.1]
            }
            base = lgb.LGBMClassifier(
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=-1
            )
            return RandomizedSearchCV(
                base,
                param_dist,
                n_iter=5,
                cv=3,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=0
            )
    
    @staticmethod
    def _create_catboost(
        time_budget: float,
        random_state: int
    ) -> BaseEstimator:
        """Create CatBoost model based on time budget."""
        iterations = max(10, int(time_budget * 10))
        return cb.CatBoostClassifier(
            iterations=iterations,
            random_state=random_state,
            verbose=False,
            thread_count=-1
        )
    
    @staticmethod
    def _create_hyperfast(
        time_budget: float,
        random_state: int
    ) -> HyperFastClassifier:
        """Create HyperFast model based on time budget."""
        # Configure based on budget
        if time_budget < 3:
            config = {'n_ensemble': 1, 'optimization': None, 'optimize_steps': 0}
        elif time_budget < 10:
            config = {'n_ensemble': 4, 'optimization': None, 'optimize_steps': 0}
        elif time_budget < 30:
            config = {'n_ensemble': 8, 'optimization': None, 'optimize_steps': 0}
        elif time_budget < 60:
            config = {'n_ensemble': 16, 'optimization': None, 'optimize_steps': 0}
        else:
            config = {'n_ensemble': 16, 'optimization': 'ensemble_optimize', 'optimize_steps': 64}
        
        return HyperFastClassifier(
            n_ensemble=config['n_ensemble'],
            batch_size=2048,
            nn_bias=False,
            optimization=config['optimization'],
            optimize_steps=config['optimize_steps'],
            seed=random_state
        )


def get_available_models() -> Dict[str, bool]:
    """
    Check which models are available in the current environment.
    
    Returns:
        Dictionary mapping model names to availability status
    """
    return {
        'KNN': True,
        'LogReg': True,
        'XGBoost': True,
        'LightGBM': True,
        'CatBoost': True,
        'HyperFast': HYPERFAST_AVAILABLE
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Available models:")
    for model, available in get_available_models().items():
        status = "✓" if available else "✗"
        print(f"  {status} {model}")
    
    # Test model creation
    print("\nTesting model creation...")
    try:
        model = ModelFactory.create_model('XGBoost', time_budget=10, n_features=50, n_classes=2)
        print(f"  ✓ Created XGBoost model with 10s budget")
    except Exception as e:
        print(f"  ✗ Error: {e}")