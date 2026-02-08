# prepare_mini_datasets.py
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

os.makedirs('data_mini', exist_ok=True)

# Configuration: Select 5 datasets that meet mini-test criteria
# (≤1000 train, ≤100 features, ≤10 classes)
DATASETS = [
    {'name': 'blood-transfusion-service-center', 'openml_id': 1464, 'max_samples': 1000},
    {'name': 'ilpd', 'openml_id': 1480, 'max_samples': 1000},
    {'name': 'diabetes', 'openml_id': 37, 'max_samples': 1000},
    {'name': 'kc2', 'openml_id': 1063, 'max_samples': 1000},
]

def prepare_dataset(name, openml_id, max_samples=1000):
    """Download and prepare a single dataset"""
    print(f"\nProcessing: {name}")
    
    try:
        # Download from OpenML
        X, y = fetch_openml(
            data_id=openml_id,
            return_X_y=True,
            as_frame=True,
            parser='auto'
        )
        
        # Handle missing values in target
        if y.isna().any():
            mask = ~y.isna()
            X, y = X[mask], y[mask]
        
        # Sample if too large
        if len(X) > max_samples * 1.5:  # 1.5 to account for train/test split
            X, _, y, _ = train_test_split(
                X, y, train_size=max_samples * 1.5, 
                random_state=42, stratify=y
            )
        
        # Limit features to 100
        if X.shape[1] > 100:
            # Select first 100 features (or use feature selection)
            X = X.iloc[:, :100]
        
        # Limit classes to 10
        value_counts = y.value_counts()
        if len(value_counts) > 10:
            # Keep top 10 most frequent classes
            top_classes = value_counts.head(10).index
            mask = y.isin(top_classes)
            X, y = X[mask], y[mask]
        
        # Create train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Ensure train set is ≤1000
        if len(X_train) > max_samples:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=max_samples,
                random_state=42, stratify=y_train
            )
        
        # Save to CSV
        X_train.to_csv(f'data_mini/{name}_X_train.csv', index=False)
        X_test.to_csv(f'data_mini/{name}_X_test.csv', index=False)
        pd.Series(y_train).to_csv(f'data_mini/{name}_y_train.csv', index=False)
        pd.Series(y_test).to_csv(f'data_mini/{name}_y_test.csv', index=False)
        
        print(f"  ✓ Train: {X_train.shape}, Test: {X_test.shape}, "
              f"Features: {X_train.shape[1]}, Classes: {y_train.nunique()}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

# Process all datasets
success_count = 0
for ds in DATASETS:
    if prepare_dataset(ds['name'], ds['openml_id'], ds['max_samples']):
        success_count += 1

print(f"\n{'='*60}")
print(f"✓ Successfully prepared {success_count}/{len(DATASETS)} datasets")
print(f"{'='*60}")