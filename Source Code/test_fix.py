#!/usr/bin/env python3
"""
Quick test to verify XGBoost callback fix works.

Usage:
    python test_fix.py
"""

import sys
import numpy as np

def test_model_creation():
    """Test that models can be created."""
    print("\n" + "="*70)
    print("Testing Model Creation")
    print("="*70)
    
    from time_budget_models_improved import ModelFactory, get_available_models
    
    # Check available models
    available = get_available_models()
    print("\nAvailable models:")
    for model, status in available.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {model}")
    
    # Test creating each model
    print("\nTesting model creation:")
    for model_name in ['KNN', 'LogReg', 'XGBoost', 'LightGBM', 'CatBoost']:
        if not available.get(model_name, False):
            print(f"  ⊘ {model_name:12s} - Skipped (not available)")
            continue
        
        try:
            model = ModelFactory.create_model(
                model_name=model_name,
                time_budget=10,
                n_features=50,
                n_classes=2
            )
            print(f"  ✓ {model_name:12s} - Created successfully")
        except Exception as e:
            print(f"  ✗ {model_name:12s} - Failed: {str(e)[:50]}")
            return False
    
    return True


def test_model_training():
    """Test that models can train."""
    print("\n" + "="*70)
    print("Testing Model Training")
    print("="*70)
    
    from time_budget_models_improved import ModelFactory
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.rand(100, 20)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 20)
    
    # Test XGBoost specifically (the problematic one)
    print("\nTesting XGBoost training (the fix target):")
    try:
        model = ModelFactory.create_model(
            model_name='XGBoost',
            time_budget=5,
            n_features=20,
            n_classes=2
        )
        
        print("  - Fitting model...")
        model.fit(X_train, y_train)
        
        print(f"  - Fit time: {model.metrics.fit_time:.3f}s")
        print(f"  - Budget exceeded: {model.metrics.budget_exceeded}")
        
        print("  - Making predictions...")
        y_pred = model.predict(X_test)
        
        print(f"  - Predict time: {model.metrics.predict_time:.3f}s")
        print(f"  - Total time: {model.metrics.total_time:.3f}s")
        print(f"  - Predictions shape: {y_pred.shape}")
        
        print("  ✓ XGBoost training successful!")
        return True
        
    except Exception as e:
        print(f"  ✗ XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_models():
    """Test training all available models."""
    print("\n" + "="*70)
    print("Testing All Models")
    print("="*70)
    
    from time_budget_models_improved import ModelFactory, get_available_models
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.rand(10, 10)
    
    available = get_available_models()
    results = {}
    
    for model_name in ['KNN', 'LogReg', 'XGBoost', 'LightGBM', 'CatBoost']:
        if not available.get(model_name, False):
            results[model_name] = 'skipped'
            continue
        
        try:
            print(f"\n{model_name}:")
            model = ModelFactory.create_model(
                model_name=model_name,
                time_budget=3,
                n_features=10,
                n_classes=2
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            print(f"  ✓ Fit: {model.metrics.fit_time:.3f}s")
            print(f"  ✓ Predict: {model.metrics.predict_time:.3f}s")
            print(f"  ✓ Budget OK: {not model.metrics.budget_exceeded}")
            
            results[model_name] = 'success'
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:60]}")
            results[model_name] = 'failed'
    
    # Summary
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    success = sum(1 for v in results.values() if v == 'success')
    failed = sum(1 for v in results.values() if v == 'failed')
    skipped = sum(1 for v in results.values() if v == 'skipped')
    
    for model, status in results.items():
        symbol = {'success': '✓', 'failed': '✗', 'skipped': '⊘'}[status]
        print(f"  {symbol} {model:12s} - {status}")
    
    print(f"\nSuccess: {success}, Failed: {failed}, Skipped: {skipped}")
    
    return failed == 0


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("XGBoost Callback Fix - Verification Tests")
    print("="*70)
    
    try:
        # Test 1: Model creation
        if not test_model_creation():
            print("\n✗ Model creation test failed")
            return 1
        
        # Test 2: XGBoost training (specific fix target)
        if not test_model_training():
            print("\n✗ Model training test failed")
            return 1
        
        # Test 3: All models
        if not test_all_models():
            print("\n✗ Some models failed")
            return 1
        
        # Success!
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED - Fix is working!")
        print("="*70)
        print("\nYou can now run the full benchmark:")
        print("  python run_benchmark_improved.py")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())