# Reproducibility Of HyerFast Paper

Note you have to manually download the Model from [hyperfast/](https://github.com/AI-sandbox/HyperFast/blob/main/hyperfast/config.py) and make sure to clone their repository

Make sure to have install these libraries

```bash

pip install hyperfast xgboost lightbm catboost seaborn tqdm numpy pandas scikit-learn openml

```



### Core Modules
* config.py - Centralized configuration management
* time_budget_models_improved.py - Enhanced model wrappers with error handling
* run_benchmark_improved.py - Main benchmark runner with progress tracking
* analyze_results_improved.py - Comprehensive results analysis
* prepare_mini_datasets.py - Download the OpenML datasets
* quickstart.py - Quick start script for the complete pipeline





### Run the complete pipeline

```bash
python quickstart.py
```



Or run steps individually:

```bash
python run_benchmark_improved.py    # Run benchmark

python analyze_results_improved.py  # Analyze results
```

Author
Mohsin Ali Mirza

