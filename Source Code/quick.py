#!/usr/bin/env python3
"""
Quick Start Guide for HyperFast Benchmark
==========================================

This script demonstrates the complete workflow:
1. Configure experiment
2. Run benchmark
3. Analyze results
4. Generate visualizations

Usage:
    python quickstart.py
"""

import sys
from pathlib import Path

def main():
    """Run complete benchmark pipeline."""
    
    print("\n" + "="*70)
    print("HyperFast Mini-Benchmark - Quick Start")
    print("="*70)
    
    # Step 1: Configuration
    print("\n[Step 1/4] Setting up configuration...")
    try:
        from config import ExperimentConfig
        
        config = ExperimentConfig()
        
        # Customize if needed
        config.experiment_name = "quickstart_demo"
        config.description = "Quick start demonstration"
        config.benchmark.budgets = [1, 3, 10, 30, 60]  # Shorter for demo
        
        # Create directories
        config.paths.create_directories()
        
        # Save configuration
        config_path = config.save()
        print(f"✓ Configuration saved to: {config_path}")
        print(f"  - Data directory: {config.paths.data_dir}")
        print(f"  - Results directory: {config.paths.results_dir}")
        print(f"  - Budgets: {config.benchmark.budgets}")
        print(f"  - Models: {config.benchmark.models}")
        
    except Exception as e:
        print(f"✗ Error in configuration: {e}")
        return 1
    
    # Step 2: Check data availability
    print("\n[Step 2/4] Checking data availability...")
    try:
        from run_benchmark_improved import DataProcessor
        
        processor = DataProcessor(config.paths.data_dir)
        
        # Check for datasets
        dataset_files = list(config.paths.data_dir.glob('*_X_train.csv'))
        
        if len(dataset_files) == 0:
            print("⚠ No datasets found!")
            print(f"  Please run prepare_datasets.py first to download data")
            print(f"  Or place your datasets in: {config.paths.data_dir}")
            
            response = input("\nContinue with dataset preparation? (y/n): ")
            if response.lower() == 'y':
                print("\nPreparing datasets...")
                import prepare_mini_datasets
                # This would run the dataset preparation
            else:
                print("Exiting. Please prepare datasets and run again.")
                return 0
        else:
            print(f"✓ Found {len(dataset_files)} datasets")
            
    except Exception as e:
        print(f"⚠ Warning: {e}")
        print("  Continuing anyway...")
    
    # Step 3: Run benchmark
    print("\n[Step 3/4] Running benchmark...")
    print("This may take a while depending on number of datasets and budgets.")
    
    try:
        from run_benchmark_improved import BenchmarkRunner
        
        # Create runner
        runner = BenchmarkRunner(config)
        
        # Check if we have datasets
        if len(runner.datasets) == 0:
            print("✗ No datasets available. Exiting.")
            return 1
        
        print(f"  - Datasets: {len(runner.datasets)}")
        print(f"  - Models: {len(runner.models)}")
        print(f"  - Budgets: {len(config.benchmark.budgets)}")
        print(f"  - Total experiments: {len(runner.datasets) * len(runner.models) * len(config.benchmark.budgets)}")
        
        # Ask for confirmation
        response = input("\nProceed with benchmark? (y/n): ")
        if response.lower() != 'y':
            print("Benchmark cancelled.")
            return 0
        
        # Run experiments
        df_results = runner.run_all_experiments(save_interval=5)
        
        print(f"\n✓ Benchmark complete!")
        print(f"  - Successful experiments: {len(df_results)}")
        print(f"  - Results saved to: {config.paths.results_dir / 'benchmark_results.json'}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user.")
        print("  Partial results have been saved to checkpoint file.")
        return 130
    except Exception as e:
        print(f"\n✗ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Analyze results
    print("\n[Step 4/4] Analyzing results...")
    try:
        from analyze_results_improved import ResultsAnalyzer
        
        results_path = config.paths.results_dir / 'benchmark_results.json'
        
        # Create analyzer
        analyzer = ResultsAnalyzer(results_path, config)
        
        # Print summary
        print("\n" + "-"*70)
        analyzer.print_summary()
        
        # Save detailed analysis
        print("\nSaving detailed analysis...")
        saved_files = analyzer.save_analysis()
        print(f"✓ Saved {len(saved_files)} analysis files")
        
        # Create visualizations
        print("\nCreating visualizations...")
        figures = analyzer.create_visualizations()
        print(f"✓ Created {len(figures)} figures:")
        for fig_path in figures:
            print(f"  - {fig_path.name}")
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Success!
    print("\n" + "="*70)
    print("SUCCESS - All steps completed!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  Results: {config.paths.results_dir}")
    print(f"  Figures: {config.paths.figures_dir}")
    print("\nNext steps:")
    print("  1. Review results in results_mini/")
    print("  2. Check visualizations in results_mini/figures/")
    print("  3. Read analysis_summary.json for detailed statistics")
    print("  4. Compare with paper results")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())