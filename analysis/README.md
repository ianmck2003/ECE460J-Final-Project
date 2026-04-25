# Analysis Workspace

This directory contains all experiments, reporting, and validation utilities for the ASL project.

## Safety Boundary

- Do not edit runtime code in `final-submission/` from analysis scripts.
- Analysis scripts can read from `final-submission/landmarks.csv`.
- Only final approved artifacts are promoted into `final-submission/`:
  - `model.pkl`
  - `label_encoder.pkl`

## Setup

```powershell
py -m pip install -r analysis/requirements-analysis.txt
```

## Pipeline

Run everything through one orchestrator:

```powershell
py analysis/run_analysis_suite.py
```

Useful toggles:

```powershell
# Skip model training (expects existing comparison artifacts)
py analysis/run_analysis_suite.py --skip-model-comparison

# Skip manual-trial utilities
py analysis/run_analysis_suite.py --skip-manual-matrix --skip-manual-summary

# Show what would run without executing
py analysis/run_analysis_suite.py --dry-run
```

1. Train and compare models:
```powershell
py analysis/experiments/model_comparison.py
```
2. Generate plots:
```powershell
py analysis/plots/generate_visualizations.py
```
3. Benchmark inference:
```powershell
py analysis/benchmarks/benchmark_inference.py
```
4. Kaggle 29-image test (if dataset is present):
```powershell
py analysis/experiments/evaluate_kaggle29.py
```
5. Promote best model into `final-submission/`:
```powershell
py analysis/experiments/promote_best_model.py
```

6. Compile one comprehensive markdown report with links:
```powershell
py analysis/reports/compile_report.py
```

Manual webcam sweeps are documented in `analysis/manual_trials/`.
