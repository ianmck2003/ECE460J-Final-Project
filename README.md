# ECE460J Final Project - ASL Fingerspelling Recognition

This repository contains two connected workflows:

- `analysis/`: model comparison, plots, benchmarks, report generation, and artifact promotion.
- `final-submission/`: webcam inference runtime and standalone training scripts.

## Prerequisites

- Windows 10/11 (project commands are written for PowerShell)
- Python 3.10+ (the repo currently runs with `py`)
- A webcam (for real-time inference)
- ASL alphabet image dataset at:
  - `asl_data/asl_alphabet_train/asl_alphabet_train/<CLASS_FOLDER>/*.jpg`
- MediaPipe hand landmarker task file:
  - `final-submission/hand_landmarker.task`

## Quick Start (Run Inference)

From the repository root:

```powershell
py -m pip install -r final-submission/requirements.txt
```

Make sure these files exist inside `final-submission/`:

- `model.pkl`
- `label_encoder.pkl`
- `hand_landmarker.task`

Then run:

```powershell
Set-Location final-submission
py .\inference.py
```

Inference controls:

- `Backspace`: delete last character
- `C`: clear caption
- `Q` or `Esc`: quit

## Full Analysis Suite

Install analysis dependencies:

```powershell
py -m pip install -r analysis/requirements-analysis.txt
```

Run the full pipeline:

```powershell
py .\analysis\run_analysis_suite.py
```

This runs:

1. Model comparison (`analysis/experiments/model_comparison.py`)
2. Plot generation (`analysis/plots/generate_visualizations.py`)
3. Inference benchmark (`analysis/benchmarks/benchmark_inference.py`)
4. Kaggle 29-image eval (`analysis/experiments/evaluate_kaggle29.py`)
5. Manual trial matrix + summary (`analysis/manual_trials/*`)
6. Promote best artifacts (`analysis/experiments/promote_best_model.py`)
7. Compile report (`analysis/reports/compile_report.py`)

Useful flags:

```powershell
# Print planned steps only
py .\analysis\run_analysis_suite.py --dry-run

# Skip model comparison (requires existing comparison artifacts)
py .\analysis\run_analysis_suite.py --skip-model-comparison

# Skip manual trial utilities
py .\analysis\run_analysis_suite.py --skip-manual-matrix --skip-manual-summary
```

## Training From Scratch (Final Submission Scripts)

If you need to regenerate `landmarks.csv` from image folders:

```powershell
Set-Location final-submission
py .\extract_landmarks.py
```

Then train and save inference artifacts:

```powershell
py .\CV_model.py
```

This writes:

- `final-submission/model.pkl`
- `final-submission/label_encoder.pkl`

## Important Paths

- Runtime inference entrypoint: `final-submission/inference.py`
- Analysis orchestrator: `analysis/run_analysis_suite.py`
- Main analysis report output: `analysis/reports/analysis_report.md`
- Generated plots: `analysis/plots/outputs/`
- Model-comparison artifacts: `analysis/reports/model_comparison/`

## Troubleshooting

- `ModuleNotFoundError`:
  - Re-run the matching `pip install -r ...` command for the workflow you are using.
- `Could not open webcam`:
  - Close other apps using the camera and retry.
- Missing `hand_landmarker.task`:
  - Place the model file at `final-submission/hand_landmarker.task`.
- Inference starts but no predictions stabilize:
  - Improve lighting/background contrast and keep one hand clearly in-frame.
