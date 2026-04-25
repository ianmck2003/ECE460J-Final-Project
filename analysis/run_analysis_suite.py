from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ANALYSIS_ROOT / "reports" / "model_comparison"


def run_python_step(step_name: str, script_path: Path, dry_run: bool) -> None:
    command = [sys.executable, str(script_path)]
    print(f"\n=== {step_name} ===")
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=ANALYSIS_ROOT.parent)


def require_file(path: Path, step_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{step_name} requires {path}, but it was not found. "
            "Run model comparison first or include it in this suite run."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run analysis pipeline steps in sequence with optional toggles."
    )
    parser.add_argument("--skip-model-comparison", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-kaggle", action="store_true")
    parser.add_argument("--skip-manual-matrix", action="store_true")
    parser.add_argument("--skip-manual-summary", action="store_true")
    parser.add_argument("--skip-promote", action="store_true")
    parser.add_argument("--skip-compile-report", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining steps after a failed step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifacts_needed = [
        REPORTS_DIR / "best_model.pkl",
        REPORTS_DIR / "label_encoder.pkl",
        REPORTS_DIR / "best_model_summary.json",
    ]

    steps: list[tuple[str, Path, bool]] = [
        (
            "Model comparison",
            ANALYSIS_ROOT / "experiments" / "model_comparison.py",
            not args.skip_model_comparison,
        ),
        (
            "Visualizations",
            ANALYSIS_ROOT / "plots" / "generate_visualizations.py",
            not args.skip_visualizations,
        ),
        (
            "Inference benchmark",
            ANALYSIS_ROOT / "benchmarks" / "benchmark_inference.py",
            not args.skip_benchmark,
        ),
        (
            "Kaggle 29-image evaluation",
            ANALYSIS_ROOT / "experiments" / "evaluate_kaggle29.py",
            not args.skip_kaggle,
        ),
        (
            "Manual trial matrix generation",
            ANALYSIS_ROOT / "manual_trials" / "generate_trial_matrix.py",
            not args.skip_manual_matrix,
        ),
        (
            "Manual trial summarization",
            ANALYSIS_ROOT / "manual_trials" / "summarize_manual_trials.py",
            not args.skip_manual_summary,
        ),
        (
            "Promote best model artifacts",
            ANALYSIS_ROOT / "experiments" / "promote_best_model.py",
            not args.skip_promote,
        ),
        (
            "Compile comprehensive markdown report",
            ANALYSIS_ROOT / "reports" / "compile_report.py",
            not args.skip_compile_report,
        ),
    ]

    for step_name, script_path, enabled in steps:
        if not enabled:
            print(f"\n--- Skipping {step_name} (flag disabled) ---")
            continue

        requires_comparison_artifacts = step_name in {
            "Visualizations",
            "Inference benchmark",
            "Kaggle 29-image evaluation",
            "Promote best model artifacts",
            "Compile comprehensive markdown report",
        }
        if args.skip_model_comparison and requires_comparison_artifacts:
            for path in artifacts_needed:
                require_file(path, step_name)

        try:
            run_python_step(step_name, script_path, args.dry_run)
        except Exception as exc:  # noqa: BLE001
            print(f"\n!!! Step failed: {step_name}\n{exc}")
            if not args.continue_on_error:
                raise

    print("\nAnalysis suite run complete.")


if __name__ == "__main__":
    main()
