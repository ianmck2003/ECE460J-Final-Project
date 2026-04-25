from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "analysis"
REPORTS_DIR = ANALYSIS_DIR / "reports"
MODEL_COMPARISON_DIR = REPORTS_DIR / "model_comparison"
PLOTS_DIR = ANALYSIS_DIR / "plots" / "outputs"
MANUAL_DIR = ANALYSIS_DIR / "manual_trials"
MANUAL_OUTPUTS_DIR = MANUAL_DIR / "outputs"
FINAL_SUBMISSION_DIR = REPO_ROOT / "final-submission"


def rel_link(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    href = Path(os.path.relpath(path, start=REPORTS_DIR)).as_posix()
    return f"[`{rel}`]({href})"


def bullet_links(paths: Iterable[Path]) -> str:
    lines = []
    for p in sorted(paths):
        if p.exists():
            lines.append(f"- {rel_link(p)}")
    return "\n".join(lines) if lines else "- (none found)"


def image_embeds(paths: Iterable[Path]) -> str:
    blocks = []
    for p in sorted(paths):
        if not p.exists():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            continue
        rel = p.relative_to(REPO_ROOT).as_posix()
        href = Path(os.path.relpath(p, start=REPORTS_DIR)).as_posix()
        blocks.append(f"#### `{p.name}`\n\n![{rel}]({href})")
    return "\n\n".join(blocks) if blocks else "_No images found._"


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    safe_df = df.fillna("")
    cols = list(safe_df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in safe_df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, separator, *rows])


def top_hardest_pairs(files: List[Path]) -> pd.DataFrame:
    rows = []
    for csv_path in files:
        if not csv_path.exists():
            continue
        model_name = csv_path.stem.replace("hardest_confusions_", "")
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["model"] = model_name
        rows.append(df.head(3))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["model", "true_label", "pred_label", "count"]]


def main() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = REPORTS_DIR / "analysis_report.md"

    metrics_path = MODEL_COMPARISON_DIR / "metrics_summary.csv"
    best_summary_path = MODEL_COMPARISON_DIR / "best_model_summary.json"
    benchmark_path = REPORTS_DIR / "benchmark_results.json"
    kaggle_summary_path = REPORTS_DIR / "kaggle29_summary.json"
    promotion_log_path = REPORTS_DIR / "promotion_log.json"

    metrics_df = safe_read_csv(metrics_path)
    best_summary = safe_read_json(best_summary_path)
    benchmark = safe_read_json(benchmark_path)
    kaggle = safe_read_json(kaggle_summary_path)
    promotion = safe_read_json(promotion_log_path)

    hardest_files = [
        MODEL_COMPARISON_DIR / "hardest_confusions_mlp.csv",
        MODEL_COMPARISON_DIR / "hardest_confusions_random_forest.csv",
        MODEL_COMPARISON_DIR / "hardest_confusions_svm_rbf.csv",
        MODEL_COMPARISON_DIR / "hardest_confusions_knn.csv",
    ]
    hardest_df = top_hardest_pairs(hardest_files)

    plot_files = list(PLOTS_DIR.glob("*"))
    confusion_plot_files = sorted(MODEL_COMPARISON_DIR.glob("confusion_matrix_*.png"))
    manual_output_files = list(MANUAL_OUTPUTS_DIR.glob("*"))

    source_files = [
        ANALYSIS_DIR / "common.py",
        ANALYSIS_DIR / "experiments" / "model_comparison.py",
        ANALYSIS_DIR / "plots" / "generate_visualizations.py",
        ANALYSIS_DIR / "benchmarks" / "benchmark_inference.py",
        ANALYSIS_DIR / "experiments" / "evaluate_kaggle29.py",
        ANALYSIS_DIR / "manual_trials" / "generate_trial_matrix.py",
        ANALYSIS_DIR / "manual_trials" / "summarize_manual_trials.py",
        ANALYSIS_DIR / "experiments" / "promote_best_model.py",
        ANALYSIS_DIR / "run_analysis_suite.py",
    ]

    data_inputs = [
        FINAL_SUBMISSION_DIR / "landmarks.csv",
        FINAL_SUBMISSION_DIR / "hand_landmarker.task",
        REPO_ROOT / "kaggle-test-images",
        MANUAL_DIR / "manual_trial_log.csv",
        MANUAL_DIR / "trial_matrix.csv",
    ]

    report = f"""# Comprehensive Analysis Report

Generated: `{now}`

## Executive Summary

- Best model: `{best_summary.get("best_model", "unknown")}`
- Best validation accuracy: `{best_summary.get("best_accuracy", "unknown")}`
- Train/val samples: `{best_summary.get("num_train_samples", "unknown")}` / `{best_summary.get("num_val_samples", "unknown")}`
- Kaggle status: `{kaggle.get("status", "unknown")}`

## Model Comparison Metrics

{to_markdown_table(metrics_df)}

## Hardest Confused Pairs (Top 3 per model)

{to_markdown_table(hardest_df)}

## Runtime Benchmark Summary

- Classifier-only mean (ms/sample): `{benchmark.get("classifier_only", {}).get("ms_per_sample_mean", "n/a")}`
- Classifier-only std (ms/sample): `{benchmark.get("classifier_only", {}).get("ms_per_sample_std", "n/a")}`
- End-to-end status: `{benchmark.get("end_to_end", {}).get("status", "n/a")}`
- End-to-end mean (ms/frame): `{benchmark.get("end_to_end", {}).get("ms_per_frame_mean", "n/a")}`
- End-to-end std (ms/frame): `{benchmark.get("end_to_end", {}).get("ms_per_frame_std", "n/a")}`
- End-to-end samples used: `{benchmark.get("end_to_end", {}).get("samples_used", "n/a")}`

## Kaggle 29-image Evaluation

- Status: `{kaggle.get("status", "unknown")}`
- Reason: `{kaggle.get("reason", "n/a")}`
- Images total / scored: `{kaggle.get("num_images_total", "n/a")}` / `{kaggle.get("num_images_scored", "n/a")}`
- Accuracy: `{kaggle.get("accuracy", "n/a")}`

## Promotion Summary

- Promoted model path: `{promotion.get("copied_model_to", "n/a")}`
- Promoted encoder path: `{promotion.get("copied_encoder_to", "n/a")}`
- Source model artifact: `{promotion.get("source_model", "n/a")}`
- Source encoder artifact: `{promotion.get("source_encoder", "n/a")}`

## Artifact Links

### Model Comparison Artifacts

{bullet_links([
    MODEL_COMPARISON_DIR / "metrics_summary.csv",
    MODEL_COMPARISON_DIR / "classification_reports.json",
    MODEL_COMPARISON_DIR / "best_model_summary.json",
    MODEL_COMPARISON_DIR / "mlp_loss_curve.json",
    MODEL_COMPARISON_DIR / "best_model.pkl",
    MODEL_COMPARISON_DIR / "label_encoder.pkl",
    *hardest_files,
])}

### Confusion Matrix Plots

{bullet_links(confusion_plot_files)}

### Confusion Matrix Displays

{image_embeds(confusion_plot_files)}

### Visualization Plots

{bullet_links(plot_files)}

### Visualization Displays

{image_embeds(plot_files)}

### Benchmark And Evaluation Outputs

{bullet_links([
    REPORTS_DIR / "benchmark_results.json",
    REPORTS_DIR / "kaggle29_summary.json",
    REPORTS_DIR / "promotion_log.json",
])}

### Manual Trial Outputs

{bullet_links([
    MANUAL_DIR / "manual_trial_template.csv",
    MANUAL_DIR / "manual_trial_log.csv",
    MANUAL_DIR / "trial_matrix.csv",
    *manual_output_files,
])}

### Manual Trial Diagram Displays

{image_embeds(manual_output_files)}

## Source Scripts Used

{bullet_links(source_files)}

## Data Inputs Used

{bullet_links(data_inputs)}
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote comprehensive report to {report_path}")


if __name__ == "__main__":
    main()
