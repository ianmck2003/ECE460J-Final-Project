from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.append(str(Path(__file__).resolve().parents[2]))

from analysis.common import Paths, ensure_output_dirs, load_landmark_data, make_models, split_data


def hardest_confusions(cm: np.ndarray, labels: List[str], top_k: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                rows.append({"true_label": labels[i], "pred_label": labels[j], "count": count})
    rows.sort(key=lambda r: r["count"], reverse=True)
    return pd.DataFrame(rows[:top_k])


def plot_confusion(cm: np.ndarray, labels: List[str], out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def timed_predict_ms_per_sample(model, x_val: np.ndarray, loops: int = 5) -> float:
    # Repeat predictions to smooth measurement noise.
    elapsed = []
    for _ in range(loops):
        start = time.perf_counter()
        _ = model.predict(x_val)
        end = time.perf_counter()
        elapsed.append((end - start) * 1000.0 / len(x_val))
    return float(np.mean(elapsed))


def main() -> None:
    paths = Paths()
    ensure_output_dirs(paths)

    comparison_dir = paths.reports_dir / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    x, y, le = load_landmark_data(paths.landmarks_csv)
    x_train, x_val, y_train, y_val = split_data(x, y)
    labels = list(le.classes_)

    models = make_models()
    metrics_rows: List[Dict[str, object]] = []
    reports_json: Dict[str, dict] = {}
    best_model_name = None
    best_model = None
    best_score = -1.0

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        train_start = time.perf_counter()
        model.fit(x_train, y_train)
        train_s = time.perf_counter() - train_start

        y_pred = model.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")
        ms_per_sample = timed_predict_ms_per_sample(model, x_val)

        cm = confusion_matrix(y_val, y_pred)
        plot_confusion(
            cm,
            labels,
            comparison_dir / f"confusion_matrix_{model_name}.png",
            f"Confusion Matrix ({model_name})",
        )

        hardest = hardest_confusions(cm, labels)
        hardest.to_csv(comparison_dir / f"hardest_confusions_{model_name}.csv", index=False)

        report = classification_report(
            y_val,
            y_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        reports_json[model_name] = report

        if model_name == "mlp" and hasattr(model, "loss_curve_"):
            (comparison_dir / "mlp_loss_curve.json").write_text(
                json.dumps({"loss_curve": [float(v) for v in model.loss_curve_]}, indent=2),
                encoding="utf-8",
            )

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "train_seconds": train_s,
                "predict_ms_per_sample": ms_per_sample,
            }
        )

        if acc > best_score:
            best_score = acc
            best_model_name = model_name
            best_model = model

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["accuracy", "macro_f1"], ascending=False
    )
    metrics_df.to_csv(comparison_dir / "metrics_summary.csv", index=False)
    (comparison_dir / "classification_reports.json").write_text(
        json.dumps(reports_json, indent=2),
        encoding="utf-8",
    )

    if best_model is None or best_model_name is None:
        raise RuntimeError("No model was trained.")

    with (comparison_dir / "best_model.pkl").open("wb") as f:
        pickle.dump(best_model, f)
    with (comparison_dir / "label_encoder.pkl").open("wb") as f:
        pickle.dump(le, f)

    summary = {
        "best_model": best_model_name,
        "best_accuracy": float(best_score),
        "num_train_samples": int(len(x_train)),
        "num_val_samples": int(len(x_val)),
        "source_landmarks_csv": str(paths.landmarks_csv),
    }
    (comparison_dir / "best_model_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_model_name} ({best_score:.4f})")


if __name__ == "__main__":
    main()
