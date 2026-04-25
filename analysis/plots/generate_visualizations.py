from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import seaborn as sns
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

sys.path.append(str(Path(__file__).resolve().parents[2]))

from analysis.common import Paths, ensure_output_dirs, load_landmark_data, split_data

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


DATA_DIR = Path("./asl_data/asl_alphabet_train/asl_alphabet_train")
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _load_model_results(paths: Paths) -> Tuple[pd.DataFrame, Dict[str, dict], str]:
    comparison_dir = paths.reports_dir / "model_comparison"
    metrics = pd.read_csv(comparison_dir / "metrics_summary.csv")
    reports = json.loads((comparison_dir / "classification_reports.json").read_text(encoding="utf-8"))
    best_model = json.loads((comparison_dir / "best_model_summary.json").read_text(encoding="utf-8"))["best_model"]
    return metrics, reports, best_model


def plot_class_distribution(paths: Paths) -> pd.Series:
    df = pd.read_csv(paths.landmarks_csv)
    counts = df["label"].value_counts().sort_index()
    plt.figure(figsize=(12, 5))
    counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "class_distribution.png", dpi=150)
    plt.close()
    return counts


def plot_per_class_accuracy(paths: Paths, reports: Dict[str, dict], best_model: str) -> Dict[str, float]:
    best_report = reports[best_model]
    class_rows = {
        cls: vals.get("recall", 0.0)
        for cls, vals in best_report.items()
        if isinstance(vals, dict) and "recall" in vals
    }
    per_class = dict(sorted(class_rows.items(), key=lambda kv: kv[0]))
    plt.figure(figsize=(12, 5))
    plt.bar(list(per_class.keys()), list(per_class.values()))
    plt.ylim(0, 1)
    plt.title(f"Per-class Accuracy (Recall) - {best_model}")
    plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "per_class_accuracy.png", dpi=150)
    plt.close()
    return per_class


def plot_mlp_loss_curve(paths: Paths) -> None:
    mlp_loss_path = paths.reports_dir / "model_comparison" / "mlp_loss_curve.json"
    if not mlp_loss_path.exists():
        print("Skipping MLP loss curve: mlp_loss_curve.json not found.")
        return

    losses = json.loads(mlp_loss_path.read_text(encoding="utf-8")).get("loss_curve", [])
    if not losses:
        print("Skipping MLP loss curve: empty curve.")
        return
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("MLP Training Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "mlp_loss_curve.png", dpi=150)
    plt.close()


def plot_rf_feature_importance(paths: Paths, x: np.ndarray, y: np.ndarray) -> None:
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(x, y)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]
    top_labels = [f"f{i}" for i in top_idx]
    top_vals = importances[top_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(top_labels[::-1], top_vals[::-1])
    plt.title("Random Forest Top-20 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "rf_feature_importance_top20.png", dpi=150)
    plt.close()


def _normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    wrist = coords[0]
    centered = coords - wrist
    max_val = np.max(np.abs(centered))
    if max_val <= 0:
        max_val = 1.0
    return centered / max_val


def plot_before_after_normalization(paths: Paths) -> None:
    if not DATA_DIR.exists():
        print(f"Skipping normalization plot: missing dataset dir {DATA_DIR}")
        return

    image_paths = sorted(DATA_DIR.glob("*/*"))
    if not image_paths:
        print("Skipping normalization plot: no images found.")
        return

    base_options = python.BaseOptions(model_asset_path=str(paths.hand_landmarker_task))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    raw = None
    normalized = None
    for image_path in image_paths[:200]:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            normalized = _normalize_landmarks(raw)
            break
    detector.close()
    if raw is None or normalized is None:
        print("Skipping normalization plot: no detectable hand sample found.")
        return

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.scatter(raw[:, 0], raw[:, 1], raw[:, 2], c=np.arange(21), cmap="viridis")
    ax1.set_title("Before normalization")
    ax2.scatter(normalized[:, 0], normalized[:, 1], normalized[:, 2], c=np.arange(21), cmap="viridis")
    ax2.set_title("After normalization")
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "before_after_normalization.png", dpi=150)
    plt.close()


def plot_embedding(paths: Paths, x: np.ndarray, y: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    sample_n = min(4000, len(x))
    indices = rng.choice(len(x), size=sample_n, replace=False)
    x_sample = x[indices]
    y_sample = y[indices]

    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
    x_tsne = tsne.fit_transform(x_sample)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_sample, cmap="tab20", s=5, alpha=0.7)
    plt.title("t-SNE of Landmark Feature Space")
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "landmark_tsne.png", dpi=150)
    plt.close()

    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        x_umap = reducer.fit_transform(x_sample)
        plt.figure(figsize=(8, 6))
        plt.scatter(x_umap[:, 0], x_umap[:, 1], c=y_sample, cmap="tab20", s=5, alpha=0.7)
        plt.title("UMAP of Landmark Feature Space")
        plt.tight_layout()
        plt.savefig(paths.plots_dir / "landmark_umap.png", dpi=150)
        plt.close()


def plot_asl_grid(paths: Paths, per_class_accuracy: Dict[str, float]) -> None:
    classes = sorted(per_class_accuracy.keys())
    n = len(classes)
    cols = 6
    rows = math.ceil(n / cols)
    fig, ax = plt.subplots(figsize=(cols * 2.3, rows * 1.7))
    ax.axis("off")
    for i, label in enumerate(classes):
        r = i // cols
        c = i % cols
        x = c / cols
        y = 1 - (r + 1) / rows
        acc = per_class_accuracy[label]
        ax.text(
            x + 0.02,
            y + 0.04,
            f"{label}\n{acc:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    ax.set_title("ASL Classes with Per-letter Accuracy")
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "asl_accuracy_grid.png", dpi=150)
    plt.close()


def render_mediapipe_overlay(paths: Paths) -> None:
    if not DATA_DIR.exists():
        print(f"Skipping skeleton overlay: missing dataset dir {DATA_DIR}")
        return

    image_paths = sorted(DATA_DIR.glob("*/*"))
    if not image_paths:
        print("Skipping skeleton overlay: no images found.")
        return

    base_options = python.BaseOptions(model_asset_path=str(paths.hand_landmarker_task))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    connections = HAND_CONNECTIONS

    output = None
    for image_path in image_paths[:200]:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
        if not result.hand_landmarks:
            continue
        h, w = img.shape[:2]
        pts = []
        for point in result.hand_landmarks[0]:
            px = int(point.x * w)
            py = int(point.y * h)
            pts.append((px, py))
            cv2.circle(img, (px, py), 4, (0, 255, 0), -1)
        for a, b in connections:
            cv2.line(img, pts[a], pts[b], (255, 0, 0), 2)
        output = img
        break
    detector.close()
    if output is None:
        print("Skipping skeleton overlay: no detectable hand sample found.")
        return
    cv2.imwrite(str(paths.plots_dir / "mediapipe_skeleton_overlay.png"), output)


def main() -> None:
    paths = Paths()
    ensure_output_dirs(paths)

    metrics, reports, best_model = _load_model_results(paths)
    _ = metrics  # loaded for dependency check and future expansion

    x, y, _le = load_landmark_data(paths.landmarks_csv)

    plot_class_distribution(paths)
    per_class_accuracy = plot_per_class_accuracy(paths, reports, best_model)
    plot_mlp_loss_curve(paths)
    plot_rf_feature_importance(paths, x, y)
    plot_before_after_normalization(paths)
    plot_embedding(paths, x, y)
    plot_asl_grid(paths, per_class_accuracy)
    render_mediapipe_overlay(paths)
    print(f"Saved plot outputs to {paths.plots_dir}")


if __name__ == "__main__":
    main()
