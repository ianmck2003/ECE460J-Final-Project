from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


@dataclass(frozen=True)
class Paths:
    repo_root: Path = Path(__file__).resolve().parents[1]
    analysis_root: Path = repo_root / "analysis"
    reports_dir: Path = analysis_root / "reports"
    plots_dir: Path = analysis_root / "plots" / "outputs"
    experiments_dir: Path = analysis_root / "experiments"
    benchmarks_dir: Path = analysis_root / "benchmarks"
    manual_trials_dir: Path = analysis_root / "manual_trials"
    final_submission_dir: Path = repo_root / "final-submission"
    landmarks_csv: Path = final_submission_dir / "landmarks.csv"
    model_out: Path = final_submission_dir / "model.pkl"
    label_encoder_out: Path = final_submission_dir / "label_encoder.pkl"
    hand_landmarker_task: Path = final_submission_dir / "hand_landmarker.task"
    kaggle_test_dir: Path = repo_root / "kaggle-test-images"


RANDOM_STATE = 42
TEST_SIZE = 0.15


def ensure_output_dirs(paths: Paths) -> None:
    for directory in (
        paths.reports_dir,
        paths.plots_dir,
        paths.manual_trials_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_landmark_data(landmarks_csv: Path) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    if not landmarks_csv.exists():
        raise FileNotFoundError(f"Could not find landmarks CSV at {landmarks_csv}")

    df = pd.read_csv(landmarks_csv)
    if "label" not in df.columns:
        raise ValueError(f"Expected 'label' column in {landmarks_csv}")

    x = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values

    # Mirror augmentation for left-hand robustness.
    x_mirrored = x.copy()
    x_mirrored[:, 0::3] *= -1
    x_aug = np.concatenate([x, x_mirrored], axis=0)
    y_aug = np.concatenate([y, y], axis=0)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_aug)
    return x_aug, y_encoded, label_encoder


def split_data(x: np.ndarray, y_encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        x,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )


def make_models() -> Dict[str, ClassifierMixin]:
    return {
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=False,
            random_state=RANDOM_STATE,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        ),
    }
