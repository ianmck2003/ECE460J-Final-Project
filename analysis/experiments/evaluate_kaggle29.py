from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import List

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).resolve().parents[2]))

from analysis.common import Paths, ensure_output_dirs


def extract_landmarks(result):
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
    coords = []
    for point in lm:
        coords += [point.x - wrist_x, point.y - wrist_y, point.z - wrist_z]
    max_val = max(abs(v) for v in coords) or 1.0
    return [v / max_val for v in coords]


def infer_label_from_filename(path: Path) -> str:
    # Default convention: filenames start with label, e.g. A_01.jpg, space_3.png, del_2.jpg.
    stem = path.stem
    token = stem.split("_")[0].split("-")[0]
    return token


def main() -> None:
    paths = Paths()
    ensure_output_dirs(paths)
    summary_path = paths.reports_dir / "kaggle29_summary.json"
    kaggle_dir = paths.kaggle_test_dir
    if not kaggle_dir.exists():
        summary_path.write_text(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": f"directory_not_found:{kaggle_dir}",
                    "num_images_total": 0,
                    "num_images_scored": 0,
                    "accuracy": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Kaggle test directory not found at {kaggle_dir}, skipping.")
        return

    model_path = paths.reports_dir / "model_comparison" / "best_model.pkl"
    encoder_path = paths.reports_dir / "model_comparison" / "label_encoder.pkl"
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("Run model_comparison.py before evaluate_kaggle29.py")

    with model_path.open("rb") as f:
        model = pickle.load(f)
    with encoder_path.open("rb") as f:
        le = pickle.load(f)

    if not paths.hand_landmarker_task.exists():
        raise FileNotFoundError(f"Missing hand landmarker task at {paths.hand_landmarker_task}")

    image_paths = sorted(
        [p for p in kaggle_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        print(f"No test images found in {kaggle_dir}, skipping.")
        return

    base_options = python.BaseOptions(model_asset_path=str(paths.hand_landmarker_task))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    rows: List[dict] = []
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            rows.append(
                {
                    "image": image_path.name,
                    "expected": infer_label_from_filename(image_path),
                    "predicted": None,
                    "status": "unreadable",
                }
            )
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
        features = extract_landmarks(result)
        expected = infer_label_from_filename(image_path)
        if features is None:
            rows.append(
                {"image": image_path.name, "expected": expected, "predicted": None, "status": "no_hand_detected"}
            )
            continue
        pred_idx = model.predict([features])[0]
        predicted = le.inverse_transform([pred_idx])[0]
        rows.append(
            {
                "image": image_path.name,
                "expected": expected,
                "predicted": predicted,
                "correct": expected == predicted,
                "status": "ok",
            }
        )
    detector.close()

    out_df = pd.DataFrame(rows)
    out_path = paths.reports_dir / "kaggle29_results.csv"
    out_df.to_csv(out_path, index=False)

    valid = out_df[out_df["status"] == "ok"]
    if len(valid) > 0:
        acc = accuracy_score(valid["expected"], valid["predicted"])
        summary = {
            "num_images_total": int(len(out_df)),
            "num_images_scored": int(len(valid)),
            "accuracy": float(acc),
        }
    else:
        summary = {
            "num_images_total": int(len(out_df)),
            "num_images_scored": 0,
            "accuracy": None,
        }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)
    print(f"Saved Kaggle evaluation to {out_path}")


if __name__ == "__main__":
    main()
