from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.append(str(Path(__file__).resolve().parents[2]))

from analysis.common import Paths, ensure_output_dirs, load_landmark_data


def extract_landmarks(result) -> np.ndarray | None:
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
    coords = []
    for point in lm:
        coords += [point.x - wrist_x, point.y - wrist_y, point.z - wrist_z]
    max_val = max(abs(v) for v in coords) or 1.0
    return np.asarray([v / max_val for v in coords], dtype=np.float32)


def benchmark_classifier_only(model, x_val: np.ndarray, loops: int = 20) -> Dict[str, float]:
    timings = []
    for _ in range(loops):
        start = time.perf_counter()
        _ = model.predict(x_val)
        end = time.perf_counter()
        timings.append((end - start) * 1000 / len(x_val))
    return {
        "ms_per_sample_mean": float(np.mean(timings)),
        "ms_per_sample_std": float(np.std(timings)),
    }


def benchmark_end_to_end(paths: Paths, model, max_images: int = 200) -> Dict[str, float]:
    data_dir = Path("./asl_data/asl_alphabet_train/asl_alphabet_train")
    if not data_dir.exists() or not paths.hand_landmarker_task.exists():
        return {"status": "skipped", "reason": "dataset or hand_landmarker.task missing"}

    image_paths = list(data_dir.glob("*/*"))[:max_images]
    if not image_paths:
        return {"status": "skipped", "reason": "no images found"}

    base_options = python.BaseOptions(model_asset_path=str(paths.hand_landmarker_task))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    latencies = []
    used = 0
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        start = time.perf_counter()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
        features = extract_landmarks(result)
        if features is not None:
            _ = model.predict([features])
            used += 1
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)
    detector.close()

    if not latencies:
        return {"status": "skipped", "reason": "no detectable hands in sampled images"}
    return {
        "status": "ok",
        "samples_used": used,
        "ms_per_frame_mean": float(np.mean(latencies)),
        "ms_per_frame_std": float(np.std(latencies)),
    }


def main() -> None:
    paths = Paths()
    ensure_output_dirs(paths)
    model_path = paths.reports_dir / "model_comparison" / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "Missing analysis best_model.pkl. Run analysis/experiments/model_comparison.py first."
        )
    with model_path.open("rb") as f:
        model = pickle.load(f)

    x, y, _ = load_landmark_data(paths.landmarks_csv)
    # Use the tail as proxy validation set for benchmark repeatability.
    x_val = x[int(0.85 * len(x)) :]
    if len(x_val) == 0:
        raise RuntimeError("No validation samples available for benchmark.")

    classifier_stats = benchmark_classifier_only(model, x_val)
    end_to_end_stats = benchmark_end_to_end(paths, model)

    out = {
        "classifier_only": classifier_stats,
        "end_to_end": end_to_end_stats,
    }
    out_path = paths.reports_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Saved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
