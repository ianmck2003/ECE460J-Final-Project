from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    manual_dir = Path(__file__).resolve().parent
    log_path = manual_dir / "manual_trial_log.csv"
    if not log_path.exists():
        print(f"Missing {log_path}. Create it from manual_trial_template.csv first.")
        return

    df = pd.read_csv(log_path)
    if "correct" not in df.columns:
        raise ValueError("manual_trial_log.csv must include a 'correct' column.")

    df["correct"] = df["correct"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    out_dir = manual_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_window = df.groupby("window_size")["correct"].mean().sort_index()
    by_confirm = df.groupby("confirm_frames")["correct"].mean().sort_index()

    by_window.to_csv(out_dir / "accuracy_by_window.csv", header=["accuracy"])
    by_confirm.to_csv(out_dir / "accuracy_by_confirm_frames.csv", header=["accuracy"])

    plt.figure(figsize=(6, 4))
    plt.plot(by_window.index, by_window.values, marker="o")
    plt.ylim(0, 1)
    plt.title("Accuracy vs Smoothing Window Size")
    plt.xlabel("Window size")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_window_size.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(by_confirm.index, by_confirm.values, marker="o")
    plt.ylim(0, 1)
    plt.title("Accuracy vs Confirmation Threshold")
    plt.xlabel("Confirm frames")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_confirm_frames.png", dpi=150)
    plt.close()

    print(f"Saved manual-trial summary outputs to {out_dir}")


if __name__ == "__main__":
    main()
