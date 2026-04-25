from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd


def main() -> None:
    windows = [3, 5, 7, 10]
    confirms = [30, 40, 60]
    prompts = ["HELLO", "WORLD", "SIGN", "CLASS", "TEXAS"]

    rows = []
    for window, confirm, prompt in itertools.product(windows, confirms, prompts):
        rows.append(
            {
                "window_size": window,
                "confirm_frames": confirm,
                "prompt_word": prompt,
                "tester": "",
                "location": "",
                "lighting": "",
                "predicted_text": "",
                "correct": "",
                "error_notes": "",
            }
        )
    out_path = Path(__file__).resolve().parent / "trial_matrix.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} trial configs to {out_path}")


if __name__ == "__main__":
    main()
