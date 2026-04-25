from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from analysis.common import Paths, ensure_output_dirs


def main() -> None:
    paths = Paths()
    ensure_output_dirs(paths)

    comparison_dir = paths.reports_dir / "model_comparison"
    src_model = comparison_dir / "best_model.pkl"
    src_encoder = comparison_dir / "label_encoder.pkl"
    summary_path = comparison_dir / "best_model_summary.json"

    if not src_model.exists() or not src_encoder.exists():
        raise FileNotFoundError("Best artifacts missing. Run model_comparison.py first.")

    shutil.copy2(src_model, paths.model_out)
    shutil.copy2(src_encoder, paths.label_encoder_out)

    payload = {
        "copied_model_to": str(paths.model_out),
        "copied_encoder_to": str(paths.label_encoder_out),
        "source_model": str(src_model),
        "source_encoder": str(src_encoder),
        "summary_exists": summary_path.exists(),
    }
    (paths.reports_dir / "promotion_log.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
