#!/usr/bin/env python3
"""Score eval_run's staged files with repository TrackEval in the eval venv."""

import csv
import json
from pathlib import Path
import sys


def main():
    stage = Path(sys.argv[1])
    config = json.loads((stage / "config.json").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/TrackEval"))
    import numpy as np
    for name, alias in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, alias)  # The vendored TrackEval predates NumPy 2.
    import trackeval

    evaluator = trackeval.Evaluator({**trackeval.Evaluator.get_default_eval_config(),
        "USE_PARALLEL": False, "PRINT_CONFIG": False, "PLOT_CURVES": False,
        "TIME_PROGRESS": False, "BREAK_ON_ERROR": True, "RETURN_ON_ERROR": False,
        "OUTPUT_SUMMARY": True, "OUTPUT_DETAILED": True, "LOG_ON_ERROR": str(stage / "error.log")})
    dataset = trackeval.datasets.MotChallenge2DBox(config["dataset"])
    _, messages = evaluator.evaluate([dataset], [trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False})])
    if messages[dataset.get_name()] != {"v1": "Success"}:
        raise RuntimeError("TrackEval evaluation failed")
    with (stage / "results/v1/pedestrian_detailed.csv").open(newline="") as source:
        detailed = {row["seq"]: row for row in csv.DictReader(source)}
    rows = []
    for name in [*config["dataset"]["SEQ_INFO"], "COMBINED"]:
        detail = detailed[name]
        row = {"sequence": name, "frames": int(detail["CLR_Frames"])}
        row.update({key: 100 * float(detail[column]) for key, column in (
            ("HOTA", "HOTA___AUC"), ("DetA", "DetA___AUC"), ("AssA", "AssA___AUC"),
            ("IDF1", "IDF1"), ("MOTA", "MOTA"))})
        row["IDSW"] = int(detail["IDSW"])
        rows.append(row)
    (stage / "quality.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
