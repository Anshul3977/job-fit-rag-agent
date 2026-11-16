"""Fail CI when the labeled fit eval regresses.

Usage: EVAL_OUT_DIR=/tmp/x python eval/run_eval.py && python eval/gate.py /tmp/x/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FLOORS = json.loads((Path(__file__).resolve().parent / "floors.json").read_text(encoding="utf-8"))["fit"]


def check(metrics: dict) -> list[str]:
    problems = []
    if metrics["failures"] > FLOORS["failures_max"]:
        problems.append(f"failures {metrics['failures']} > {FLOORS['failures_max']}")
    for key, floor in FLOORS.items():
        if not key.endswith("_min"):
            continue
        name = key[: -len("_min")]
        value = metrics.get(name)
        if value is None or value + 1e-9 < floor:
            problems.append(f"{name} {value} < {floor}")
    return problems


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results.json")
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    problems = check(metrics)
    for key in (
        "verdict_accuracy",
        "citation_validity",
        "unsupported_skill_rejection_rate",
        "skill_hit_rate",
    ):
        print(f"{key}: {metrics.get(key)}")
    if problems:
        print("Eval regression:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("Eval gate: OK")


if __name__ == "__main__":
    main()
