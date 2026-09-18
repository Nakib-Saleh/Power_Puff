"""Offline check of the energy engine against the 10 public sample cases.

Runs two things, neither of which needs an API key or a running server:

  1. Replays the ORGANIZER'S OWN reference plans through our validator.
     All 10 must pass -- if one fails, our reading of the rules is wrong.
  2. Feeds the organizer's expected directives straight into our optimizer and
     compares our cost against theirs.

Usage:  python scripts/check_samples.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.directives import Directive  # noqa: E402
from app.optimizer import optimize  # noqa: E402
from app.schemas import ScenarioIn  # noqa: E402
from app.validator import validate  # noqa: E402

SAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
)


def directives_from_expected(expected: List[Dict[str, Any]]) -> List[Directive]:
    out = []
    for entry in expected:
        adj = entry.get("structured_adjustment") or {}
        out.append(
            Directive(
                note_index=entry["note_index"],
                directive_type=entry["directive_type"],
                hours=list(adj.get("hours", [])),
                factor=adj.get("factor"),
                minimum_energy_kwh=adj.get("minimum_energy_kwh"),
                max_grid_kwh=adj.get("max_grid_kwh"),
                explanation=entry.get("explanation", ""),
            )
        )
    return out


def main() -> int:
    with open(SAMPLES, encoding="utf-8") as fh:
        pack = json.load(fh)

    cases = pack["cases"]
    failures = 0

    print("=" * 78)
    print("PART 1  -- do the organizer's own reference plans pass our checker?")
    print("=" * 78)
    for case in cases:
        scenario = ScenarioIn(**case["input"])
        exp = case["expected_output"]
        dirs = directives_from_expected(exp["directive_interpretation"])
        errs = validate(
            scenario.ordered_hours(),
            scenario.battery,
            dirs,
            exp["hourly_plan"],
            totals={
                "total_grid_kwh": exp["total_grid_kwh"],
                "total_cost_bdt": exp["total_cost_bdt"],
                "peak_grid_kwh": exp["peak_grid_kwh"],
            },
        )
        if errs:
            failures += 1
            print(f"  {case['id']}   FAIL")
            for e in errs[:4]:
                print(f"             - {e}")
        else:
            print(f"  {case['id']}   PASS")

    print()
    print("=" * 78)
    print("PART 2  -- does our optimizer match or beat the organizer's cost?")
    print("=" * 78)
    print(f"  {'case':<12}{'valid':<8}{'our cost':>12}{'their cost':>13}{'ratio':>9}{'ms':>7}")
    total_ratio = 0.0
    for case in cases:
        scenario = ScenarioIn(**case["input"])
        exp = case["expected_output"]
        dirs = directives_from_expected(exp["directive_interpretation"])

        t0 = time.perf_counter()
        result = optimize(scenario.ordered_hours(), scenario.battery, dirs)
        ms = (time.perf_counter() - t0) * 1000

        errs = validate(
            scenario.ordered_hours(),
            scenario.battery,
            dirs,
            result["hourly_plan"],
            totals={k: result[k] for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh")},
        )
        ours = result["total_cost_bdt"]
        theirs = float(exp["total_cost_bdt"])
        ratio = (theirs / ours) if ours > 0 else 1.0
        total_ratio += min(1.0, ratio)

        flag = "YES" if not errs else "NO"
        if errs:
            failures += 1
        print(
            f"  {case['id']:<12}{flag:<8}{ours:>12.2f}{theirs:>13.2f}"
            f"{ratio:>9.3f}{ms:>7.0f}"
        )
        for e in errs[:3]:
            print(f"             - {e}")

    score = 10.0 * total_ratio / len(cases)
    print()
    print(f"  projected optimization score: {score:.2f} / 10")
    print()
    if failures:
        print(f"RESULT: {failures} problem(s) found.")
        return 1
    print("RESULT: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
