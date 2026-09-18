"""End-to-end test against a running service (local or deployed).

Checks, for all 10 public samples:
  * HTTP status and response shape
  * one interpretation entry per note, in order, with legal applies/adjustment
  * interpretation vs the organizer's expected answer
  * plan validity replayed against the EXPECTED (ground-truth) directives
  * our cost vs the organizer's optimal cost
  * response time

Usage:
    python scripts/test_api.py                        # http://127.0.0.1:8000
    python scripts/test_api.py https://your-app.com
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from app.schemas import ScenarioIn  # noqa: E402
from app.validator import validate  # noqa: E402
from scripts.check_samples import directives_from_expected  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")

REQUIRED_TOP = [
    "scenario_id", "directive_interpretation", "hourly_plan",
    "total_grid_kwh", "total_cost_bdt", "peak_grid_kwh", "plan_summary",
]
REQUIRED_DI = ["note_index", "applies", "directive_type", "structured_adjustment", "explanation"]
REQUIRED_HP = ["hour", "grid_kwh", "solar_used_kwh", "battery_action",
               "battery_kwh", "battery_energy_after_kwh"]


def shape_errors(body: Dict[str, Any], scenario_id: str, n_notes: int) -> List[str]:
    errs = []
    for f in REQUIRED_TOP:
        if f not in body:
            errs.append(f"missing top-level field {f}")
    if errs:
        return errs
    if body["scenario_id"] != scenario_id:
        errs.append("scenario_id was not echoed back")

    di = body["directive_interpretation"]
    if not isinstance(di, list) or len(di) != n_notes:
        errs.append(f"expected {n_notes} interpretation entries, got {len(di) if isinstance(di, list) else '?'}")
    else:
        for i, entry in enumerate(di):
            for f in REQUIRED_DI:
                if f not in entry:
                    errs.append(f"interpretation[{i}] missing {f}")
            if entry.get("note_index") != i:
                errs.append(f"interpretation[{i}] has note_index {entry.get('note_index')} (out of order)")
            applies, dtype = entry.get("applies"), entry.get("directive_type")
            adj = entry.get("structured_adjustment")
            if dtype == "no_op" and (applies is not False or adj is not None):
                errs.append(f"interpretation[{i}] no_op must have applies=false and null adjustment")
            if dtype != "no_op" and applies is not True:
                errs.append(f"interpretation[{i}] non-no_op must have applies=true")
            if isinstance(adj, dict) and "hours" in adj:
                hrs = adj["hours"]
                if hrs != sorted(set(hrs)) or any(not (0 <= h <= 23) for h in hrs):
                    errs.append(f"interpretation[{i}] hours must be unique 0-23 ascending, got {hrs}")

    hp = body["hourly_plan"]
    if not isinstance(hp, list) or len(hp) != 24:
        errs.append("hourly_plan must have 24 entries")
    else:
        for i, row in enumerate(hp):
            for f in REQUIRED_HP:
                if f not in row:
                    errs.append(f"hourly_plan[{i}] missing {f}")
                    break
    return errs


def compare_directive(got: Dict[str, Any], want: Dict[str, Any]) -> str:
    if got.get("directive_type") != want.get("directive_type"):
        return f"type {got.get('directive_type')} != {want.get('directive_type')}"
    g, w = got.get("structured_adjustment"), want.get("structured_adjustment")
    if w is None:
        return "" if g is None else "expected null adjustment"
    if not isinstance(g, dict):
        return "adjustment missing"
    if list(g.get("hours", [])) != list(w.get("hours", [])):
        return f"hours {g.get('hours')} != {w.get('hours')}"
    for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if key in w:
            if key not in g:
                return f"missing {key}"
            if abs(float(g[key]) - float(w[key])) > 0.01:
                return f"{key} {g[key]} != {w[key]}"
    return ""


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    with open(SAMPLES, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]

    print(f"target: {base}\n")
    try:
        h = httpx.get(f"{base}/health", timeout=30)
        print(f"GET /health -> {h.status_code} {h.text.strip()[:60]}")
        if h.status_code != 200 or h.json().get("status") != "ok":
            print("HEALTH CHECK FAILED")
            return 1
    except Exception as exc:
        print(f"cannot reach {base}: {exc}")
        return 1

    print()
    print(f"  {'case':<12}{'http':<6}{'shape':<8}{'notes':<9}{'plan':<7}{'cost ratio':>11}{'sec':>7}")
    print("  " + "-" * 62)

    notes_right = notes_total = 0
    bad = 0
    latencies = []
    ratios = []

    for case in cases:
        scenario = ScenarioIn(**case["input"])
        exp = case["expected_output"]

        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{base}/optimize-energy", json=case["input"], timeout=35)
        except Exception as exc:
            print(f"  {case['id']:<12}ERROR  {exc}")
            bad += 1
            continue
        dt = time.perf_counter() - t0
        latencies.append(dt)

        if r.status_code != 200:
            print(f"  {case['id']:<12}{r.status_code:<6}{r.text[:60]}")
            bad += 1
            continue

        body = r.json()
        serrs = shape_errors(body, case["input"]["scenario_id"], len(case["input"]["operator_notes"]))

        details = []
        ok_notes = 0
        want_all = exp["directive_interpretation"]
        got_all = body.get("directive_interpretation", [])
        for i, want in enumerate(want_all):
            notes_total += 1
            got = got_all[i] if i < len(got_all) else {}
            diff = compare_directive(got, want)
            if diff:
                details.append(f"note {i}: {diff}")
            else:
                ok_notes += 1
                notes_right += 1

        # Replay OUR plan against the GROUND-TRUTH directives, as the judge does.
        truth = directives_from_expected(want_all)
        perrs = validate(
            scenario.ordered_hours(), scenario.battery, truth, body["hourly_plan"],
            totals={k: body[k] for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh")},
        )

        ours = float(body["total_cost_bdt"])
        theirs = float(exp["total_cost_bdt"])
        ratio = min(1.0, theirs / ours) if ours > 0 else 1.0
        ratios.append(ratio if not perrs else 0.0)

        print(
            f"  {case['id']:<12}{r.status_code:<6}"
            f"{('OK' if not serrs else 'BAD'):<8}"
            f"{f'{ok_notes}/{len(want_all)}':<9}"
            f"{('VALID' if not perrs else 'INVALID'):<7}"
            f"{ratio:>11.3f}{dt:>7.2f}"
        )
        if serrs or perrs or details:
            bad += 1
        for line in (serrs[:2] + details[:3] + perrs[:3]):
            print(f"               - {line}")

    print()
    lat = sorted(latencies)
    if lat:
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        print(f"  interpretation accuracy : {notes_right}/{notes_total}")
        print(f"  optimization score      : {10 * sum(ratios) / len(ratios):.2f} / 10")
        print(f"  slowest 5% (p95)        : {p95:.2f} s   (needs to stay under 5s)")
        print(f"  slowest single request  : {lat[-1]:.2f} s   (hard limit 30s)")
    print()
    print("RESULT: all good." if bad == 0 else f"RESULT: {bad} case(s) need attention.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
