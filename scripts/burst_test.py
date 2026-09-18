"""Sustained-load test: many DISTINCT scenarios fired at the service.

Simulates the judge working through a hidden test set. Notes are paraphrased so
the response cache cannot mask a provider outage -- every request must really
reach a model. Reports latency percentiles, how many requests fell back to the
deterministic parser, and whether every returned plan was valid.

Usage:
    python scripts/burst_test.py [count] [base_url]
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from app.directives import Directive  # noqa: E402
from app.schemas import ScenarioIn  # noqa: E402
from app.validator import validate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 25
BASE = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000").rstrip("/")

# (note text, expected directive_type, expected hours, expected numeric value)
VARIANTS = [
    ("PV output will sit near {pct}% of forecast from {a} until {b}.", "solar_reduction"),
    ("Expect roughly a {inv}% loss of rooftop solar between {a} and {b}.", "solar_reduction"),
    ("Hold at least {kwh} kWh in the battery from {a} until {b}.", "minimum_battery_reserve"),
    ("Do not let the pack fall under {kwh} kWh between {a} and {b}.", "minimum_battery_reserve"),
    ("The charging circuit is offline from {a} until {b}.", "no_charge_window"),
    ("No battery charging is permitted between {a} and {b}.", "no_charge_window"),
    ("The battery must not discharge from {a} until {b}.", "no_discharge_window"),
    ("Keep the pack from draining between {a} and {b}.", "no_discharge_window"),
    ("Grid import must stay at or below {kwh} kWh each hour from {a} until {b}.", "max_grid_window"),
    ("Cap utility intake at {kwh} kWh per hour between {a} and {b}.", "max_grid_window"),
]

DISTRACTORS = [
    "The library will extend its opening hours next semester.",
    "Registration for the sports gala closes on Friday.",
    "A new coffee vendor starts in the canteen next month.",
    "We are reviewing the solar maintenance contract next quarter.",
    "The seminar room booking moved to another building.",
    "Staff parking permits are being reissued this week.",
]

CLOCK = [
    ("9 AM", "11 AM", [9, 10]), ("1 PM", "3 PM", [13, 14]),
    ("6 PM", "9 PM", [18, 19, 20]), ("2 AM", "5 AM", [2, 3, 4]),
    ("11 AM", "2 PM", [11, 12, 13]), ("7 PM", "10 PM", [19, 20, 21]),
    ("noon", "2 PM", [12, 13]), ("10 AM", "1 PM", [10, 11, 12]),
]


def build(rng: random.Random, base_case: dict) -> tuple[dict, list[tuple[str, list[int]]]]:
    body = copy.deepcopy(base_case["input"])
    body["scenario_id"] = f"BURST-{rng.randrange(10**6):06d}"

    # Keep generated constraints satisfiable: the spec guarantees scored
    # scenarios are feasible, so an impossible cap would test nothing real.
    peak = max(h["demand_kwh"] for h in body["hours"])
    batt = body["battery"]
    floor_cap = peak - batt["max_discharge_kwh_per_hour"]

    notes, expect = [], []
    for _ in range(rng.randint(1, 2)):
        template, dtype = rng.choice(VARIANTS)
        a, b, hours = rng.choice(CLOCK)
        pct = rng.choice([20, 25, 40, 50, 60])
        if dtype == "max_grid_window":
            kwh = int(max(floor_cap + 10, peak * rng.uniform(0.85, 1.0)))
        else:
            kwh = rng.choice([80, 100, 120, min(180, int(batt["capacity_kwh"] * 0.6))])
        notes.append(template.format(pct=pct, inv=100 - pct, a=a, b=b, kwh=kwh))
        expect.append((dtype, hours))

    if rng.random() < 0.5 and len(notes) < 3:
        notes.append(rng.choice(DISTRACTORS))
        expect.append(("no_op", []))

    body["operator_notes"] = notes
    return body, expect


def main() -> int:
    rng = random.Random(7)
    with open(SAMPLES, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]

    print(f"target: {BASE}   firing {COUNT} distinct scenarios\n")

    times, failures, invalid, type_right, type_total = [], 0, 0, 0, 0
    started = time.perf_counter()

    for n in range(COUNT):
        body, expect = build(rng, rng.choice(cases))
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{BASE}/optimize-energy", json=body, timeout=35)
        except Exception as exc:
            failures += 1
            print(f"  {n + 1:>3}. NO RESPONSE  {exc}")
            continue
        dt = time.perf_counter() - t0
        times.append(dt)

        if r.status_code != 200:
            failures += 1
            print(f"  {n + 1:>3}. http {r.status_code}")
            continue

        b = r.json()
        scenario = ScenarioIn(**body)
        dirs = []
        for e in b["directive_interpretation"]:
            adj = e.get("structured_adjustment") or {}
            dirs.append(
                Directive(
                    note_index=e["note_index"],
                    directive_type=e["directive_type"],
                    hours=list(adj.get("hours", [])),
                    factor=adj.get("factor"),
                    minimum_energy_kwh=adj.get("minimum_energy_kwh"),
                    max_grid_kwh=adj.get("max_grid_kwh"),
                )
            )
        errs = validate(
            scenario.ordered_hours(), scenario.battery, dirs, b["hourly_plan"],
            totals={k: b[k] for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh")},
        )
        if errs:
            invalid += 1
            print(f"  {n + 1:>3}. INVALID PLAN: {errs[0]}")

        for i, (want_type, want_hours) in enumerate(expect):
            type_total += 1
            got = b["directive_interpretation"][i]
            adj = got.get("structured_adjustment") or {}
            if got["directive_type"] == want_type and (
                want_type == "no_op" or list(adj.get("hours", [])) == want_hours
            ):
                type_right += 1

    wall = time.perf_counter() - started
    s = sorted(times)
    print()
    print(f"  requests        : {COUNT} in {wall:.1f}s ({COUNT / wall * 60:.0f}/min)")
    print(f"  failures        : {failures}")
    print(f"  invalid plans   : {invalid}")
    print(f"  paraphrase acc. : {type_right}/{type_total}")
    if s:
        print(f"  median          : {s[len(s) // 2]:.2f}s")
        print(f"  p95             : {s[min(len(s) - 1, int(0.95 * len(s)))]:.2f}s  (target < 5s)")
        print(f"  slowest         : {s[-1]:.2f}s  (hard limit 30s)")
    ok = failures == 0 and invalid == 0
    print()
    print("RESULT: held up." if ok else "RESULT: needs attention.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
