"""Concurrency test: many DISTINCT scenarios fired at the same time.

The sequential burst test spreads token usage over a minute. A judge harness
running requests in parallel compresses that into seconds, which is a much
harsher test of the rate-limit strategy. This fires N requests with a given
concurrency level and reports what actually happened.

Usage:
    python scripts/parallel_test.py [count] [concurrency] [base_url]
"""

from __future__ import annotations

import asyncio
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
from scripts.burst_test import build  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CONCURRENCY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BASE = (sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8000").rstrip("/")


async def one(client, sem, body, expect, results):
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{BASE}/optimize-energy", json=body, timeout=35)
        except Exception as exc:
            results.append({"ok": False, "why": f"no response: {exc}", "t": None})
            return
        dt = time.perf_counter() - t0

        if r.status_code != 200:
            results.append({"ok": False, "why": f"http {r.status_code}", "t": dt})
            return

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

        right = 0
        for i, (want_type, want_hours) in enumerate(expect):
            got = b["directive_interpretation"][i]
            adj = got.get("structured_adjustment") or {}
            if got["directive_type"] == want_type and (
                want_type == "no_op" or list(adj.get("hours", [])) == want_hours
            ):
                right += 1

        results.append(
            {
                "ok": not errs,
                "why": errs[0] if errs else "",
                "t": dt,
                "right": right,
                "total": len(expect),
            }
        )


async def main() -> int:
    rng = random.Random(int(os.getenv("SEED", "0")) or time.time_ns())
    with open(SAMPLES, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]

    jobs = [build(rng, rng.choice(cases)) for _ in range(COUNT)]
    results: list = []
    sem = asyncio.Semaphore(CONCURRENCY)

    print(f"target: {BASE}")
    print(f"firing {COUNT} distinct scenarios, {CONCURRENCY} at a time\n")

    started = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, sem, b, e, results) for b, e in jobs))
    wall = time.perf_counter() - started

    times = sorted(r["t"] for r in results if r["t"] is not None)
    failures = [r for r in results if not r["ok"]]
    right = sum(r.get("right", 0) for r in results)
    total = sum(r.get("total", 0) for r in results)

    for r in failures[:8]:
        print(f"  FAILED: {r['why']}")

    print()
    print(f"  requests        : {COUNT} in {wall:.1f}s  ({COUNT / wall * 60:.0f}/min effective)")
    print(f"  concurrency     : {CONCURRENCY}")
    print(f"  failures        : {len(failures)}")
    print(f"  interpretation  : {right}/{total}")
    if times:
        print(f"  median          : {times[len(times) // 2]:.2f}s")
        print(f"  p95             : {times[min(len(times) - 1, int(0.95 * len(times)))]:.2f}s"
              f"  (target < 5s)")
        print(f"  slowest         : {times[-1]:.2f}s  (hard limit 30s)")
    print()
    ok = not failures and right == total
    print("RESULT: held up under parallel load." if ok else "RESULT: needs attention.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
