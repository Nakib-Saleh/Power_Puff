"""Benchmark candidate interpreter models against the 18 public sample notes.

Runs the REAL system prompt and guardrail pipeline for each candidate model and
scores it on directive_type, hours and numeric value against organizer ground
truth. Use it to choose OPENAI_COMPAT_MODEL / GEMINI_MODEL.

Usage:
    python scripts/pick_model.py                       # all default candidates
    python scripts/pick_model.py openai/gpt-oss-120b   # just these
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")

DEFAULT_CANDIDATES = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]


def load_env() -> None:
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def expected_of(entry):
    adj = entry.get("structured_adjustment") or {}
    return (
        entry["directive_type"],
        tuple(adj.get("hours", ())),
        adj.get("factor"),
        adj.get("minimum_energy_kwh"),
        adj.get("max_grid_kwh"),
    )


def got_of(d):
    adj = d.structured_adjustment() or {}
    return (
        d.directive_type,
        tuple(adj.get("hours", ())),
        adj.get("factor"),
        adj.get("minimum_energy_kwh"),
        adj.get("max_grid_kwh"),
    )


def same(a, b) -> bool:
    if a[0] != b[0] or a[1] != b[1]:
        return False
    for x, y in zip(a[2:], b[2:]):
        if (x is None) != (y is None):
            return False
        if x is not None and abs(float(x) - float(y)) > 0.01:
            return False
    return True


async def run_model(model: str, cases) -> None:
    os.environ["OPENAI_COMPAT_MODEL"] = model
    # Force the Groq path only, so Gemini quota is untouched during benchmarking.
    saved = os.environ.pop("GEMINI_API_KEYS", None)
    os.environ.pop("GEMINI_API_KEY", None)

    import importlib

    import app.llm as llm_mod

    importlib.reload(llm_mod)
    interp = llm_mod.Interpreter()

    right = total = 0
    times = []
    misses = []

    for case in cases:
        notes = case["input"]["operator_notes"]
        cap = float(case["input"]["battery"]["capacity_kwh"])
        want = case["expected_output"]["directive_interpretation"]

        t0 = time.perf_counter()
        try:
            dirs, source = await interp.interpret(notes, cap)
        except Exception as exc:
            misses.append(f"{case['id']}: EXCEPTION {exc}")
            total += len(want)
            continue
        times.append(time.perf_counter() - t0)

        if source not in ("openai-compatible", "cache"):
            misses.append(f"{case['id']}: served by {source}, not the model")

        for i, w in enumerate(want):
            total += 1
            if i < len(dirs) and same(got_of(dirs[i]), expected_of(w)):
                right += 1
            else:
                g = got_of(dirs[i]) if i < len(dirs) else ("<missing>",)
                misses.append(
                    f"{case['id']} note {i}: got {g[0]} {g[1] if len(g) > 1 else ''} "
                    f"| want {expected_of(w)[0]} {expected_of(w)[1]}"
                )

    if saved:
        os.environ["GEMINI_API_KEYS"] = saved

    p95 = sorted(times)[min(len(times) - 1, int(0.95 * len(times)))] if times else 0
    print(f"\n  {model}")
    print(f"    accuracy : {right}/{total}")
    print(f"    median   : {statistics.median(times):.2f}s" if times else "    median   : n/a")
    print(f"    p95      : {p95:.2f}s")
    for m in misses[:6]:
        print(f"    - {m}")


async def main() -> None:
    load_env()
    if not os.getenv("OPENAI_COMPAT_API_KEYS") and not os.getenv("OPENAI_COMPAT_API_KEY"):
        print("No Groq key found in .env (OPENAI_COMPAT_API_KEYS).")
        return

    with open(SAMPLES, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]

    candidates = sys.argv[1:] or DEFAULT_CANDIDATES
    print("Benchmarking against all 18 public sample notes")
    print("=" * 60)
    for model in candidates:
        await run_model(model, cases)


if __name__ == "__main__":
    asyncio.run(main())
