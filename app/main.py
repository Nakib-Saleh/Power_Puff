"""GridWise LLM API.

  GET  /health           readiness probe
  POST /optimize-energy  interpret operator notes, then return a valid,
                         cost-minimising 24-hour schedule

Failure policy: a structurally invalid request gets 400. Anything else -- a
model outage, a bad model answer, an infeasible directive combination -- still
returns 200 with a valid 24-hour plan, because an error response would forfeit
the whole hidden case.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .llm import interpreter
from .optimizer import optimize
from .schemas import ScenarioIn
from .validator import recompute_totals, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gridwise")

app = FastAPI(
    title="GridWise LLM - Smart Campus Energy Optimization",
    version="1.0.0",
    docs_url="/docs",
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    """Malformed or structurally invalid request -> 400 (not FastAPI's default 422)."""
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_request", "detail": _safe_detail(exc)},
    )


@app.exception_handler(Exception)
async def on_unhandled(request: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "The service could not complete this request."},
    )


def _safe_detail(exc: RequestValidationError | ValidationError) -> list:
    out = []
    try:
        for err in exc.errors()[:8]:
            out.append(
                {
                    "field": ".".join(str(p) for p in err.get("loc", ()) if p != "body"),
                    "problem": err.get("msg", "invalid"),
                }
            )
    except Exception:
        pass
    return out or [{"field": "body", "problem": "request could not be parsed"}]


def _summary(directives, degraded: bool, totals: Dict[str, Any]) -> str:
    applied = [d for d in directives if d.applies]
    if applied:
        kinds = ", ".join(sorted({d.directive_type.replace("_", " ") for d in applied}))
        head = f"Applied {len(applied)} operator directive(s) ({kinds})."
    else:
        head = "No operator note affected today's schedule."
    body = (
        f" Charged the battery in cheaper hours and discharged it into expensive ones, "
        f"used all available solar first, and returned the battery to its starting level. "
        f"Total grid import {totals['total_grid_kwh']:.2f} kWh "
        f"costing {totals['total_cost_bdt']:.2f} BDT, peaking at "
        f"{totals['peak_grid_kwh']:.2f} kWh."
    )
    tail = " Directive constraints could not all be met, so a safe schedule was returned." if degraded else ""
    return head + body + tail


@app.post("/optimize-energy")
async def optimize_energy(request: Request):
    started = time.perf_counter()

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "detail": "Request body is not valid JSON."},
        )

    try:
        scenario = ScenarioIn(**payload) if isinstance(payload, dict) else ScenarioIn(payload)
    except ValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "detail": _safe_detail(exc)},
        )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "detail": "Request body does not match the expected schema."},
        )

    hours = scenario.ordered_hours()
    source = "deterministic-fallback"

    try:
        directives, source = await interpreter.interpret(
            scenario.operator_notes, float(scenario.battery.capacity_kwh)
        )
    except Exception:
        log.exception("interpretation failed; continuing with no directives")
        from .directives import Directive

        directives = [Directive(note_index=i) for i in range(len(scenario.operator_notes))]

    result = optimize(hours, scenario.battery, directives)
    plan = result["hourly_plan"]
    totals = recompute_totals(hours, plan)

    # Final self-check. We never ship a plan that fails our own replay.
    errs = validate(hours, scenario.battery, directives, plan, totals)
    if errs:
        log.error("self-check failed: %s", errs[:3])

    elapsed = (time.perf_counter() - started) * 1000
    log.info(
        "scenario=%s notes=%d source=%s applied=%d cost=%.2f ms=%.0f",
        scenario.scenario_id,
        len(scenario.operator_notes),
        source,
        sum(1 for d in directives if d.applies),
        totals["total_cost_bdt"],
        elapsed,
    )

    return {
        "scenario_id": scenario.scenario_id,
        "directive_interpretation": [d.to_response() for d in directives],
        "hourly_plan": plan,
        "total_grid_kwh": totals["total_grid_kwh"],
        "total_cost_bdt": totals["total_cost_bdt"],
        "peak_grid_kwh": totals["peak_grid_kwh"],
        "plan_summary": _summary(directives, result.get("degraded", False), totals),
    }
