"""Supported operator directives and the deterministic guardrails applied to
whatever the language model returns.

The LLM emits a FLAT structure (hours/factor/minimum_energy_kwh/max_grid_kwh as
sibling fields) because flat JSON is far more reliable from a model than a
polymorphic nested object. This module converts that flat output into the exact
`structured_adjustment` shape the Problem Statement requires, clamping every
value into a legal range on the way through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SOLAR_REDUCTION = "solar_reduction"
MIN_RESERVE = "minimum_battery_reserve"
NO_CHARGE = "no_charge_window"
NO_DISCHARGE = "no_discharge_window"
MAX_GRID = "max_grid_window"
NO_OP = "no_op"

SUPPORTED = {SOLAR_REDUCTION, MIN_RESERVE, NO_CHARGE, NO_DISCHARGE, MAX_GRID, NO_OP}

NEEDS_HOURS = {SOLAR_REDUCTION, MIN_RESERVE, NO_CHARGE, NO_DISCHARGE, MAX_GRID}


@dataclass
class Directive:
    note_index: int
    directive_type: str = NO_OP
    hours: List[int] = field(default_factory=list)
    factor: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_grid_kwh: Optional[float] = None
    explanation: str = "This note does not affect today's 24-hour energy schedule."

    @property
    def applies(self) -> bool:
        return self.directive_type != NO_OP

    def structured_adjustment(self) -> Optional[Dict[str, Any]]:
        if self.directive_type == NO_OP:
            return None
        if self.directive_type == SOLAR_REDUCTION:
            return {"hours": self.hours, "factor": self.factor}
        if self.directive_type == MIN_RESERVE:
            return {"hours": self.hours, "minimum_energy_kwh": self.minimum_energy_kwh}
        if self.directive_type == MAX_GRID:
            return {"hours": self.hours, "max_grid_kwh": self.max_grid_kwh}
        return {"hours": self.hours}

    def to_response(self) -> Dict[str, Any]:
        return {
            "note_index": self.note_index,
            "applies": self.applies,
            "directive_type": self.directive_type,
            "structured_adjustment": self.structured_adjustment(),
            "explanation": self.explanation,
        }


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _clean_hours(raw: Any) -> List[int]:
    if not isinstance(raw, (list, tuple)):
        return []
    out = set()
    for item in raw:
        if isinstance(item, bool):
            continue
        try:
            h = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= h <= 23:
            out.add(h)
    return sorted(out)


def normalize(raw: Any, note_index: int, capacity_kwh: float) -> Directive:
    """Turn one raw LLM entry into a legal Directive. Anything we cannot trust
    degrades to no_op rather than inventing a constraint."""
    noop = Directive(note_index=note_index)
    if not isinstance(raw, dict):
        return noop

    dtype = raw.get("directive_type")
    dtype = dtype.strip() if isinstance(dtype, str) else ""
    if dtype not in SUPPORTED:
        return noop

    applies = raw.get("applies")
    if dtype == NO_OP or applies is False:
        return noop

    explanation = raw.get("explanation")
    explanation = (
        explanation.strip()[:400]
        if isinstance(explanation, str) and explanation.strip()
        else f"Interpreted as {dtype}."
    )

    hours = _clean_hours(raw.get("hours"))
    if dtype in NEEDS_HOURS and not hours:
        # A window directive with no usable hours cannot be applied to anything.
        return noop

    d = Directive(
        note_index=note_index,
        directive_type=dtype,
        hours=hours,
        explanation=explanation,
    )

    if dtype == SOLAR_REDUCTION:
        f = _as_float(raw.get("factor"))
        if f is None:
            return noop
        if f > 1.0:
            # Model reported the reduction instead of the remainder; also covers
            # percentages expressed as 0..100.
            f = (100.0 - f) / 100.0 if f <= 100.0 else 0.0
        d.factor = min(1.0, max(0.0, f))

    elif dtype == MIN_RESERVE:
        m = _as_float(raw.get("minimum_energy_kwh"))
        if m is None:
            return noop
        d.minimum_energy_kwh = min(capacity_kwh, max(0.0, m))

    elif dtype == MAX_GRID:
        g = _as_float(raw.get("max_grid_kwh"))
        if g is None:
            return noop
        d.max_grid_kwh = max(0.0, g)

    return d


def normalize_all(raw_list: Any, n_notes: int, capacity_kwh: float) -> List[Directive]:
    """Guarantee exactly one entry per note, in note_index order 0..n-1."""
    by_index: Dict[int, Directive] = {}
    if isinstance(raw_list, list):
        for position, item in enumerate(raw_list):
            idx = None
            if isinstance(item, dict):
                try:
                    idx = int(item.get("note_index"))
                except (TypeError, ValueError):
                    idx = None
            if idx is None or not (0 <= idx < n_notes):
                idx = position
            if not (0 <= idx < n_notes) or idx in by_index:
                continue
            by_index[idx] = normalize(item, idx, capacity_kwh)

    return [by_index.get(i) or Directive(note_index=i) for i in range(n_notes)]
