"""Independent replay of an hourly plan -- our own copy of the judge.

Used two ways:
  1. as a self-check before we return any response, and
  2. as a test harness against the organizer's own reference plans.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .directives import (
    MAX_GRID,
    MIN_RESERVE,
    NO_CHARGE,
    NO_DISCHARGE,
    SOLAR_REDUCTION,
    Directive,
)

TOL = 0.01


def effective_solar(base_solar: Sequence[float], directives: Sequence[Directive]) -> List[float]:
    eff = [float(s) for s in base_solar]
    for d in directives:
        if d.directive_type == SOLAR_REDUCTION and d.factor is not None:
            for h in d.hours:
                eff[h] *= d.factor
    return eff


def active_reserve(base_min: float, directives: Sequence[Directive]) -> List[float]:
    res = [float(base_min)] * 24
    for d in directives:
        if d.directive_type == MIN_RESERVE and d.minimum_energy_kwh is not None:
            for h in d.hours:
                res[h] = max(res[h], d.minimum_energy_kwh)
    return res


def window_hours(directives: Sequence[Directive], dtype: str) -> set:
    out = set()
    for d in directives:
        if d.directive_type == dtype:
            out.update(d.hours)
    return out


def grid_caps(directives: Sequence[Directive]) -> Dict[int, float]:
    caps: Dict[int, float] = {}
    for d in directives:
        if d.directive_type == MAX_GRID and d.max_grid_kwh is not None:
            for h in d.hours:
                caps[h] = min(caps.get(h, float("inf")), d.max_grid_kwh)
    return caps


def _num(v: Any) -> float:
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError("non-finite")
    return f


def validate(
    hours_in: Sequence[Any],
    battery: Any,
    directives: Sequence[Directive],
    plan: Sequence[Dict[str, Any]],
    totals: Dict[str, float] | None = None,
) -> List[str]:
    """Return a list of violations. Empty list means the plan is valid.

    `hours_in` entries need .demand_kwh/.solar_kwh/.tariff_bdt_per_kwh and .hour;
    `battery` needs the five battery fields. Both work with the Pydantic models
    or with any simple object exposing those attributes.
    """
    errs: List[str] = []

    if len(plan) != 24:
        return [f"hourly_plan must contain 24 entries, found {len(plan)}"]
    if sorted(int(r["hour"]) for r in plan) != list(range(24)):
        return ["hourly_plan must contain each hour 0..23 exactly once"]

    plan = sorted(plan, key=lambda r: int(r["hour"]))
    demand = [float(h.demand_kwh) for h in hours_in]
    tariff = [float(h.tariff_bdt_per_kwh) for h in hours_in]
    eff = effective_solar([h.solar_kwh for h in hours_in], directives)
    reserve = active_reserve(battery.minimum_energy_kwh, directives)
    no_chg = window_hours(directives, NO_CHARGE)
    no_dis = window_hours(directives, NO_DISCHARGE)
    caps = grid_caps(directives)

    energy = float(battery.initial_energy_kwh)
    total_grid = 0.0
    total_cost = 0.0
    peak = 0.0

    for h in range(24):
        row = plan[h]
        try:
            grid = _num(row["grid_kwh"])
            solar = _num(row["solar_used_kwh"])
            amount = _num(row["battery_kwh"])
            after = _num(row["battery_energy_after_kwh"])
        except (KeyError, TypeError, ValueError) as exc:
            errs.append(f"h{h}: missing or non-finite numeric field ({exc})")
            continue

        action = row.get("battery_action")
        if action not in ("charge", "discharge", "idle"):
            errs.append(f"h{h}: battery_action must be charge/discharge/idle, got {action!r}")
            continue

        if grid < -TOL:
            errs.append(f"h{h}: grid_kwh is negative ({grid})")
        if solar < -TOL:
            errs.append(f"h{h}: solar_used_kwh is negative ({solar})")
        if amount < -TOL:
            errs.append(f"h{h}: battery_kwh is negative ({amount})")

        if action == "idle" and abs(amount) > TOL:
            errs.append(f"h{h}: battery_kwh must be 0 when idle, got {amount}")

        charge = amount if action == "charge" else 0.0
        discharge = amount if action == "discharge" else 0.0

        if charge > battery.max_charge_kwh_per_hour + TOL:
            errs.append(
                f"h{h}: charge {charge} exceeds max_charge_kwh_per_hour "
                f"{battery.max_charge_kwh_per_hour}"
            )
        if discharge > battery.max_discharge_kwh_per_hour + TOL:
            errs.append(
                f"h{h}: discharge {discharge} exceeds max_discharge_kwh_per_hour "
                f"{battery.max_discharge_kwh_per_hour}"
            )

        if h in no_chg and charge > TOL:
            errs.append(f"h{h}: charging {charge} during a no_charge_window")
        if h in no_dis and discharge > TOL:
            errs.append(f"h{h}: discharging {discharge} during a no_discharge_window")

        if solar > eff[h] + TOL:
            errs.append(f"h{h}: solar_used {solar} exceeds effective solar {eff[h]:.4f}")

        if h in caps and grid > caps[h] + TOL:
            errs.append(f"h{h}: grid {grid} exceeds max_grid_kwh {caps[h]}")

        expected_after = energy + charge - discharge
        if abs(after - expected_after) > TOL:
            errs.append(
                f"h{h}: battery_energy_after {after} does not follow from "
                f"{energy} {'+' if charge else '-'} {amount} (expected {expected_after})"
            )
        energy = after

        if after < reserve[h] - TOL:
            errs.append(f"h{h}: battery {after} is below the active reserve {reserve[h]}")
        if after > battery.capacity_kwh + TOL:
            errs.append(f"h{h}: battery {after} exceeds capacity {battery.capacity_kwh}")

        lhs = grid + solar + discharge
        rhs = demand[h] + charge
        if abs(lhs - rhs) > TOL:
            errs.append(
                f"h{h}: energy balance broken -- grid+solar+discharge={lhs:.4f} "
                f"but demand+charge={rhs:.4f}"
            )

        total_grid += grid
        total_cost += grid * tariff[h]
        peak = max(peak, grid)

    if abs(energy - battery.initial_energy_kwh) > TOL:
        errs.append(
            f"end of day: battery {energy} must return to initial "
            f"{battery.initial_energy_kwh}"
        )

    if totals is not None:
        for name, computed in (
            ("total_grid_kwh", total_grid),
            ("total_cost_bdt", total_cost),
            ("peak_grid_kwh", peak),
        ):
            reported = totals.get(name)
            if reported is None:
                errs.append(f"totals: {name} is missing")
            elif abs(float(reported) - computed) > TOL:
                errs.append(
                    f"totals: {name} reported {reported} but plan recalculates to {computed:.4f}"
                )

    return errs


def recompute_totals(hours_in: Sequence[Any], plan: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    tariff = {int(h.hour): float(h.tariff_bdt_per_kwh) for h in hours_in}
    total_grid = sum(float(r["grid_kwh"]) for r in plan)
    total_cost = sum(float(r["grid_kwh"]) * tariff[int(r["hour"])] for r in plan)
    peak = max((float(r["grid_kwh"]) for r in plan), default=0.0)
    return {
        "total_grid_kwh": round(total_grid, 4),
        "total_cost_bdt": round(total_cost, 4),
        "peak_grid_kwh": round(peak, 4),
    }
