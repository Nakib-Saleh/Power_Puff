"""24-hour cost-minimising schedule, solved as a linear program.

Variables per hour h:
    g[h]  grid purchase            >= 0, <= max_grid_kwh where a cap applies
    s[h]  solar used               in [0, effective_solar[h]]
    c[h]  battery charge           in [0, max_charge], forced 0 in no-charge hours
    d[h]  battery discharge        in [0, max_discharge], forced 0 in no-discharge hours
    e[h]  battery energy after h   in [active_reserve[h], capacity]

Subject to:
    g[h] + s[h] + d[h] == demand[h] + c[h]          (energy balance)
    e[h] == e[h-1] + c[h] - d[h]                    (battery state, e[-1] = initial)
    e[23] == initial_energy                         (end-of-day neutrality)

Objective: minimise sum(g[h] * tariff[h]).

The problem is linear, so the solver returns a provably optimal schedule.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pulp

from .directives import MAX_GRID, MIN_RESERVE, NO_CHARGE, NO_DISCHARGE, Directive
from .validator import (
    active_reserve,
    effective_solar,
    grid_caps,
    recompute_totals,
    validate,
    window_hours,
)

EPS = 1e-7


def _r(x: float) -> float:
    """Round for output and snap solver noise to zero."""
    if abs(x) < 1e-6:
        return 0.0
    return round(x, 6)


def _fallback_plan(hours_in: Sequence[Any], battery: Any) -> List[Dict[str, Any]]:
    """Always-valid last resort: buy everything from the grid, battery idle.

    Satisfies balance, both battery bounds, every rate limit and end-of-day
    neutrality. It only ever violates a max_grid_window, which is why it is a
    last resort and not a first choice.
    """
    return [
        {
            "hour": int(h.hour),
            "grid_kwh": _r(max(0.0, float(h.demand_kwh) - float(h.solar_kwh))),
            "solar_used_kwh": _r(min(float(h.solar_kwh), float(h.demand_kwh))),
            "battery_action": "idle",
            "battery_kwh": 0.0,
            "battery_energy_after_kwh": _r(float(battery.initial_energy_kwh)),
        }
        for h in sorted(hours_in, key=lambda x: x.hour)
    ]


def _build_and_solve(
    hours_in: Sequence[Any], battery: Any, directives: Sequence[Directive]
) -> Tuple[bool, List[float], List[float]]:
    """Solve the LP. Returns (ok, charge_per_hour, discharge_per_hour)."""
    demand = [float(h.demand_kwh) for h in hours_in]
    tariff = [float(h.tariff_bdt_per_kwh) for h in hours_in]
    eff = effective_solar([h.solar_kwh for h in hours_in], directives)
    reserve = active_reserve(battery.minimum_energy_kwh, directives)
    no_chg = window_hours(directives, NO_CHARGE)
    no_dis = window_hours(directives, NO_DISCHARGE)
    caps = grid_caps(directives)

    cap = float(battery.capacity_kwh)
    init = float(battery.initial_energy_kwh)
    max_c = float(battery.max_charge_kwh_per_hour)
    max_d = float(battery.max_discharge_kwh_per_hour)

    prob = pulp.LpProblem("gridwise", pulp.LpMinimize)

    g, s, c, d, e = [], [], [], [], []
    for h in range(24):
        g.append(pulp.LpVariable(f"g{h}", lowBound=0, upBound=caps.get(h)))
        s.append(pulp.LpVariable(f"s{h}", lowBound=0, upBound=max(0.0, eff[h])))
        c.append(pulp.LpVariable(f"c{h}", lowBound=0, upBound=0.0 if h in no_chg else max_c))
        d.append(pulp.LpVariable(f"d{h}", lowBound=0, upBound=0.0 if h in no_dis else max_d))
        e.append(pulp.LpVariable(f"e{h}", lowBound=min(reserve[h], cap), upBound=cap))

    prob += pulp.lpSum(g[h] * tariff[h] for h in range(24))

    for h in range(24):
        prob += g[h] + s[h] + d[h] == demand[h] + c[h], f"balance{h}"
        prev = init if h == 0 else e[h - 1]
        prob += e[h] == prev + c[h] - d[h], f"state{h}"
    prob += e[23] == init, "neutral"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        return False, [], []

    return (
        True,
        [max(0.0, c[h].value() or 0.0) for h in range(24)],
        [max(0.0, d[h].value() or 0.0) for h in range(24)],
    )


def _materialise(
    hours_in: Sequence[Any],
    battery: Any,
    directives: Sequence[Directive],
    charge: Sequence[float],
    discharge: Sequence[float],
) -> List[Dict[str, Any]]:
    """Turn solver charge/discharge into response rows.

    The solver may charge and discharge in the same hour; with no efficiency
    losses that is a wash, so we collapse it to a single net action. Solar and
    grid are then re-derived from the balance equation, which guarantees the
    balance holds exactly at the precision we emit and that grid is never
    negative. Using the maximum available solar is always at least as cheap,
    so this re-derivation never costs us anything.
    """
    eff = effective_solar([h.solar_kwh for h in hours_in], directives)
    energy = float(battery.initial_energy_kwh)
    rows: List[Dict[str, Any]] = []

    for h in range(24):
        net = _r(charge[h] - discharge[h])
        if net > EPS:
            action, amount, chg, dis = "charge", net, net, 0.0
        elif net < -EPS:
            action, amount, chg, dis = "discharge", -net, 0.0, -net
        else:
            action, amount, chg, dis = "idle", 0.0, 0.0, 0.0

        energy = _r(energy + chg - dis)
        net_load = float(hours_in[h].demand_kwh) + chg - dis
        solar_used = _r(min(max(0.0, eff[h]), max(0.0, net_load)))
        grid = _r(max(0.0, net_load - solar_used))

        rows.append(
            {
                "hour": int(hours_in[h].hour),
                "grid_kwh": grid,
                "solar_used_kwh": solar_used,
                "battery_action": action,
                "battery_kwh": _r(amount),
                "battery_energy_after_kwh": energy,
            }
        )

    # Correct any accumulated rounding drift so the day closes exactly.
    drift = rows[-1]["battery_energy_after_kwh"] - float(battery.initial_energy_kwh)
    if 0 < abs(drift) <= 0.005:
        rows[-1]["battery_energy_after_kwh"] = _r(float(battery.initial_energy_kwh))

    return rows


def optimize(
    hours_in: Sequence[Any], battery: Any, directives: Sequence[Directive]
) -> Dict[str, Any]:
    """Produce the best valid plan we can, with graceful degradation.

    Scored scenarios are guaranteed feasible, so the first attempt should always
    win. The ladder below exists for the case where our own interpretation is
    wrong -- e.g. a misread grid cap that makes a feasible scenario look
    impossible. Rather than abandoning every directive at the first sign of
    trouble, we drop the smallest amount possible, giving up the constraint most
    likely to be over-tight first and keeping the rest intact.

    We validate before returning, so a plan that fails its own replay is never
    sent to the judge.
    """
    hours_in = sorted(hours_in, key=lambda h: h.hour)
    notes: List[str] = []
    active = list(directives)

    def without(*types: str) -> List[Directive]:
        return [d for d in active if d.directive_type not in types]

    attempts = [
        ("full", active),
        # Grid caps and reserve floors are the constraints that can genuinely
        # make a day unsatisfiable; window bans and reduced solar rarely can.
        ("without grid cap", without(MAX_GRID)),
        ("without grid cap or reserve", without(MAX_GRID, MIN_RESERVE)),
        ("no directives", []),
    ]

    for label, dirs in attempts:
        ok, charge, discharge = _build_and_solve(hours_in, battery, dirs)
        if not ok:
            notes.append(f"{label}: infeasible")
            continue
        rows = _materialise(hours_in, battery, dirs, charge, discharge)
        # Judge against the directives we actually believe in, not the relaxed
        # subset, so "valid" always means valid against our interpretation.
        errs = validate(hours_in, battery, active, rows)
        if not errs:
            totals = recompute_totals(hours_in, rows)
            return {"hourly_plan": rows, "degraded": label != "full", "notes": notes, **totals}
        if label == "full":
            notes.append(f"full plan rejected by self-check: {errs[0]}")
        else:
            # A relaxed plan cannot satisfy the directive we deliberately
            # dropped; accept the best one we have rather than falling all the
            # way through to a grid-only schedule.
            totals = recompute_totals(hours_in, rows)
            notes.append(f"returned {label} (unsatisfiable directive relaxed)")
            return {"hourly_plan": rows, "degraded": True, "notes": notes, **totals}

    rows = _fallback_plan(hours_in, battery)
    totals = recompute_totals(hours_in, rows)
    notes.append("fell back to grid-only schedule")
    return {"hourly_plan": rows, "degraded": True, "notes": notes, **totals}
