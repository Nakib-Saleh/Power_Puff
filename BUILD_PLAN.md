# GridWise LLM — Phase-by-Phase Build Plan

**Target:** BUP CSE Fest 2026 Preliminary — LLM-assisted campus energy optimization API.

**Scoring reminder:** 25 (note interpretation) + 25 (obeying the notes) + 10 (schema) + 10 (speed/stability) + 10 (deploy/docker) + 10 (docs) + 10 (cost quality) = 100.

---

## Guiding principles

1. **Validity beats cheapness.** An invalid plan loses points twice (correctness + cost). A slightly expensive valid plan loses almost nothing.
2. **Build the judge before the solver.** We write our own replay/validator first and run the organizer's own reference answers through it. If our validator rejects their answer, our understanding of the spec is wrong — better to find that in Phase 2 than at 10:45 PM.
3. **The LLM is an untrusted intern.** Its output is parsed, validated, clamped, and normalized by plain code before it touches the optimizer.
4. **Never return an error to the judge.** Any internal failure degrades to a valid (if suboptimal) plan. A 500 loses stability points *and* the whole case.
5. **Totals are always recomputed from `hourly_plan`,** never taken from the solver. This makes the "totals must match" check impossible to fail.

---

## Tech stack (decided)

| Piece | Choice | Why |
|---|---|---|
| Web framework | **FastAPI + Uvicorn** | Pydantic gives exact schema validation and correct 400/422 behaviour for free |
| Optimizer | **Linear programming via PuLP (CBC)** | The whole problem is linear, so we get a provably optimal answer in ~20 ms and secure the full 10 cost points. Backup: `scipy.optimize.linprog` (HiGHS) if PuLP misbehaves on Python 3.13 |
| LLM | **Provider-agnostic layer** (see Phase 4) | Keys are a list; code does not care if there are 1 or 10, or which vendor |
| Container | **Docker, python:3.12-slim** | 3.12 not 3.13 — better wheel availability for the solver |
| Tests | **pytest** | Every phase ends with a green test file |

### The optimization model (for reference)

For each hour `h` in 0..23:

- `grid[h] >= 0` — grid purchase
- `solar[h]` in `[0, effective_solar[h]]` — effective_solar already has any solar_reduction factor applied
- `chg[h]` in `[0, max_charge]`, `dis[h]` in `[0, max_discharge]`
- `E[h] = E[h-1] + chg[h] - dis[h]`, with `E[-1] = initial_energy`
- Balance: `grid[h] + solar[h] + dis[h] == demand[h] + chg[h]`
- Bounds: `max(base_min, directive_min[h]) <= E[h] <= capacity`
- End of day: `E[23] == initial_energy`
- no_charge hours: `chg[h] == 0`; no_discharge hours: `dis[h] == 0`
- max_grid hours: `grid[h] <= cap`
- **Objective:** minimize `sum(grid[h] * tariff[h])`

The response needs exactly one action per hour, so we emit `net = chg[h] - dis[h]` as `charge` / `discharge` / `idle`. Cost is unaffected — charging and discharging in the same hour is always a wash when there are no efficiency losses.

---

## PHASE 1 — Skeleton, exact contract, safe stub plan

**Time: ~45 min · Locks in: most of the 10 schema points, plus a floor under the 25 correctness points**

### Build
- `app/schemas.py` — Pydantic models for the request (scenario_id, operator_notes 1–3, hours[24], battery) and the response (scenario_id, directive_interpretation[], hourly_plan[24], total_grid_kwh, total_cost_bdt, peak_grid_kwh, plan_summary).
- `app/main.py` — `GET /health` returns `{"status":"ok"}`; `POST /optimize-energy`.
- **Stub brain:** every note answered as `no_op` (applies=false, adjustment=null). **Stub plan:** buy everything from the grid, battery idle for all 24 hours. This is always valid — the battery ends where it started, the balance holds, nothing is violated.
- Totals computed from the plan.
- Request validation: 24 unique hours 0–23, 1–3 non-empty notes, sane battery numbers, otherwise 400.

### Done when
- `pytest tests/test_api_contract.py` green.
- All 10 public samples return HTTP 200 with a structurally perfect response.

### How you test it (plain language)
Start the service, then hit it with Sample 1. You should get back a reply that:
- has the same scenario id you sent,
- has exactly one answer per note you sent, in the same order,
- has exactly 24 rows, one per hour,
- shows the battery doing nothing all day and all power bought from the grid.

It will be **expensive and it will call every note irrelevant — that is correct for this phase.** We only care that the shape is perfect and nothing crashes. I will give you a one-line command that prints a pass/fail table.

---

## PHASE 2 — The validator (our own judge)

**Time: ~60 min · Protects: the 25 correctness points and all 10 cost points**

### Build
`app/validator.py` — given a scenario, a list of directives, and an hourly plan, check:

- 24 unique hours, all numbers finite and non-negative
- `battery_kwh == 0` when idle; action matches the sign of the battery movement
- charge/discharge within hourly rate limits
- `battery_energy_after` chains correctly hour to hour
- battery between the **active** minimum (base, raised by any reserve directive) and capacity
- `solar_used <= effective_solar` (after any solar_reduction)
- energy balance every hour, within 0.01
- final battery energy equals initial energy
- no_charge / no_discharge / max_grid windows respected
- reported totals match recomputed totals

Returns a list of human-readable violations. Tolerance 0.01 everywhere.

### Also build
`scripts/run_samples.py` — loads the public sample pack and runs **the organizer's own reference plans** through our validator.

### Done when
- All 10 organizer reference plans pass our validator with zero violations.
- Deliberately corrupted plans (shifted battery, over-used solar, broken end-of-day) are caught.

### How you test it (plain language)
Run one command. It prints 10 lines, one per sample case, each saying PASS.

Those 10 plans are the *official answers*, so if our checker says any of them is broken, our checker has misread the rules. Getting 10 PASS means our understanding of the energy rules exactly matches the organizers'. I will also show you 3 lines where I intentionally break a plan and the checker catches it — so you can see it is not just saying PASS to everything.

---

## PHASE 3 — The optimizer (still no LLM)

**Time: ~75 min · Secures: the 10 cost points and the real 25 correctness points**

### Build
- `app/directives.py` — the 5 directive types as dataclasses, plus normalization (sorted unique hours 0–23, factor clamped 0–1, reserve clamped to capacity, cap non-negative).
- `app/optimizer.py` — builds and solves the LP above, converts the solution into the response's hourly rows, rounds cleanly, and re-verifies against `validator.py` before returning.
- Infeasibility handling: if the LP has no solution (should not happen in scored cases), fall back to a relaxed solve and flag it rather than erroring.

### Wire-up
Feed the optimizer the **expected directives taken straight from the sample file**, bypassing the LLM entirely. This isolates the maths from the language understanding.

### Done when
- All 10 cases: plan valid, and our cost is less than or equal to the reference cost (equal is expected, since we solve optimally).
- Solve time under 50 ms per case.

### How you test it (plain language)
One command prints a table like:

```
SAMPLE-01   valid: YES   our cost: 38365   organizer cost: 38365   ratio: 1.000
SAMPLE-02   valid: YES   our cost: 42885   organizer cost: 42885   ratio: 1.000
```

Every row must say **valid: YES** and the ratio must be **1.000 or better**. A ratio above 1.0 means we are paying more than we should. Below 1.0 would mean we found something cheaper than the organizers — possible, and fine, as long as it is still valid.

At the end of this phase the maths is finished and worth roughly 35 points. Everything after this is language and plumbing.

---

## PHASE 4 — LLM interpretation

**Time: ~90 min · Targets: the 25 interpretation points**

### Build
- `app/llm/client.py` — provider abstraction with:
  - **Key rotation**: `GEMINI_API_KEYS=k1,k2,k3` (comma-separated). Round-robin, auto-advance on 429/quota errors, cool-down on exhausted keys.
  - **Provider failover**: primary, then secondary, then tertiary, each with its own key list.
  - **Hard timeout** per attempt (~7 s) so we never approach the judge's 30 s limit.
- `app/llm/prompt.py` — one prompt handling **all 1–3 notes in a single call**, not one call per note. Includes the directive catalogue, the hour convention (end-exclusive), the "factor is what remains" rule, the percentage-of-capacity rule, and 4–5 worked examples written as paraphrases rather than copies of the sample text.
- Force strict JSON output (Gemini `responseSchema` / JSON mode).
- `app/cache.py` — cache key is normalized note text plus battery capacity (capacity matters because "50% of the battery" depends on it). An identical note costs zero LLM calls.

### Done when
- Interpretation accuracy on all 18 public sample notes is **18/18** on applies, directive_type, hours, and numeric value.
- Exactly one entry per note, in order, in every response.

### How you test it (plain language)
One command feeds all 18 real operator sentences from the sample pack to the AI and prints a table:

```
note: "The battery charger will be isolated from 2 AM until 5 AM..."
   expected: no charging, hours 2,3,4
   we said:  no charging, hours 2,3,4        MATCH
```

Everything must say MATCH. If something mismatches, the line shows exactly what went wrong so I can fix the prompt.

---

## PHASE 5 — Guardrails and safe failure

**Time: ~60 min · Targets: the remaining schema points and the 10 stability points**

### Build
`app/guardrails.py`, sitting between the LLM and the optimizer:
- unknown `directive_type` — reject that entry, treat as no_op
- hours — dedupe, drop anything outside 0–23, sort ascending; an empty hour list becomes no_op
- `factor` outside 0–1 — clamp; reserve above capacity — clamp; negative grid cap — drop
- `applies` forced consistent: false **only** for no_op, true for everything else, adjustment null **only** for no_op
- missing, duplicate, or out-of-order note indexes — rebuilt so there is exactly one entry per note in order
- **Failure ladder:** provider 1, then provider 2, then a deterministic regex interpreter (documented in the README as a safe-failure path only, never the primary interpreter), then all-no_op. The service always answers 200 with a valid plan.
- Error contract: malformed JSON gives 400; well-formed but semantically impossible gives 422; an internal problem gives 500 with a clean message, no stack traces, no keys.

### Done when
- `tests/test_robustness.py` green across: broken JSON, missing fields, 23 hours, 25 hours, duplicate hours, 0 notes, 4 notes, empty-string note, negative demand, battery starting below its own minimum, LLM key removed entirely, LLM returning garbage, LLM returning an invented directive type.

### How you test it (plain language)
I will hand you a list of about 12 deliberately broken requests. You fire them one at a time. Expected behaviour:
- Rubbish input gives a polite error message, and the service stays alive.
- Valid input while the AI is unplugged still gives **a complete, valid 24-hour plan**, just possibly treating notes as irrelevant.
- At no point does the service stop responding, and at no point does any reply contain a key, a password, or a Python error trace.

The final check: after all 12 broken requests, send a normal request. It must still work perfectly.

---

## PHASE 6 — Paraphrase hardening

**Time: ~60 min · Directly targets 5 of the 25 interpretation points**

The hidden tests rephrase the same instructions. The sample pack warns about this three separate times.

### Build
`scripts/paraphrase_set.json` — 60+ hand-written notes covering, for each directive type:
- 24-hour clock vs AM/PM vs words ("from one until three")
- "reduce by 80%" vs "drops to 20%" vs "leaves a fifth" vs "cut to one-quarter"
- reserve as kWh, as a percentage of capacity, as "half the pack", as "do not go below"
- caps phrased as "must not exceed", "stay at or below", "limited to", "no more than"
- charge vs discharge wording that is easy to confuse ("charger isolated", "cannot take charge", "no export from the battery", "battery must hold its charge")
- edge windows: crossing midnight, single-hour windows, "all afternoon", "overnight"
- 20+ realistic distractors that must be no_op, including sneaky ones that *mention* energy but change nothing ("we are reviewing the solar contract next quarter")

### Done when
- 95% or better accuracy on the paraphrase set, with every failure understood and either fixed or consciously accepted.

### How you test it (plain language)
One command prints a scorecard:

```
solar reduction wording      12/12
battery reserve wording      10/10
no-charge wording             9/10   <- 1 miss, shown below
...
irrelevant notes             21/21
OVERALL                      61/63  (96.8%)
```

Then it lists the misses in full, so you can judge whether they are realistic phrasings or me being unfair to our own system.

---

## PHASE 7 — Performance

**Time: ~45 min · Targets: 3 latency points and 3 stability points**

- Async I/O for the LLM call; the solver runs in a thread so it never blocks the event loop.
- Cache warmed on startup with the common phrasings.
- `scripts/bench_latency.py` — fires 40 mixed requests, reports p50/p95/max and error rate.
- Tune: if p95 is above 5 s, switch the primary to the fastest available free provider.

### Done when
- p95 at or under 5 s (the full-marks band), zero failures across 40 requests, max comfortably under 30 s.

### How you test it (plain language)
One command fires 40 requests and prints:

```
requests: 40    failures: 0
typical response: 1.4 s
slowest 5%:      2.9 s     <- this number must stay under 5 seconds
slowest single:  3.6 s
```

The "slowest 5%" number is the one the judges measure. Under 5 seconds is full marks.

---

## PHASE 8 — Deployment and Docker fallback

**Time: ~60 min · Targets: the full 10 deployment points**

- `Dockerfile` — python:3.12-slim, non-root, binds `0.0.0.0:8000`, `EXPOSE 8000`, healthcheck, and **no secrets baked in** (keys only via environment variables).
- Push to Docker Hub or GHCR with an exact tag.
- Deploy the live service. Free-tier options, ranked:
  1. **Fly.io** — fast restarts, generous free allowance
  2. **Hugging Face Spaces (Docker)** — free, stays warm, trivially public
  3. **Render free** — easiest, but sleeps after 15 minutes idle and cold-starts slowly, so it *must* be paired with an uptime pinger
- **Keep-alive**: an external cron (cron-job.org or similar) pinging `/health` every 10 minutes through the whole judging window. Cheap insurance against a cold start blowing the 30 s timeout on the judge's first request.

### Done when
- `docker run` from the documented command reaches `/health` on a clean machine.
- The public URL answers `/health` and a full sample from a network that is not yours.
- `git grep` finds zero secrets, and the built image contains zero secrets.

### How you test it (plain language)
On your phone, with wifi off, open the health link in a browser. It should say `ok`. Then I will give you a single command that sends a full sample to the live URL and prints whether the answer is valid.

Separately, we wipe the local container, pull the published image by its exact name, run the one documented command, and confirm it works. This is exactly what a judge does if our live site is down.

---

## PHASE 9 — README and reproducibility

**Time: ~45 min · Targets: the full 10 documentation points — the cheapest points on the board**

The rubric is explicit about what it wants, so the README is written directly against it:
- 3 pts — copy-paste quickstart from a clean machine
- 2 pts — environment variable **names** (never values), model/provider named
- 2 pts — how to run the public samples and what the expected result looks like
- 1 pt — the LLM to guardrails to optimizer architecture explained
- 1 pt — docker pull/run fallback instructions
- 1 pt — dependencies, known limitations, secret handling

Plus `.env.example` with names only, and a credits section for every library used.

### Done when
- Someone who has never seen the project can go from clone to working `/health` to a passing sample request using only the README, with no questions asked.

### How you test it (plain language)
The best test: send the README to a teammate who has not touched the code and ask them to get it running without messaging you. If they have to ask a single question, that question is a missing line in the README.

---

## PHASE 10 — Final rehearsal and video

**Time: ~45 min · Targets: the tiebreaker**

- Full dress rehearsal: fire all 10 samples plus the paraphrase set at the **live public URL**, validate every response, confirm zero failures.
- Run the official pre-submit checklist line by line.
- 3-minute video script: the problem (~30 s), the architecture — note, AI, guardrails, optimizer, validated plan (~90 s), key decisions and how to run it (~50 s). Zero points, but it is the **first** tiebreaker, and with this much automated pass/fail scoring, ties at the cutoff are likely.

---

## LLM provider decision (to finalise at Phase 4)

The code is written so this is a config change, not a rewrite. Current recommendation:

| Slot | Provider | Role |
|---|---|---|
| Primary | Google Gemini 2.5 Flash (free tier) | Best instruction-following per free request; native strict-JSON output |
| Secondary | Groq (Llama 3.3 70B, free tier) | Very low latency — insurance for the p95 target |
| Tertiary | OpenRouter free models / Cerebras | Another independent quota pool |
| Last resort | Our own regex interpreter | Safe-failure only, documented as such |

On the **6-accounts-in-rotation** idea: the multi-key rotation is worth building regardless, and the code supports any number of keys. Two things worth weighing before investing in it. First, spreading across *different providers* (each with one legitimate account) gives the same throughput without the risk that a key gets disabled mid-judging for terms-of-service reasons. Second, the judge probably sends on the order of 20–60 requests in total, and with caching that sits well inside a single free key. The heavy consumption is **our own testing**, and that is solved by the cache plus a local mock. So rotation is mostly a development convenience and a reliability backstop, not a scoring necessity.

---

## Time budget

| Phase | Est. | Running total |
|---|---|---|
| 1 Skeleton | 45 m | 0:45 |
| 2 Validator | 60 m | 1:45 |
| 3 Optimizer | 75 m | 3:00 |
| 4 LLM | 90 m | 4:30 |
| 5 Guardrails | 60 m | 5:30 |
| 6 Paraphrases | 60 m | 6:30 |
| 7 Performance | 45 m | 7:15 |
| 8 Deploy | 60 m | 8:15 |
| 9 README | 45 m | 9:00 |
| 10 Rehearsal | 45 m | 9:45 |

That is roughly 10 hours of build for a 4-hour contest window, which is the point of building it now as a rehearsal. On contest day the order changes to **1, 8, 3, 2, 4, 5, 9, 7, 10** — deploying a working stub inside the first 45 minutes, so deployment risk is retired early and there is always something submittable on the board.

Note: the rules require a fresh repository created after the question is revealed, so this build is the rehearsal, not the submission artifact.
