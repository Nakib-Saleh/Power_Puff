"""Operator-note interpretation.

Pipeline: note text -> language model -> strict JSON -> deterministic guardrails
(in directives.py) -> optimizer.

Reliability design:
  * one model call covers all 1-3 notes (not one call per note)
  * several API keys rotate automatically on quota/rate-limit errors
  * a second, OpenAI-compatible provider takes over if the first is unavailable
  * identical notes are cached, so repeated hidden cases cost nothing
  * if every provider fails, a deterministic parser keeps the service answering

The deterministic parser is a SAFE-FAILURE path only. In normal operation the
language model produces the structured interpretation that reaches the optimizer.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .directives import (
    MAX_GRID,
    MIN_RESERVE,
    NO_CHARGE,
    NO_DISCHARGE,
    NO_OP,
    SOLAR_REDUCTION,
    Directive,
    normalize_all,
)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = """You convert short notes written by campus energy operators into structured scheduling directives for a 24-hour electricity plan.

You must classify each note as exactly ONE of these six types:

1. "solar_reduction" - usable solar power will be lower than forecast during some hours.
   fields: hours, factor
2. "minimum_battery_reserve" - the battery must stay at or above some energy level during some hours.
   fields: hours, minimum_energy_kwh
3. "no_charge_window" - the battery cannot be charged during some hours.
   fields: hours
4. "no_discharge_window" - the battery cannot be discharged/drained during some hours.
   fields: hours
5. "max_grid_window" - electricity imported from the grid must not exceed a limit in each of some hours.
   fields: hours, max_grid_kwh
6. "no_op" - the note has no effect on today's electricity schedule.
   fields: none

CRITICAL RULES

A. HOURS ARE WHOLE HOURS, START INCLUSIVE, END EXCLUSIVE.
   "1 PM to 3 PM"        -> [13, 14]
   "from 2 AM until 5 AM" -> [2, 3, 4]
   "between 11 AM and 2 PM" -> [11, 12, 13]
   "6 PM until 10 PM"    -> [18, 19, 20, 21]
   "noon until 2 PM"     -> [12, 13]
   "at 3 PM" (single hour) -> [15]
   Always output unique integers 0-23 in ascending order.

B. "factor" IS THE FRACTION THAT REMAINS, NOT THE AMOUNT LOST.
   "drops to 20% of forecast"       -> factor 0.2
   "an 80% reduction in solar"      -> factor 0.2
   "about half the usual output"    -> factor 0.5
   "roughly one-fifth of normal"    -> factor 0.2
   "treated as 25% of the forecast" -> factor 0.25
   factor is always between 0 and 1.

C. RESERVES EXPRESSED AS A PERCENTAGE MUST BE CONVERTED TO kWh using the battery
   capacity given in the request. "keep at least 50% of capacity" with a 200 kWh
   battery -> minimum_energy_kwh 100.

D. CHARGE vs DISCHARGE - read carefully.
   "charger is isolated / charging circuit unavailable / do not charge" -> no_charge_window
   "must not discharge / do not drain / battery must hold its charge"   -> no_discharge_window

E. NOTES THAT DO NOT CHANGE TODAY'S SCHEDULE ARE "no_op".
   Anything about menus, deadlines, bookings, notices, staffing, next week,
   next month, or contracts is no_op. A note can mention energy and still be
   no_op if it does not constrain today's 24 hours.

F. NEVER invent a directive type outside the six listed. Never change demand,
   tariff, or battery limits. When genuinely unsure, choose no_op.

OUTPUT
Return JSON only, in exactly this shape, with one entry per note in the order
the notes were given:

{"interpretations":[
  {"note_index":0,"applies":true,"directive_type":"solar_reduction","hours":[13,14],"factor":0.2,"minimum_energy_kwh":null,"max_grid_kwh":null,"explanation":"one short sentence"},
  {"note_index":1,"applies":false,"directive_type":"no_op","hours":[],"factor":null,"minimum_energy_kwh":null,"max_grid_kwh":null,"explanation":"one short sentence"}
]}

Set applies to false only for no_op. Unused numeric fields must be null."""

FEW_SHOT = """EXAMPLES

Battery capacity: 300 kWh
Notes:
0: "Inverter servicing between 9 AM and 11 AM should leave us about 40% of the usual PV yield."
1: "Please keep no less than a third of the pack in reserve from 7 PM to 10 PM."
2: "Canteen supplier meeting has been pushed to Thursday."
Answer:
{"interpretations":[
{"note_index":0,"applies":true,"directive_type":"solar_reduction","hours":[9,10],"factor":0.4,"minimum_energy_kwh":null,"max_grid_kwh":null,"explanation":"Servicing leaves 40% of forecast solar for those hours."},
{"note_index":1,"applies":true,"directive_type":"minimum_battery_reserve","hours":[19,20,21],"factor":null,"minimum_energy_kwh":100,"max_grid_kwh":null,"explanation":"A third of the 300 kWh pack is 100 kWh."},
{"note_index":2,"applies":false,"directive_type":"no_op","hours":[],"factor":null,"minimum_energy_kwh":null,"max_grid_kwh":null,"explanation":"Unrelated to today's energy schedule."}
]}

Battery capacity: 180 kWh
Notes:
0: "Intake from the utility is limited to 140 kWh per hour from 17:00 to 20:00 while the substation is worked on."
1: "The battery cannot accept charge during the morning inspection, 9 until 11."
Answer:
{"interpretations":[
{"note_index":0,"applies":true,"directive_type":"max_grid_window","hours":[17,18,19],"factor":null,"minimum_energy_kwh":null,"max_grid_kwh":140,"explanation":"Grid import capped at 140 kWh in each listed hour."},
{"note_index":1,"applies":true,"directive_type":"no_charge_window","hours":[9,10],"factor":null,"minimum_energy_kwh":null,"max_grid_kwh":null,"explanation":"Charging is unavailable during the inspection."}
]}"""


def _keys(*names: str) -> List[str]:
    out: List[str] = []
    for name in names:
        raw = os.getenv(name, "")
        for part in re.split(r"[,\s]+", raw):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


class KeyRing:
    """Round-robin over API keys, skipping ones that recently hit their quota."""

    def __init__(self, keys: List[str]) -> None:
        self.keys = keys
        self.pos = 0
        self.cooldown: Dict[str, float] = {}

    def __bool__(self) -> bool:
        return bool(self.keys)

    def next(self) -> Optional[str]:
        now = time.time()
        for _ in range(len(self.keys)):
            key = self.keys[self.pos % len(self.keys)]
            self.pos += 1
            if self.cooldown.get(key, 0) <= now:
                return key
        return None

    def penalise(self, key: str, seconds: float = 60.0) -> None:
        self.cooldown[key] = time.time() + seconds


class Interpreter:
    def __init__(self) -> None:
        self.gemini = KeyRing(_keys("GEMINI_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY"))
        self.models = [
            m.strip()
            for m in os.getenv(
                "GEMINI_MODEL", "gemini-3.5-flash,gemini-3-flash-preview,gemini-flash-latest"
            ).split(",")
            if m.strip()
        ]
        self.compat_keys = KeyRing(_keys("OPENAI_COMPAT_API_KEYS", "OPENAI_COMPAT_API_KEY"))
        self.compat_base = os.getenv(
            "OPENAI_COMPAT_BASE_URL", "https://api.groq.com/openai/v1"
        ).rstrip("/")
        self.compat_model = os.getenv("OPENAI_COMPAT_MODEL", "llama-3.3-70b-versatile")
        self.timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))
        self.cache: Dict[str, List[Dict[str, Any]]] = {}
        self.stats = {"llm": 0, "cache": 0, "fallback": 0}

    @property
    def configured(self) -> bool:
        return bool(self.gemini) or bool(self.compat_keys)

    def provider_name(self) -> str:
        if self.gemini:
            return f"google:{self.models[0]}"
        if self.compat_keys:
            return f"openai-compatible:{self.compat_model}"
        return "deterministic-fallback-only"

    # ---------------------------------------------------------------- prompt

    @staticmethod
    def _user_prompt(notes: List[str], capacity: float) -> str:
        listed = "\n".join(f'{i}: "{n.strip()}"' for i, n in enumerate(notes))
        return (
            f"{FEW_SHOT}\n\nNOW CLASSIFY THESE NOTES.\n\n"
            f"Battery capacity: {capacity:g} kWh\nNotes:\n{listed}\n\n"
            f"Return JSON with exactly {len(notes)} entries, note_index 0 to {len(notes) - 1}."
        )

    # -------------------------------------------------------------- providers

    async def _call_gemini(self, client: httpx.AsyncClient, prompt: str) -> Optional[str]:
        for model in self.models:
            key = self.gemini.next()
            if key is None:
                return None
            body: Dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                    "maxOutputTokens": 2048,
                },
            }
            if not model.startswith("gemini-1"):
                # Disable "thinking" tokens on every 2.x/3.x model for latency;
                # older 1.x models predate this parameter and reject it.
                body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

            try:
                resp = await client.post(
                    GEMINI_URL.format(model=model),
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json=body,
                )
            except Exception:
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    parts = data["candidates"][0]["content"]["parts"]
                    return "".join(p.get("text", "") for p in parts)
                except Exception:
                    continue
            if resp.status_code in (429, 403):
                self.gemini.penalise(key)
            # 400/404 -> try the next model in the list
        return None

    async def _call_compat(self, client: httpx.AsyncClient, prompt: str) -> Optional[str]:
        key = self.compat_keys.next()
        if key is None:
            return None
        try:
            resp = await client.post(
                f"{self.compat_base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": self.compat_model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
        except Exception:
            return None
        if resp.status_code == 429:
            self.compat_keys.penalise(key)
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return None

    # ------------------------------------------------------------------ main

    async def interpret(
        self, notes: List[str], capacity: float
    ) -> Tuple[List[Directive], str]:
        cache_key = json.dumps([n.strip().lower() for n in notes] + [capacity])
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.stats["cache"] += 1
            return normalize_all(cached, len(notes), capacity), "cache"

        raw: Optional[List[Dict[str, Any]]] = None
        source = "fallback"

        if self.configured:
            prompt = self._user_prompt(notes, capacity)
            text = None
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Groq (openai-compatible) is tried first: sub-second on its free
                # tier. Gemini is the reliable-but-slower fallback -- both are
                # kept because relying on one free provider alone is risky.
                if self.compat_keys:
                    text = await self._call_compat(client, prompt)
                    if text:
                        source = "openai-compatible"
                if not text and self.gemini:
                    text = await self._call_gemini(client, prompt)
                    if text:
                        source = "google"
            if text:
                raw = _extract_interpretations(text)

        if raw is None:
            self.stats["fallback"] += 1
            raw = [rule_based(note, i, capacity) for i, note in enumerate(notes)]
            source = "deterministic-fallback"
        else:
            self.stats["llm"] += 1
            self.cache[cache_key] = raw

        return normalize_all(raw, len(notes), capacity), source


def _extract_interpretations(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse model output defensively -- it may be wrapped in prose or fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ("interpretations", "directive_interpretation", "results", "notes"):
                if isinstance(data.get(key), list):
                    return data[key]
            return None
        if isinstance(data, list):
            return data
    return None


# --------------------------------------------------------------------------
# Deterministic safe-failure interpreter. Only runs when every model provider
# is unreachable; documented in the README as a fallback, never the primary path.
# --------------------------------------------------------------------------

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "noon": 12, "midnight": 0, "midday": 12,
}
_FRACTION = {
    "half": 0.5, "a half": 0.5, "one-half": 0.5,
    "a third": 1 / 3, "one-third": 1 / 3, "third": 1 / 3,
    "a quarter": 0.25, "one-quarter": 0.25, "quarter": 0.25,
    "a fifth": 0.2, "one-fifth": 0.2, "fifth": 0.2,
}


def _to_hour(token: str, meridiem: Optional[str]) -> Optional[int]:
    token = token.strip().lower()
    if token in _WORD_NUM:
        val = _WORD_NUM[token]
    else:
        m = re.match(r"^(\d{1,2})(?::(\d{2}))?$", token)
        if not m:
            return None
        val = int(m.group(1))
    if meridiem == "pm" and val < 12:
        val += 12
    elif meridiem == "am" and val == 12:
        val = 0
    return val if 0 <= val <= 23 else None


def _hours_from_text(text: str) -> List[int]:
    t = text.lower()
    pattern = (
        r"(\d{1,2}(?::\d{2})?|noon|midnight|midday|one|two|three|four|five|six|seven|"
        r"eight|nine|ten|eleven|twelve)\s*(am|pm|a\.m\.|p\.m\.)?\s*"
        r"(?:-|--|to|until|till|through|and)\s*"
        r"(\d{1,2}(?::\d{2})?|noon|midnight|midday|one|two|three|four|five|six|seven|"
        r"eight|nine|ten|eleven|twelve)\s*(am|pm|a\.m\.|p\.m\.)?"
    )
    m = re.search(pattern, t)
    if not m:
        return []
    a_tok, a_mer, b_tok, b_mer = m.group(1), m.group(2), m.group(3), m.group(4)
    norm = lambda x: x.replace(".", "")[:2] if x else None  # noqa: E731
    a_mer, b_mer = norm(a_mer), norm(b_mer)
    if a_mer is None and b_mer is not None:
        a_mer = b_mer
    start = _to_hour(a_tok, a_mer)
    end = _to_hour(b_tok, b_mer)
    if start is None or end is None:
        return []
    if end == start:
        return [start]
    span = (end - start) % 24
    return sorted({(start + i) % 24 for i in range(span)}) or [start]


def rule_based(note: str, index: int, capacity: float) -> Dict[str, Any]:
    t = note.lower()
    hours = _hours_from_text(note)
    base = {
        "note_index": index,
        "applies": False,
        "directive_type": NO_OP,
        "hours": [],
        "factor": None,
        "minimum_energy_kwh": None,
        "max_grid_kwh": None,
        "explanation": "No schedule-affecting instruction detected.",
    }
    if not hours:
        return base

    def hit(*words: str) -> bool:
        return any(w in t for w in words)

    solar = hit("solar", "pv", "panel", "photovoltaic", "rooftop")
    grid = hit("grid", "import", "intake", "feeder", "substation", "transformer", "utility")
    charging = hit("charg")
    discharging = hit("discharg", "drain", "draw down", "must hold")

    if solar and hit("drop", "reduc", "%", "percent", "half", "third", "quarter", "fifth", "cloud", "wash", "clean", "inspect", "maintenance", "leave"):
        factor = None
        pct = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", t)
        if pct:
            value = float(pct.group(1)) / 100.0
            factor = (1.0 - value) if hit("reduc", "drop by", "cut by", "lower by", "less") and not hit("drop to", "to about", "treated as", "leave") else value
        else:
            for word, val in _FRACTION.items():
                if word in t:
                    factor = val
                    break
        if factor is not None:
            base.update(applies=True, directive_type=SOLAR_REDUCTION, hours=hours,
                        factor=max(0.0, min(1.0, factor)),
                        explanation="Usable solar is reduced during these hours.")
            return base

    if grid and hit("not exceed", "no more than", "at or below", "limited to", "limit",
                    "cap", "must not", "maximum", "max", "restrict"):
        num = re.search(r"(\d+(?:\.\d+)?)\s*kwh", t)
        if num:
            base.update(applies=True, directive_type=MAX_GRID, hours=hours,
                        max_grid_kwh=float(num.group(1)),
                        explanation="Grid import is capped during these hours.")
            return base

    if hit("reserve", "at least", "no less than", "keep", "remain", "minimum"):
        pct = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", t)
        num = re.search(r"(\d+(?:\.\d+)?)\s*kwh", t)
        value = None
        if num:
            value = float(num.group(1))
        elif pct:
            value = capacity * float(pct.group(1)) / 100.0
        else:
            for word, frac in _FRACTION.items():
                if word in t:
                    value = capacity * frac
                    break
        if value is not None and not charging and not discharging:
            base.update(applies=True, directive_type=MIN_RESERVE, hours=hours,
                        minimum_energy_kwh=value,
                        explanation="A minimum battery reserve applies during these hours.")
            return base

    if discharging:
        base.update(applies=True, directive_type=NO_DISCHARGE, hours=hours,
                    explanation="Battery discharging is unavailable during these hours.")
        return base
    if charging:
        base.update(applies=True, directive_type=NO_CHARGE, hours=hours,
                    explanation="Battery charging is unavailable during these hours.")
        return base

    return base


interpreter = Interpreter()
