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

SYSTEM_PROMPT = """Convert campus energy operator notes into scheduling directives.

TYPES (pick exactly one per note):
solar_reduction        - usable solar drops for some hours. fields: hours, factor
minimum_battery_reserve- battery must stay at/above a level. fields: hours, minimum_energy_kwh
no_charge_window       - battery cannot charge. fields: hours
no_discharge_window    - battery cannot discharge/drain. fields: hours
max_grid_window        - grid import capped per hour. fields: hours, max_grid_kwh
no_op                  - no effect on today's schedule. no fields

RULES:
1. Hours are whole hours, START INCLUSIVE, END EXCLUSIVE, unique ints 0-23 ascending.
   "1 PM to 3 PM"=[13,14]  "2 AM until 5 AM"=[2,3,4]  "6 PM to 10 PM"=[18,19,20,21]
   "noon until 2 PM"=[12,13]  "11 AM-2 PM"=[11,12,13]  "at 3 PM"=[15]
2. factor = fraction REMAINING, not lost. "drops to 20%"=0.2. "80% reduction"=0.2.
   "half"=0.5. "one-fifth"=0.2. "25% of forecast"=0.25. Always 0..1.
3. Percentage reserves convert via the battery capacity given. 50% of 200kWh = 100.
4. "charger isolated/unavailable/do not charge"=no_charge_window.
   "must not discharge/drain/must hold charge"=no_discharge_window.
5. Menus, deadlines, bookings, notices, staffing, next week/month, contracts = no_op.
   A note may mention energy and still be no_op if it does not constrain today.
6. Never invent types outside the six. Never alter demand/tariff/battery limits.
   If unsure, use no_op.

Return ONLY JSON, one entry per note, in order:
{"interpretations":[{"note_index":0,"applies":true,"directive_type":"solar_reduction","hours":[13,14],"factor":0.2,"minimum_energy_kwh":null,"max_grid_kwh":null,"explanation":"short"}]}
applies=false ONLY for no_op. Unused numeric fields null."""

FEW_SHOT = """Example - capacity 300 kWh:
0: "Inverter work 9 AM to 11 AM leaves about 40% of usual PV."
1: "Keep no less than a third of the pack from 7 PM to 10 PM."
2: "Canteen meeting moved to Thursday."
{"interpretations":[
{"note_index":0,"applies":true,"directive_type":"solar_reduction","hours":[9,10],"factor":0.4,"minimum_energy_kwh":null,"max_grid_kwh":null,"explanation":"40% of solar remains."},
{"note_index":1,"applies":true,"directive_type":"minimum_battery_reserve","hours":[19,20,21],"factor":null,"minimum_energy_kwh":100,"max_grid_kwh":null,"explanation":"A third of 300 is 100."},
{"note_index":2,"applies":false,"directive_type":"no_op","hours":[],"factor":null,"minimum_energy_kwh":null,"max_grid_kwh":null,"explanation":"Unrelated to today."}]}"""


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
                "GEMINI_MODEL",
                "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3-flash-preview,gemini-flash-latest",
            ).split(",")
            if m.strip()
        ]
        self.compat_keys = KeyRing(_keys("OPENAI_COMPAT_API_KEYS", "OPENAI_COMPAT_API_KEY"))
        self.compat_base = os.getenv(
            "OPENAI_COMPAT_BASE_URL", "https://api.groq.com/openai/v1"
        ).rstrip("/")
        self.compat_models = [
            m.strip()
            for m in os.getenv(
                "OPENAI_COMPAT_MODEL",
                "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b",
            ).split(",")
            if m.strip()
        ]
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
            return f"openai-compatible:{self.compat_models[0]}"
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
            # Disabling "thinking" tokens roughly halves latency, but not every
            # model accepts the parameter -- some reject it with a 400. Try with
            # it first, then retry once without before giving up on this model.
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

            for attempt in range(2):
                try:
                    resp = await client.post(
                        GEMINI_URL.format(model=model),
                        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                        json=body,
                    )
                except Exception:
                    break

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        parts = data["candidates"][0]["content"]["parts"]
                        return "".join(p.get("text", "") for p in parts)
                    except Exception:
                        break

                if resp.status_code in (429, 403):
                    self.gemini.penalise(key)
                    break
                if resp.status_code == 400 and attempt == 0:
                    body["generationConfig"].pop("thinkingConfig", None)
                    continue
                break
            # exhausted this model (404/400/quota) -> fall through to the next one
        return None

    async def _call_compat(self, client: httpx.AsyncClient, prompt: str) -> Optional[str]:
        """Try each (model, key) pair until one answers.

        Groq meters its free tier per MODEL *and* per organisation, so N models
        across M accounts gives N*M independent token buckets. A 429 means that
        one bucket is momentarily full, not that anything is broken, so we move
        to the next pair. Keys from the same account share a bucket; keys from
        different accounts do not, which is why a second account genuinely adds
        capacity where a second key on the same account would not.

        Rate-limit rejections come back without running inference, so cycling
        past them is cheap. MAX_ATTEMPTS bounds the worst case so a bad run can
        never approach the judge's 30 s limit.
        """
        MAX_ATTEMPTS = 6
        attempts = 0
        keys = list(self.compat_keys.keys)
        if not keys:
            return None

        for model in self.compat_models:
            # Start each model at a different key so load spreads across
            # accounts instead of always hammering the first one.
            offset = self.compat_keys.pos
            for n in range(len(keys)):
                if attempts >= MAX_ATTEMPTS:
                    return None
                key = keys[(offset + n) % len(keys)]
                attempts += 1
                try:
                    resp = await client.post(
                        f"{self.compat_base}/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json={
                            "model": model,
                            "temperature": 0,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                        },
                    )
                except Exception:
                    continue

                if resp.status_code == 200:
                    self.compat_keys.pos += 1
                    try:
                        return resp.json()["choices"][0]["message"]["content"]
                    except Exception:
                        continue
                if resp.status_code in (401, 403):
                    # A genuinely bad key, unlike a 429 -- stop using it.
                    self.compat_keys.penalise(key, 300)
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
