"""Request schema for POST /optimize-energy.

Validation here is deliberately strict against the Problem Statement so that
structurally invalid requests are rejected with 400 rather than silently
producing a nonsense plan.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


class HourIn(BaseModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float

    @field_validator("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh")
    @classmethod
    def finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("value must be finite")
        return v


class BatteryIn(BaseModel):
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

    @model_validator(mode="after")
    def coherent(self):
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh exceeds capacity_kwh")
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh exceeds capacity_kwh")
        if self.initial_energy_kwh < self.minimum_energy_kwh:
            raise ValueError("initial_energy_kwh is below minimum_energy_kwh")
        return self


class ScenarioIn(BaseModel):
    scenario_id: str = Field(min_length=1)
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourIn] = Field(min_length=24, max_length=24)
    battery: BatteryIn

    @field_validator("operator_notes")
    @classmethod
    def notes_non_empty(cls, v: List[str]) -> List[str]:
        for n in v:
            if not isinstance(n, str) or not n.strip():
                raise ValueError("operator_notes entries must be non-empty strings")
        return v

    @field_validator("hours")
    @classmethod
    def hours_complete(cls, v: List[HourIn]) -> List[HourIn]:
        seen = sorted(h.hour for h in v)
        if seen != list(range(24)):
            raise ValueError("hours must contain each hour 0..23 exactly once")
        return v

    def ordered_hours(self) -> List[HourIn]:
        return sorted(self.hours, key=lambda h: h.hour)
