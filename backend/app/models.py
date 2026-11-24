from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FireType = Literal["низовой", "верховой", "почвенный"]


class Fire(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    detected_at: str                       # ISO
    area_ha: float = Field(0.1, ge=0)
    fire_type: FireType = "низовой"
    wind_ms: float = Field(0, ge=0)
    weather_class: int = Field(3, ge=1, le=5)   # КПО по условиям погоды (приказ № 287)
    natural_class: int | None = Field(None, ge=1, le=5)
    species: str | None = None
    lesnichestvo: str | None = None
    uch_lesnichestvo: str | None = None
    source: str = "оператор"


class Station(BaseModel):
    id: str
    name: str
    type: str
    settlement: str | None = None
    lat: float
    lon: float
    groups: int
    people: int
    vehicles: dict[str, int]
    available: bool = True
    note: str | None = None


class Scenario(BaseModel):
    title: str
    description: str
    now: str
    fires: list[Fire]
    stations: list[Station]


class OverrideRequest(BaseModel):
    fire_id: str
    station_ids: list[str]


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
