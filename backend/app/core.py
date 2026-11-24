"""Общие объекты: конфигурация, нормативы, справочники, геометрия, модель разрастания."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("LESRESURS_DATA", BASE / "data"))
APP = Path(__file__).resolve().parent

R_EARTH = 6371.0

REGIONS = {
    "region_a": {"slug": "region_a", "name": "Северотаёжный модельный край", "center": [92.5, 59.3], "zoom": 6.7},
    "region_b": {"slug": "region_b", "name": "Лесостепная модельная область", "center": [38.5, 52.0], "zoom": 7.6},
}
DEFAULT_REGION = "region_a"


@lru_cache
def norms() -> dict:
    return yaml.safe_load(open(APP / "norms.yaml", encoding="utf-8"))


def ref(r: dict) -> dict:
    """Ссылка на пункт НПА в читаемом виде."""
    act = norms()["acts"][r["act"]]
    return {"act": act["short"], "point": r["point"], "title": act["title"]}


@lru_cache
def reference(region: str = DEFAULT_REGION) -> dict:
    return json.load(open(DATA / "regions" / region / "reference.json", encoding="utf-8"))


def region_path(region: str) -> Path:
    return DATA / "regions" / region


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def nearest(lat: float, lon: float, items: list[dict]) -> tuple[dict | None, float]:
    best, bd = None, float("inf")
    for it in items:
        d = haversine_km(lat, lon, it["lat"], it["lon"])
        if d < bd:
            best, bd = it, d
    return best, bd


def weather_class_from_index(nesterov: float) -> int:
    th = norms()["weather_class"]["thresholds"]
    for i, t in enumerate(th):
        if nesterov <= t:
            return i + 1
    return 5


def natural_class_for(species: str | None) -> int:
    m = norms()["natural_class_by_species"]
    if not species:
        return m["default"]
    for k, v in m.items():
        if k not in ("default", "ref") and k.lower() in species.lower():
            return v
    return m["default"]


@dataclass
class GrowthModel:
    """Обёртка над CatBoost: P(итоговая площадь ≥ 25 га | признаки, задержка) и ожидаемая итоговая площадь."""
    model: object | None
    meta: dict
    area_model: object | None = None

    FEATURES = ["lat", "lon", "month", "doy", "hour", "wind_ms", "np_dist_km", "species", "subject",
                "delay_h", "n_active_lesn", "n_active_subj"]

    def predict(self, fire: dict, delays_h: list[float], n_active_lesn: int, n_active_subj: int) -> list[float]:
        if self.model is None:
            # запасной вариант: эмпирическая кривая по реестру (доля ≥ 25 га от задержки)
            base = {0.5: 0.121, 1: 0.155, 2: 0.171, 3: 0.21, 99: 0.264}
            out = []
            for d in delays_h:
                for k in sorted(base):
                    if d <= k:
                        out.append(base[k])
                        break
            return out
        t = pd.Timestamp(fire["detected_at"])
        rows = []
        for d in delays_h:
            rows.append({
                "lat": fire["lat"], "lon": fire["lon"], "month": t.month, "doy": t.dayofyear, "hour": t.hour,
                "wind_ms": fire.get("wind_ms") or 0.0, "np_dist_km": fire.get("np_dist_km") or 5.0,
                "species": fire.get("species") or "нет данных", "subject": fire.get("subject") or REGIONS[DEFAULT_REGION]["name"],
                "delay_h": float(min(max(d, 0), 48)), "n_active_lesn": n_active_lesn, "n_active_subj": n_active_subj,
            })
        from catboost import Pool
        df = pd.DataFrame(rows)[self.FEATURES]
        pool = Pool(df, cat_features=["species", "subject"])
        p = self.model.predict_proba(pool)[:, 1]
        self._last_area = (np.maximum.accumulate(np.expm1(self.area_model.predict(pool))).tolist()
                           if self.area_model is not None else None)
        return [float(x) for x in np.maximum.accumulate(p)]  # гарантия монотонности по задержке

    def last_area(self) -> list[float] | None:
        """Ожидаемая итоговая площадь (га) для задержек из последнего вызова predict()."""
        return getattr(self, "_last_area", None)


@lru_cache
def growth_model() -> GrowthModel:
    meta_p = DATA / "growth_model_meta.json"
    meta = json.load(open(meta_p, encoding="utf-8")) if meta_p.exists() else {}
    path = DATA / "growth_model.cbm"
    if not path.exists():
        return GrowthModel(None, meta)
    from catboost import CatBoostClassifier, CatBoostRegressor
    m = CatBoostClassifier()
    m.load_model(str(path))
    area = None
    if (DATA / "area_model.cbm").exists():
        area = CatBoostRegressor()
        area.load_model(str(DATA / "area_model.cbm"))
    return GrowthModel(m, meta, area)
