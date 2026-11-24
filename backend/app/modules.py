"""Модули 1, 2, 4: градация риска, калькулятор потребности, маршрутизация."""
from __future__ import annotations

import math
import os

import httpx

from .core import growth_model, haversine_km, natural_class_for, nearest, norms, ref, reference

# ---------------------------------------------------------------- Модуль 4. Маршрутизация

TORTUOSITY = 1.35           # коэффициент извилистости дорог относительно прямой (без графа OSM)
MOBILIZATION_H = 0.25       # сбор группы в месте дислокации
SPEED_KMH = {"АЦЛ": 50, "АГП1": 55, "ТЛП": 35, "ВЛП": 30}


def route(station: dict, fire: dict) -> dict:
    """Время прибытия первой группы и тяжёлой техники. Если задан OSRM_URL — по дорожному графу OSM."""
    osrm = os.environ.get("OSRM_URL")
    geom = [[station["lon"], station["lat"]], [fire["lon"], fire["lat"]]]
    straight = haversine_km(station["lat"], station["lon"], fire["lat"], fire["lon"])
    road_km, method = straight * TORTUOSITY, "по прямой × 1,35"
    if osrm:
        try:
            r = httpx.get(f"{osrm}/route/v1/driving/{station['lon']},{station['lat']};{fire['lon']},{fire['lat']}",
                          params={"overview": "simplified", "geometries": "geojson"}, timeout=3)
            j = r.json()["routes"][0]
            road_km, geom, method = j["distance"] / 1000, j["geometry"]["coordinates"], "OSRM (OpenStreetMap)"
        except Exception:
            pass
    t_first = MOBILIZATION_H + road_km / SPEED_KMH["АЦЛ"]
    t_heavy = MOBILIZATION_H + road_km / SPEED_KMH["ТЛП"]
    return {"km": round(road_km, 1), "straight_km": round(straight, 1), "t_first_h": round(t_first, 3),
            "t_heavy_h": round(t_heavy, 3), "method": method, "geometry": geom}


# ---------------------------------------------------------------- Модуль 1. Градация риска

LEVELS = ["low", "medium", "high"]
LEVEL_RU = {"low": "низкая", "medium": "средняя", "high": "высокая"}


ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}


def _c(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def assess_risk(fire: dict, nominal_delay_h: float, n_active_lesn: int, n_active_subj: int) -> dict:
    N = norms()["risk"]
    refd = reference(fire.get("region") or "region_a")
    np_, np_km = nearest(fire["lat"], fire["lon"], refd["settlements"])
    oo, oo_km = nearest(fire["lat"], fire["lon"], refd["oopt"])
    if oo is None:
        oo, oo_km = {"name": "нет в справочнике"}, 999.0
    crown = fire.get("fire_type") == "верховой"
    nat = fire.get("natural_class") or natural_class_for(fire.get("species"))
    wc = fire.get("weather_class", 3)
    wind = fire.get("wind_ms") or 0

    criteria = []

    def crit(text, hit, axis, r):
        criteria.append({"text": text, "triggered": bool(hit), "axis": axis, "ref": r})

    sev = "low"
    sh, sm = N["severity"]["high"], N["severity"]["medium"]
    r248 = ref(N["ref"])
    if np_km <= sh["np_km_max"] or oo_km <= sh["oopt_km_max"] or crown:
        sev = "high"
    elif np_km <= sm["np_km_max"] or oo_km <= sm["oopt_km_max"] or nat <= 2:
        sev = "medium"
    crit(f"До населённого пункта «{np_['name']}» {_c(np_km)} км (порог 5/10 км)", np_km <= sm["np_km_max"], "тяжесть", r248)
    crit(f"До ООПТ «{oo['name']}» {oo_km:.0f} км (порог 10/20 км)", oo_km <= sm["oopt_km_max"], "тяжесть", r248)
    crit(f"Вид пожара: {fire.get('fire_type', 'низовой')}", crown, "тяжесть", ref(norms()["spread"]["ref"]))
    crit(f"Природная пожарная опасность: класс {ROMAN[nat]}", nat <= 2, "тяжесть", ref(norms()["natural_class_by_species"]["ref"]))

    gm = growth_model()
    p = gm.predict({**fire, "np_dist_km": np_km}, [nominal_delay_h], n_active_lesn, n_active_subj)[0]
    ph, pm = N["probability"]["high"], N["probability"]["medium"]
    prob = "low"
    if p >= ph["p_min"] or wind >= ph["wind_ms_min"] or wc >= ph["weather_class_min"]:
        prob = "high"
    elif p >= pm["p_min"] or wind >= pm["wind_ms_min"] or wc >= pm["weather_class_min"]:
        prob = "medium"
    crit(f"ML-прогноз: P(площадь ≥ 25 га) = {p:.0%} при задержке {nominal_delay_h * 60:.0f} мин", p >= pm["p_min"], "вероятность",
         {"act": "Модель CatBoost", "point": f"ROC-AUC {str(gm.meta.get('roc_auc', '—')).replace('.', ',')} на 2024–2025 гг.", "title": "Модель разрастания"})
    crit(f"Ветер {_c(wind, 0)} м/с (порог 5/8 м/с)", wind >= pm["wind_ms_min"], "вероятность", ref(norms()["reinforcement"]["ref"]))
    crit(f"Класс пожарной опасности по погоде: {ROMAN[wc]}", wc >= pm["weather_class_min"], "вероятность", ref(norms()["weather_class"]["ref"]))

    cat = N["matrix"][sev][prob]
    return {
        "category": cat, "weight": N["weights"][cat], "severity": sev, "probability": prob,
        "severity_ru": LEVEL_RU[sev], "probability_ru": LEVEL_RU[prob],
        "p_growth": round(p, 3), "nearest_np": {"name": np_["full_name"], "km": round(np_km, 1), "lat": np_["lat"], "lon": np_["lon"]},
        "nearest_oopt": {"name": oo["name"], "km": round(oo_km, 1)},
        "natural_class": nat, "criteria": criteria,
    }


# ---------------------------------------------------------------- Модуль 2. Калькулятор потребности

def perimeter_m(area_ha: float) -> float:
    return 2 * math.sqrt(math.pi * max(area_ha, 0.01) * 1e4) * 1.15


def compute_demand(fire: dict, risk: dict) -> dict:
    n = norms()
    sp = n["spread"]
    wc = fire.get("weather_class", 3)
    wind = fire.get("wind_ms") or 0
    # значения ветра > 20 м/с в реестре — ошибки ввода; для расчёта скорости распространения ограничиваем
    head = sp["head_ros_m_h_by_weather_class"][wc] * math.exp(sp["wind_coef"] * min(wind, 20))
    if fire.get("fire_type") == "верховой":
        head *= sp["crown_multiplier"]
    v_edge = sp["perimeter_to_head"] * head
    P0 = perimeter_m(fire.get("area_ha", 0.1))
    v = v_edge + P0 / sp["localization_horizon_h"]
    p = n["productivity_m_h_per_group"]["value"]
    groups = max(1, math.ceil(v / p))
    if risk["category"] in ("высокий", "чрезвычайно высокий"):
        groups = max(groups, 2)

    rf = n["reinforcement"]
    nat = risk["natural_class"]
    hard_weather = (wind > rf["wind_ms_over"] or wc >= rf["weather_class_min"]) and nat in rf["natural_classes"]
    limit = n["arrival_limit_h"]["by_natural_class"][min(nat, 5)]
    limit_red = limit * rf["time_factor"] if hard_weather else limit

    area = fire.get("area_ha", 0.1)
    veh = {
        "АЦЛ": max(1, math.ceil(groups / 2)),
        "ТЛП": 2 if area >= 25 else (1 if area >= 2 or fire.get("fire_type") == "верховой" else 0),
        "ВЛП": 1 if risk["nearest_np"]["km"] > 10 else 0,
    }
    veh["ТБ"] = 1 if (area >= 25 or fire.get("fire_type") == "верховой") else 0
    veh["АГП1"] = max(0, groups - veh["АЦЛ"])  # каждая группа обеспечивается транспортом
    gs = n["group"]["size"]
    vref = n["vehicles"]
    lines = [
        {"resource": "Люди (группы / чел.)", "code": "groups", "qty": groups, "people": groups * gs,
         "norm": f"N = ⌈v/p⌉ = ⌈{v:.0f}/{p}⌉; группа — {gs} чел.", "ref": ref(n["group"]["ref"])},
    ]
    for code in ("АЦЛ", "ТЛП", "ВЛП", "ТБ", "АГП1"):
        if veh[code] > 0:
            lines.append({"resource": f"{code} ({vref[code]['name']})", "code": code, "qty": veh[code],
                          "norm": vref[code]["rule"], "ref": ref(vref[code]["ref"])})
    return {
        "groups": groups, "people": groups * gs, "vehicles": veh,
        "v_m_h": round(v, 1), "v_edge_m_h": round(v_edge, 1), "head_ros_m_h": round(head, 1),
        "perimeter_m": round(P0), "p_m_h": p, "k_if_late": rf["force_factor"] if hard_weather else 1,
        "hard_weather": hard_weather,
        "arrival_limit_h": limit, "arrival_limit_reduced_h": limit_red,
        "arrival_ref": ref(n["arrival_limit_h"]["ref"]), "reinforcement_ref": ref(rf["ref"]),
        "lines": lines,
    }


def expected_damage(p_growth: float) -> float:
    d = norms()["damage"]
    return d["small_fire_rub"] + p_growth * (d["large_fire_rub"] - d["small_fire_rub"])
