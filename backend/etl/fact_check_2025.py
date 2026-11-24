"""Сверка «рекомендация калькулятора против факта» по реестру 2025 г. (все субъекты, районы НСС).

Для каждого пожара калькулятор (модуль 2) рассчитывает нормативную потребность в группах и предельное
время прибытия (пп. 51–52 Правил тушения). Затем фактические силы и задержка начала тушения из реестра
сравниваются с рекомендацией, и для групп «в нормативе / ниже норматива» сравниваются наблюдаемые
исходы: доля пожаров, достигших 25 га, медианная итоговая площадь, медианные затраты.

Запуск: python -I fact_check_2025.py <data_dir>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import natural_class_for  # noqa: E402
from app.modules import compute_demand  # noqa: E402


def main(data_dir: str) -> None:
    df = pd.read_parquet(Path(data_dir) / "registry.parquet")
    g = df[(df.year == 2025) & (df.force_district == "НСС")].dropna(subset=["det_dt", "area_total_ha"]).copy()
    g["ppl_all"] = g[["ppl_ground", "ppl_avia", "ppl_mchs_total", "ppl_users", "ppl_other"]].fillna(0).sum(axis=1)
    g["delay_h"] = (g.start_dt - g.det_dt).dt.total_seconds() / 3600
    rows = []
    for r in g.itertuples():
        area0 = r.area_detect_ha if pd.notna(r.area_detect_ha) and r.area_detect_ha > 0 else None
        if area0 is None:
            continue
        nat = natural_class_for(r.species if isinstance(r.species, str) else None)
        fire = {"area_ha": float(area0), "wind_ms": float(r.wind_ms) if pd.notna(r.wind_ms) else 0.0,
                "weather_class": 4 if r.det_dt.month in (4, 5, 10) else 3,
                "fire_type": "верховой" if (pd.notna(r.area_crown_ha) and r.area_crown_ha > 0) else "низовой"}
        risk = {"category": "средний", "natural_class": nat,
                "nearest_np": {"km": float(r.np_dist_km) if pd.notna(r.np_dist_km) else 5.0}}
        d = compute_demand(fire, risk)
        delay = r.delay_h if pd.notna(r.delay_h) and 0 <= r.delay_h <= 72 else None
        late_red = delay is not None and d["hard_weather"] and delay > d["arrival_limit_reduced_h"]
        need = d["groups"] * (d["k_if_late"] if late_red else 1)
        have = math.ceil(r.ppl_all / 6) if r.ppl_all > 0 else 0
        rows.append({"need": need, "have": have, "enough": have >= need, "delay": delay,
                     "in_time": delay is not None and delay <= d["arrival_limit_h"],
                     "large": r.area_total_ha >= 25, "area": r.area_total_ha, "area0": area0,
                     "cost": r.cost_rub if pd.notna(r.cost_rub) else None})
    x = pd.DataFrame(rows)
    x = x[x.delay.notna()]

    def grp(m: pd.Series, label: str) -> dict:
        s = x[m]
        return {"group": label, "n": int(len(s)), "large_share": round(float(s.large.mean()), 4),
                "median_area_ha": round(float(s.area.median()), 2),
                "median_cost_thousand_rub": round(float(s.cost.median() / 1000), 1) if s.cost.notna().any() else None}

    small = x.area0 < 1  # одинаковые стартовые условия: при обнаружении < 1 га
    res = {
        "n_fires_2025_with_area_at_detection_and_delay": int(len(x)),
        "share_forces_below_norm": round(float((~x.enough).mean()), 4),
        "share_arrival_beyond_norm": round(float((~x.in_time).mean()), 4),
        "by_arrival": [grp(x.in_time, "прибытие в нормативе (п. 51)"), grp(~x.in_time, "прибытие сверх норматива")],
        "by_arrival_small_at_detection": [grp(small & x.in_time, "в нормативе, < 1 га при обнаружении"),
                                          grp(small & ~x.in_time, "сверх норматива, < 1 га при обнаружении")],
        "by_forces_small_at_detection": [grp(small & x.enough, "силы ≥ рекомендации, < 1 га при обнаружении"),
                                         grp(small & ~x.enough, "силы < рекомендации, < 1 га при обнаружении")],
        "by_forces_and_time_small": [grp(small & x.enough & x.in_time, "силы ≥ рекомендации и в нормативе (< 1 га)"),
                                     grp(small & ~(x.enough & x.in_time), "иначе (< 1 га)")],
    }
    json.dump(res, open(Path(data_dir) / "fact_check_2025.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
