"""Подготовка данных модельного субъекта (синтетический набор).

1. Справочник населённых пунктов: координаты восстанавливаются из реестра
   (координаты пожара + азимут и удалённость до ближайшего н.п.; азимут
   отсчитывается от н.п. к пожару), берётся медиана по всем пожарам.
2. ООПТ: центроиды пожаров в лесничествах-ООПТ.
3. Условный справочник лесопожарных станций: число и типы — по нормативу
   приказа 1605-р для субъекта (состав станций — в app/norms.yaml, station_composition),
   размещение — жадная задача
   о покрытии спроса (исторические пожары) в заданном радиусе по прямой.
   Фактическая дислокация ПХС подлежит уточнению.
4. Исторические сценарии: моменты, когда в крае одновременно действовало
   несколько пожаров в районах наземных сил (для демо и ретро-симуляции).

Запуск: python -I build_region.py <registry.parquet> <data_dir> <region_a|region_b>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R_EARTH = 6371.0

REGIONS = {
    "region_a": {"name": "Северотаёжный модельный край", "bbox": (57.0, 61.0, 88.0, 98.0), "cover_km": 40.0,
                 "spacing_km": 25.0, "np_min_fires": 3, "cand_min_fires": 8},
    "region_b": {"name": "Лесостепная модельная область", "bbox": (50.0, 54.0, 36.0, 42.0), "cover_km": 30.0,
                 "spacing_km": 15.0, "np_min_fires": 1, "cand_min_fires": 2},
}


def move(lat, lon, az_deg, d_km):
    a = np.radians(az_deg)
    dl = d_km / R_EARTH
    la, lo = np.radians(lat), np.radians(lon)
    la2 = np.arcsin(np.sin(la) * np.cos(dl) + np.cos(la) * np.sin(dl) * np.cos(a))
    lo2 = lo + np.arctan2(np.sin(a) * np.sin(dl) * np.cos(la), np.cos(dl) - np.sin(la) * np.sin(la2))
    return np.degrees(la2), np.degrees(lo2)


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def clean_np(name: str) -> str:
    n = re.sub(r"^(пгт|кп|ж/д ст|г|с|п|д|ст|х|кордон|рп|нп|м)\.?\s+", "", str(name).strip(), flags=re.I)
    return n.strip().strip('"«»').strip()


def main(src: str, data_dir: str, slug: str) -> None:
    import yaml
    cfg = REGIONS[slug]
    REGION = cfg["name"]
    COVER_KM = cfg["cover_km"]
    norms = yaml.safe_load(open(Path(__file__).parents[1] / "app" / "norms.yaml", encoding="utf-8"))
    rn = norms["region_norms"][REGION]
    STATION_PLAN = [(t, rn.get(t, 0)) for t in ("ЛПС-3", "ЛПС-2", "ЛПС-1") if rn.get(t, 0)]
    composition = rn.get("station_composition", norms["station_types"])
    out_dir = Path(data_dir) / "regions" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)
    la0, la1, lo0, lo1 = cfg["bbox"]
    pr = df[(df.subject == REGION) & df.lat.between(la0, la1) & df.lon.between(lo0, lo1)].copy()
    pr["np_clean"] = pr.np_name.map(lambda s: clean_np(s) if pd.notna(s) else None)

    # 1. населённые пункты
    s = pr.dropna(subset=["np_clean", "np_azimuth", "np_dist_km"]).copy()
    s = s[s.np_dist_km.between(0, 60)]
    la, lo = move(s.lat.values, s.lon.values, s.np_azimuth.values + 180.0, s.np_dist_km.values)
    s["np_lat"], s["np_lon"] = la, lo
    g = s.groupby("np_clean").agg(
        lat=("np_lat", "median"), lon=("np_lon", "median"), n_fires=("np_lat", "size"),
        full_name=("np_name", lambda x: x.mode().iat[0]),
    ).reset_index()
    g = g[g.n_fires >= cfg["np_min_fires"]].sort_values("n_fires", ascending=False)
    settlements = [
        {"id": f"np{i}", "name": r.np_clean, "full_name": r.full_name, "lat": round(r.lat, 5),
         "lon": round(r.lon, 5), "n_fires": int(r.n_fires)}
        for i, r in enumerate(g.itertuples())
    ]

    # 2. ООПТ (центроиды пожаров в соответствующих лесничествах)
    oopt_mask = pr.lesnichestvo.str.contains(r"ГПБЗ|ГПЗ|НП |НП«|НП «|заповедник|национальный", case=False, na=False)
    oopt = (pr[oopt_mask].groupby("lesnichestvo").agg(lat=("lat", "median"), lon=("lon", "median"), n=("lat", "size"))
            .reset_index())
    oopt_list = [{"name": r.lesnichestvo, "lat": round(r.lat, 4), "lon": round(r.lon, 4)} for r in oopt.itertuples()]

    # 3. станции: жадное покрытие спроса
    ground = pr[pr.force_district == "НСС"]
    cand = g[g.n_fires >= cfg["cand_min_fires"]].reset_index(drop=True)
    D = hav(cand.lat.values[:, None], cand.lon.values[:, None], ground.lat.values[None, :], ground.lon.values[None, :])
    covered = np.zeros(len(ground), bool)
    chosen = []
    for _ in range(sum(n for _, n in STATION_PLAN)):
        gain = ((D <= COVER_KM) & ~covered[None, :]).sum(axis=1).astype(float)
        # не ставить две станции ближе заданного расстояния
        for c in chosen:
            gain[hav(cand.lat.values, cand.lon.values, cand.lat[c], cand.lon[c]) < cfg["spacing_km"]] = -1
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            # покрытие уже полное — добираем станции по нормативу, выбирая максимальный охват
            raw = (D <= COVER_KM).sum(axis=1).astype(float)
            for c in chosen:
                raw[hav(cand.lat.values, cand.lon.values, cand.lat[c], cand.lon[c]) < cfg["spacing_km"]] = -1
            best = int(np.argmax(raw))
            if raw[best] <= 0:
                break
        chosen.append(best)
        covered |= D[best] <= COVER_KM
    load = [int((D[c] <= COVER_KM).sum()) for c in chosen]
    order = [c for _, c in sorted(zip(load, chosen), reverse=True)]
    types = [t for t, n in STATION_PLAN for _ in range(n)]
    stations = []
    for i, (c, t) in enumerate(zip(order, types)):
        comp = composition[t]
        r = cand.iloc[c]
        stations.append({
            "id": f"st{i + 1}", "name": f"{t} «{r.np_clean}»", "type": t, "settlement": r.full_name,
            "lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5),
            "groups": comp["groups"], "people": comp["groups"] * norms["group"]["size"],
            "vehicles": {k: comp.get(k, 0) for k in ("АЦЛ", "ТЛП", "ВЛП", "ТБ", "АГП1")},
            "demand_covered": int((D[c] <= COVER_KM).sum()),
        })
    print(f"покрыто спроса в радиусе {COVER_KM:g} км:", round(covered.mean(), 3))

    # 4. исторические сценарии
    h = ground.dropna(subset=["det_dt"]).copy()
    h["end_dt"] = h.loc_dt.fillna(h.liq_dt).fillna(h.det_dt + pd.Timedelta(hours=12))
    ground = ground.copy()
    ground["ppl_all"] = ground[["ppl_ground", "ppl_avia", "ppl_mchs_total", "ppl_users", "ppl_other"]].fillna(0).sum(axis=1)
    ground["veh_all"] = ground[["veh_tractor", "veh_truck", "veh_firetruck", "veh_allterrain"]].fillna(0).sum(axis=1)
    h = ground.dropna(subset=["det_dt"]).copy()
    h["end_dt"] = h.loc_dt.fillna(h.liq_dt).fillna(h.det_dt + pd.Timedelta(hours=12))
    keep = ["ppl_all", "veh_all", "lesnichestvo", "uch_lesnichestvo", "np_name", "np_dist_km", "lat", "lon", "det_dt", "start_dt",
            "loc_dt", "liq_dt", "end_dt", "wind_ms", "area_detect_ha", "area_total_ha", "area_crown_ha",
            "species", "ppl_ground", "veh_firetruck", "veh_tractor", "veh_allterrain", "year"]
    h = h[keep]
    # одно возгорание может быть записано несколькими строками (участки контура) с одинаковыми
    # координатами и временем обнаружения — объединяем в один пожар с суммарной площадью
    n0 = len(h)
    h = (h.groupby(["lesnichestvo", "lat", "lon", "det_dt"], as_index=False, dropna=False)
          .agg(uch_lesnichestvo=("uch_lesnichestvo", "first"), np_name=("np_name", "first"),
               np_dist_km=("np_dist_km", "min"), start_dt=("start_dt", "min"), loc_dt=("loc_dt", "max"),
               liq_dt=("liq_dt", "max"), end_dt=("end_dt", "max"), wind_ms=("wind_ms", "max"),
               area_detect_ha=("area_detect_ha", lambda x: x.sum(min_count=1)), area_total_ha=("area_total_ha", "sum"),
               area_crown_ha=("area_crown_ha", lambda x: x.sum(min_count=1)), species=("species", "first"),
               ppl_ground=("ppl_ground", "sum"), veh_firetruck=("veh_firetruck", "sum"),
               veh_tractor=("veh_tractor", "sum"), veh_allterrain=("veh_allterrain", "sum"), year=("year", "first"),
               ppl_all=("ppl_all", "sum"), veh_all=("veh_all", "sum")))
    print(f"записей: {n0}, после объединения участков одного возгорания: {len(h)}")
    h = h.sort_values("det_dt").reset_index(drop=True)
    h.to_parquet(out_dir / "fires.parquet")
    # моменты одновременности считаются заметающей прямой: для каждого пожара — сколько
    # пожаров действовало в момент его обнаружения (иначе O(n^2) на синтетическом наборе)
    import heapq
    snaps = []
    heap = []          # времена окончания действующих пожаров
    for i, r in h.iterrows():
        while heap and heap[0] <= r.det_dt:
            heapq.heappop(heap)
        heapq.heappush(heap, r.end_dt)
        n_active = len(heap)
        if n_active >= 3:
            snaps.append({"t": r.det_dt.isoformat(), "n_active": int(n_active)})
    print("моментов с ≥3 одновременными пожарами:", len(snaps))

    json.dump({"settlements": settlements, "oopt": oopt_list, "stations": stations},
              open(out_dir / "reference.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(snaps, open(out_dir / "snapshots.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"н.п.: {len(settlements)}, ООПТ: {len(oopt_list)}, станций: {len(stations)}")
    for st in stations:
        print(st["name"], st["lat"], st["lon"], st["demand_covered"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
