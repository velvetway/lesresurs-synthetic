"""Событийная ретроспективная симуляция: ассистент (CP-SAT) против FIFO и против факта по синтетическому набору.

Пожары обрабатываются в порядке обнаружения. При каждом обнаружении политика распределяет свободные
станции между ещё не обслуженными действующими пожарами (уже направленные силы не отзываются).
Станция занята до локализации пожара по синтетическому набору плюс время обратного пути.

Факт: задержка начала тушения (графа «Начало тушения» − обнаружение) и численность сил по синтетическому набору.
Все три варианта оцениваются одной ML-моделью: P(итоговая площадь ≥ 25 га) и ожидаемая площадь
при задержке начала тушения, ожидаемый ущерб — по медианам реестра 2025 (приказ № 566).

Рассматриваются эпизоды (непрерывные периоды с действующими пожарами в районах НСС), в которых
одновременно действовало не менее 3 пожаров.

Запуск: python -I retro_events.py <data_dir> <region_a|region_b>
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import REGIONS, growth_model, region_path  # noqa: E402
from app.modules import expected_damage  # noqa: E402
from app.optimizer import fifo, optimize  # noqa: E402
from app.planner import build_context, stations_default  # noqa: E402

UNSERVED_H = 8.0


def episodes(h: pd.DataFrame) -> list[pd.DataFrame]:
    h = h.sort_values("det_dt").reset_index(drop=True)
    out, cur, cur_end = [], [], None
    for i, r in h.iterrows():
        if cur and r.det_dt >= cur_end:
            out.append(h.loc[cur])
            cur = []
        cur.append(i)
        cur_end = r.end_dt if len(cur) == 1 else max(cur_end, r.end_dt)
    if cur:
        out.append(h.loc[cur])
    return out


def max_concurrency(ep: pd.DataFrame) -> int:
    st, en = ep.det_dt.values, ep.end_dt.values
    return int(max(((st <= t) & (en > t)).sum() for t in st))


def concurrency(h: pd.DataFrame) -> dict:
    """Фактическое число других действующих пожаров при обнаружении (в лесничестве и в субъекте)."""
    out = {}
    st, en, les = h.det_dt.values, h.end_dt.values, h.lesnichestvo.astype(object).to_numpy()
    for i, r in h.iterrows():
        act = (st <= np.datetime64(r.det_dt)) & (en > np.datetime64(r.det_dt))
        n_s = int(act.sum()) - 1
        n_l = int((act & (les == r.lesnichestvo)).astype(bool).sum()) - 1
        out[i] = (max(n_l, 0), max(n_s, 0))
    return out


CONC: dict = {}


def fire_dict(idx, r) -> dict:
    wc = 4 if r.det_dt.month in (4, 5, 10) else 3
    tot = float(r.area_total_ha) if pd.notna(r.area_total_ha) and r.area_total_ha > 0 else 1.0
    area = r.area_detect_ha if pd.notna(r.area_detect_ha) and r.area_detect_ha > 0 else min(tot, 3.0)
    return {"id": f"h{idx}", "name": f"Пожар {idx}", "lat": float(r.lat), "lon": float(r.lon),
            "detected_at": r.det_dt.isoformat(), "area_ha": float(area),
            "fire_type": "верховой" if (pd.notna(r.area_crown_ha) and r.area_crown_ha > 0) else "низовой",
            "wind_ms": float(r.wind_ms) if pd.notna(r.wind_ms) else 0.0, "weather_class": wc,
            "species": r.species if isinstance(r.species, str) else None, "lesnichestvo": r.lesnichestvo,
            "n_active_lesn": CONC[int(idx)][0], "n_active_subj": CONC[int(idx)][1]}


def run_policy(ep: pd.DataFrame, region: str, policy: str) -> dict:
    stations = stations_default(region)
    st_by = {s["id"]: s for s in stations}
    busy_until = {s["id"]: pd.Timestamp.min for s in stations}
    served: dict[str, dict] = {}
    rows = {f"h{i}": r for i, r in ep.iterrows()}
    solve_times = []
    for t in sorted(ep.det_dt.unique()):
        t = pd.Timestamp(t)
        open_ids = [fid for fid, r in rows.items() if r.det_dt <= t and r.end_dt > t and fid not in served]
        if not open_ids:
            continue
        free = [dict(st_by[s], available=True) for s, bu in busy_until.items() if bu <= t]
        if not free:
            continue
        scn = {"now": t.isoformat(), "region": region, "fires": [fire_dict(fid[1:], rows[fid]) for fid in open_ids],
               "stations": free}
        ctx = build_context(scn)
        t0 = time.time()
        assign = optimize(ctx, time_limit_s=2)[0] if policy == "assistant" else fifo(ctx)
        solve_times.append(time.time() - t0)
        fmap = {f["id"]: f for f in ctx["fires"]}
        for fid, sids in assign.items():
            if not sids:
                continue
            f = fmap[fid]
            lead = min(sids, key=lambda s: ctx["t"][s][fid])
            arr = ctx["elapsed"][fid] + ctx["t"][lead][fid]
            late = arr > f["demand"]["arrival_limit_reduced_h"] and f["demand"]["hard_weather"]
            k = f["demand"]["k_if_late"] if late else 1
            need_g = f["demand"]["groups"] * k
            sup_g = sum(st_by[s]["groups"] for s in sids)
            served[fid] = {"arrival_h": arr, "P": ctx["P"][fid][lead], "A": ctx["A"][fid][lead],
                           "norm_ok": arr <= f["demand"]["arrival_limit_h"], "deficit_groups": max(0, need_g - sup_g),
                           "need_groups": need_g, "limit_h": f["demand"]["arrival_limit_h"]}
            for s in sids:
                back = ctx["t"][s][fid]
                busy_until[s] = rows[fid].end_dt + pd.Timedelta(hours=back)
    # не обслуженные за эпизод
    out = {}
    for fid, r in rows.items():
        if fid in served:
            out[fid] = served[fid]
        else:
            scn = {"now": (r.det_dt + pd.Timedelta(hours=UNSERVED_H)).isoformat(), "region": region,
                   "fires": [fire_dict(fid[1:], r)], "stations": stations[:1]}
            ctx = build_context(scn)
            f = ctx["fires"][0]
            out[fid] = {"arrival_h": None, "P": ctx["P_inf"][f["id"]], "A": ctx["A_inf"][f["id"]], "norm_ok": False,
                        "deficit_groups": f["demand"]["groups"], "need_groups": f["demand"]["groups"],
                        "limit_h": f["demand"]["arrival_limit_h"]}
    return {"fires": out, "solve_times": solve_times}


def fact(ep: pd.DataFrame, region: str, ref: dict) -> dict:
    """Факт по синтетическому набору: та же модель, задержка и силы — фактические."""
    out = {}
    for i, r in ep.iterrows():
        fid = f"h{i}"
        delay = (r.start_dt - r.det_dt).total_seconds() / 3600 if pd.notna(r.start_dt) else None
        if delay is not None and (delay < 0 or delay > 72):
            delay = None
        d = delay if delay is not None else UNSERVED_H
        scn = {"now": r.det_dt.isoformat(), "region": region, "fires": [fire_dict(i, r)], "stations": ref[:1]}
        ctx = build_context(scn)
        f = ctx["fires"][0]
        gm = growth_model()
        p = gm.predict(f, [d], f["n_active_lesn"], f["n_active_subj"])[0]
        a = max((gm.last_area() or [0.0])[0], f["area_ha"])
        need = f["demand"]["groups"]
        out[fid] = {"arrival_h": delay, "P": p, "A": a,
                    "norm_ok": delay is not None and delay <= f["demand"]["arrival_limit_h"],
                    "deficit_groups": max(0, need - math.ceil((r.ppl_all or 0) / 6)), "need_groups": need,
                    "limit_h": f["demand"]["arrival_limit_h"], "observed_large": bool(r.area_total_ha >= 25),
                    "observed_area": float(r.area_total_ha)}
    return out


def summarize(res: dict, name: str) -> dict:
    vals = list(res.values())
    n = len(vals)
    arr = [v["arrival_h"] for v in vals if v["arrival_h"] is not None]
    P = np.array([v["P"] for v in vals])
    A = np.array([v["A"] for v in vals])
    D = np.array([expected_damage(p) for p in P])
    s = {"policy": name, "n_fires": n,
         "served_share": round(len(arr) / n, 4),
         "mean_arrival_h": round(float(np.mean(arr)), 3) if arr else None,
         "median_arrival_h": round(float(np.median(arr)), 3) if arr else None,
         "norm_ok_share": round(float(np.mean([v["norm_ok"] for v in vals])), 4),
         "expected_large_share": round(float(P.mean()), 4),
         "expected_mean_area_ha": round(float(A.mean()), 2),
         "expected_damage_mln": round(float(D.sum() / 1e6), 3),
         "deficit_groups": int(sum(v["deficit_groups"] for v in vals))}
    if vals and "observed_large" in vals[0]:
        s["observed_large_share"] = round(float(np.mean([v["observed_large"] for v in vals])), 4)
        s["observed_mean_area_ha"] = round(float(np.mean([v["observed_area"] for v in vals])), 2)
    return s


def main(data_dir: str, region: str) -> None:
    h = pd.read_parquet(region_path(region) / "fires.parquet").sort_values("det_dt").reset_index(drop=True)
    CONC.update(concurrency(h))
    eps = [e for e in episodes(h) if max_concurrency(e) >= 3]
    print(REGIONS[region]["name"], "эпизодов с ≥3 одновременными пожарами:", len(eps), "пожаров:", sum(len(e) for e in eps), flush=True)
    ref = stations_default(region)
    acc = {"assistant": {}, "fifo": {}, "fact": {}}
    years = {}
    solve = []
    t0 = time.time()
    for k, ep in enumerate(eps):
        for pol in ("assistant", "fifo"):
            r = run_policy(ep, region, pol)
            acc[pol].update(r["fires"])
            if pol == "assistant":
                solve += r["solve_times"]
        acc["fact"].update(fact(ep, region, ref))
        for i, r in ep.iterrows():
            years[f"h{i}"] = int(r.det_dt.year)
        if k % 20 == 0:
            print(k, len(eps), round(time.time() - t0), "s", flush=True)
    summary = {"region": REGIONS[region]["name"], "episodes": len(eps),
               "solve_time_median_s": round(float(np.median(solve)), 4) if solve else None,
               "solve_time_p95_s": round(float(np.percentile(solve, 95)), 4) if solve else None,
               "all_years": [summarize(acc[p], p) for p in ("assistant", "fifo", "fact")]}
    ids25 = [fid for fid, y in years.items() if y == 2025]
    if ids25:
        summary["year_2025"] = [summarize({k: acc[p][k] for k in ids25}, p) for p in ("assistant", "fifo", "fact")]
    rows = [{"fire": fid, "year": years[fid], **{f"{p}_{k}": acc[p][fid].get(k) for p in acc for k in ("arrival_h", "P", "A", "norm_ok", "deficit_groups")}}
            for fid in years]
    out = region_path(region)
    pd.DataFrame(rows).to_csv(out / "retro_events.csv", index=False)
    json.dump(summary, open(out / "retro_events_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
