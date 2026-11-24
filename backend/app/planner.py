"""Оркестрация: обстановка → риск → потребность → маршруты → оптимизация → расчётная ведомость."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from .core import DEFAULT_REGION, REGIONS, growth_model, norms, ref, reference, region_path
from .modules import assess_risk, compute_demand, expected_damage, route
from .optimizer import T_MAX_H, evaluate, fifo, optimize


def _active_counts(fires: list[dict]) -> tuple[dict, int]:
    by_l = {}
    for f in fires:
        by_l[f.get("lesnichestvo")] = by_l.get(f.get("lesnichestvo"), 0) + 1
    return {f["id"]: by_l[f.get("lesnichestvo")] - 1 for f in fires}, len(fires) - 1


def build_context(scn: dict) -> dict:
    now = pd.Timestamp(scn["now"])
    region = scn.get("region", DEFAULT_REGION)
    fires = [dict(f, region=region, subject=REGIONS[region]["name"]) for f in scn["fires"]]
    stations = [dict(s) for s in scn["stations"]]
    avail = [s for s in stations if s.get("available", True)]
    n_lesn, n_subj = _active_counts(fires)
    routes = {s["id"]: {f["id"]: route(s, f) for f in fires} for s in stations}
    t = {sid: {fid: r["t_first_h"] for fid, r in rs.items()} for sid, rs in routes.items()}
    elapsed = {f["id"]: max(0.0, (now - pd.Timestamp(f["detected_at"])).total_seconds() / 3600) for f in fires}
    gm = growth_model()
    P, D, P_inf, D_inf, A, A_inf = {}, {}, {}, {}, {}, {}
    unserved = norms()["damage"]["unserved_delay_h"]
    for f in fires:
        fid = f["id"]
        # фактическая одновременность (если передана, например в ретро-симуляции) имеет приоритет
        n_lesn[fid] = f.get("n_active_lesn", n_lesn[fid])
        n_subj_f = f.get("n_active_subj", n_subj)
        nearest_t = min((t[s["id"]][fid] for s in avail), default=unserved)
        risk = assess_risk(f, elapsed[fid] + nearest_t, n_lesn[fid], n_subj_f)
        f["risk"] = risk
        f["demand"] = compute_demand(f, risk)
        f["np_dist_km"] = risk["nearest_np"]["km"]
        sids = [s["id"] for s in stations]
        delays = [elapsed[fid] + t[s][fid] for s in sids] + [elapsed[fid] + unserved]
        # модель гарантирует монотонность по возрастанию задержки — подаём задержки отсортированными
        order = sorted(range(len(delays)), key=lambda i: delays[i])
        pr_sorted = gm.predict(f, [delays[i] for i in order], n_lesn[fid], n_subj_f)
        ar_sorted = gm.last_area() or [0.0] * len(delays)
        pr = [0.0] * len(delays)
        ar = [0.0] * len(delays)
        for rank, i in enumerate(order):
            pr[i] = pr_sorted[rank]
            ar[i] = max(ar_sorted[rank], f.get("area_ha", 0))
        A[fid] = {sid: round(ar[i], 2) for i, sid in enumerate(sids)}
        A_inf[fid] = round(ar[-1], 2)
        P[fid] = {sid: round(pr[i], 4) for i, sid in enumerate(sids)}
        D[fid] = {sid: expected_damage(pr[i]) for i, sid in enumerate(sids)}
        P_inf[fid] = round(pr[-1], 4)
        D_inf[fid] = expected_damage(pr[-1])
    return {"fires": fires, "stations": stations, "routes": routes, "t": t, "elapsed": elapsed,
            "P": P, "D": D, "P_inf": P_inf, "D_inf": D_inf, "A": A, "A_inf": A_inf, "now": scn["now"]}


def _statement(ctx: dict, ev: dict) -> list[dict]:
    """Расчётная ведомость: по каждому пожару — потребность, назначение, обоснование."""
    st = {s["id"]: s for s in ctx["stations"]}
    out = []
    for f in sorted(ctx["fires"], key=lambda f: -f["risk"]["weight"]):
        pf = ev["per_fire"][f["id"]]
        d = f["demand"]
        lines = []
        for ln in d["lines"]:
            code = ln["code"]
            need = pf["need"]["groups" if code == "groups" else code]
            sup = pf["supply"]["groups" if code == "groups" else code]
            row = dict(ln)
            row.update({"need": need, "assigned": sup, "deficit": pf["deficit"]["groups" if code == "groups" else code]})
            if code == "groups":
                row["need_people"] = need * norms()["group"]["size"]
                row["assigned_people"] = sum(st[s]["people"] for s in pf["stations"])
            lines.append(row)
        if pf["late_reinforced"]:
            lines.append({"resource": "Усиление ×2 (прибытие позже сокращённого норматива)", "code": "reinf",
                          "qty": d["k_if_late"], "need": d["k_if_late"], "assigned": None, "deficit": 0,
                          "norm": f"ветер > 5 м/с или КПО ≥ 4 → предельное время {d['arrival_limit_reduced_h']:g} ч, иначе силы ×2",
                          "ref": d["reinforcement_ref"]})
        units = [{"id": s, "name": st[s]["name"], "travel_h": ctx["t"][s][f["id"]],
                  "km": ctx["routes"][s][f["id"]]["km"], "groups": st[s]["groups"], "people": st[s]["people"],
                  "vehicles": st[s]["vehicles"], "geometry": ctx["routes"][s][f["id"]]["geometry"],
                  "method": ctx["routes"][s][f["id"]]["method"],
                  "heavy_h": ctx["routes"][s][f["id"]]["t_heavy_h"] if (st[s]["vehicles"].get("ТЛП", 0) or st[s]["vehicles"].get("ТБ", 0)) else None}
                 for s in pf["stations"]]
        out.append({
            "fire_id": f["id"], "fire_name": f["name"], "lesnichestvo": f.get("lesnichestvo"),
            "risk": f["risk"], "demand": {k: v for k, v in d.items() if k != "lines"},
            "lines": lines, "units": units, "arrival_h": pf["arrival_h"], "elapsed_h": round(ctx["elapsed"][f["id"]], 2),
            "norm_ok": pf["norm_ok"], "late_reinforced": pf["late_reinforced"],
            "p_growth_at_arrival": pf["p_growth_at_arrival"], "expected_damage_rub": pf["expected_damage_rub"],
            "expected_area_ha": pf["expected_area_ha"],
            "deficit": pf["deficit"],
        })
    return out


def make_plan(scn: dict, fixed: dict | None = None) -> dict:
    ctx = build_context(scn)
    assign, info = optimize(ctx, fixed=fixed)
    ev = evaluate(assign, ctx)
    base_assign = fifo(ctx)
    ev_base = evaluate(base_assign, ctx)
    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario_now": scn["now"],
        "solver": info,
        "assignment": assign,
        "metrics": {k: v for k, v in ev.items() if k != "per_fire"},
        "baseline": {"name": "FIFO (по очереди обнаружения, ближайшие свободные силы)",
                     "assignment": base_assign, "metrics": {k: v for k, v in ev_base.items() if k != "per_fire"},
                     "per_fire": {fid: {"stations": v["stations"], "arrival_h": v["arrival_h"], "deficit": v["deficit"]}
                                  for fid, v in ev_base["per_fire"].items()}},
        "statement": _statement(ctx, ev),
        "stations": [{**s, "assigned_to": next((fid for fid, ss in assign.items() if s["id"] in ss), None)}
                     for s in ctx["stations"]],
        "fires": [{k: f[k] for k in ("id", "name", "lat", "lon", "detected_at", "area_ha", "fire_type", "wind_ms",
                                     "weather_class", "lesnichestvo", "species")} | {"category": f["risk"]["category"]}
                  for f in ctx["fires"]],
        "fixed": fixed or {},
        "region": scn.get("region", DEFAULT_REGION),
    }
    plan["deficit_summary"] = summarize_deficit(plan)
    return plan


def summarize_deficit(plan: dict) -> dict:
    tot = {"groups": 0, "АЦЛ": 0, "ТЛП": 0, "ВЛП": 0, "ТБ": 0, "АГП1": 0}
    fires = []
    for row in plan["statement"]:
        if any(row["deficit"].values()):
            fires.append(row["fire_name"])
        for k in tot:
            tot[k] += row["deficit"].get(k, 0)
    return {"total": tot, "fires": fires, "has_deficit": bool(fires),
            "message": ("Ресурсов недостаточно: требуется усиление (межрайонное маневрирование, авиация, привлечение сил МЧС)."
                        if fires else "Потребность всех пожаров обеспечена.")}


# ---------------------------------------------------------------- сценарии

DEMO = {
    "region_a": {"now": "2019-05-12T14:10:00", "window_min": 40, "weather_class": 4,
                 "title": "Синтетический сценарий: Северотаёжный модельный край"},
    "region_b": {"now": "2023-09-18T16:50:00", "window_min": 45, "weather_class": 4,
                 "title": "Синтетический сценарий: Лесостепная модельная область"},
}


def stations_default(region: str = DEFAULT_REGION) -> list[dict]:
    return [dict(s, available=True) for s in reference(region)["stations"]]


def _fires(region: str) -> pd.DataFrame:
    return pd.read_parquet(region_path(region) / "fires.parquet")


def demo_scenario(region: str = DEFAULT_REGION) -> dict:
    """Ретроспективный сценарий из реестра: момент с наибольшим числом одновременных пожаров."""
    cfg = DEMO[region]
    h = _fires(region)
    now = pd.Timestamp(cfg["now"])
    act = h[(h.det_dt <= now) & (h.end_dt > now)].sort_values("det_dt").copy()
    n, w = len(act), cfg["window_min"]
    # время обнаружения приводится к одному диспетчерскому окну с сохранением порядка
    act["det_dt"] = [now - pd.Timedelta(minutes=w - round(w * i / max(n - 1, 1))) for i in range(n)]
    name = REGIONS[region]["name"]
    return scenario_from_rows(act, now, cfg["title"],
                              f"{n} пожаров, одновременно действовавших {now:%d.%m.%Y} в районах наземных сил ({name}): "
                              "координаты, лесничества, ветер и площадь при обнаружении — из синтетического набора; время обнаружения "
                              f"приведено к одному диспетчерскому окну ({w} мин). КПО по условиям погоды принят "
                              f"{['I', 'II', 'III', 'IV', 'V'][cfg['weather_class'] - 1]} (пик весеннего сезона) — метеоданных в реестре нет.",
                              cfg["weather_class"], region)


def scenario_from_rows(act: pd.DataFrame, now: pd.Timestamp, title: str, desc: str, weather_class: int = 4,
                       region: str = DEFAULT_REGION) -> dict:
    fires = []
    for i, r in enumerate(act.itertuples(), 1):
        tot = float(r.area_total_ha) if pd.notna(r.area_total_ha) and r.area_total_ha > 0 else 1.0
        area = r.area_detect_ha if pd.notna(r.area_detect_ha) and r.area_detect_ha > 0 else min(tot, 3.0)
        crown = pd.notna(r.area_crown_ha) and r.area_crown_ha > 0
        fires.append({
            "id": f"f{i}", "name": f"Пожар № {i} — {r.lesnichestvo}",
            "lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5),
            "detected_at": r.det_dt.isoformat(), "area_ha": round(float(area), 2),
            "fire_type": "верховой" if crown else "низовой", "wind_ms": float(r.wind_ms) if pd.notna(r.wind_ms) else 0.0,
            "weather_class": weather_class, "species": r.species if isinstance(r.species, str) else None,
            "lesnichestvo": r.lesnichestvo, "uch_lesnichestvo": r.uch_lesnichestvo if isinstance(r.uch_lesnichestvo, str) else None,
            "source": "синтетический набор",
        })
    return {"title": title, "description": desc, "now": now.isoformat(), "fires": fires,
            "stations": stations_default(region), "region": region}


def historical_snapshots(region: str = DEFAULT_REGION, limit: int = 300) -> list[dict]:
    snaps = json.load(open(region_path(region) / "snapshots.json", encoding="utf-8"))
    seen, out = set(), []
    for s in snaps:
        key = (s["t"][:13], s["n_active"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"t": s["t"], "n_active": s["n_active"]})
    return out[-limit:]


def historical_scenario(t: str, region: str = DEFAULT_REGION) -> dict:
    h = _fires(region)
    now = pd.Timestamp(t)
    act = h[(h.det_dt <= now) & (h.end_dt > now)].sort_values("det_dt")
    m = now.month
    wc = 4 if m in (4, 5, 10) else 3
    return scenario_from_rows(act, now, f"Ретроспективный сценарий: {REGIONS[region]['name']}, {now:%d.%m.%Y %H:%M}",
                              f"{len(act)} одновременно действующих пожаров (синтетический набор). КПО по погоде принят {wc}.", wc, region)


def validate_override(scn: dict, fire_id: str, station_ids: list[str]) -> list[str]:
    """Проверка ручного назначения диспетчера: возвращает список ошибок (пустой — назначение корректно).

    Проверяется, что пожар есть в текущей обстановке, подразделения есть в справочнике, доступны
    и успевают прибыть за предельное время T_max. Назначение «наугад» иначе молча терялось бы:
    недостижимое подразделение не попадает в модель и не влияет на план.
    """
    fires = {f["id"]: f for f in scn["fires"]}
    stations = {s["id"]: s for s in scn["stations"]}
    if fire_id not in fires:
        return [f"пожар «{fire_id}» отсутствует в текущей обстановке"]
    errors = []
    for sid in station_ids:
        st = stations.get(sid)
        if st is None:
            errors.append(f"подразделение «{sid}» не найдено в справочнике")
            continue
        if not st.get("available", True):
            errors.append(f"подразделение «{st['name']}» недоступно")
            continue
        r = route(st, fires[fire_id])
        if r["t_first_h"] > T_MAX_H:
            errors.append(f"«{st['name']}» не успевает: {r['t_first_h']:.1f} ч в пути (предел {T_MAX_H:g} ч)")
    return errors



def acts_list() -> list[dict]:
    return [{"code": k, **v} for k, v in norms()["acts"].items()]


__all__ = ["make_plan", "demo_scenario", "historical_scenario", "historical_snapshots", "acts_list", "ref"]
