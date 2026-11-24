"""Модуль 3. Оптимизатор распределения сил (OR-Tools CP-SAT) и базовый алгоритм FIFO.

Постановка (минимизация взвешенного ожидаемого ущерба, приказ № 566):

  min  Σ_f w_f · [ Σ_s z_sf · (D_f(t_sf) + λ_n·[t_sf > T51_f]) + u_f · (D_f(∞) + λ_n) ]
       + λ_g Σ_f w_f · дефицит_групп_f + λ_v Σ_f w_f · дефицит_техники_f + Σ y_sf (ε + ε_t·t_sf + ρ·c_s)

  где ρ·c_s — цена вывода станции из резерва (c_s — доля исторических пожаров, прикрываемых станцией,
  к средней доле): без неё оптимизатор «близоруко» отправляет лишние станции ради одной единицы техники
  и оставляет без сил последующие пожары (выявлено в событийной ретро-симуляции).

  при ограничениях:
   (1) Σ_f y_sf ≤ 1              — одно подразделение направляется на один пожар;
   (2) z_sf ≤ y_sf, Σ_s z_sf + u_f = 1, u_f + y_sf ≤ 1 — у обслуживаемого пожара одно «ведущее»
       подразделение (определяет время прибытия);
   (3) Σ_s y_sf · ресурс_s + дефицит_f ≥ потребность_f · (1 + (k−1)·Σ_s z_sf·late_sf)
                                 — силы ≥ потребности, удвоение при опоздании (п. 52 Правил);
   (4) y_sf = 0, если t_sf > T_max  — недоступные подразделения не рассматриваются;
   (5) прибытие ≤ норматива п. 51 (T51_f = 1/2/3 ч) — штрафное ограничение λ_n (приоритет после дефицита сил).

D_f(t) = D_мал + P_f(t)·(D_крупн − D_мал), где P_f(t) — вероятность разрастания ≥ 25 га от ML-модели
при задержке начала тушения t (с учётом времени, уже прошедшего с обнаружения).
"""
from __future__ import annotations

from ortools.sat.python import cp_model

VEH = ("АЦЛ", "ТЛП", "ВЛП", "ТБ", "АГП1")
LAMBDA_GROUP = 1000     # тыс. руб. за недостающую группу (× вес риска): обеспечение потребности — приоритет
LAMBDA_VEH = 60        # тыс. руб. за недостающую единицу техники
LAMBDA_NORM = 1000     # тыс. руб. за прибытие позже норматива п. 51 (или отсутствие сил)
EPS_STATION = 15        # базовая «стоимость» вывода подразделения из резерва
EPS_PER_HOUR = 10
RESERVE_COST = 1000     # цена вывода станции из резерва × (доля прикрываемых ею исторических пожаров) / средняя доля
T_MAX_H = 6.0


def _need(fx: dict, late: bool) -> dict:
    k = fx["demand"]["k_if_late"] if late else 1
    d = fx["demand"]
    return {"groups": d["groups"] * k, **{v: d["vehicles"][v] * k for v in VEH}}


def evaluate(assign: dict[str, list[str]], ctx: dict) -> dict:
    """Метрики плана: одинаковые для оптимизатора, FIFO и ручной корректировки диспетчером."""
    fires, st = ctx["fires"], {s["id"]: s for s in ctx["stations"]}
    per_fire, tot_w, tot_d, def_g, def_v, viol, uncovered = {}, 0.0, 0.0, 0, 0, 0, 0
    exp_large, exp_area = 0.0, 0.0
    for f in fires:
        fid = f["id"]
        sids = sorted(assign.get(fid, []), key=lambda s: ctx["t"][s][fid])
        if sids:
            lead = sids[0]
            t_arr = ctx["elapsed"][fid] + ctx["t"][lead][fid]
            D = ctx["D"][fid][lead]
        else:
            lead, t_arr, D = None, None, ctx["D_inf"][fid]
            uncovered += 1
        late = t_arr is not None and t_arr > f["demand"]["arrival_limit_reduced_h"] and f["demand"]["hard_weather"]
        need = _need(f, late)
        sup = {"groups": sum(st[s]["groups"] for s in sids), **{v: sum(st[s]["vehicles"].get(v, 0) for s in sids) for v in VEH}}
        deficit = {"groups": max(0, need["groups"] - sup["groups"])}
        transport_def = max(0, need["groups"] - (sup["АЦЛ"] + sup["АГП1"]))
        for v in ("АЦЛ", "ТЛП", "ВЛП", "ТБ"):
            deficit[v] = max(0, need[v] - sup[v])
        deficit["АГП1"] = max(0, transport_def - deficit["АЦЛ"])
        norm_ok = t_arr is not None and t_arr <= f["demand"]["arrival_limit_h"]
        if not norm_ok:
            viol += 1
        w = f["risk"]["weight"]
        p_arr = ctx["P"][fid][lead] if lead else ctx["P_inf"][fid]
        exp_large += p_arr
        exp_area += (ctx["A"][fid][lead] if lead else ctx["A_inf"][fid]) if ctx.get("A") else 0.0
        tot_w += w * D
        tot_d += D
        def_g += deficit["groups"]
        def_v += sum(deficit[v] for v in VEH)
        per_fire[fid] = {
            "stations": sids, "lead": lead, "arrival_h": None if t_arr is None else round(t_arr, 2),
            "travel_h": None if lead is None else ctx["t"][lead][fid], "late_reinforced": late,
            "need": need, "supply": sup, "deficit": deficit, "norm_ok": norm_ok,
            "p_growth_at_arrival": p_arr,
            "expected_area_ha": round((ctx["A"][fid][lead] if lead else ctx["A_inf"][fid]), 2) if ctx.get("A") else None,
            "expected_damage_rub": round(D),
        }
    return {"per_fire": per_fire, "weighted_damage": round(tot_w), "expected_damage_rub": round(tot_d),
            "deficit_groups": def_g, "deficit_vehicles": def_v, "norm_violations": viol, "uncovered": uncovered,
            "stations_used": sum(len(v) for v in assign.values()),
            "expected_large_fires": round(exp_large, 3), "expected_area_ha": round(exp_area, 1),
            "n_fires": len(fires)}


def optimize(ctx: dict, fixed: dict[str, list[str]] | None = None, time_limit_s: float = 5.0) -> tuple[dict, dict]:
    fires, stations = ctx["fires"], [s for s in ctx["stations"] if s.get("available", True)]
    m = cp_model.CpModel()
    y, z, u = {}, {}, {}
    for f in fires:
        fid = f["id"]
        u[fid] = m.NewBoolVar(f"u_{fid}")
        for s in stations:
            if ctx["t"][s["id"]][fid] <= T_MAX_H:
                y[s["id"], fid] = m.NewBoolVar(f"y_{s['id']}_{fid}")
                z[s["id"], fid] = m.NewBoolVar(f"z_{s['id']}_{fid}")
                m.Add(z[s["id"], fid] <= y[s["id"], fid])
        m.Add(sum(z[k] for k in z if k[1] == fid) + u[fid] == 1)
        for k in y:
            if k[1] == fid:
                m.Add(u[fid] + y[k] <= 1)   # если силы направлены, у пожара есть ведущее подразделение
    for s in stations:
        m.Add(sum(y[k] for k in y if k[0] == s["id"]) <= 1)

    obj = []
    for f in fires:
        fid, w = f["id"], f["risk"]["weight"]
        hw = f["demand"]["hard_weather"]
        late_terms = [z[k] for k in z if k[1] == fid
                      and ctx["elapsed"][fid] + ctx["t"][k[0]][fid] > f["demand"]["arrival_limit_reduced_h"] and hw]
        late = sum(late_terms) if late_terms else 0
        kx = f["demand"]["k_if_late"] - 1
        base = _need(f, False)
        sup = {r: sum(y[k] * (next(s for s in stations if s["id"] == k[0])["groups"] if r == "groups"
                              else next(s for s in stations if s["id"] == k[0])["vehicles"].get(r, 0))
                      for k in y if k[1] == fid) for r in ("groups",) + VEH}
        dg = m.NewIntVar(0, 100000, f"dg_{fid}")
        m.Add(sup["groups"] + dg >= base["groups"] + base["groups"] * kx * late)
        obj.append(LAMBDA_GROUP * w * dg)
        for r in ("АЦЛ", "ТЛП", "ВЛП", "ТБ"):
            dv = m.NewIntVar(0, 100000, f"d{r}_{fid}")
            m.Add(sup[r] + dv >= base[r] + base[r] * kx * late)
            obj.append(LAMBDA_VEH * w * dv)
        dt = m.NewIntVar(0, 100000, f"dtr_{fid}")
        m.Add(sup["АЦЛ"] + sup["АГП1"] + dt >= base["groups"] + base["groups"] * kx * late)
        obj.append(LAMBDA_VEH * w * dt)
        lim = f["demand"]["arrival_limit_h"]
        for k in z:
            if k[1] == fid:
                late51 = ctx["elapsed"][fid] + ctx["t"][k[0]][fid] > lim
                obj.append(int(w * ctx["D"][fid][k[0]] / 1000 + (LAMBDA_NORM * w if late51 else 0)) * z[k])
        obj.append(int(w * ctx["D_inf"][fid] / 1000 + LAMBDA_NORM * w) * u[fid])
    cov = {s["id"]: s.get("demand_covered", 1) for s in stations}
    mean_cov = max(sum(cov.values()) / max(len(cov), 1), 1)
    for k in y:
        reserve = RESERVE_COST * cov[k[0]] / mean_cov
        obj.append(int(EPS_STATION + EPS_PER_HOUR * ctx["t"][k[0]][k[1]] + reserve) * y[k])

    if fixed:
        for fid, sids in fixed.items():
            for k in y:
                if k[1] == fid:
                    m.Add(y[k] == (1 if k[0] in sids else 0))
    m.Minimize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 8
    status = solver.Solve(m)
    assign = {f["id"]: [] for f in fires}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (sid, fid), var in y.items():
            if solver.Value(var):
                assign[fid].append(sid)
    info = {"status": solver.StatusName(status), "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            "wall_time_s": round(solver.WallTime(), 3), "n_vars": len(y) * 2 + len(u)}
    return assign, info


def fifo(ctx: dict) -> dict[str, list[str]]:
    """Базовый способ: пожары обслуживаются по очереди обнаружения, ближайшими свободными силами."""
    free = {s["id"] for s in ctx["stations"] if s.get("available", True)}
    st = {s["id"]: s for s in ctx["stations"]}
    assign = {}
    for f in sorted(ctx["fires"], key=lambda f: f["detected_at"]):
        fid = f["id"]
        cand = sorted((s for s in free if ctx["t"][s][fid] <= T_MAX_H), key=lambda s: ctx["t"][s][fid])
        chosen = []
        for s in cand:
            if not chosen:
                lead_t = ctx["elapsed"][fid] + ctx["t"][s][fid]
                late = lead_t > f["demand"]["arrival_limit_reduced_h"] and f["demand"]["hard_weather"]
                need = _need(f, late)
            chosen.append(s)
            g = sum(st[x]["groups"] for x in chosen)
            a = sum(st[x]["vehicles"].get("АЦЛ", 0) for x in chosen)
            if g >= need["groups"] and a >= need["АЦЛ"]:
                break
        for s in chosen:
            free.discard(s)
        assign[fid] = chosen
    return assign
