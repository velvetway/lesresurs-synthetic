"""Проверка независимыми моделями: не является ли выигрыш артефактом оптимизации по той же модели.

Оптимизатор строит план по основной модели разрастания, а ущерб плана пересчитывается двумя
независимо обученными моделями (иные seed и разбиение по годам). Если снижение ущерба
сохраняется, вывод не зависит от конкретной модели.

Оценка идёт на всех доступных моментах (по одному на сутки); для каждого варианта модели
сохраняются seed, разбиение по годам и ROC-AUC на отложенной части, для снижения ущерба —
95 % бутстрэп-интервал по моментам.

Запуск: python -I alt_model_check.py <registry.parquet> <data_dir> [моментов] [регион]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import optimizer, planner                                    # noqa: E402
from app.modules import expected_damage, norms                        # noqa: E402
from etl.train_growth import CAT, FEATURES, prepare                   # noqa: E402

VARIANTS = {"alt_a": dict(seed=7, train_max=2021, val_year=2022),
            "alt_b": dict(seed=202, train_max=2023, val_year=2024)}


def train_variants(registry: str, data_dir: str) -> tuple[dict[str, str], dict[str, dict]]:
    from sklearn.metrics import roc_auc_score

    g = prepare(pd.read_parquet(registry))
    mono = [1 if f == "delay_h" else 0 for f in FEATURES]
    paths, meta = {}, {}
    for name, cfg in VARIANTS.items():
        tr, va = g[g.year <= cfg["train_max"]], g[g.year == cfg["val_year"]]
        te = g[g.year > cfg["val_year"]]
        m = CatBoostClassifier(iterations=600, depth=5, learning_rate=0.06, loss_function="Logloss",
                               monotone_constraints=mono, verbose=0, random_seed=cfg["seed"],
                               thread_count=-1)
        m.fit(Pool(tr[FEATURES], tr.y, cat_features=CAT),
              eval_set=Pool(va[FEATURES], va.y, cat_features=CAT))
        path = str(Path(data_dir) / f"growth_model_{name}.cbm")
        m.save_model(path)
        paths[name] = path
        auc = float("nan")
        if te.y.nunique() > 1:
            p_te = m.predict_proba(Pool(te[FEATURES], cat_features=CAT))[:, 1]
            auc = round(float(roc_auc_score(te.y, p_te)), 3)
        meta[name] = {"seed": cfg["seed"], "train_years_max": cfg["train_max"],
                      "val_year": cfg["val_year"], "n_train": int(len(tr)),
                      "roc_auc_holdout": auc}
        print(f"модель {name}: seed={cfg['seed']}, обучение до {cfg['train_max']}, ROC-AUC {auc}")
    return paths, meta


class AltModel:
    def __init__(self, path: str):
        self.m = CatBoostClassifier()
        self.m.load_model(path)

    def p(self, fire: dict, delay_h: float, n_lesn: int, n_subj: int) -> float:
        t = pd.Timestamp(fire["detected_at"])
        row = {"lat": fire["lat"], "lon": fire["lon"], "month": t.month, "doy": t.day_of_year,
               "hour": t.hour, "wind_ms": fire.get("wind_ms") or 0.0,
               "np_dist_km": fire.get("np_dist_km") or 5.0,
               "species": fire.get("species") or "нет данных",
               "subject": fire.get("subject") or planner.REGIONS[planner.DEFAULT_REGION]["name"],
               "delay_h": float(min(max(delay_h, 0), 48)),
               "n_active_lesn": n_lesn, "n_active_subj": n_subj}
        return float(self.m.predict_proba(Pool(pd.DataFrame([row])[FEATURES], cat_features=CAT))[0, 1])


def damage_under(alt: AltModel, ctx: dict, assign: dict) -> float:
    total = 0.0
    unserved = norms()["damage"]["unserved_delay_h"]
    for f in ctx["fires"]:
        fid = f["id"]
        sids = sorted(assign.get(fid, []), key=lambda s: ctx["t"][s][fid])
        t_arr = (ctx["elapsed"][fid] + ctx["t"][sids[0]][fid]) if sids else (ctx["elapsed"][fid] + unserved)
        p = alt.p(f, t_arr, f.get("n_active_lesn", 1), f.get("n_active_subj", 1))
        total += expected_damage(p)
    return total


def main(registry: str, data_dir: str, limit: int = 100000, region: str = "region_a",
         n_boot: int = 4000) -> None:
    paths, variants_meta = train_variants(registry, data_dir)
    alts = {n: AltModel(p) for n, p in paths.items()}
    snaps = planner.historical_snapshots(region, limit=100000)
    df = pd.DataFrame(snaps)
    df["day"] = df.t.str[:10]
    df = df.sort_values(["day", "n_active", "t"], ascending=[True, False, True]).drop_duplicates("day")
    df = df.head(limit)

    acc = {"main": {"opt": 0.0, "fifo": 0.0, "rows": []},
           **{n: {"opt": 0.0, "fifo": 0.0, "better": 0, "rows": []} for n in alts}}
    for _, row in df.iterrows():
        ctx = planner.build_context(planner.historical_scenario(row.t, region))
        a_opt, _ = optimizer.optimize(ctx)
        a_fifo = optimizer.fifo(ctx)
        m_opt = optimizer.evaluate(a_opt, ctx)["expected_damage_rub"]
        m_fifo = optimizer.evaluate(a_fifo, ctx)["expected_damage_rub"]
        acc["main"]["opt"] += m_opt
        acc["main"]["fifo"] += m_fifo
        acc["main"]["rows"].append((m_opt, m_fifo))
        for name, alt in alts.items():
            d_opt, d_fifo = damage_under(alt, ctx, a_opt), damage_under(alt, ctx, a_fifo)
            acc[name]["opt"] += d_opt
            acc[name]["fifo"] += d_fifo
            acc[name]["rows"].append((d_opt, d_fifo))
            acc[name]["better"] = acc[name].get("better", 0) + int(d_opt <= d_fifo)

    rng = np.random.default_rng(2026)

    def boot_ci(rows, n=n_boot):
        arr = np.array(rows, dtype=float)
        idx = rng.integers(0, len(arr), size=(n, len(arr)))
        red = 100 * (1 - arr[idx, 0].sum(axis=1) / arr[idx, 1].sum(axis=1))
        return [round(float(np.percentile(red, 2.5)), 2), round(float(np.percentile(red, 97.5)), 2)]

    out = {"region": region, "moments": len(df), "n_boot": n_boot,
           "variants": variants_meta,
           "main_model_ci95": boot_ci(acc["main"]["rows"]), "main_model": {
        "opt_mln": round(acc["main"]["opt"] / 1e6, 1), "fifo_mln": round(acc["main"]["fifo"] / 1e6, 1),
        "reduction_pct": round(100 * (1 - acc["main"]["opt"] / acc["main"]["fifo"]), 2)},
        "independent_models": {}}
    for name in alts:
        r = acc[name]
        out["independent_models"][name] = {
            "opt_mln": round(r["opt"] / 1e6, 1), "fifo_mln": round(r["fifo"] / 1e6, 1),
            "reduction_pct": round(100 * (1 - r["opt"] / r["fifo"]), 2),
            "ci95": boot_ci(r["rows"]),
            "share_moments_better": round(r["better"] / len(df), 3)}
    json.dump(out, open(Path(data_dir) / "alt_model_check.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1], int(args[2]) if len(args) > 2 else 250,
         args[3] if len(args) > 3 else "region_a")
