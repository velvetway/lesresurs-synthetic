"""ML-модель вероятности разрастания пожара до 25 га и более.

Признаки известны на момент обнаружения + задержка начала тушения (управляемый
параметр: её сокращает оптимизатор). Для задержки задано монотонное
ограничение (больше задержка → не меньше вероятность).
Разбиение по времени: обучение 2016–2022, валидация 2023, тест 2024–2025.

Запуск: python -I train_growth.py <registry.parquet> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score

FEATURES = ["lat", "lon", "month", "doy", "hour", "wind_ms", "np_dist_km", "species", "subject",
            "delay_h", "n_active_lesn", "n_active_subj"]
CAT = ["species", "subject"]
THRESH_HA = 25.0


def active_counts(df: pd.DataFrame, key: list[str]) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int32)
    for _, grp in df.groupby(key):
        st = grp.det_dt.values
        en = grp.end_dt.values
        o = np.argsort(st)
        st_s, en_s = st[o], en[o]
        cnt = np.zeros(len(st_s), dtype=np.int32)
        for i in range(len(st_s)):
            j0 = max(0, i - 400)  # окно для скорости
            cnt[i] = int(((st_s[j0:i] <= st_s[i]) & (en_s[j0:i] > st_s[i])).sum())
        res = np.empty_like(cnt)
        res[o] = cnt
        out[grp.index.values] = res
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df.force_district == "НСС"].dropna(subset=["det_dt", "area_total_ha", "lat", "lon"]).copy()
    g = g.reset_index(drop=True)
    g["end_dt"] = g.loc_dt.fillna(g.liq_dt).fillna(g.det_dt + pd.Timedelta(hours=12))
    g["delay_h"] = ((g.start_dt - g.det_dt).dt.total_seconds() / 3600).clip(lower=0, upper=48)
    g = g[g.delay_h.notna()].reset_index(drop=True)
    g["month"] = g.det_dt.dt.month
    g["doy"] = g.det_dt.dt.dayofyear
    g["hour"] = g.det_dt.dt.hour
    g["species"] = g.species.fillna("нет данных").astype(str)
    g["subject"] = g.subject.astype(str)
    g["n_active_lesn"] = active_counts(g, ["subject", "lesnichestvo"])
    g["n_active_subj"] = active_counts(g, ["subject"])
    g["y"] = (g.area_total_ha >= THRESH_HA).astype(int)
    return g


def main(src: str, out: str) -> None:
    out_dir = Path(out)
    g = prepare(pd.read_parquet(src))
    tr, va, te = g[g.year <= 2022], g[g.year == 2023], g[g.year >= 2024]
    print("train", len(tr), "test", len(te), "pos rate", round(tr.y.mean(), 3), round(te.y.mean(), 3))
    mono = [1 if f == "delay_h" else 0 for f in FEATURES]
    model = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.06, loss_function="Logloss",
                               monotone_constraints=mono, verbose=200, random_seed=42, thread_count=-1)
    model.fit(Pool(tr[FEATURES], tr.y, cat_features=CAT), eval_set=Pool(va[FEATURES], va.y, cat_features=CAT))
    p = model.predict_proba(Pool(te[FEATURES], cat_features=CAT))[:, 1]
    pr_mask = te.subject == "Северотаёжный модельный край"
    metrics = {
        "target": f"итоговая площадь ≥ {THRESH_HA:g} га",
        "train_years": "2016–2022", "val_years": "2023 (ранняя остановка)", "test_years": "2024–2025",
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "positive_rate_test": round(float(te.y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(te.y, p)), 4),
        "pr_auc": round(float(average_precision_score(te.y, p)), 4),
        "brier": round(float(brier_score_loss(te.y, p)), 4),
        "roc_auc_region_a": round(float(roc_auc_score(te.y[pr_mask], p[pr_mask.values])), 4) if pr_mask.sum() > 50 else None,
        "n_test_region_a": int(pr_mask.sum()),
    }
    imp = model.get_feature_importance(prettified=True)
    metrics["feature_importance"] = {r["Feature Id"]: round(float(r["Importances"]), 2) for _, r in imp.iterrows()}
    # чувствительность к задержке (средняя по тестовой выборке модельного субъекта)
    base = te[pr_mask].copy() if pr_mask.sum() > 50 else te.copy()
    curve = {}
    for d in [0, 0.5, 1, 2, 3, 5]:
        base["delay_h"] = d
        curve[str(d)] = round(float(model.predict_proba(Pool(base[FEATURES], cat_features=CAT))[:, 1].mean()), 4)
    metrics["mean_p_by_delay_h"] = curve
    # --- регрессия итоговой площади (для метрики «площадь при локализации»)
    reg = CatBoostRegressor(iterations=1200, depth=6, learning_rate=0.05, loss_function="RMSE",
                            monotone_constraints=mono, verbose=0, random_seed=42, thread_count=-1)
    ytr, yva, yte = np.log1p(tr.area_total_ha), np.log1p(va.area_total_ha), np.log1p(te.area_total_ha)
    reg.fit(Pool(tr[FEATURES], ytr, cat_features=CAT), eval_set=Pool(va[FEATURES], yva, cat_features=CAT), early_stopping_rounds=100)
    pa = reg.predict(Pool(te[FEATURES], cat_features=CAT))
    base_mae = mean_absolute_error(yte, np.full(len(yte), np.median(ytr)))
    metrics["area_model"] = {
        "target": "ln(1 + итоговая площадь, га)",
        "mae_log": round(float(mean_absolute_error(yte, pa)), 4),
        "mae_log_baseline_median": round(float(base_mae), 4),
        "r2_log": round(float(r2_score(yte, pa)), 4),
        "median_abs_error_ha": round(float(np.median(np.abs(np.expm1(pa) - te.area_total_ha))), 3),
        "spearman": round(float(pd.Series(pa).corr(pd.Series(yte.values), method="spearman")), 4),
    }
    base2 = te[pr_mask].copy() if pr_mask.sum() > 50 else te.copy()
    area_curve = {}
    for d in [0, 0.5, 1, 2, 3, 5]:
        base2["delay_h"] = d
        area_curve[str(d)] = round(float(np.expm1(reg.predict(Pool(base2[FEATURES], cat_features=CAT))).mean()), 3)
    metrics["area_model"]["mean_area_by_delay_h"] = area_curve
    reg.save_model(str(out_dir / "area_model.cbm"))
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    model.save_model(str(out_dir / "growth_model.cbm"))
    json.dump({"features": FEATURES, "cat_features": CAT, **metrics},
              open(out_dir / "growth_model_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
