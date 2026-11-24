"""Ретроспективная симуляция: план ассистента (CP-SAT) против FIFO на исторических моментах
с ≥ 3 одновременными пожарами в районах наземных сил субъекта (синтетический набор 2016–2025).

Обе стратегии оцениваются одной функцией evaluate() при одинаковых допущениях
(условный справочник станций 1605-р, время в пути по прямой × 1,35, ML-модель разрастания).

Запуск: python -I retro_sim.py <data_dir> <region_a|region_b> [max_snapshots]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.optimizer import evaluate, fifo, optimize  # noqa: E402
from app.planner import build_context, historical_scenario, historical_snapshots  # noqa: E402


def main(out: str, region: str, limit: int = 100000) -> None:
    snaps = historical_snapshots(region, limit=100000)
    # один момент на сутки: момент с максимальным числом одновременных пожаров
    df = pd.DataFrame(snaps)
    df["day"] = df.t.str[:10]
    df = df.sort_values(["day", "n_active", "t"], ascending=[True, False, True]).drop_duplicates("day")
    df = df.head(limit)
    rows = []
    t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        scn = historical_scenario(r.t, region)
        if len(scn["fires"]) < 3:
            continue
        ctx = build_context(scn)
        a, info = optimize(ctx, time_limit_s=3)
        e_opt, e_fifo = evaluate(a, ctx), evaluate(fifo(ctx), ctx)
        rows.append({"t": r.t, "n_fires": len(scn["fires"]), "solver": info["status"], "solve_s": info["wall_time_s"],
                     **{f"opt_{k}": v for k, v in e_opt.items() if k != "per_fire"},
                     **{f"fifo_{k}": v for k, v in e_fifo.items() if k != "per_fire"}})
        if i % 25 == 0:
            print(i, len(df), round(time.time() - t0), "s", flush=True)
    res = pd.DataFrame(rows)
    out = Path(out) / "regions" / region
    res.to_csv(out / "retro_sim.csv", index=False)

    def agg(d: pd.DataFrame) -> dict:
        return {
            "n_moments": int(len(d)),
            "mean_fires": round(float(d.n_fires.mean()), 2),
            "expected_damage_opt_mln": round(float(d.opt_expected_damage_rub.sum() / 1e6), 2),
            "expected_damage_fifo_mln": round(float(d.fifo_expected_damage_rub.sum() / 1e6), 2),
            "damage_reduction_pct": round(float(100 * (1 - d.opt_expected_damage_rub.sum() / d.fifo_expected_damage_rub.sum())), 2),
            "weighted_damage_reduction_pct": round(float(100 * (1 - d.opt_weighted_damage.sum() / d.fifo_weighted_damage.sum())), 2),
            "norm_violations_opt": int(d.opt_norm_violations.sum()), "norm_violations_fifo": int(d.fifo_norm_violations.sum()),
            "deficit_groups_opt": int(d.opt_deficit_groups.sum()), "deficit_groups_fifo": int(d.fifo_deficit_groups.sum()),
            "deficit_vehicles_opt": int(d.opt_deficit_vehicles.sum()), "deficit_vehicles_fifo": int(d.fifo_deficit_vehicles.sum()),
            "share_moments_opt_better_or_equal": round(float((d.opt_weighted_damage <= d.fifo_weighted_damage + 1).mean()), 3),
            "solver_optimal_share": round(float((d.solver == "OPTIMAL").mean()), 3),
            "solve_time_median_s": round(float(d.solve_s.median()), 3),
            "solve_time_max_s": round(float(d.solve_s.max()), 3),
        }
    summary = {"all": agg(res)}
    for lo, hi, name in ((3, 4, "3–4 пожара"), (5, 7, "5–7 пожаров"), (8, 999, "8 и более")):
        sub = res[(res.n_fires >= lo) & (res.n_fires <= hi)]
        if len(sub):
            summary[name] = agg(sub)
    json.dump(summary, open(Path(out) / "retro_sim_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100000)
