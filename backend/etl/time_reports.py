"""Замер времени формирования расчётной ведомости (PDF и XLSX) на демонстрационных сценариях.

Запуск: python -I time_reports.py <data_dir> [прогонов]
Результат: <data_dir>/report_timing.json — min/median/max по сценариям и форматам.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import planner, report  # noqa: E402


def main(data_dir: str, runs: int = 6) -> None:
    out = {"runs": runs, "scenarios": {}}
    for region in ("region_a", "region_b"):
        scn = planner.demo_scenario(region)
        plan = planner.make_plan(scn)
        res = {}
        for name, fn in (("pdf", lambda: report.pdf(plan, scn, "черновик")),
                         ("xlsx", lambda: report.xlsx(plan, scn))):
            times, size = [], 0
            for _ in range(runs):
                t0 = time.perf_counter()
                payload = fn()
                times.append(time.perf_counter() - t0)
                size = len(payload)
            res[name] = {"min_s": round(min(times), 3), "median_s": round(statistics.median(times), 3),
                         "max_s": round(max(times), 3), "size_kb": round(size / 1024)}
        res["fires"] = len(scn["fires"])
        res["stations"] = len(scn["stations"])
        out["scenarios"][region] = res
        print(region, res)
    json.dump(out, open(Path(data_dir) / "report_timing.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], int(args[1]) if len(args) > 1 else 6)
