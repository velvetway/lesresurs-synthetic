"""Регрессионные тесты прототипа «ЛесРесурс» (pytest).

Покрывают: целостность плана и ведомости, валидацию ручного назначения, локализацию
ответа ассистента, устойчивость парсера реестра к падежу заголовков, воспроизводимость
ретро-симуляции и выгрузки. Запуск: pytest backend/tests -q
"""
from __future__ import annotations

import json

import pandas as pd


# ---------------------------------------------------------------- API и расчёт

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model"] is True          # ML-модель разрастания загружена


def test_regions_available(client):
    j = client.get("/api/regions").json()
    assert {r["slug"] for r in j["regions"]} >= {"region_a", "region_b"}


def test_plan_metrics_match_statement(client, scn):
    plan = client.get("/api/plan").json()
    assert plan["solver"]["status"] in ("OPTIMAL", "FEASIBLE")
    assert len(plan["statement"]) == len(scn["fires"])
    deficit = sum(row["deficit"]["groups"] for row in plan["statement"])
    assert deficit == plan["metrics"]["deficit_groups"]
    assert plan["metrics"]["weighted_damage"] >= plan["metrics"]["expected_damage_rub"] > 0
    assert all(row["deficit"]["groups"] >= 0 for row in plan["statement"])


def test_recalculate_is_deterministic(client):
    first = client.post("/api/plan/recalculate").json()["metrics"]
    second = client.post("/api/plan/recalculate").json()["metrics"]
    assert first == second


# ---------------------------------------------------------------- ручное назначение

def test_override_rejects_unknown_fire(client, scn):
    r = client.post("/api/plan/override", json={"fire_id": "NOPE", "station_ids": []})
    assert r.status_code == 400 and "NOPE" in r.json()["detail"]


def test_override_rejects_unknown_station(client, scn):
    fid = scn["fires"][0]["id"]
    r = client.post("/api/plan/override", json={"fire_id": fid, "station_ids": ["st999"]})
    assert r.status_code == 400 and "st999" in r.json()["detail"]


def test_override_rejects_unavailable_station(client, scn):
    fid, sid = scn["fires"][0]["id"], scn["stations"][0]["id"]
    client.put(f"/api/stations/{sid}", json={"available": False})
    try:
        r = client.post("/api/plan/override", json={"fire_id": fid, "station_ids": [sid]})
        assert r.status_code == 400 and "недоступно" in r.json()["detail"]
    finally:
        client.put(f"/api/stations/{sid}", json={"available": True})


def test_override_accepts_valid_assignment(client, scn):
    plan = client.get("/api/plan").json()
    fid = next(row["fire_id"] for row in plan["statement"] if row["units"])
    sid = plan["assignment"][fid][0]
    r = client.post("/api/plan/override", json={"fire_id": fid, "station_ids": [sid]})
    assert r.status_code == 200
    assert r.json()["status"] == "изменён диспетчером"
    client.post("/api/plan/recalculate")


# ---------------------------------------------------------------- ассистент

def test_assistant_deficit_answer_is_localized(client):
    client.post("/api/region/region_a")                     # 10 пожаров — дефицит сил есть
    client.get("/api/plan")
    j = client.post("/api/assistant/chat", json={"message": "Есть ли дефицит сил?", "history": []}).json()
    assert "группы:" in j["answer"]
    assert "groups:" not in j["answer"]                      # внутренние ключи в ответ не попадают
    client.post("/api/region/region_a")


# ---------------------------------------------------------------- парсер реестра

def _registry_file(tmp_path, det_header):
    sys_path_registry = __import__("etl.load_registry", fromlist=["load_file"])
    header = ["Субъект РФ", "Лесничество", det_header, "Обнаружение / Время", "Ближайший населённый пункт",
              "Азимут, град", "Удалённость, км", "Координаты, град / Широта", "Координаты, град / Долгота",
              "Площадь, га / Момент обнаружения", "Площадь, га / Всего", "Скорость ветра, м/с"]
    rows = [["Северотаёжный модельный край", "Ключевское", "30.04.2025", "09:40", "с Ольхинка", 120, 4.0,
             "43° 50' 12\" N", "132° 16' 40\" E", 3.0, 4.0, 5]]
    raw = pd.DataFrame([["Реестр лесных пожаров 2025"], header,
                        ["Федеральный округ"] + [None] * (len(header) - 1)] + rows)
    path = tmp_path / "registry.xlsx"
    raw.to_excel(path, sheet_name="Реестр", header=False, index=False)
    return path, sys_path_registry


def test_registry_header_any_case(tmp_path):
    """Заголовок встречается и как «Дата обнаружения», и как «Обнаружение / Дата»."""
    for det_header in ("Дата обнаружения", "Обнаружение / Дата"):
        path, registry = _registry_file(tmp_path, det_header)
        df, rep = registry.load_file(path)
        assert rep["rows"] == 1
        assert str(df.iloc[0]["lesnichestvo"]) == "Ключевское"
        assert df.iloc[0]["det_dt"] == pd.Timestamp("2025-04-30 09:40")
        assert 43.8 < float(df.iloc[0]["lat"]) < 43.9
        assert float(df.iloc[0]["area_detect_ha"]) == 3.0


def test_registry_reports_missing_headers(tmp_path):
    path = tmp_path / "wrong.xlsx"
    pd.DataFrame([["Реестр лесных пожаров"], ["Федеральный округ", "что-то"],
                  ["Северотаёжный модельный край", "1"]]).to_excel(path, sheet_name="Реестр", header=False, index=False)
    _, registry = _registry_file(tmp_path, "Дата обнаружения")
    try:
        registry.load_file(path)
    except ValueError as exc:
        assert "обязательные колонки" in str(exc)
    else:                                                      # pragma: no cover
        raise AssertionError("файл без обязательных колонок должен отклоняться")


# ---------------------------------------------------------------- воспроизводимость

def test_retro_simulation_row_reproduced():
    """Строка закоммиченной ретро-симуляции пересчитывается тем же кодом."""
    from pathlib import Path

    from app.optimizer import evaluate, fifo, optimize
    from app.planner import build_context, historical_scenario

    base = Path(__file__).resolve().parents[1] / "data" / "regions" / "region_a"
    row = pd.read_csv(base / "retro_sim.csv").iloc[0]
    ctx = build_context(historical_scenario(row.t, "region_a"))
    a, info = optimize(ctx, time_limit_s=3)
    e_opt, e_fifo = evaluate(a, ctx), evaluate(fifo(ctx), ctx)
    assert info["status"] == "OPTIMAL"
    for key in ("expected_damage_rub", "weighted_damage", "norm_violations", "deficit_groups"):
        assert abs(e_opt[key] - row[f"opt_{key}"]) <= 1
        assert abs(e_fifo[key] - row[f"fifo_{key}"]) <= 1


def test_documented_metrics_of_models():
    """Метрики моделей в документации совпадают с сохранёнными."""
    from pathlib import Path

    meta = json.load(open(Path(__file__).resolve().parents[1] / "data" / "growth_model_meta.json", encoding="utf-8"))
    assert 0.70 < meta["roc_auc"] < 0.80
    assert meta["area_model"]["spearman"] > 0.40


# ---------------------------------------------------------------- выгрузки

def test_reports_render(client, scn):
    client.get("/api/plan")
    pdf = client.get("/api/plan/pdf")
    xlsx = client.get("/api/plan/xlsx")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert xlsx.status_code == 200 and xlsx.content[:2] == b"PK"
