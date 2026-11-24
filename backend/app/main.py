"""ЛесРесурс — API ассистента поддержки принятия решений по ресурсному обеспечению пожаротушения."""
from __future__ import annotations

import copy
import itertools
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import assistant, report, storage
from .core import DEFAULT_REGION, REGIONS, growth_model, norms, reference
from .models import ChatRequest, Fire, OverrideRequest
from .planner import acts_list, demo_scenario, historical_scenario, historical_snapshots, make_plan, validate_override

app = FastAPI(title="ЛесРесурс API", version="0.9.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE: dict = {"region": DEFAULT_REGION, "scenario": None, "plan": None, "status": "черновик", "journal": []}
_ids = itertools.count(100)


def log(event: str, detail: str = "") -> None:
    STATE["journal"].insert(0, {"t": datetime.now().strftime("%H:%M:%S"), "event": event, "detail": detail})
    del STATE["journal"][60:]
    storage.log_event(STATE["region"], event, detail)


def scenario() -> dict:
    if STATE["scenario"] is None:
        STATE["scenario"] = demo_scenario(STATE["region"])
        log("Загружен сценарий", STATE["scenario"]["title"])
    return STATE["scenario"]


def recompute(fixed: dict | None = None, reason: str = "Пересчёт") -> dict:
    STATE["plan"] = make_plan(scenario(), fixed=fixed)
    STATE["status"] = "черновик" if not fixed else "изменён диспетчером"
    m = STATE["plan"]["metrics"]
    log(reason, f"{len(scenario()['fires'])} пож., {m['stations_used']} подр., нарушений норматива: {m['norm_violations']}")
    return STATE["plan"]


@app.get("/api/health")
def health():
    return {"ok": True, "model": growth_model().model is not None, "llm": assistant.provider_name()}


@app.get("/api/regions")
def get_regions():
    return {"current": STATE["region"], "regions": list(REGIONS.values())}


@app.post("/api/region/{slug}")
def set_region(slug: str):
    if slug not in REGIONS:
        raise HTTPException(404, "регион не найден")
    STATE["region"] = slug
    STATE["scenario"] = demo_scenario(slug)
    log("Выбран регион", REGIONS[slug]["name"])
    return recompute(reason="Автоматический расчёт")


@app.get("/api/reference")
def get_reference():
    region = STATE["region"]
    ref = reference(region)
    gm = growth_model().meta
    name = REGIONS[region]["name"]
    return {"settlements": ref["settlements"][:120], "oopt": ref["oopt"], "acts": acts_list(),
            "region": name, "region_slug": region, "region_norms": norms()["region_norms"][name],
            "model": {k: gm.get(k) for k in ("target", "roc_auc", "pr_auc", "brier", "roc_auc_region_a", "n_train", "n_test",
                                              "train_years", "test_years", "feature_importance", "mean_p_by_delay_h", "area_model")},
            "llm": assistant.provider_name()}


@app.get("/api/scenario")
def get_scenario():
    return scenario()


@app.post("/api/scenario/demo")
def load_demo():
    STATE["scenario"] = demo_scenario(STATE["region"])
    log("Загружен сценарий", STATE["scenario"]["title"])
    return recompute(reason="Автоматический расчёт")


@app.get("/api/scenario/history")
def list_history():
    return historical_snapshots(STATE["region"])


@app.post("/api/scenario/history/{t}")
def load_history(t: str):
    STATE["scenario"] = historical_scenario(t, STATE["region"])
    log("Загружен сценарий", STATE["scenario"]["title"])
    return recompute(reason="Автоматический расчёт")


@app.post("/api/fires")
def add_fire(f: dict):
    scn = scenario()
    fid = f"f{next(_ids)}"
    n = len(scn["fires"]) + 1
    base = {"id": fid, "name": f.get("name") or f"Пожар № {n} — новый", "detected_at": scn["now"], "source": "оператор"}
    fire = Fire(**{**base, **{k: v for k, v in f.items() if k != "id"}}).model_dump()
    scn["fires"].append(fire)
    log("Новый пожар", f"{fire['name']}: {fire['area_ha']} га, ветер {fire['wind_ms']} м/с")
    return recompute(reason="Автоматический пересчёт (новый пожар)")


@app.put("/api/fires/{fid}")
def update_fire(fid: str, f: dict):
    scn = scenario()
    for i, x in enumerate(scn["fires"]):
        if x["id"] == fid:
            scn["fires"][i] = Fire(**{**x, **f, "id": fid}).model_dump()
            log("Изменены параметры пожара", scn["fires"][i]["name"])
            return recompute(reason="Автоматический пересчёт")
    raise HTTPException(404, "пожар не найден")


@app.delete("/api/fires/{fid}")
def delete_fire(fid: str):
    scn = scenario()
    scn["fires"] = [x for x in scn["fires"] if x["id"] != fid]
    log("Пожар снят с учёта", fid)
    return recompute(reason="Автоматический пересчёт")


@app.put("/api/stations/{sid}")
def update_station(sid: str, s: dict):
    for x in scenario()["stations"]:
        if x["id"] == sid:
            x.update({k: v for k, v in s.items() if k in ("available", "groups", "people", "vehicles", "note")})
            log("Изменено подразделение", f"{x['name']}: {'доступно' if x.get('available', True) else 'недоступно'}")
            return recompute(reason="Автоматический пересчёт")
    raise HTTPException(404, "подразделение не найдено")


@app.get("/api/plan")
def get_plan():
    if STATE["plan"] is None:
        recompute(reason="Автоматический расчёт")
    return {**STATE["plan"], "status": STATE["status"], "journal": STATE["journal"], "scenario": scenario(),
            "region": STATE["region"], "region_name": REGIONS[STATE["region"]]["name"]}


@app.post("/api/plan/recalculate")
def plan_recalc():
    recompute(reason="Пересчёт по запросу диспетчера")
    return get_plan()


@app.post("/api/plan/override")
def plan_override(req: OverrideRequest):
    errors = validate_override(scenario(), req.fire_id, req.station_ids)
    if errors:
        raise HTTPException(400, "; ".join(errors))
    fixed = copy.deepcopy(STATE["plan"].get("fixed") or {}) if STATE["plan"] else {}
    busy = {sid: fid for fid, sids in fixed.items() if fid != req.fire_id for sid in sids}
    clash = [f"{sid} (уже назначено на {busy[sid]})" for sid in req.station_ids if sid in busy]
    if clash:
        raise HTTPException(409, "подразделение нельзя назначить на два пожара: " + ", ".join(clash))
    fixed[req.fire_id] = req.station_ids
    recompute(fixed=fixed, reason="Изменение диспетчером")
    if STATE["plan"]["solver"]["status"] not in ("OPTIMAL", "FEASIBLE"):
        recompute(reason="Откат ручного назначения: задача неразрешима")   # план пересчитан без изменений
        raise HTTPException(409, "назначение несовместимо с ограничениями задачи (силы ≥ потребности, "
                                 "одно подразделение — один пожар); план пересчитан без изменений")
    return get_plan()


@app.post("/api/plan/approve")
def plan_approve():
    if STATE["plan"] is None:
        raise HTTPException(400, "план не рассчитан")
    STATE["status"] = "утверждён"
    log("План утверждён", f"в {datetime.now():%H:%M}")
    storage.save_plan(STATE["region"], "утверждён", scenario()["title"], STATE["plan"])
    return get_plan()


@app.get("/api/plan/pdf")
def plan_pdf():
    p = STATE["plan"] or recompute()
    data = report.pdf(p, scenario(), STATE["status"])
    log("Выгружена ведомость", "PDF")
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=vedomost_lesresurs.pdf"})


@app.get("/api/plan/xlsx")
def plan_xlsx():
    p = STATE["plan"] or recompute()
    log("Выгружена ведомость", "XLSX")
    return Response(report.xlsx(p, scenario()),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": "attachment; filename=vedomost_lesresurs.xlsx"})


@app.post("/api/assistant/chat")
def chat(req: ChatRequest):
    res = assistant.answer(req.message, STATE["plan"], req.history)
    if res.get("verified") is False:
        log("Ответ LLM отклонён проверкой", ", ".join(str(x) for x in res.get("rejected_numbers", [])[:5]))
    return res


@app.get("/api/journal/history")
def journal_history():
    return storage.history()


@app.post("/api/scenario/upload")
async def upload_scenario(file: UploadFile = File(...), moment: str | None = Form(None)):
    """Обстановка из выгрузки реестра лесных пожаров / ИСДМ-Рослесхоз (.xls/.xlsx, форма реестра)."""
    import tempfile

    import numpy as np
    import pandas as pd

    from etl.load_registry import load_file
    from .planner import scenario_from_rows
    suffix = Path(file.filename or "x.xlsx").suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    try:
        df, rep = load_file(path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Не удалось разобрать файл как реестр лесных пожаров: {e}")
    finally:
        path.unlink(missing_ok=True)
    name = REGIONS[STATE["region"]]["name"]
    df = df[(df.subject == name) & df.det_dt.notna() & df.lat.notna()]
    if df.empty:
        raise HTTPException(400, f"В файле нет пожаров региона «{name}». Выберите другой регион или файл.")
    df = df[df.force_district.fillna("НСС").eq("НСС")]
    df["end_dt"] = df.loc_dt.fillna(df.liq_dt)
    state = df.state.fillna("").str.lower()
    live = ~state.str.startswith("ликвид") & df.liq_dt.isna()
    if moment:
        now = pd.Timestamp(moment)
        mode = "на заданный момент"
    elif live.any():
        now = df.det_dt.max()
        mode = "текущая обстановка (неликвидированные пожары)"
    else:
        # архивный реестр: момент с наибольшим числом одновременно действовавших пожаров
        st, en = df.det_dt.values, df.end_dt.fillna(df.det_dt + pd.Timedelta(hours=12)).values
        counts = [int(((st <= t) & (en > t)).sum()) for t in st]
        now = pd.Timestamp(st[int(np.argmax(counts))])
        mode = "момент с наибольшим числом одновременных пожаров в файле"
    end = df.end_dt.fillna(df.det_dt + pd.Timedelta(hours=12))
    act = df[(df.det_dt <= now) & ((end > now) | (live & (df.det_dt <= now)))]
    if act.empty:
        raise HTTPException(400, f"На {now:%d.%m.%Y %H:%M} действующих пожаров в районах наземных сил нет.")
    act = (act.groupby(["lesnichestvo", "lat", "lon", "det_dt"], as_index=False, dropna=False)
              .agg(uch_lesnichestvo=("uch_lesnichestvo", "first"), np_dist_km=("np_dist_km", "min"),
                   wind_ms=("wind_ms", "max"), area_detect_ha=("area_detect_ha", lambda x: x.sum(min_count=1)),
                   area_total_ha=("area_total_ha", "sum"), area_crown_ha=("area_crown_ha", lambda x: x.sum(min_count=1)),
                   species=("species", "first")).sort_values("det_dt"))
    wc = 4 if now.month in (4, 5, 10) else 3
    STATE["scenario"] = scenario_from_rows(act, now, f"Загруженная обстановка: {file.filename}, на {now:%d.%m.%Y %H:%M}",
                                           f"{len(act)} действующих пожаров из файла {file.filename} ({mode}). КПО по погоде принят {wc}.",
                                           wc, STATE["region"])
    log("Загружена обстановка из файла", f"{file.filename}: {len(act)} пожаров")
    return recompute(reason="Автоматический расчёт")


# --- статический фронтенд (собранный) ---
DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
