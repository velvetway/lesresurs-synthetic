"""Журналирование решений (SQLite): события и утверждённые планы сохраняются между перезапусками."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from .core import DATA

DB = Path(os.environ.get("LESRESURS_DB", DATA / "lesresurs.db"))


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS journal (ts TEXT, region TEXT, event TEXT, detail TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS plans (ts TEXT, region TEXT, status TEXT, scenario TEXT, metrics TEXT, assignment TEXT, statement TEXT)")
    return c


def log_event(region: str, event: str, detail: str) -> None:
    try:
        with _conn() as c:
            c.execute("INSERT INTO journal VALUES (?,?,?,?)", (datetime.now().isoformat(timespec="seconds"), region, event, detail))
    except sqlite3.Error:
        pass


def save_plan(region: str, status: str, scenario_title: str, plan: dict) -> None:
    try:
        with _conn() as c:
            c.execute("INSERT INTO plans VALUES (?,?,?,?,?,?,?)", (
                datetime.now().isoformat(timespec="seconds"), region, status, scenario_title,
                json.dumps(plan["metrics"], ensure_ascii=False), json.dumps(plan["assignment"], ensure_ascii=False),
                json.dumps([{"fire": r["fire_name"], "category": r["risk"]["category"], "units": [u["name"] for u in r["units"]],
                             "arrival_h": r["arrival_h"], "deficit": r["deficit"]} for r in plan["statement"]], ensure_ascii=False)))
    except sqlite3.Error:
        pass


def history(limit: int = 200) -> dict:
    try:
        with _conn() as c:
            j = c.execute("SELECT ts, region, event, detail FROM journal ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
            p = c.execute("SELECT ts, region, status, scenario, metrics FROM plans ORDER BY ts DESC LIMIT 50").fetchall()
        return {"journal": [dict(zip(("ts", "region", "event", "detail"), r)) for r in j],
                "plans": [{"ts": r[0], "region": r[1], "status": r[2], "scenario": r[3], "metrics": json.loads(r[4])} for r in p]}
    except sqlite3.Error:
        return {"journal": [], "plans": []}
