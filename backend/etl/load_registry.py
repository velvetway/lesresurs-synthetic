"""Загрузка реестров лесных пожаров 2016–2025 в единую таблицу.

Форматы файлов отличаются по годам (разное число колонок, разная разбивка
заголовка), поэтому колонки сопоставляются по ключевым словам в заголовке,
а не по номеру.

Запуск:  python -I load_registry.py <папка_с_реестрами> <выходной.parquet>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEADER_SCAN_ROWS = 16


def _norm(s: str) -> str:
    s = str(s).lower().replace("\xad", "").replace("ё", "е").replace("\n", " ")
    s = re.sub(r"(\w)-\s*(\w)", r"\1\2", s)  # «удален-ность» → «удаленность»
    return re.sub(r"\s+", " ", s).strip()


def _find_data_start(raw: pd.DataFrame) -> int:
    for i in range(HEADER_SCAN_ROWS):
        v = raw.iat[i, 0]
        if isinstance(v, str) and "федеральн" in _norm(v):
            return i
    raise ValueError("в первых строках файла нет строки «Федеральный округ» — это не выгрузка реестра лесных пожаров")


def _headers(raw: pd.DataFrame, start: int) -> list[str]:
    hdr = []
    for c in range(raw.shape[1]):
        parts = [_norm(raw.iat[r, c]) for r in range(start) if pd.notna(raw.iat[r, c])]
        parts = [p for p in parts if not re.fullmatch(r"[а-яa-z]?\d*", p) and not p.startswith("реестр лесных")]
        hdr.append(" / ".join(parts))
    return hdr


def _first(hdr, *keys, after: int = -1, exclude=()):
    for i, h in enumerate(hdr):
        if i <= after:
            continue
        if all(k in h for k in keys) and not any(e in h for e in exclude):
            return i
    return None


def map_columns(hdr: list[str]) -> dict[str, int | None]:
    m: dict[str, int | None] = {}
    m["subject"] = 0
    m["lesnichestvo"] = _first(hdr, "лесничеств", exclude=("участков", "номер"))
    m["uch_lesnichestvo"] = _first(hdr, "участков")
    det = _first(hdr, "обнаруж")
    m["det_date"], m["det_time"] = det, (det + 1 if det is not None else None)
    np_ = _first(hdr, "ближайший насел")
    m["np_name"] = np_
    m["np_azimuth"] = _first(hdr, "азимут")
    m["np_dist_km"] = _first(hdr, "удален")
    lat = _first(hdr, "координат")
    m["lat_dms"], m["lon_dms"] = lat, (lat + 1 if lat is not None else None)
    m["det_method"] = _first(hdr, "способ обнаруж")
    m["zone"] = _first(hdr, "зона охран")
    m["force_district"] = _first(hdr, "район примен")
    m["species"] = _first(hdr, "пород")
    m["state"] = _first(hdr, "состоян")
    m["wind_ms"] = _first(hdr, "скорость ветра")
    a = _first(hdr, "площад")
    m["area_detect_ha"] = _first(hdr, "момент обнаруж", after=(a or 0) - 1)
    m["area_total_ha"] = _first(hdr, "всего", after=m["area_detect_ha"] or 0)
    m["area_crown_ha"] = _first(hdr, "верхов")
    m["area_surface_ha"] = _first(hdr, "низов")
    m["area_ground_ha"] = _first(hdr, "почвен")
    forces = _first(hdr, "силы")
    m["ppl_ground"] = _first(hdr, "наземных лесопожар", after=(forces or 0) - 1)
    m["ppl_avia"] = _first(hdr, "авиапожар", after=(forces or 0) - 1)
    m["ppl_mchs_total"] = _first(hdr, "пожарной охран", after=(forces or 0) - 1)
    m["ppl_users"] = _first(hdr, "лица, использующ")
    m["ppl_other"] = _first(hdr, "не указан", after=(forces or 0) - 1)
    m["veh_tractor"] = _first(hdr, "трактор")
    m["veh_truck"] = _first(hdr, "грузовой")
    m["veh_firetruck"] = _first(hdr, "патрульный")
    m["veh_allterrain"] = _first(hdr, "вездеход")
    m["cause"] = _first(hdr, "причин")
    st = _first(hdr, "начало тушен")
    m["start_date"], m["start_time"] = st, (st + 1 if st is not None else None)
    lo = _first(hdr, "локализац")
    m["loc_date"], m["loc_time"] = lo, (lo + 1 if lo is not None else None)
    li = _first(hdr, "ликвидац")
    m["liq_date"], m["liq_time"] = li, (li + 1 if li is not None else None)
    m["damage_rub"] = _first(hdr, "ущерб")
    m["cost_rub"] = _first(hdr, "затраты на тушен")
    return m


DMS_RE = re.compile(r"(\d+)\s*°\s*(\d+)?\s*'?\s*([\d.,]+)?\s*\"?\s*([NSEWСЮВЗ])?", re.I)


def dms_to_dd(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("’", "'").replace("″", '"').replace("''", '"')
    mt = DMS_RE.search(s)
    if not mt:
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return np.nan
    d = float(mt.group(1))
    mnt = float(mt.group(2) or 0)
    sec = float((mt.group(3) or "0").replace(",", "."))
    dd = d + mnt / 60 + sec / 3600
    if (mt.group(4) or "").upper() in ("S", "W", "Ю", "З"):
        dd = -dd
    return dd


def _to_dt(d, t):
    def one(dv, tv):
        if pd.isna(dv):
            return pd.NaT
        ds = dv.strftime("%d.%m.%Y") if hasattr(dv, "strftime") else str(dv).strip()
        if tv is None or pd.isna(tv):
            ts = "00:00"
        elif hasattr(tv, "strftime"):
            ts = tv.strftime("%H:%M")
        else:
            ts = str(tv).strip()[:5]
        return pd.to_datetime(f"{ds[:10]} {ts}", format="%d.%m.%Y %H:%M", errors="coerce")
    return [one(a, b) for a, b in zip(d, t)]


NUMERIC = [
    "np_azimuth", "np_dist_km", "wind_ms", "area_detect_ha", "area_total_ha", "area_crown_ha",
    "area_surface_ha", "area_ground_ha", "ppl_ground", "ppl_avia", "ppl_mchs_total", "ppl_users",
    "ppl_other", "veh_tractor", "veh_truck", "veh_firetruck", "veh_allterrain", "damage_rub", "cost_rub",
]


def load_file(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object)
    start = _find_data_start(raw)
    hdr = _headers(raw, start)
    m = map_columns(hdr)
    # обязательные колонки: без времени обнаружения и лесничества файл не является выгрузкой реестра
    REQUIRED = {"det_date": "дата обнаружения", "det_time": "время обнаружения", "lesnichestvo": "лесничество"}
    absent = [name for key, name in REQUIRED.items() if m.get(key) is None]
    if absent:
        seen = "; ".join(h for h in hdr[:10] if h) or "заголовки не распознаны"
        raise ValueError("в заголовке не найдены обязательные колонки: " + ", ".join(absent)
                         + f". Распознанные заголовки: {seen}")
    body = raw.iloc[start:].reset_index(drop=True)
    # строки-разделители (федеральный округ / итоги) не содержат даты обнаружения
    det = body.iloc[:, m["det_date"]]
    body = body[det.notna() & body.iloc[:, m["lesnichestvo"]].notna()].reset_index(drop=True)
    out = pd.DataFrame()
    for k, c in m.items():
        out[k] = body.iloc[:, c] if c is not None else np.nan
    for k in NUMERIC:
        out[k] = pd.to_numeric(out[k].astype(str).str.replace(",", ".").str.replace(" ", ""), errors="coerce")
    out["lat"] = out["lat_dms"].map(dms_to_dd)
    out["lon"] = out["lon_dms"].map(dms_to_dd)
    for p in ("det", "start", "loc", "liq"):
        out[f"{p}_dt"] = _to_dt(out[f"{p}_date"], out[f"{p}_time"])
        out.drop(columns=[f"{p}_date", f"{p}_time"], inplace=True)
    out.drop(columns=["lat_dms", "lon_dms"], inplace=True)
    for k in ("subject", "lesnichestvo", "uch_lesnichestvo", "np_name", "det_method", "zone",
              "force_district", "species", "state", "cause"):
        out[k] = out[k].astype("string").str.strip()
    out["source_file"] = path.name
    missing = [k for k, v in m.items() if v is None]
    return out, {"file": path.name, "rows": len(out), "missing": missing}


def main(src: str, dst: str) -> None:
    frames, report = [], []
    for f in sorted(Path(src).glob("Реестр*.xls*")):
        df, rep = load_file(f)
        frames.append(df)
        report.append(rep)
        print(rep)
    all_ = pd.concat(frames, ignore_index=True)
    all_ = all_[all_["det_dt"].notna()]
    all_["year"] = all_["det_dt"].dt.year
    all_ = all_.drop_duplicates(subset=["subject", "lesnichestvo", "det_dt", "lat", "lon", "area_total_ha"])
    all_.reset_index(drop=True).to_parquet(dst)
    print("итого записей:", len(all_))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
