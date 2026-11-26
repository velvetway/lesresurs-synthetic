"""Генератор синтетического набора данных публичной версии «ЛесРесурса».

Выгрузка реестра лесных пожаров не публикуется: данные реестра нельзя приводить
в статье и выкладывать в открытый репозиторий. Поэтому публичная версия прототипа
работает на синтетическом наборе, который:

  * воспроизводит схему реестра (37 колонок, 2016–2025 гг., деление на районы наземных
    и авиационных сил, площадь при обнаружении только в 2025 г. — как в исходных данных);
  * откалиброван по опубликованным агрегированным показателям: доля пожаров ≥ 25 га
    растёт с задержкой начала тушения (12 → 27 % при задержке 0 → 5 ч), около 30 %
    пожаров обслуживается силами ниже рекомендации калькулятора, медианные затраты
    на тушение 50 и 180 тыс. ₽ для пожаров меньше и больше 25 га;
  * полностью детерминирован: одинаковый seed даёт одинаковый файл.

Ни одна строка исходного реестра не используется: координаты, названия лесничеств,
населённых пунктов и результаты пожаров сгенерированы.

Запуск: python -I make_synthetic.py <registry.parquet> [--seed 42] [--scale 1.0]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Модельные субъекты: два пилотных региона прототипа и два «фоновых» субъекта
REGIONS = [
    # slug, subject, bbox (lat0, lat1, lon0, lon1), n_fires, force_district, лесничества
    ("region_a", "Северотаёжный модельный край", (58.0, 60.6, 90.0, 95.0), 9000, "НСС", 10),
    ("region_b", "Лесостепная модельная область", (51.0, 53.0, 37.0, 40.0), 2500, "НСС", 8),
    ("region_c", "Заозёрный модельный округ", (52.0, 56.0, 28.0, 34.0), 1300, "АСС", 6),
    ("region_d", "Южнотаёжный модельный округ", (56.0, 60.0, 76.0, 84.0), 1200, "ЗК", 6),
]

STEMS = ["Крут", "Бел", "Черн", "Сух", "Малин", "Заозёр", "Ключев", "Сосн", "Кедр", "Верхн",
         "Нижн", "Камен", "Листв", "Ольх", "Берез", "Ягодн", "Тих", "Светл", "Глух", "Дальн"]
LESN_TAILS = ["овское", "инское", "ское", "янское"]
NP_TAILS = ["овка", "ино", "ово", "иха", "янка", "ское", "яны"]
NP_PREFIX = ["с", "п", "д", "пгт", "рп"]
SPECIES = ["Сосна", "Лиственница", "Кедр", "Берёза", "Осина", "Ель", "Дуб", "Смешанный"]
METHODS = ["МН", "ЛП", "АВ", "ИСДМ", "ДПС", "МЧС"]
STATES = ["Ликвид.", "Локализ.", "Действ."]
CAUSES = ["гроза", "неосторожное обращение с огнём", "сельхозпал", "линии электропередачи", "неустановленные"]

# сезонность и суточный ход (доли)
MONTH_W = np.array([0.006, 0.006, 0.020, 0.160, 0.200, 0.090, 0.060, 0.070, 0.120, 0.150, 0.012, 0.006])
HOUR_W = np.array([1, 1, 1, 1, 2, 3, 5, 7, 9, 11, 13, 15, 16, 16, 15, 14, 12, 10, 8, 6, 4, 3, 2, 1], float)
YEAR_W = np.array([10984, 11042, 12131, 14433, 14810, 15111, 12527, 12766, 10216, 7135], float)

# калибровка исхода: доля пожаров ≥ 25 га по задержке начала тушения 0 / 0,5 / 1 / 2 / 3 / 5 ч
# соответствует опубликованным 12,1 / 15,5 / 17,1 / 21,0 / 26,8 %
DELAY_BETA = 0.60
TARGET_LARGE_NSS = 0.17      # доля пожаров ≥ 25 га в районах наземных сил
TARGET_LARGE_AVIA = 0.20


def _names(rng, n, count, tails=None):
    tails = tails or LESN_TAILS
    out = set()
    while len(out) < n:
        out.add(rng.choice(STEMS) + rng.choice(tails))
    return sorted(out)[:n]


def _calibrate_shift(z, target, lo=-12.0, hi=12.0, iters=60):
    """Сдвиг логита, при котором средняя вероятность попадания в класс >=25 га равна target."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _sigmoid(z + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit_large(delay, area0, wind, weather, np_dist, hazard, month, rng):
    """Логит вероятности разрастания до 25 га.

    Сигнал задан так, чтобы структура была обучаемой (как в исходном реестре): площадь при
    обнаружении, задержка начала тушения, ветер и КПО, сезон, плюс гладкое пространственное
    поле пожарной опасности — оно же даёт «горячие точки» на карте.
    """
    spring = np.isin(month, (4, 5)).astype(float)        # весенний сезон: сухая трава, ветры
    return (DELAY_BETA * delay + 0.55 * np.log1p(area0) + 0.16 * wind
            + 0.40 * (wind >= 5) + 0.60 * (weather >= 4) + 0.18 * (np_dist < 5)
            + hazard + 1.05 * spring + rng.normal(0, 0.22, size=len(delay)))


def _perimeter_rate(area_ha, wind, weather, crown):
    """Скорость прироста кромки по той же формуле, что в модуле потребности прототипа."""
    r0 = np.choose(np.asarray(weather, dtype=int) - 1, [5.0, 12.0, 25.0, 45.0, 70.0])
    r = r0 * np.exp(0.18 * np.asarray(wind, dtype=float)) * np.where(crown, 5.0, 1.0)
    p0 = 4.0 * np.sqrt(np.maximum(area_ha, 0.01) * 10000.0)          # периметр, м
    return 1.5 * r + p0 / 6.0


def make_region(rng, subject, bbox, n, force_district, n_lesn):
    la0, la1, lo0, lo1 = bbox

    # лесничества: центр в границах региона + устойчивый латентный риск (1 — норматив)
    lesn_centers = [(rng.uniform(la0 + 0.2, la1 - 0.2), rng.uniform(lo0 + 0.2, lo1 - 0.2)) for _ in range(n_lesn)]
    # устойчивое пространственное поле пожарной опасности: несколько «горячих точек» региона
    spots = [(rng.uniform(la0 + 0.3, la1 - 0.3), rng.uniform(lo0 + 0.3, lo1 - 0.3),
              rng.normal(0.0, 1.45), rng.uniform(1.0, 2.0)) for _ in range(3)]
    names = _names(rng, n_lesn, n_lesn)
    lesn = [f"{nm}" for nm in names]
    lesn[-1] = f"{rng.choice(STEMS)}{rng.choice(['овский', 'инский'])} ГПЗ"      # одно лесничество-ООПТ
    uchs = _names(rng, 4 * n_lesn, 4 * n_lesn)

    # населённые пункты: рядом с центрами лесничеств (лесничество → его пункты)
    settlements = []
    for li in range(n_lesn):
        cla, clo = lesn_centers[li]
        for _ in range(max(1, n // (n_lesn * 12))):
            settlements.append({
                "name": f"{rng.choice(NP_PREFIX)} {rng.choice(STEMS)}{rng.choice(NP_TAILS)}",
                "lat": float(np.clip(cla + rng.normal(0, 0.28), la0 + 0.05, la1 - 0.05)),
                "lon": float(np.clip(clo + rng.normal(0, 0.35), lo0 + 0.05, lo1 - 0.05)),
                "lesn": li,
            })
    while len(settlements) < 18:                      # в малых регионах — не меньше 18 пунктов
        li = rng.integers(0, n_lesn)
        cla, clo = lesn_centers[li]
        settlements.append({"name": f"{rng.choice(NP_PREFIX)} {rng.choice(STEMS)}{rng.choice(NP_TAILS)}",
                            "lat": float(np.clip(cla + rng.normal(0, 0.3), la0 + 0.05, la1 - 0.05)),
                            "lon": float(np.clip(clo + rng.normal(0, 0.35), lo0 + 0.05, lo1 - 0.05)),
                            "lesn": int(li)})

    years = rng.choice(np.arange(2016, 2026), size=n, p=YEAR_W / YEAR_W.sum())
    months = rng.choice(np.arange(12), size=n, p=MONTH_W / MONTH_W.sum())
    days = rng.integers(1, 29, size=n)
    hours = rng.choice(np.arange(24), size=n, p=HOUR_W / HOUR_W.sum())
    minutes = rng.choice([0, 10, 15, 20, 30, 40, 45, 50], size=n)
    det = pd.Series(pd.to_datetime({"year": years, "month": months + 1, "day": days, "hour": hours,
                                    "minute": minutes})).astype("datetime64[ns]")
    # кластеризация обнаружений: в пожароопасные дни пожары выявляются «пачками» (тяжёлый хвост
    # по числу одновременных пожаров, как в реальных реестрах: единичные дни с 10–20 пожарами)
    n_events = max(8, n // 300)
    sizes = 3 + rng.poisson(4.5, size=n_events)                     # 3..~15 пожаров в «пачке»
    centers = det.iloc[rng.choice(n, size=n_events, replace=False)].to_numpy()
    order = rng.permutation(n)[: int(sizes.sum())]
    det = pd.Series(det.to_numpy().copy())
    pos = 0
    for e in range(n_events):
        take = order[pos:pos + int(sizes[e])]
        if len(take) == 0:
            break
        shift = (rng.normal(0.0, 0.9, size=len(take)) * 3600e9).astype("timedelta64[ns]")
        det.iloc[take] = pd.Series(centers[e] + shift)
        pos += len(take)
    det = det.sort_values().reset_index(drop=True).astype("datetime64[ns]")

    # положение пожара: около случайного населённого пункта своего лесничества
    np_idx = rng.integers(0, len(settlements), size=n)
    np_dist = np.clip(rng.lognormal(mean=np.log(6.0), sigma=0.8, size=n), 0.2, 60.0)
    azimuth = rng.uniform(0, 360, size=n)
    lat = np.array([settlements[i]["lat"] for i in np_idx])
    lon = np.array([settlements[i]["lon"] for i in np_idx])
    dphi = np.radians(np_dist / 111.0)
    lat_f = np.clip(lat + dphi * np.cos(np.radians(azimuth)), la0, la1)
    lon_f = np.clip(lon + dphi * np.sin(np.radians(azimuth)) / np.maximum(np.cos(np.radians(lat)), 0.2), lo0, lo1)
    lesn_idx = np.array([settlements[i]["lesn"] for i in np_idx])
    hazard = np.zeros(n)
    for cla, clo, amp, rad in spots:                     # гладкое поле: выучивается по координатам
        hazard += amp * np.exp(-(((lat_f - cla) ** 2 + (lon_f - clo) ** 2) / (2 * rad ** 2)))

    area0 = np.clip(rng.lognormal(mean=np.log(1.2), sigma=0.9, size=n), 0.01, 12.0)
    wind = np.clip(rng.gamma(shape=2.2, scale=1.9, size=n), 0.0, 20.0)
    weather = rng.choice([2, 3, 4, 5], size=n, p=[0.12, 0.38, 0.34, 0.16])
    crown = rng.random(n) < np.clip(0.02 + 0.25 * (area0 / 10.0), 0, 0.5)
    species = rng.choice(SPECIES, size=n)

    # задержка начала тушения: больше для удалённых и труднодоступных пожаров
    delay = np.clip(rng.lognormal(mean=np.log(0.8), sigma=0.8, size=n) + 0.06 * np_dist, 0.0, 48.0)
    delay = np.where(rng.random(n) < 0.25, 0.0, delay)     # часть пожаров тушат сразу при обнаружении
    if force_district != "НСС":
        delay = np.clip(delay * 2.2, 0.0, 48.0)

    z = _logit_large(delay, area0, wind, weather, np_dist, hazard, months + 1, rng)
    # автокалибровка: доля пожаров ≥ 25 га приводится к опубликованному агрегату
    target = TARGET_LARGE_NSS if force_district == "НСС" else TARGET_LARGE_AVIA
    if force_district != "НСС":
        target *= 0.9
    delta = _calibrate_shift(z, target)        # бисекция: mean(sigmoid(z+delta)) == target
    p_large = _sigmoid(z + delta)
    large = rng.random(n) < p_large
    # масштаб крупного пожара тоже зависит от обстановки -> выучиваемая регрессия площади
    mu_large = (np.log(55.0) + 0.42 * np.log1p(delay) + 0.065 * (wind - 6.0)
                + 0.22 * (weather - 3.2) + 0.30 * np.log1p(area0) + 0.14 * hazard)
    mu_small = (np.log(1.1) + 0.42 * np.log1p(delay) + 0.050 * (wind - 6.0)
                + 0.17 * (weather - 3.2) + 0.70 * np.log1p(area0) + 0.10 * hazard)
    area_total = np.where(large,
                          np.maximum(25.0, rng.lognormal(mean=mu_large, sigma=0.26, size=n)),
                          np.clip(rng.lognormal(mean=mu_small, sigma=0.30, size=n), 0.01, 24.0))

    start = det + pd.to_timedelta(delay * 3600.0, unit="s").round("s")
    loc = start + pd.to_timedelta((0.6 + 0.12 * np.sqrt(area_total) + rng.gamma(1.5, 1.2, n)) * 3600.0, unit="s").round("s")
    liq = loc + pd.to_timedelta((1.2 + 0.25 * np.sqrt(area_total) + rng.gamma(1.5, 1.5, n)) * 3600.0, unit="s").round("s")

    # силы на пожаре: медиана около рекомендации, часть пожаров — ниже неё
    rate = _perimeter_rate(area_total, wind, weather, crown)
    need_groups = np.maximum(1, np.ceil(rate / 130.0)).astype(int)
    fact_groups = np.maximum(1, np.round(need_groups * rng.lognormal(mean=np.log(0.92), sigma=0.42, size=n)).astype(int))
    ppl_ground = fact_groups * 6.0
    veh_firetruck = np.maximum(1, np.ceil(fact_groups / 2.0))
    veh_tractor = np.where(fact_groups >= 2, np.ceil(fact_groups / 4.0), 0.0)

    cause = rng.choice(CAUSES, size=n)
    state = np.where(pd.Timestamp("2025-12-31") > liq, "Ликвид.", "Действ.")

    df = pd.DataFrame({
        "subject": subject,
        "lesnichestvo": np.array(lesn)[lesn_idx],
        "uch_lesnichestvo": rng.choice(uchs, size=n),
        "np_name": [settlements[i]["name"] for i in np_idx],
        "np_azimuth": azimuth.round(1),
        "np_dist_km": np_dist.round(1),
        "det_method": rng.choice(METHODS, size=n, p=[0.55, 0.15, 0.1, 0.1, 0.05, 0.05]),
        "zone": np.where(force_district == "НСС", "наземная", "авиационная"),
        "force_district": force_district,
        "species": np.where(species == "Смешанный", None, species),
        "state": state,
        "wind_ms": wind.round(1),
        "area_detect_ha": np.round(area0, 1),
        "area_total_ha": np.round(area_total, 1),
        "area_crown_ha": np.where(crown, np.round(area_total * 0.4, 1), 0.0),
        "area_surface_ha": np.where(crown, np.round(area_total * 0.6, 1), np.round(area_total, 1)),
        "area_ground_ha": np.where(rng.random(n) < 0.05, np.round(area_total * 0.2, 1), np.nan),
        "ppl_ground": ppl_ground,
        "ppl_avia": np.where(force_district == "НСС", np.nan, rng.integers(2, 8, n)),
        "ppl_mchs_total": np.where(rng.random(n) < 0.15, rng.integers(2, 10, n), np.nan),
        "ppl_users": np.where(rng.random(n) < 0.25, rng.integers(1, 12, n), np.nan),
        "ppl_other": np.where(rng.random(n) < 0.15, rng.integers(1, 6, n), np.nan),
        "veh_tractor": veh_tractor,
        "veh_truck": np.where(rng.random(n) < 0.1, 1.0, np.nan),
        "veh_firetruck": veh_firetruck,
        "veh_allterrain": np.where(rng.random(n) < 0.12, 1.0, np.nan),
        "cause": cause,
        "damage_rub": np.nan,
        "cost_rub": np.nan,
        "lat": lat_f.round(5),
        "lon": lon_f.round(5),
        "det_dt": det,
        "start_dt": start,
        "loc_dt": loc,
        "liq_dt": liq,
        "source_file": "синтетический_набор_2026.synthetic",
        "year": years.astype("int32"),
    })

    # заявленный в реестре ущерб и затраты есть только за 2025 год
    m2025 = df.year.eq(2025)
    k = int(m2025.sum())
    if k:
        big = (df.loc[m2025, "area_total_ha"] >= 25).to_numpy()
        cost = np.where(big, rng.lognormal(np.log(180_000.0), 0.75, k), rng.lognormal(np.log(50_000.0), 0.8, k))
        dmg = np.where(big, rng.lognormal(np.log(370_200.0), 0.85, k), rng.lognormal(np.log(47_100.0), 0.95, k))
        df.loc[m2025, "cost_rub"] = np.round(cost)
        df.loc[m2025, "damage_rub"] = np.round(dmg)
    return df


def main(dst: str, seed: int = 42, scale: float = 1.0) -> None:
    rng = np.random.default_rng(seed)
    frames = []
    for _slug, subject, bbox, n, fd, nl in REGIONS:
        frames.append(make_region(rng, subject, bbox, max(60, int(n * scale)), fd, nl))
    df = pd.concat(frames, ignore_index=True)
    # часть возгораний в реестре записана несколькими участками контура: дублируем ~1,5 % строк
    dup = df.sample(frac=0.015, random_state=seed).copy()
    dup["area_total_ha"] = np.round(dup["area_total_ha"] * 0.45, 1)
    dup["area_detect_ha"] = np.round(dup["area_detect_ha"] * 0.45, 2)
    df = pd.concat([df, dup], ignore_index=True).sort_values("det_dt").reset_index(drop=True)
    df["year"] = df.det_dt.dt.year.astype("int32")
    for c in ("subject", "lesnichestvo", "uch_lesnichestvo", "np_name", "det_method", "zone",
              "force_district", "species", "state", "cause"):
        df[c] = df[c].astype("string")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst)
    nss = df.force_district.eq("НСС")
    print("строк:", len(df), "| районы НСС:", int(nss.sum()))
    print("доля ≥ 25 га:", round(float((df.area_total_ha >= 25).mean()), 3),
          "| НСС:", round(float((df[nss].area_total_ha >= 25).mean()), 3))
    d = (df.start_dt - df.det_dt).dt.total_seconds() / 3600
    for lo, hi in ((0, 0.1), (0.4, 0.6), (0.9, 1.1), (1.9, 2.1), (2.9, 3.1)):
        m = nss & d.between(lo, hi)
        print(f"  задержка {lo:g}–{hi:g} ч: доля ≥ 25 га {float((df[m].area_total_ha >= 25).mean()):.3f} (n={int(m.sum())})")
    print("2025: медианные затраты, тыс. ₽:",
          {("≥25" if b else "<25"): round(float(df[(df.year == 2025) & ((df.area_total_ha >= 25) == b)].cost_rub.median()) / 1000, 1)
           for b in (True, False)})


if __name__ == "__main__":
    args = sys.argv[1:]
    dst = args[0]
    seed = 42
    scale = 1.0
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])
    if "--scale" in args:
        scale = float(args[args.index("--scale") + 1])
    main(dst, seed, scale)
