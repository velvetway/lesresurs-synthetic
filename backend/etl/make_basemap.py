"""Генератор синтетической фоновой карты для модельных субъектов.

Проект использует только синтетические данные, поэтому реальная географическая подложка
(Natural Earth) в него не входит: границы, береговая линия и реки для модельных субъектов
рисуются процедурно в окрестности bbox субъекта. Геометрия не соответствует никакой реальной
местности — это условная карта вымышленного региона.

Запуск: python -I make_basemap.py <frontend_public_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etl.make_synthetic import REGIONS  # noqa: E402

N_EDGE = 80        # точек на сторону границы субъекта
MARGIN = 1.6        # градусов «соседних» территорий вокруг субъекта


def _wavy_edge(p0, p1, rng, amp=0.05, n=N_EDGE, harmonics=3):
    """Волнистая линия между двумя точками (граница или берег)."""
    t = np.linspace(0.0, 1.0, n)
    x = p0[0] + (p1[0] - p0[0]) * t
    y = p0[1] + (p1[1] - p0[1]) * t
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = float(np.hypot(dx, dy)) or 1.0
    nx, ny = -dy / length, dx / length              # нормаль к отрезку
    off = np.zeros(n)
    for k in range(1, harmonics + 1):
        off += rng.normal(0.0, amp / k) * np.sin(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
    off *= np.sin(np.pi * t) ** 0.7                  # края закреплены
    return [[float(x[i] + nx * off[i]), float(y[i] + ny * off[i])] for i in range(n)]


def _region_outline(bbox, rng):
    """Замкнутая волнистая граница модельного субъекта."""
    la0, la1, lo0, lo1 = bbox
    c = [(lo0, la0), (lo1, la0), (lo1, la1), (lo0, la1)]
    pts = []
    for i in range(4):
        seg = _wavy_edge(c[i], c[(i + 1) % 4], rng, amp=0.045)
        pts.extend(seg[:-1])
    pts.append(pts[0])
    return pts


def _neighbours(bbox, rng):
    """Полосы соседних территорий вокруг субъекта (для фона карты)."""
    la0, la1, lo0, lo1 = bbox
    m = MARGIN
    out = []
    sides = [
        ((lo0 - m, la1), (lo1 + m, la1), (lo1 + m, la1 + m), (lo0 - m, la1 + m)),   # север
        ((lo1, la0 - m), (lo1 + m, la0 - m), (lo1 + m, la1 + m), (lo1, la1)),       # восток
        ((lo0 - m, la0 - m), (lo1 + m, la0 - m), (lo1 + m, la0), (lo0 - m, la0)),   # юг
        ((lo0 - m, la0 - m), (lo0, la0 - m), (lo0, la1), (lo0 - m, la1 + m)),       # запад
    ]
    for s in sides:
        pts = []
        for i in range(4):
            seg = _wavy_edge(s[i], s[(i + 1) % 4], rng, amp=0.05)
            pts.extend(seg[:-1])
        pts.append(pts[0])
        out.append(pts)
    return out


def _river(bbox, rng):
    """Река, пересекающая субъект (для фона)."""
    la0, la1, lo0, lo1 = bbox
    n = 90
    t = np.linspace(0.0, 1.0, n)
    base = np.linspace(lo0 - 0.6, lo1 + 0.6, n)
    y = la0 + (la1 - la0) * (0.72 - 0.55 * t)
    off = sum(rng.normal(0.0, 0.10 / k) * np.sin(2 * np.pi * k * t + rng.uniform(0, 6.3))
              for k in range(1, 4))
    return [[float(base[i]), float(y[i] + off[i])] for i in range(n)]


def build(bbox, seed):
    rng = np.random.default_rng(seed)
    outline = _region_outline(bbox, rng)
    neighbours = _neighbours(bbox, rng)
    land = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"kind": "subject"}, "geometry": {"type": "Polygon", "coordinates": [outline]}},
        ] + [
            {"type": "Feature", "properties": {"kind": "neighbour", "name": f"соседняя территория {i + 1}"},
             "geometry": {"type": "Polygon", "coordinates": [pts]}}
            for i, pts in enumerate(neighbours)
        ],
    }
    borders = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"kind": "boundary"}, "geometry": {"type": "LineString", "coordinates": outline}},
            {"type": "Feature", "properties": {"kind": "river"}, "geometry": {"type": "LineString", "coordinates": _river(bbox, rng)}},
        ],
    }
    return {"land": land, "borders": borders}


def main(dst: str) -> None:
    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    for i, (slug, _subject, bbox, *_rest) in enumerate(REGIONS):
        bm = build(bbox, seed=100 + i)
        path = out / f"basemap_{slug}.json"
        path.write_text(json.dumps(bm, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"{path} ({path.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../../frontend/public")
