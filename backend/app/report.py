"""Выгрузка расчётной ведомости: PDF (ReportLab) и XLSX (openpyxl)."""
from __future__ import annotations

import io
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIRS = [Path(__file__).parent / "fonts", Path("/usr/share/fonts/truetype/dejavu"), Path("C:/Windows/Fonts")]


def _register_fonts() -> tuple[str, str]:
    for d in FONT_DIRS:
        for reg, bold in (("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"), ("arial.ttf", "arialbd.ttf")):
            if (d / reg).exists() and (d / bold).exists():
                pdfmetrics.registerFont(TTFont("F", str(d / reg)))
                pdfmetrics.registerFont(TTFont("FB", str(d / bold)))
                return "F", "FB"
    return "Helvetica", "Helvetica-Bold"


def _hm(x):
    if x is None:
        return "—"
    m = round(x * 60)
    return f"{m // 60} ч {m % 60:02d} мин" if m >= 60 else f"{m} мин"


def pdf(plan: dict, scenario: dict, status: str) -> bytes:
    F, FB = _register_fonts()
    st = {
        "h1": ParagraphStyle("h1", fontName=FB, fontSize=15, leading=19, spaceAfter=4),
        "h2": ParagraphStyle("h2", fontName=FB, fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=3),
        "p": ParagraphStyle("p", fontName=F, fontSize=9, leading=12),
        "s": ParagraphStyle("s", fontName=F, fontSize=8, leading=10, textColor=colors.HexColor("#555555")),
        "c": ParagraphStyle("c", fontName=F, fontSize=8.3, leading=10.3),
    }
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
                            title="Расчётная ведомость сил и средств", author="ЛесРесурс")
    el = [Paragraph("Расчётная ведомость сил и средств пожаротушения", st["h1"]),
          Paragraph(f"{scenario['title']}. Расчёт: {plan['created_at'].replace('T', ' ')}. Статус плана: <b>{status}</b>.", st["p"]),
          Paragraph("Сформировано ассистентом «ЛесРесурс». Рекомендация требует утверждения руководителем тушения.", st["s"]),
          Spacer(1, 4)]
    m, b = plan["metrics"], plan["baseline"]["metrics"]
    summ = [["Показатель", "План ассистента", "Базовый (FIFO)"],
            ["Ожидаемый ущерб, ₽", f"{m['expected_damage_rub']:,}".replace(",", " "), f"{b['expected_damage_rub']:,}".replace(",", " ")],
            ["Нарушений норматива прибытия (п. 51)", m["norm_violations"], b["norm_violations"]],
            ["Дефицит групп / техники", f"{m['deficit_groups']} / {m['deficit_vehicles']}", f"{b['deficit_groups']} / {b['deficit_vehicles']}"],
            ["Задействовано подразделений", m["stations_used"], b["stations_used"]]]
    t = Table(summ, colWidths=[80 * mm, 45 * mm, 45 * mm])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), F, 8.5), ("FONT", (0, 0), (-1, 0), FB, 8.5),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4d3a")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c4bd")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    el += [t, Paragraph(plan["deficit_summary"]["message"], st["p"])]

    for r in plan["statement"]:
        el.append(Paragraph(f"{r['fire_name']} — категория риска: {r['risk']['category'].upper()}", st["h2"]))
        crit = "; ".join(c["text"] for c in r["risk"]["criteria"] if c["triggered"]) or "—"
        el.append(Paragraph(f"Сработавшие критерии: {crit}.", st["c"]))
        units = ", ".join(f"{u['name']} ({_hm(u['travel_h'])}, {u['km']} км)" for u in r["units"]) or "не назначены"
        el.append(Paragraph(f"Назначены: {units}. Прибытие с момента обнаружения: {_hm(r['arrival_h'])} "
                            f"(норматив {r['demand']['arrival_limit_h']} ч{', сокращённый ' + str(r['demand']['arrival_limit_reduced_h']) + ' ч' if r['demand']['hard_weather'] else ''}) — "
                            f"{'соответствует' if r['norm_ok'] else 'ПРЕВЫШЕН'}.", st["c"]))
        rows = [["Ресурс", "Потребность", "Назначено", "Дефицит", "Норматив / расчёт", "Ссылка на НПА"]]
        for ln in r["lines"]:
            need = f"{ln['need']}" + (f" / {ln['need_people']} чел." if ln.get("need_people") else "")
            asg = "—" if ln.get("assigned") is None else f"{ln['assigned']}" + (f" / {ln['assigned_people']} чел." if ln.get("assigned_people") is not None else "")
            rows.append([Paragraph(ln["resource"], st["c"]), need, asg, ln.get("deficit", 0) or "—",
                         Paragraph(ln["norm"], st["c"]), Paragraph(f"{ln['ref']['act']}, {ln['ref']['point']}", st["c"])])
        tt = Table(rows, colWidths=[60 * mm, 26 * mm, 26 * mm, 21 * mm, 68 * mm, 64 * mm], repeatRows=1)
        tt.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), F, 8.3), ("FONT", (0, 0), (-1, 0), FB, 8.3),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eee9")),
                                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c4bd")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        el.append(tt)
    el += [PageBreak(), Paragraph("Нормативные правовые акты", st["h2"])]
    from .core import norms
    for a in norms()["acts"].values():
        el.append(Paragraph(f"• {a['title']}", st["c"]))
    el.append(Spacer(1, 6))
    el.append(Paragraph("Допущения прототипа: дислокация станций — условный справочник по нормативам 1605-р; время в пути — "
                        "по прямой с коэффициентом извилистости 1,35 (или OSRM при подключении графа OSM); производительность группы "
                        "(130 м/ч) и параметры скорости распространения откалиброваны по реестру 2016–2025 и требуют экспертной проверки.", st["s"]))
    doc.build(el)
    return buf.getvalue()


def xlsx(plan: dict, scenario: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Ведомость"
    hdr = ["Пожар", "Категория риска", "Ресурс", "Потребность", "Назначено", "Дефицит", "Норматив / расчёт", "НПА", "Пункт",
           "Подразделения", "Прибытие, ч", "Норматив, ч"]
    ws.append([f"Расчётная ведомость — {scenario['title']}"])
    ws["A1"].font = Font(bold=True, size=13)
    ws.append(hdr)
    for c in ws[2]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4D3A")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for r in plan["statement"]:
        units = "; ".join(u["name"] for u in r["units"])
        for ln in r["lines"]:
            # у строки «усиление ×2» назначение не считается в единицах ресурса — так же, как в PDF, ставим прочерк
            assigned = ln.get("assigned")
            ws.append([r["fire_name"], r["risk"]["category"], ln["resource"], ln["need"],
                       "—" if assigned is None else assigned, ln.get("deficit"),
                       ln["norm"], ln["ref"]["act"], ln["ref"]["point"], units, r["arrival_h"], r["demand"]["arrival_limit_h"]])
    for col, w in zip("ABCDEFGHIJKL", (34, 18, 34, 12, 12, 10, 48, 22, 22, 46, 12, 12)):
        ws.column_dimensions[col].width = w
    ws2 = wb.create_sheet("Сравнение с FIFO")
    ws2.append(["Показатель", "План ассистента", "FIFO"])
    for k, name in (("expected_damage_rub", "Ожидаемый ущерб, ₽"), ("weighted_damage", "Взвешенный ущерб"),
                    ("norm_violations", "Нарушения норматива прибытия"), ("deficit_groups", "Дефицит групп"),
                    ("deficit_vehicles", "Дефицит техники"), ("stations_used", "Подразделений")):
        ws2.append([name, plan["metrics"][k], plan["baseline"]["metrics"][k]])
    ws2.column_dimensions["A"].width = 36
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
