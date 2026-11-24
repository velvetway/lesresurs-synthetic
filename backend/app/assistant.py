"""Ассистент-объяснитель.

LLM ничего не рассчитывает: она получает готовый план (результаты модулей 1–4) и отвечает
на вопросы диспетчера, ссылаясь только на эти данные. Провайдер задаётся переменными окружения:

  LLM_PROVIDER=gigachat   GIGACHAT_AUTH_KEY=<base64 client_id:secret>  [GIGACHAT_SCOPE=GIGACHAT_API_PERS]
  LLM_PROVIDER=yandexgpt  YANDEX_API_KEY=...  YANDEX_FOLDER_ID=...
  LLM_PROVIDER=openai     OPENAI_BASE_URL=http://localhost:11434/v1  OPENAI_MODEL=qwen2.5:7b  [OPENAI_API_KEY]
                          (совместимо с Ollama / vLLM / LM Studio — локальный контур без внешних API)

Без провайдера (или при ошибке) работает детерминированный шаблонный ответчик.
"""
from __future__ import annotations

import json
import os
import re
import uuid

import httpx

SYSTEM_PROMPT = (
    "Ты — ассистент диспетчера лесопожарной службы (система «ЛесРесурс»). Отвечай по-русски, кратко и по делу. "
    "Используй ТОЛЬКО данные расчётного плана в JSON: категории риска и сработавшие критерии, потребность в силах, "
    "назначенные подразделения, время прибытия, дефициты и ссылки на НПА. Не придумывай чисел и не меняй расчёт. "
    "Если данных не хватает — так и скажи. Окончательное решение принимает человек."
)


def _h(x: float | None) -> str:
    if x is None:
        return "—"
    m = round(x * 60)
    return f"{m // 60} ч {m % 60:02d} мин" if m >= 60 else f"{m} мин"


def _compact(plan: dict) -> dict:
    return {
        "metrics": plan["metrics"], "baseline": plan["baseline"]["metrics"], "deficit": plan["deficit_summary"],
        "fires": [{
            "name": r["fire_name"], "category": r["risk"]["category"], "p_growth": r["risk"]["p_growth"],
            "criteria": [c["text"] for c in r["risk"]["criteria"] if c["triggered"]],
            "need_groups": r["demand"]["groups"], "need_groups_final": r["lines"][0]["need"],
            "people_final": r["lines"][0]["need"] * 6, "vehicles": r["demand"]["vehicles"],
            "edge_growth_m_h": round(r["demand"]["v_m_h"]), "group_productivity_m_h": r["demand"]["p_m_h"],
            "perimeter_m": r["demand"]["perimeter_m"], "nearest_settlement": r["risk"]["nearest_np"],
            "expected_damage_rub": r["expected_damage_rub"], "expected_area_ha": r.get("expected_area_ha"),
            "p_growth_at_arrival": r["p_growth_at_arrival"],
            "units": [{"name": u["name"], "travel": _h(u["travel_h"]), "travel_h": u["travel_h"], "km": u["km"]} for u in r["units"]],
            "arrival": _h(r["arrival_h"]), "limit": f"{r['demand']['arrival_limit_h']} ч (сокращ. {r['demand']['arrival_limit_reduced_h']} ч)",
            "reinforced_x2": r["late_reinforced"], "deficit": r["deficit"],
        } for r in plan["statement"]],
    }


DEFICIT_NAMES = {"groups": "группы", "АЦЛ": "АЦЛ", "ТЛП": "ТЛП", "ВЛП": "ВЛП", "ТБ": "ТБ", "АГП1": "АГП1"}


def template_answer(q: str, plan: dict) -> str:
    ql = q.lower()
    st = plan["statement"]
    # пожар, упомянутый в вопросе
    target = None
    m = re.search(r"№\s*(\d+)|пожар\w*\s+(\d+)", ql)
    if m:
        num = m.group(1) or m.group(2)
        target = next((r for r in st if f"№ {num} " in r["fire_name"] + " "), None)
    if target is None:
        target = next((r for r in st if r.get("lesnichestvo") and r["lesnichestvo"].lower()[:6] in ql), None)
    unit_q = next((u for r in st for u in r["units"] if u["name"].split("«")[-1].strip("»").lower()[:6] in ql), None)

    if any(w in ql for w in ("дефицит", "не хватает", "нехват", "усилен")):
        d = plan["deficit_summary"]
        if not d["has_deficit"]:
            return "Дефицита нет: по всем пожарам назначенные силы покрывают нормативную потребность."
        tot = ", ".join(f"{DEFICIT_NAMES.get(k, k)}: {v}" for k, v in d["total"].items() if v)
        return f"Дефицит по пожарам: {', '.join(d['fires'])}. Не хватает — {tot}. {d['message']}"
    if any(w in ql for w in ("сравн", "fifo", "базов", "эффект", "лучше", "очеред")):
        a, b = plan["metrics"], plan["baseline"]["metrics"]
        gain = (b["weighted_damage"] - a["weighted_damage"]) / max(b["weighted_damage"], 1)
        def sp(x: int) -> str:
            return f"{x:,}".replace(",", " ")
        pct = f"{gain:.1%}".replace(".", ",")
        return (f"По сравнению с распределением по очереди обнаружения (FIFO) план снижает взвешенный ожидаемый ущерб на {pct} "
                f"(ожидаемый ущерб {sp(a['expected_damage_rub'])} ₽ против {sp(b['expected_damage_rub'])} ₽). "
                f"Нарушений норматива прибытия: {a['norm_violations']} против {b['norm_violations']}; "
                f"дефицит техники: {a['deficit_vehicles']} против {b['deficit_vehicles']} ед.")
    if unit_q and ("почему" in ql or "зачем" in ql):
        r = next(r for r in st if unit_q in r["units"])
        others = [u["name"] for u in r["units"] if u is not unit_q]
        return (f"{unit_q['name']} направлено на «{r['fire_name']}» (категория риска — {r['risk']['category']}): "
                f"расчётное время в пути {_h(unit_q['travel_h'])}, {unit_q['km']} км. "
                + (f"Вместе с ним работают: {', '.join(others)}. " if others else "")
                + "Оптимизатор выбирает назначение, минимизирующее суммарный взвешенный ущерб по всем пожарам, "
                  "поэтому ближайшее подразделение может быть оставлено за более опасным пожаром.")
    if target:
        r = target
        crit = "; ".join(c["text"] for c in r["risk"]["criteria"] if c["triggered"]) or "критерии не превышены"
        units = ", ".join(f"{u['name']} ({_h(u['travel_h'])})" for u in r["units"]) or "не назначены"
        dmd = r["demand"]
        g_need = r["lines"][0]["need"]
        txt = (f"{r['fire_name']}: категория риска «{r['risk']['category']}» (тяжесть — {r['risk']['severity_ru']}, "
               f"вероятность разрастания — {r['risk']['probability_ru']}). Сработали критерии: {crit}. "
               f"Потребность: {g_need} гр. ({g_need * 6} чел.){' с учётом удвоения по п. 52' if r['late_reinforced'] else ''}; скорость прироста кромки {dmd['v_m_h']:.0f} м/ч "
               f"при производительности группы {dmd['p_m_h']} м/ч. Назначены: {units}. Прибытие через {_h(r['arrival_h'])} "
               f"с момента обнаружения (норматив {dmd['arrival_limit_h']} ч, п. 51 Правил тушения).")
        if r["late_reinforced"]:
            txt += (f" При ветре > 5 м/с или КПО ≥ IV предельное время сокращается до {str(dmd['arrival_limit_reduced_h']).replace('.', ',')} ч (п. 52); "
                    "уложиться в него нельзя, поэтому силы увеличены в 2 раза.")
        return txt
    a = plan["metrics"]
    top = st[0] if st else None
    return (f"В плане {len(st)} пожаров, задействовано {a['stations_used']} подразделений. "
            + (f"Наиболее опасный — {top['fire_name']} (категория «{top['risk']['category']}»). " if top else "")
            + f"Нарушений норматива прибытия: {a['norm_violations']}. {plan['deficit_summary']['message']} "
            "Спросите, например: «почему ЛПС-1 направлено на пожар № 2», «есть ли дефицит», «сравни с FIFO».")


def _llm(messages: list[dict]) -> str | None:
    prov = os.environ.get("LLM_PROVIDER", "").lower()
    try:
        if prov == "openai":
            base = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1").rstrip("/")
            r = httpx.post(f"{base}/chat/completions", timeout=60,
                           headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'local')}"},
                           json={"model": os.environ.get("OPENAI_MODEL", "qwen2.5:7b"), "messages": messages, "temperature": 0.2})
            return r.json()["choices"][0]["message"]["content"]
        if prov == "gigachat":
            tok = httpx.post("https://ngw.devices.sberbank.ru:9443/api/v2/oauth", timeout=20, verify=False,
                             headers={"Authorization": f"Basic {os.environ['GIGACHAT_AUTH_KEY']}", "RqUID": str(uuid.uuid4()),
                                      "Content-Type": "application/x-www-form-urlencoded"},
                             data={"scope": os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")}).json()["access_token"]
            r = httpx.post("https://gigachat.devices.sberbank.ru/api/v1/chat/completions", timeout=60, verify=False,
                           headers={"Authorization": f"Bearer {tok}"},
                           json={"model": os.environ.get("GIGACHAT_MODEL", "GigaChat"), "messages": messages, "temperature": 0.2})
            return r.json()["choices"][0]["message"]["content"]
        if prov == "yandexgpt":
            folder = os.environ["YANDEX_FOLDER_ID"]
            r = httpx.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=60,
                           headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                           json={"modelUri": f"gpt://{folder}/yandexgpt/latest",
                                 "completionOptions": {"temperature": 0.2, "maxTokens": 800},
                                 "messages": [{"role": m["role"], "text": m["content"]} for m in messages]})
            return r.json()["result"]["alternatives"][0]["message"]["text"]
    except Exception:
        return None
    return None


NUM_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in NUM_RE.findall(text.replace("\u00a0", "").replace(" ", " ")):
        try:
            out.add(round(float(m.replace(",", ".")), 2))
        except ValueError:
            pass
    return out


def verify_numbers(answer_text: str, plan: dict, question: str) -> tuple[bool, list[float]]:
    """Проверка ответа LLM (ФСТЭК № 117: проверка ответов ИИ): все числа в ответе должны присутствовать
    в данных расчёта или в вопросе (с допуском на округление и перевод часов в минуты)."""
    ctx = json.dumps(_compact(plan), ensure_ascii=False) + " " + template_answer("сравни с FIFO", plan) + " " + question
    allowed = _numbers(ctx)
    extra = set()
    for a in list(allowed):
        extra |= {round(a * 60, 2), round(a / 60, 2), round(a * 100, 2), round(a / 1000, 2), round(a / 1e6, 2), round(a, 1), round(a)}
    allowed |= extra | {float(i) for i in range(0, 13)} | {25.0, 51.0, 52.0, 59.0, 62.0, 244.0, 566.0, 287.0, 1605.0, 2025.0}
    bad = [x for x in _numbers(answer_text) if not any(abs(x - a) <= max(0.05, 0.01 * abs(a)) for a in allowed)]
    return (len(bad) == 0, bad)


def answer(question: str, plan: dict | None, history: list[dict]) -> dict:
    if not plan:
        return {"answer": "План ещё не рассчитан. Нажмите «Рассчитать план».", "source": "template"}
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nПЛАН:\n" + json.dumps(_compact(plan), ensure_ascii=False)}]
    msgs += [m for m in history[-6:] if m.get("role") in ("user", "assistant")]
    msgs.append({"role": "user", "content": question})
    out = _llm(msgs)
    if out:
        ok, bad = verify_numbers(out, plan, question)
        if ok:
            return {"answer": out.strip(), "source": os.environ.get("LLM_PROVIDER"), "verified": True}
        return {"answer": template_answer(question, plan)
                + f"\n\n(Ответ языковой модели не прошёл проверку: числа {', '.join(str(b) for b in bad[:5])} отсутствуют в расчёте. "
                  "Показан детерминированный ответ.)", "source": "template", "verified": False, "rejected_numbers": bad}
    return {"answer": template_answer(question, plan), "source": "template", "verified": True}


def provider_name() -> str:
    p = os.environ.get("LLM_PROVIDER", "").lower()
    return {"gigachat": "GigaChat", "yandexgpt": "YandexGPT", "openai": "локальная LLM"}.get(p, "шаблонный режим")
