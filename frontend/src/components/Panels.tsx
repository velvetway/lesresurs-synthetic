import { useEffect, useRef, useState } from 'react'
import { RISK_COLORS, api, hm, n, rub } from '../api'
import type { Plan, Reference, Station } from '../types'

export function FireList({ plan, selected, onSelect }: { plan: Plan; selected: string | null; onSelect: (id: string) => void }) {
  return (
    <div className="fire-list">
      {plan.statement.map((r) => {
        const f = plan.fires.find((x) => x.id === r.fire_id)!
        const def = Object.values(r.deficit).some((v) => v > 0)
        return (
          <button key={r.fire_id} className={`fire-item ${selected === r.fire_id ? 'sel' : ''}`}
            style={{ ['--c' as string]: RISK_COLORS[r.risk.category] }} onClick={() => onSelect(r.fire_id)}>
            <span className="fi-bar" />
            <span className="fi-main">
              <span className="fi-name">№ {r.fire_name.match(/№ (\d+)/)?.[1]} <span>{r.lesnichestvo}</span></span>
              <span className="fi-cat">{r.risk.category}</span>
              <span className="fi-meta">{n(f.area_ha)} га, ветер {n(f.wind_ms, 0)} м/с, н.п. {n(r.risk.nearest_np.km)} км</span>
            </span>
            <span className="fi-side">
              <b>{r.units.length ? hm(r.arrival_h) : '—'}</b>
              <small className={r.norm_ok ? 'ok' : 'bad'}>{r.norm_ok ? 'в нормативе' : 'сверх норматива'}</small>
              {def && <small className="bad">дефицит</small>}
            </span>
          </button>
        )
      })}
    </div>
  )
}

export function Comparison({ plan }: { plan: Plan }) {
  const a = plan.metrics, b = plan.baseline.metrics
  const gain = (b.expected_damage_rub - a.expected_damage_rub) / Math.max(b.expected_damage_rub, 1)
  const rows: [string, string, string, boolean][] = [
    ['Ожидаемый ущерб', rub(a.expected_damage_rub), rub(b.expected_damage_rub), a.expected_damage_rub <= b.expected_damage_rub],
    ['Ожидаемо крупных (≥ 25 га)', `${n((100 * a.expected_large_fires) / Math.max(a.n_fires, 1), 0)}%`, `${n((100 * b.expected_large_fires) / Math.max(b.n_fires, 1), 0)}%`, a.expected_large_fires <= b.expected_large_fires + 1e-9],
    ['Ожидаемая площадь, га', n(a.expected_area_ha, 0), n(b.expected_area_ha, 0), a.expected_area_ha <= b.expected_area_ha + 1e-9],
    ['Сверх норматива прибытия', `${a.norm_violations}`, `${b.norm_violations}`, a.norm_violations <= b.norm_violations],
    ['Дефицит техники, ед.', `${a.deficit_vehicles}`, `${b.deficit_vehicles}`, a.deficit_vehicles <= b.deficit_vehicles],
    ['Дефицит групп', `${a.deficit_groups}`, `${b.deficit_groups}`, a.deficit_groups <= b.deficit_groups],
  ]
  return (
    <div className="compare">
      <div className="cmp-head">
        <span />
        <span>Ассистент</span>
        <span>По очереди</span>
      </div>
      {rows.map(([k, x, y, good]) => (
        <div key={k} className="cmp-row">
          <span>{k}</span>
          <b className={good ? 'good' : ''}>{x}</b>
          <span>{y}</span>
        </div>
      ))}
      <p className="cmp-note">
        {gain > 0.0005 ? `Ожидаемый ущерб ниже на ${(gain * 100).toLocaleString('ru-RU', { maximumFractionDigits: 1 })}%. ` : ''}
        Решение CP-SAT: {plan.solver.status === 'OPTIMAL' ? 'оптимум' : plan.solver.status} за {Math.round(plan.solver.wall_time_s * 1000)} мс.
      </p>
    </div>
  )
}

export function Pool({ plan, onToggle }: { plan: Plan; onToggle: (s: Station) => void }) {
  const sent = plan.stations.filter((s) => s.assigned_to)
  const free = plan.stations.filter((s) => !s.assigned_to && s.available)
  const off = plan.stations.filter((s) => !s.available)
  const sum = (arr: Station[], k: string) => arr.reduce((a, s) => a + (k === 'groups' ? s.groups : s.vehicles[k] ?? 0), 0)
  return (
    <div className="pool">
      <div className="pool-bars">
        {[['Группы', 'groups'], ['АЦЛ', 'АЦЛ'], ['ТЛП', 'ТЛП'], ['ТБ', 'ТБ'], ['ВЛП', 'ВЛП']].filter(([, k]) => plan.stations.some((s) => k === 'groups' || (s.vehicles[k] ?? 0) > 0)).map(([label, k]) => {
          const s1 = sum(sent, k), s2 = sum(free, k), tot = s1 + s2 + sum(off, k)
          return (
            <div key={k} className="pb">
              <span className="pb-l">{label}</span>
              <span className="pb-track">
                <i className="pb-sent" style={{ width: `${(100 * s1) / Math.max(tot, 1)}%` }} />
                <i className="pb-free" style={{ width: `${(100 * s2) / Math.max(tot, 1)}%` }} />
              </span>
              <span className="pb-n"><b>{s2}</b> из {tot}</span>
            </div>
          )
        })}
      </div>
      <p className="pool-note">Направлено {sent.length} из {plan.stations.length} станций, в резерве {free.length}{off.length ? `, недоступно ${off.length}` : ''}.</p>
      <details className="pool-list">
        <summary>Станции и доступность</summary>
        {plan.stations.map((s) => (
          <label key={s.id} className={`pl-row ${s.assigned_to ? 'sent' : ''}`}>
            <input type="checkbox" checked={s.available} onChange={() => onToggle(s)} />
            <span>{s.name}</span>
            <small>{s.assigned_to ? plan.statement.find((r) => r.fire_id === s.assigned_to)?.fire_name.match(/№ \d+/)?.[0] : s.available ? 'резерв' : 'нет'}</small>
          </label>
        ))}
      </details>
    </div>
  )
}

type Msg = { role: 'user' | 'assistant'; content: string }

export function Chat({ fireName, llm }: { fireName: string | null; llm: string }) {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [q, setQ] = useState('')
  const [wait, setWait] = useState(false)
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => { end.current?.scrollIntoView({ block: 'end' }) }, [msgs])
  const ask = async (text: string) => {
    if (!text.trim()) return
    const next = [...msgs, { role: 'user' as const, content: text }]
    setMsgs(next)
    setQ('')
    setWait(true)
    try {
      const r = await api.chat(text, next.slice(0, -1))
      setMsgs([...next, { role: 'assistant', content: r.answer }])
    } catch (e) {
      setMsgs([...next, { role: 'assistant', content: `Ассистент недоступен: ${(e as Error).message}` }])
    } finally {
      setWait(false)
    }
  }
  const num = fireName?.match(/№ (\d+)/)?.[1]
  const sugg = [num ? `Объясни решение по пожару № ${num}` : 'Какой пожар самый опасный?', 'Есть ли дефицит сил?', 'Сравни с распределением по очереди']
  return (
    <div className="chat">
      <div className="chat-log">
        {msgs.length === 0 && (
          <p className="empty">Спросите, почему выбрано подразделение, где дефицит или чем план лучше распределения по очереди. Ассистент отвечает только по данным расчёта. Режим: {llm}.</p>
        )}
        {msgs.map((m, i) => <div key={i} className={`msg ${m.role}`}>{m.content}</div>)}
        {wait && <div className="msg assistant wait">Формирую ответ…</div>}
        <div ref={end} />
      </div>
      <div className="chips">{sugg.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div>
      <form className="chat-in" onSubmit={(e) => { e.preventDefault(); ask(q) }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Вопрос ассистенту" />
        <button className="btn primary" disabled={wait}>Спросить</button>
      </form>
    </div>
  )
}

export function Journal({ plan }: { plan: Plan }) {
  return (
    <ol className="journal">
      {plan.journal.map((j, i) => (
        <li key={i}><time>{j.t}</time><b>{j.event}</b><span>{j.detail}</span></li>
      ))}
    </ol>
  )
}

export function EditModal({ plan, fireId, onClose, onSave }: {
  plan: Plan; fireId: string; onClose: () => void; onSave: (ids: string[]) => void
}) {
  const row = plan.statement.find((r) => r.fire_id === fireId)!
  const [sel, setSel] = useState<string[]>(row.units.map((u) => u.id))
  const candidates = plan.stations.filter((s) => s.available)
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Изменить назначение">
        <h2>Изменить назначение: {row.fire_name}</h2>
        <p>Отметьте подразделения для этого пожара. Остальные пожары будут перераспределены оптимизатором с учётом вашего решения.</p>
        <div className="modal-list">
          {candidates.map((s) => {
            const other = s.assigned_to && s.assigned_to !== fireId
            return (
              <label key={s.id} className="pl-row">
                <input type="checkbox" checked={sel.includes(s.id)}
                  onChange={(e) => setSel(e.target.checked ? [...sel, s.id] : sel.filter((x) => x !== s.id))} />
                <span>{s.name}</span>
                <small>{s.groups} гр., АЦЛ {s.vehicles['АЦЛ']}, ТЛП {s.vehicles['ТЛП']}{other ? ' — сейчас на другом пожаре' : ''}</small>
              </label>
            )
          })}
        </div>
        <div className="actions">
          <button className="btn primary" onClick={() => onSave(sel)}>Применить и пересчитать</button>
          <button className="btn" onClick={onClose}>Отмена</button>
        </div>
      </div>
    </div>
  )
}

export function AddFireForm({ at, onCancel, onSave, initial, title, onDelete }: {
  at: { lat: number; lon: number }; onCancel: () => void; onSave: (f: Record<string, unknown>) => void
  initial?: { area_ha: number; wind_ms: number; fire_type: string; weather_class: number; species: string | null }
  title?: string; onDelete?: () => void
}) {
  const [f, setF] = useState({
    area_ha: initial?.area_ha ?? 1, wind_ms: initial?.wind_ms ?? 4, fire_type: initial?.fire_type ?? 'низовой',
    weather_class: initial?.weather_class ?? 4, species: initial?.species ?? 'Сосна',
  })
  const set = (k: string, v: unknown) => setF({ ...f, [k]: v })
  return (
    <div className="modal-bg" onClick={onCancel}>
      <form className="modal small" onClick={(e) => e.stopPropagation()}
        onSubmit={(e) => { e.preventDefault(); onSave({ ...f, lat: at.lat, lon: at.lon }) }}>
        <h2>{title ?? 'Новый пожар'}</h2>
        <p>{at.lat.toFixed(4)}° с. ш., {at.lon.toFixed(4)}° в. д.</p>
        <div className="form-grid">
          <label>Площадь, га<input type="number" step="any" min="0.01" value={f.area_ha} onChange={(e) => set('area_ha', +e.target.value)} /></label>
          <label>Ветер, м/с<input type="number" step="any" min="0" value={f.wind_ms} onChange={(e) => set('wind_ms', +e.target.value)} /></label>
          <label>Вид пожара
            <select value={f.fire_type} onChange={(e) => set('fire_type', e.target.value)}>
              <option>низовой</option><option>верховой</option><option>почвенный</option>
            </select>
          </label>
          <label>КПО по погоде
            <select value={f.weather_class} onChange={(e) => set('weather_class', +e.target.value)}>
              {[1, 2, 3, 4, 5].map((x) => <option key={x} value={x}>{x}</option>)}
            </select>
          </label>
          <label>Преобладающая порода
            <select value={f.species} onChange={(e) => set('species', e.target.value)}>
              {Array.from(new Set([f.species, 'Сосна', 'Дуб', 'Кедр', 'Лиственница', 'Ель', 'Береза', 'Пихта', 'Осина'].filter(Boolean))).map((x) => <option key={x}>{x}</option>)}
            </select>
          </label>
        </div>
        <div className="actions">
          <button className="btn primary">{initial ? 'Сохранить и пересчитать' : 'Добавить и пересчитать'}</button>
          <button type="button" className="btn" onClick={onCancel}>Отмена</button>
          {onDelete && <button type="button" className="btn danger" onClick={onDelete}>Снять пожар с учёта</button>}
        </div>
      </form>
    </div>
  )
}


export function Help({ reference, plan }: { reference: Reference | null; plan: Plan }) {
  if (!reference) return <p className="empty pad">Справочные данные загружаются.</p>
  const m = reference.model
  const rn = reference.region_norms as Record<string, unknown>
  const imp = Object.entries(m.feature_importance ?? {}).slice(0, 6)
  const names: Record<string, string> = {
    lon: 'долгота', lat: 'широта', subject: 'субъект', np_dist_km: 'расстояние до н.п.', wind_ms: 'ветер',
    n_active_subj: 'пожаров в субъекте', doy: 'день года', hour: 'час', species: 'порода', n_active_lesn: 'пожаров в лесничестве',
    month: 'месяц', delay_h: 'задержка тушения',
  }
  return (
    <div className="help">
      <section>
        <h3>Как считается план</h3>
        <ol>
          <li>Категория риска — матрица 3×3: тяжесть вреда (н.п. ≤ 5/10 км, ООПТ ≤ 10/20 км, верховой пожар, класс природной пожарной опасности) × вероятность разрастания (ML-прогноз, ветер, КПО по погоде).</li>
          <li>Потребность — N = ⌈v/p⌉ × k групп по 6 человек; техника — по правилам 1605-р и Правил тушения; k = 2, если при ветре &gt; 5 м/с или КПО ≥ IV нельзя прибыть за сокращённое время (п. 52).</li>
          <li>Распределение — OR-Tools CP-SAT: минимум взвешенного ожидаемого ущерба при ограничениях «одно подразделение — один пожар», «силы ≥ потребности», «прибытие ≤ норматива п. 51», с учётом резерва станций.</li>
          <li>Решение утверждает человек; все действия пишутся в журнал.</li>
        </ol>
      </section>
      <section>
        <h3>Нормативы региона: {reference.region}</h3>
        <p>По распоряжению № 1605-р: {String(rn.groups)} групп ({String(rn.people)} чел.), станций {String(rn.stations_total)} (ЛПС-1 — {String(rn['ЛПС-1'])}, ЛПС-2 — {String(rn['ЛПС-2'])}, ЛПС-3 — {String(rn['ЛПС-3'])}), АЦЛ {String(rn['АЦЛ'])}, ТЛП {String(rn['ТЛП'])}, ТБ {String(rn['ТБ'] ?? '—')}, ВЛП {String(rn['ВЛП'])}.</p>
        <p className="muted">Дислокация станций в прототипе условная: подобрана по истории пожаров. Для пилота нужна фактическая.</p>
      </section>
      <section>
        <h3>Модели разрастания (CatBoost)</h3>
        <table className="kv">
          <tbody>
            <tr><td>Вероятность ≥ 25 га: ROC-AUC (тест {m.test_years})</td><td>{n(m.roc_auc, 3)}</td></tr>
            <tr><td>PR-AUC / Brier</td><td>{n(m.pr_auc, 3)} / {n(m.brier, 3)}</td></tr>
            <tr><td>ROC-AUC по региону А</td><td>{n(m.roc_auc_region_a, 3)}</td></tr>
            {m.area_model && <tr><td>Итоговая площадь: R² (лог) / Спирмен</td><td>{n(m.area_model.r2_log, 2)} / {n(m.area_model.spearman, 2)}</td></tr>}
            <tr><td>Обучение / тест, пожаров</td><td>{m.n_train.toLocaleString('ru-RU')} / {m.n_test.toLocaleString('ru-RU')}</td></tr>
          </tbody>
        </table>
        <p className="muted">Главные факторы: {imp.map(([k, v]) => `${names[k] ?? k} (${n(v, 0)}%)`).join(', ')}.</p>
      </section>
      <section>
        <h3>Нормативные правовые акты</h3>
        <ul className="acts">{reference.acts.map((a) => <li key={a.code}>{a.title}</li>)}</ul>
      </section>
      <section>
        <h3>Обстановка</h3>
        <p>{plan.scenario.description}</p>
      </section>
    </div>
  )
}
