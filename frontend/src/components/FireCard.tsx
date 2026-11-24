import { RISK_COLORS, actShort, hm, n, rub } from '../api'
import type { Plan, Row } from '../types'

type Props = {
  row: Row
  plan: Plan
  onApprove: () => void
  onEdit: () => void
  onRecalc: () => void
  onParams: () => void
  busy: boolean
}

function RefLink({ r }: { r: { act: string; point: string; title: string } }) {
  return (
    <span className="ref" title={`${r.title}, ${r.point}`}>
      {actShort(r.act)}
      <small>{r.point.replace('примечания', 'прим.').replace('примечание', 'прим.')}</small>
    </span>
  )
}

export default function FireCard({ row, plan, onApprove, onEdit, onRecalc, onParams, busy }: Props) {
  const color = RISK_COLORS[row.risk.category]
  const d = row.demand
  const fire = plan.fires.find((f) => f.id === row.fire_id)!
  const triggered = row.risk.criteria.filter((c) => c.triggered)
  const fifo = plan.baseline.per_fire[row.fire_id]
  const limit = d.hard_weather ? d.arrival_limit_reduced_h : d.arrival_limit_h
  const approved = plan.status === 'утверждён'

  return (
    <div className="card">
      <header className="risk-banner" style={{ ['--c' as string]: color }}>
        <div className="rb-top">
          <button className="rb-edit" onClick={onParams} disabled={busy}>Параметры пожара</button>
          <h2>{row.fire_name}</h2>
          <span className="rb-meta">{n(fire.area_ha)} га, {fire.fire_type}, ветер {n(fire.wind_ms, 0)} м/с, КПО по погоде {['I', 'II', 'III', 'IV', 'V'][fire.weather_class - 1]}</span>
        </div>
        <div className="rb-cat">
          <span className="rb-label">Категория риска</span>
          <strong>{row.risk.category}</strong>
        </div>
        <div className="rb-axes">
          <span>Тяжесть вреда: <b>{row.risk.severity_ru}</b></span>
          <span>Вероятность разрастания: <b>{row.risk.probability_ru}</b></span>
        </div>
      </header>

      <section className="crit">
        <h3>Почему такая категория</h3>
        <ul>
          {triggered.map((c, i) => (
            <li key={i}><span className={`axis ${c.axis === 'тяжесть' ? 'sev' : 'prob'}`}>{c.axis}</span>{c.text}</li>
          ))}
        </ul>
      </section>

      <section className="calc">
        <h3>Расчёт потребности</h3>
        <div className="formula">
          <span className="f-main">N = <span className="ceil">⌈</span>{Math.round(d.v_m_h)} / {d.p_m_h}<span className="ceil">⌉</span>{row.late_reinforced ? ` × ${d.k_if_late}` : ''} = <b>{row.lines[0].need} гр.</b></span>
          <span className="f-note">скорость прироста кромки {Math.round(d.v_m_h)} м/ч (с учётом периметра {d.perimeter_m} м), производительность группы {d.p_m_h} м/ч</span>
        </div>
      </section>

      <section className="statement">
        <h3>Расчётная ведомость</h3>
        <table>
          <thead>
            <tr><th>Ресурс</th><th className="num">Нужно</th><th className="num">Назначено</th><th>Основание</th></tr>
          </thead>
          <tbody>
            {row.lines.map((l, i) => (
              <tr key={i} className={l.deficit ? 'def' : ''}>
                <td>{l.code === 'groups' ? 'Люди' : l.code === 'reinf' ? 'Усиление' : l.resource.split(' (')[0]}
                  {l.code === 'groups' && <small> группы / чел.</small>}
                  {l.code !== 'groups' && l.code !== 'reinf' && <small> {l.resource.match(/\((.*)\)/)?.[1]}</small>}
                </td>
                <td className="num">{l.code === 'groups' ? `${l.need} / ${l.need_people}` : l.code === 'reinf' ? `×${l.need}` : l.need}</td>
                <td className="num">{l.assigned === null ? '—' : l.code === 'groups' ? `${l.assigned} / ${l.assigned_people}` : l.assigned}
                  {l.deficit > 0 && <em className="deficit">−{l.deficit}</em>}</td>
                <td><RefLink r={l.ref} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="units">
        <h3>Назначенные подразделения</h3>
        {row.units.length === 0 && <p className="empty">Все подразделения в зоне досягаемости направлены на другие пожары. Нужен запрос усиления: межрайонное маневрирование, авиация или силы МЧС.</p>}
        {row.units.map((u, i) => (
          <div key={u.id} className="unit">
            <div className="u-name">{u.name}{i === 0 && <span className="lead">ведущее</span>}</div>
            <div className="u-res">{u.groups} гр. / {u.people} чел., АЦЛ {u.vehicles['АЦЛ']}{u.vehicles['ТЛП'] ? `, ТЛП ${u.vehicles['ТЛП']}` : ''}{u.vehicles['ТБ'] ? `, ТБ ${u.vehicles['ТБ']}` : ''}{u.vehicles['ВЛП'] ? `, ВЛП ${u.vehicles['ВЛП']}` : ''}
              {u.heavy_h ? <span className="u-heavy">тракторная техника через {hm(u.heavy_h)}</span> : null}</div>
            <div className="u-time"><b>{hm(u.travel_h)}</b> {n(u.km, 0)} км</div>
          </div>
        ))}
        <div className={`arrival ${row.norm_ok ? 'ok' : 'bad'}`}>
          <div>
            <span>Прибытие с момента обнаружения</span>
            <b>{hm(row.arrival_h)}</b>
          </div>
          <div>
            <span>Норматив{d.hard_weather ? ' (сокращённый, п. 52)' : ' (п. 51)'}</span>
            <b>{hm(limit)}</b>
          </div>
          <div>
            <span>По очереди (FIFO)</span>
            <b>{hm(fifo?.arrival_h)}</b>
          </div>
        </div>
        <p className="prob-note">При таком прибытии вероятность разрастания до 25 га — <b>{Math.round(row.p_growth_at_arrival * 100)}%</b>{row.expected_area_ha ? <>, ожидаемая итоговая площадь — <b>{n(row.expected_area_ha)} га</b></> : null}, ожидаемый ущерб — {rub(row.expected_damage_rub)}.</p>
      </section>

      <footer className="actions">
        <button className="btn primary" onClick={onApprove} disabled={busy || approved}>{approved ? 'План утверждён' : 'Утвердить план'}</button>
        <button className="btn" onClick={onEdit} disabled={busy}>Изменить</button>
        <button className="btn" onClick={onRecalc} disabled={busy}>{busy ? 'Считаю…' : 'Пересчитать'}</button>
        <a className="btn" href="/api/plan/pdf">PDF</a>
      </footer>
    </div>
  )
}
