import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import FireCard from './components/FireCard'
import MapView from './components/MapView'
import { AddFireForm, Chat, Comparison, EditModal, FireList, Help, Journal, Pool } from './components/Panels'
import type { Plan, Reference, Station } from './types'

type Tab = 'card' | 'chat' | 'log' | 'help'

export default function App() {
  const [plan, setPlan] = useState<Plan | null>(null)
  const [ref, setRef] = useState<Reference | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('card')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [edit, setEdit] = useState(false)
  const [params, setParams] = useState(false)
  const [addMode, setAddMode] = useState(false)
  const [newAt, setNewAt] = useState<{ lat: number; lon: number } | null>(null)
  const [history, setHistory] = useState<{ t: string; n_active: number }[]>([])
  const [regions, setRegions] = useState<{ slug: string; name: string }[]>([])

  const run = useCallback(async (fn: () => Promise<unknown>) => {
    setBusy(true)
    setErr(null)
    try {
      await fn()
      const p = await api.plan()
      setPlan(p)
      setSelected((s) => (s && p.statement.some((r) => r.fire_id === s) ? s : p.statement[0]?.fire_id ?? null))
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [])

  const loadRefs = useCallback(() => {
    api.reference().then(setRef).catch(() => undefined)
    api.history().then(setHistory).catch(() => undefined)
  }, [])

  useEffect(() => {
    run(async () => undefined)
    loadRefs()
    api.regions().then((r) => setRegions(r.regions)).catch(() => undefined)
  }, [run, loadRefs])

  const switchRegion = (slug: string) => {
    setSelected(null)
    run(() => api.setRegion(slug)).then(loadRefs)
  }

  const row = plan?.statement.find((r) => r.fire_id === selected) ?? null
  const onSelect = useCallback((id: string) => { setSelected(id); setTab('card') }, [])
  const toggleStation = (s: Station) => run(() => api.station(s.id, { available: !s.available }))

  if (!plan) {
    return <div className="loading">{err ? `Не удалось получить расчёт: ${err}` : 'Загружаю обстановку и рассчитываю план…'}</div>
  }
  const now = new Date(plan.scenario.now)

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 4l8 12h-4.5l5.5 9H7l5.5-9H8z" fill="#A9C2B1" /><circle cx="23" cy="23" r="5.2" fill="#E2562B" /></svg>
          <div>
            <b>ЛесРесурс</b>
            <span>распределение сил и средств пожаротушения</span>
          </div>
        </div>
        <div className="scn">
          <select className="scn-region" aria-label="Регион" value={plan.region} onChange={(e) => switchRegion(e.target.value)}>
            {regions.map((r) => <option key={r.slug} value={r.slug}>{r.name}</option>)}
          </select>
          <select aria-label="Сценарий" value="" onChange={(e) => {
            const v = e.target.value
            if (v === 'demo') run(api.demo)
            else if (v) run(() => api.loadHistory(v))
          }}>
            <option value="">{plan.scenario.title}</option>
            <option value="demo">Демо-сценарий региона</option>
            {history.slice().reverse().slice(0, 80).map((h) => (
              <option key={h.t} value={h.t}>{new Date(h.t).toLocaleString('ru-RU', { dateStyle: 'short', timeStyle: 'short' })}: {h.n_active} пожаров</option>
            ))}
          </select>
          <span className="scn-time">обстановка на {now.toLocaleString('ru-RU', { day: '2-digit', month: 'long', hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div className="top-actions">
          <span className={`status ${plan.status === 'утверждён' ? 'ok' : ''}`}>План: {plan.status}</span>
          <button className={`btn ghost ${addMode ? 'on' : ''}`} onClick={() => setAddMode(!addMode)}>{addMode ? 'Отменить' : 'Добавить пожар'}</button>
          <label className="btn ghost file">
            Загрузить реестр
            <input type="file" accept=".xls,.xlsx" onChange={(e) => {
              const f = e.target.files?.[0]
              e.target.value = ''
              if (f) run(() => api.upload(f))
            }} />
          </label>
          <a className="btn ghost" href="/api/plan/xlsx">Ведомость XLSX</a>
        </div>
      </header>

      <aside className="left">
        <section>
          <h3 className="sec-h">Действующие пожары <span>{plan.statement.length}</span></h3>
          <FireList plan={plan} selected={selected} onSelect={onSelect} />
        </section>
        <section>
          <h3 className="sec-h">Ресурсы края</h3>
          <Pool plan={plan} onToggle={toggleStation} />
        </section>
        <section>
          <h3 className="sec-h">Сравнение с распределением по очереди</h3>
          <Comparison plan={plan} />
        </section>
      </aside>

      <main className="center">
        <MapView plan={plan} reference={ref} selected={selected} onSelect={onSelect} addMode={addMode}
          onMapClick={(lat, lon) => { setNewAt({ lat, lon }); setAddMode(false) }} />
        {plan.deficit_summary.has_deficit && (
          <div className="deficit-bar">
            <b>Дефицит сил:</b> {Object.entries(plan.deficit_summary.total).filter(([, v]) => v).map(([k, v]) => `${k === 'groups' ? 'групп' : k} ${v}`).join(', ')} — {plan.deficit_summary.fires.join(', ')}. Требуется усиление.
          </div>
        )}
        {err && <div className="err-bar">{err}</div>}
      </main>

      <aside className="right">
        <nav className="tabs" role="tablist">
          {([['card', 'Карточка пожара'], ['chat', 'Ассистент'], ['log', 'Журнал'], ['help', 'Справка']] as [Tab, string][]).map(([k, l]) => (
            <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>{l}</button>
          ))}
        </nav>
        <div className="tab-body">
          {tab === 'card' && row && (
            <FireCard row={row} plan={plan} busy={busy}
              onApprove={() => run(api.approve)} onEdit={() => setEdit(true)} onRecalc={() => run(api.recalc)}
              onParams={() => setParams(true)} />
          )}
          {tab === 'chat' && <Chat fireName={row?.fire_name ?? null} llm={ref?.llm ?? 'шаблонный режим'} />}
          {tab === 'log' && <Journal plan={plan} />}
          {tab === 'help' && <Help reference={ref} plan={plan} />}
        </div>
      </aside>

      {edit && row && (
        <EditModal plan={plan} fireId={row.fire_id} onClose={() => setEdit(false)}
          onSave={(ids) => { setEdit(false); run(() => api.override(row.fire_id, ids)) }} />
      )}
      {params && row && (() => {
        const sf = plan.scenario.fires.find((x) => x.id === row.fire_id)
        const fr = plan.fires.find((x) => x.id === row.fire_id)!
        return (
          <AddFireForm at={{ lat: fr.lat, lon: fr.lon }} title={`Параметры: ${row.fire_name}`}
            initial={{ area_ha: sf?.area_ha ?? fr.area_ha, wind_ms: sf?.wind_ms ?? fr.wind_ms, fire_type: sf?.fire_type ?? fr.fire_type,
              weather_class: sf?.weather_class ?? fr.weather_class, species: sf?.species ?? fr.species }}
            onCancel={() => setParams(false)}
            onSave={(f) => { setParams(false); run(() => api.updateFire(row.fire_id, f)) }}
            onDelete={() => { setParams(false); setSelected(null); run(() => api.deleteFire(row.fire_id)) }} />
        )
      })()}
      {newAt && (
        <AddFireForm at={newAt} onCancel={() => setNewAt(null)}
          onSave={(f) => { setNewAt(null); run(() => api.addFire(f)) }} />
      )}
    </div>
  )
}
