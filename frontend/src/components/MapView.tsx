import maplibregl, { type Map as MLMap, type StyleSpecification } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { useEffect, useRef, useState } from 'react'
import { RISK_COLORS, hm, n } from '../api'
import type { Plan, Reference } from '../types'

type Props = {
  plan: Plan
  reference: Reference | null
  selected: string | null
  onSelect: (id: string) => void
  addMode: boolean
  onMapClick: (lat: number, lon: number) => void
}

const STYLE: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© участники OpenStreetMap',
    },
  },
  layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#D3E1E6' } }],
}

export default function MapView({ plan, reference, selected, onSelect, addMode, onMapClick }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const map = useRef<MLMap | null>(null)
  const markers = useRef<maplibregl.Marker[]>([])
  const popup = useRef<maplibregl.Popup | null>(null)
  const [ready, setReady] = useState(false)
  const [osm, setOsm] = useState(false)
  const [zoomTick, setZoomTick] = useState(0)
  const clickRef = useRef(onMapClick)
  clickRef.current = onMapClick
  const addRef = useRef(addMode)
  addRef.current = addMode

  useEffect(() => {
    if (!box.current) return
    const m = new maplibregl.Map({
      container: box.current, style: STYLE, center: [133.2, 44.4], zoom: 6.3,
      attributionControl: { compact: true }, maxZoom: 14, minZoom: 4,
    })
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    m.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'top-right')
    m.on('load', async () => {
      m.addSource('land', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      m.addSource('borders', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      m.addLayer({ id: 'osm', type: 'raster', source: 'osm', layout: { visibility: 'none' } })
      m.addLayer({
        id: 'land', type: 'fill', source: 'land',
        paint: { 'fill-color': ['case', ['==', ['get', 'name'], 'Russia'], '#EEF1E8', '#E2E5DE'] },
      })
      m.addLayer({ id: 'coast', type: 'line', source: 'land', paint: { 'line-color': '#9FB6BE', 'line-width': 0.8 } })
      m.addLayer({
        id: 'borders', type: 'line', source: 'borders',
        paint: { 'line-color': '#7D8C84', 'line-width': 1.1, 'line-dasharray': [3, 2] },
      })
      m.addSource('routes', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      m.addLayer({
        id: 'routes-casing', type: 'line', source: 'routes', layout: { 'line-cap': 'round' },
        paint: { 'line-color': '#ffffff', 'line-width': ['case', ['get', 'sel'], 6, 4], 'line-opacity': 0.9 },
      })
      const routePaint = {
        'line-color': ['get', 'color'],
        'line-width': ['case', ['get', 'sel'], 3.2, 1.8],
        'line-opacity': ['case', ['get', 'sel'], 1, 0.55],
      } as const
      m.addLayer({
        id: 'routes', type: 'line', source: 'routes', filter: ['==', ['get', 'lead'], true],
        layout: { 'line-cap': 'round' }, paint: { ...routePaint } as never,
      })
      m.addLayer({
        id: 'routes-reinf', type: 'line', source: 'routes', filter: ['==', ['get', 'lead'], false],
        paint: { ...routePaint, 'line-dasharray': [2, 1.6] } as never,
      })
      setReady(true)
    })
    m.on('zoomend', () => setZoomTick((x) => x + 1))
    m.on('click', (e) => {
      if (addRef.current) clickRef.current(e.lngLat.lat, e.lngLat.lng)
    })
    map.current = m
    return () => m.remove()
  }, [])

  // подложка модельного субъекта (синтетическая геометрия, офлайн)
  useEffect(() => {
    const m = map.current
    if (!m || !ready) return
    fetch(`/basemap_${plan.region}.json`).then((r) => r.json()).then((bm) => {
      ;(m.getSource('land') as maplibregl.GeoJSONSource).setData(bm.land)
      ;(m.getSource('borders') as maplibregl.GeoJSONSource).setData(bm.borders)
    }).catch(() => undefined)
  }, [plan.region, ready])

  useEffect(() => {
    const m = map.current
    if (!m || !ready) return
    m.setLayoutProperty('osm', 'visibility', osm ? 'visible' : 'none')
    m.setLayoutProperty('land', 'visibility', osm ? 'none' : 'visible')
  }, [osm, ready])

  useEffect(() => {
    const m = map.current
    if (!m || !ready) return
    m.getCanvas().style.cursor = addMode ? 'crosshair' : ''
  }, [addMode, ready])

  // маршруты
  useEffect(() => {
    const m = map.current
    if (!m || !ready) return
    const feats = plan.statement.flatMap((r) =>
      r.units.map((u, i) => ({
        type: 'Feature' as const,
        properties: { color: RISK_COLORS[r.risk.category], sel: r.fire_id === selected, lead: i === 0 },
        geometry: { type: 'LineString' as const, coordinates: u.geometry },
      })),
    )
    ;(m.getSource('routes') as maplibregl.GeoJSONSource).setData({ type: 'FeatureCollection', features: feats })
  }, [plan, selected, ready])

  // маркеры
  useEffect(() => {
    const m = map.current
    if (!m || !ready) return
    markers.current.forEach((mk) => mk.remove())
    markers.current = []
    const add = (el: HTMLElement, lon: number, lat: number, anchor: maplibregl.PositionAnchor = 'center') => {
      const mk = new maplibregl.Marker({ element: el, anchor }).setLngLat([lon, lat]).addTo(m)
      markers.current.push(mk)
    }
    const rowByFire = Object.fromEntries(plan.statement.map((r) => [r.fire_id, r]))
    const npShown = new Set<string>()

    // населённые пункты — ближайшие к пожарам и крупные
    const nps = plan.statement.filter((r) => r.fire_id === selected).map((r) => r.risk.nearest_np)
    nps.forEach((np) => {
      if (npShown.has(np.name)) return
      npShown.add(np.name)
      const el = document.createElement('div')
      el.className = 'mk-np near'
      el.innerHTML = `<i></i><span>${np.name.replace(/^(с|п|пгт|д|г)\s/, '')}</span>`
      add(el, np.lon, np.lat, 'left')
    })
    const km = (a: [number, number], b: [number, number]) =>
      Math.hypot((a[0] - b[0]) * 111 * Math.cos((a[1] * Math.PI) / 180), (a[1] - b[1]) * 111)
    const taken: [number, number][] = [
      ...plan.fires.map((f) => [f.lon, f.lat] as [number, number]),
      ...plan.stations.map((s) => [s.lon, s.lat] as [number, number]),
      ...nps.map((p) => [p.lon, p.lat] as [number, number]),
    ]
    const minKm = 9 * Math.pow(2, 7.5 - m.getZoom())
    ;(reference?.settlements ?? []).slice(0, 60).forEach((s) => {
      if (npShown.has(s.full_name)) return
      if (taken.some((t) => km(t, [s.lon, s.lat]) < minKm)) return
      taken.push([s.lon, s.lat])
      const el = document.createElement('div')
      el.className = 'mk-np'
      el.innerHTML = `<i></i><span>${s.name}</span>`
      add(el, s.lon, s.lat, 'left')
    })

    // станции
    plan.stations.forEach((s) => {
      const el = document.createElement('div')
      const busy = !!s.assigned_to
      const selFire = busy && s.assigned_to === selected
      el.className = `mk-st ${busy ? 'busy' : ''} ${s.available ? '' : 'off'} ${selFire ? 'sel' : ''}`
      const veh = Object.entries(s.vehicles).filter(([, v]) => v > 0).map(([k, v]) => `${k} ${v}`).join(', ')
      const target = busy ? plan.statement.find((r) => r.fire_id === s.assigned_to)?.fire_name : null
      el.title = s.name
      el.innerHTML = `<b>${s.type.replace('ЛПС-', '')}</b>`
      el.onclick = (e) => {
        e.stopPropagation()
        popup.current?.remove()
        popup.current = new maplibregl.Popup({ offset: 14, closeButton: true, maxWidth: '260px' })
          .setLngLat([s.lon, s.lat])
          .setHTML(`<div class="st-pop"><b>${s.name}</b><span>${s.settlement ?? ''}</span>`
            + `<em class="${busy ? 'busy' : s.available ? 'free' : 'off'}">${busy ? `Направлена: ${target}` : s.available ? `Свободно: ${s.groups} гр. (${s.people} чел.), ${veh}` : 'Недоступна'}</em>`
            + `<small>Состав: ${s.groups} гр., ${veh}</small></div>`)
          .addTo(m)
        if (busy) onSelect(s.assigned_to!)
      }
      add(el, s.lon, s.lat)
      if (selFire) {
        const u = rowByFire[s.assigned_to!]?.units.find((x) => x.id === s.id)
        const lab = document.createElement('div')
        lab.className = 'mk-st-label'
        lab.innerHTML = `${s.name}<em>${hm(u?.travel_h)} в пути</em>`
        add(lab, s.lon, s.lat, 'top')
      }
    })

    // пожары
    plan.statement.forEach((r) => {
      const f = plan.fires.find((x) => x.id === r.fire_id)!
      const el = document.createElement('button')
      const color = RISK_COLORS[r.risk.category]
      const lvl = Math.max(0, ['низкий', 'средний', 'значительный', 'высокий', 'чрезвычайно высокий'].indexOf(r.risk.category))
      el.className = `mk-fire ${r.fire_id === selected ? 'sel' : ''}`
      el.style.setProperty('--c', color)
      el.style.setProperty('--halo', `${18 + lvl * 7}px`)
      el.setAttribute('aria-label', r.fire_name)
      el.innerHTML = `<span class="halo"></span><span class="core">${r.fire_name.match(/№ (\d+)/)?.[1] ?? ''}</span>`
      el.onclick = (e) => { e.stopPropagation(); onSelect(r.fire_id) }
      add(el, f.lon, f.lat)
      el.title = `${r.fire_name}: ${r.risk.category}, ${n(f.area_ha)} га, ветер ${n(f.wind_ms, 0)} м/с`
    })
  }, [plan, selected, ready, reference, onSelect, zoomTick])

  // обзор всей обстановки при загрузке плана
  const planKey = plan.fires.map((f) => f.id).join(',') + plan.scenario.now
  useEffect(() => {
    const m = map.current
    if (!m || !ready || !plan.fires.length) return
    const b = new maplibregl.LngLatBounds([plan.fires[0].lon, plan.fires[0].lat], [plan.fires[0].lon, plan.fires[0].lat])
    plan.statement.forEach((r) => r.units.forEach((u) => u.geometry.forEach((c) => b.extend(c as [number, number]))))
    plan.fires.forEach((f) => b.extend([f.lon, f.lat]))
    m.fitBounds(b, { padding: { top: 70, bottom: 120, left: 60, right: 60 }, maxZoom: 9, duration: 0 })
  }, [planKey, ready]) // eslint-disable-line react-hooks/exhaustive-deps

  // фокус на выбранный пожар и его силы (после действия пользователя)
  const prevSel = useRef<string | null>(null)
  useEffect(() => {
    const m = map.current
    if (!m || !ready || !selected) return
    if (prevSel.current === null) { prevSel.current = selected; return }
    if (prevSel.current === selected) return
    prevSel.current = selected
    const r = plan.statement.find((x) => x.fire_id === selected)
    const f = plan.fires.find((x) => x.id === selected)
    if (!r || !f) return
    const b = new maplibregl.LngLatBounds([f.lon, f.lat], [f.lon, f.lat])
    r.units.forEach((u) => u.geometry.forEach((c) => b.extend(c as [number, number])))
    b.extend([r.risk.nearest_np.lon, r.risk.nearest_np.lat])
    m.fitBounds(b, { padding: { top: 80, bottom: 130, left: 80, right: 80 }, maxZoom: 9, duration: 600 })
  }, [selected, ready]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="map-wrap">
      <div ref={box} className="map" />
      <div className="map-tools">
        <label className="switch">
          <input type="checkbox" checked={osm} onChange={(e) => setOsm(e.target.checked)} />
          <span>Подложка OpenStreetMap</span>
        </label>
      </div>
      <div className="legend">
        <div className="legend-row">
          <span className="legend-title">Риск</span>
          {Object.entries(RISK_COLORS).map(([k, c]) => (
            <span key={k} className="lg-item"><i style={{ background: c }} />{k}</span>
          ))}
        </div>
        <div className="legend-row">
          <span className="legend-title">ЛПС</span>
          <span className="lg-item"><span className="lg-st busy">2</span>направлена</span>
          <span className="lg-item"><span className="lg-st">1</span>в резерве</span>
          <span className="lg-item"><span className="lg-line" />ведущее подразделение</span>
          <span className="lg-item"><span className="lg-line dash" />усиление</span>
        </div>
      </div>
      {addMode && <div className="map-hint">Укажите на карте место нового пожара</div>}
    </div>
  )
}
