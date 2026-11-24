import type { Plan, Reference } from './types'

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...init })
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`)
  return r.json()
}

export const api = {
  reference: () => j<Reference>('/api/reference'),
  regions: () => j<{ current: string; regions: { slug: string; name: string }[] }>('/api/regions'),
  setRegion: (slug: string) => j<unknown>(`/api/region/${slug}`, { method: 'POST' }),
  plan: () => j<Plan>('/api/plan'),
  recalc: () => j<Plan>('/api/plan/recalculate', { method: 'POST' }),
  approve: () => j<Plan>('/api/plan/approve', { method: 'POST' }),
  override: (fire_id: string, station_ids: string[]) =>
    j<Plan>('/api/plan/override', { method: 'POST', body: JSON.stringify({ fire_id, station_ids }) }),
  demo: () => j<unknown>('/api/scenario/demo', { method: 'POST' }),
  history: () => j<{ t: string; n_active: number }[]>('/api/scenario/history'),
  loadHistory: (t: string) => j<unknown>(`/api/scenario/history/${encodeURIComponent(t)}`, { method: 'POST' }),
  addFire: (f: Record<string, unknown>) => j<unknown>('/api/fires', { method: 'POST', body: JSON.stringify(f) }),
  deleteFire: (id: string) => j<unknown>(`/api/fires/${id}`, { method: 'DELETE' }),
  updateFire: (id: string, f: Record<string, unknown>) => j<unknown>(`/api/fires/${id}`, { method: 'PUT', body: JSON.stringify(f) }),
  upload: async (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    const r = await fetch('/api/scenario/upload', { method: 'POST', body: fd })
    if (!r.ok) throw new Error((await r.json().catch(() => ({ detail: r.statusText }))).detail)
    return r.json()
  },
  station: (id: string, body: Record<string, unknown>) =>
    j<unknown>(`/api/stations/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  chat: (message: string, history: { role: string; content: string }[]) =>
    j<{ answer: string; source: string }>('/api/assistant/chat', { method: 'POST', body: JSON.stringify({ message, history }) }),
}

export const RISK_COLORS: Record<string, string> = {
  'низкий': '#5E9473',
  'средний': '#A9A63C',
  'значительный': '#DE9A2E',
  'высокий': '#E2562B',
  'чрезвычайно высокий': '#A8231B',
}

export const RISK_ORDER = ['низкий', 'средний', 'значительный', 'высокий', 'чрезвычайно высокий']

export function hm(h: number | null | undefined): string {
  if (h === null || h === undefined) return '—'
  const m = Math.round(h * 60)
  return m >= 60 ? `${Math.floor(m / 60)} ч ${String(m % 60).padStart(2, '0')} мин` : `${m} мин`
}

export function rub(x: number): string {
  if (x >= 1e6) return `${(x / 1e6).toLocaleString('ru-RU', { maximumFractionDigits: 2 })} млн ₽`
  return `${Math.round(x / 1000).toLocaleString('ru-RU')} тыс. ₽`
}

export function shortName(name: string): string {
  return name.replace(/^Пожар № (\d+) — /, '№ $1 ')
}

export function n(x: number | null | undefined, d = 1): string {
  if (x === null || x === undefined) return '—'
  return x.toLocaleString('ru-RU', { maximumFractionDigits: d })
}

export function actShort(a: string): string {
  return a.replace('Распоряжение № ', 'Распоряжение ').replace('Приказ № ', 'Приказ ')
}
