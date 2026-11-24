export type Ref = { act: string; point: string; title: string }

export type Criterion = { text: string; triggered: boolean; axis: string; ref: Ref }

export type Risk = {
  category: string
  weight: number
  severity: string
  probability: string
  severity_ru: string
  probability_ru: string
  p_growth: number
  nearest_np: { name: string; km: number; lat: number; lon: number }
  nearest_oopt: { name: string; km: number }
  natural_class: number
  criteria: Criterion[]
}

export type Line = {
  resource: string
  code: string
  qty: number
  need: number
  assigned: number | null
  deficit: number
  need_people?: number
  assigned_people?: number
  norm: string
  ref: Ref
}

export type Unit = {
  id: string
  name: string
  travel_h: number
  km: number
  groups: number
  people: number
  vehicles: Record<string, number>
  geometry: [number, number][]
  method: string
  heavy_h: number | null
}

export type Demand = {
  groups: number
  people: number
  vehicles: Record<string, number>
  v_m_h: number
  v_edge_m_h: number
  head_ros_m_h: number
  perimeter_m: number
  p_m_h: number
  k_if_late: number
  hard_weather: boolean
  arrival_limit_h: number
  arrival_limit_reduced_h: number
  arrival_ref: Ref
  reinforcement_ref: Ref
}

export type Row = {
  fire_id: string
  fire_name: string
  lesnichestvo: string | null
  risk: Risk
  demand: Demand
  lines: Line[]
  units: Unit[]
  arrival_h: number | null
  elapsed_h: number
  norm_ok: boolean
  late_reinforced: boolean
  p_growth_at_arrival: number
  expected_damage_rub: number
  expected_area_ha: number | null
  deficit: Record<string, number>
}

export type Fire = {
  id: string
  name: string
  lat: number
  lon: number
  detected_at: string
  area_ha: number
  fire_type: string
  wind_ms: number
  weather_class: number
  lesnichestvo: string | null
  species: string | null
  category: string
}

export type Station = {
  id: string
  name: string
  type: string
  settlement: string
  lat: number
  lon: number
  groups: number
  people: number
  vehicles: Record<string, number>
  available: boolean
  assigned_to: string | null
}

export type Metrics = {
  weighted_damage: number
  expected_damage_rub: number
  deficit_groups: number
  deficit_vehicles: number
  norm_violations: number
  uncovered: number
  stations_used: number
  expected_large_fires: number
  expected_area_ha: number
  n_fires: number
}

export type Plan = {
  created_at: string
  scenario_now: string
  solver: { status: string; wall_time_s: number; n_vars: number }
  assignment: Record<string, string[]>
  metrics: Metrics
  baseline: { name: string; metrics: Metrics; per_fire: Record<string, { stations: string[]; arrival_h: number | null }> }
  statement: Row[]
  stations: Station[]
  fires: Fire[]
  fixed: Record<string, string[]>
  deficit_summary: { total: Record<string, number>; fires: string[]; has_deficit: boolean; message: string }
  status: string
  region: string
  region_name: string
  journal: { t: string; event: string; detail: string }[]
  scenario: { title: string; description: string; now: string; fires: { id: string; area_ha: number; wind_ms: number; fire_type: string; weather_class: number; species: string | null }[] }
}

export type Reference = {
  settlements: { id: string; name: string; full_name: string; lat: number; lon: number; n_fires: number }[]
  oopt: { name: string; lat: number; lon: number }[]
  acts: { code: string; title: string; short: string }[]
  region: string
  model: {
    roc_auc: number; pr_auc: number; brier: number; roc_auc_region_a: number; n_train: number; n_test: number
    train_years: string; test_years: string; feature_importance: Record<string, number>
    mean_p_by_delay_h: Record<string, number>
    area_model?: { r2_log: number; mae_log: number; spearman: number; median_abs_error_ha: number }
  }
  region_norms: Record<string, number | string | object>
  llm: string
}
