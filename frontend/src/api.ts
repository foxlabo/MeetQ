export type SessionUsage = {
  transcribed_sec: number
  stt_requests: number
  llm_input_tokens: number
  llm_output_tokens: number
  llm_requests: number
}

export type MeetSession = {
  id: number
  title: string
  source_type: string
  status: string
  stage: string
  progress: number
  error: string
  duration_sec: number
  language: string
  glossary: string
  mode: string
  live: boolean
  summary: string
  corrections: string
  created_at: string
  segment_count: number
  question_count: number
  usage: SessionUsage
}

export type Segment = {
  id: number
  idx: number
  start_sec: number
  end_sec: number
  text: string
  channel: string
}

export type QuestionType = 'clarify' | 'unresolved' | 'contradiction' | 'risk' | 'comprehension'

export type Question = {
  id: number
  text: string
  type: QuestionType
  priority: number
  evidence_quote: string
  evidence_start_sec: number
  evidence_segment_ids: number[]
  mode: string
  round: number
}

export type LiveStatus = {
  running: boolean
  rounds: number
  reconnects: number
  finals: number
  interval_sec?: number
}

export type AppSettings = {
  stt_model: string
  stt_live_model: string
  llm_model: string
  language: string
  glossary: string
  corrections: string
  api_key_configured: boolean
}

export type CorrectionExample = {
  segment_id: number
  start_sec: number
  before: string
  match: string
  after: string
}

export type CorrectionPreview = {
  errors: string[]
  total: number
  rules: { wrong: string; right: string; count: number; examples: CorrectionExample[] }[]
}

export type CorrectionResult = {
  applied: boolean
  segments_changed: number
  replacements: number
  questions_changed: number
  reanchored: number
}

export type DeviceInfo = {
  index: number
  name: string
  kind: string
  channels: number
  default_sample_rate: number
  is_default: boolean
}

export type Devices = {
  available: boolean
  reason?: string
  loopback: DeviceInfo[]
  input: DeviceInfo[]
  default_loopback_index: number | null
  default_input_index: number | null
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      /* keep statusText */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export const api = {
  listSessions: () => fetch('/api/sessions').then(json<MeetSession[]>),
  getSession: (id: number) => fetch(`/api/sessions/${id}`).then(json<MeetSession>),
  deleteSession: (id: number) =>
    fetch(`/api/sessions/${id}`, { method: 'DELETE' }).then(json<{ deleted: number }>),
  segments: (id: number) => fetch(`/api/sessions/${id}/segments`).then(json<Segment[]>),
  questions: (id: number) => fetch(`/api/sessions/${id}/questions`).then(json<Question[]>),
  usage: (id: number) => fetch(`/api/sessions/${id}/usage`).then(json<SessionUsage>),
  settings: () => fetch('/api/settings').then(json<AppSettings>),
  saveSettings: (payload: Partial<AppSettings>) =>
    fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json<AppSettings>),
  devices: () => fetch('/api/devices').then(json<Devices>),
  upload: (form: FormData) =>
    fetch('/api/sessions/upload', { method: 'POST', body: form }).then(json<MeetSession>),
  regenerate: (id: number, mode: string) =>
    fetch(`/api/sessions/${id}/regenerate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    }).then(json<{ started: boolean; mode: string }>),
  startRecording: (payload: Record<string, unknown>) =>
    fetch('/api/recording/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json<MeetSession>),
  stopRecording: (id: number) =>
    fetch(`/api/recording/${id}/stop`, { method: 'POST' }).then(json<MeetSession>),
  activeRecordings: () => fetch('/api/recording/active').then(json<{ session_ids: number[] }>),
  generateNow: (id: number) =>
    fetch(`/api/sessions/${id}/generate-now`, { method: 'POST' }).then(
      json<{ started: boolean; rounds: number }>,
    ),
  liveStatus: (id: number) => fetch(`/api/sessions/${id}/live`).then(json<LiveStatus>),
  previewCorrections: (id: number, rules: string) =>
    fetch(`/api/sessions/${id}/corrections/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rules }),
    }).then(json<CorrectionPreview>),
  applyCorrections: (id: number, rules: string, saveGlobal: boolean) =>
    fetch(`/api/sessions/${id}/corrections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rules, save_global: saveGlobal }),
    }).then(json<CorrectionResult>),
}

export function formatTime(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}
