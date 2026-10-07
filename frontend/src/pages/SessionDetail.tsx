import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import CorrectionsPanel from '../components/CorrectionsPanel'
import {
  api,
  formatTime,
  type LiveStatus,
  type MeetSession,
  type Question,
  type QuestionType,
  type Segment,
} from '../api'

const TYPES: QuestionType[] = ['clarify', 'unresolved', 'contradiction', 'risk', 'comprehension']

type LivePayload = {
  type?: 'segment' | 'questions' | 'delta'
  status?: string
  stage?: string
  progress?: number
  error?: string
  segment?: Segment
  channel?: string
  text?: string
  round?: number
  added?: number
}

export default function SessionDetail() {
  const { id } = useParams()
  const sessionId = Number(id)
  const [session, setSession] = useState<MeetSession | null>(null)
  const [segments, setSegments] = useState<Segment[]>([])
  const [questions, setQuestions] = useState<Question[]>([])
  const [filter, setFilter] = useState<QuestionType | 'all'>('all')
  const [highlighted, setHighlighted] = useState<number[]>([])
  const [error, setError] = useState('')
  const [regenerating, setRegenerating] = useState(false)
  const [partial, setPartial] = useState('')
  const [generating, setGenerating] = useState(false)
  const [live, setLive] = useState<LiveStatus | null>(null)
  const segmentRefs = useRef<Record<number, HTMLDivElement | null>>({})

  const refresh = useCallback(async () => {
    try {
      const [s, segs, qs] = await Promise.all([
        api.getSession(sessionId),
        api.segments(sessionId),
        api.questions(sessionId),
      ])
      setSession(s)
      setSegments(segs)
      setQuestions(qs)
      if (s.status === 'done' || s.status === 'error') setRegenerating(false)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [sessionId])

  const refreshLive = useCallback(async () => {
    try {
      setLive(await api.liveStatus(sessionId))
    } catch {
      /* live status is best effort */
    }
  }, [sessionId])

  useEffect(() => {
    void refresh()
    void refreshLive()
  }, [refresh, refreshLive])

  // WebSocket: progress (F5) plus live segments and rolling rounds (F8/F9).
  // A reverse proxy or tunnel in front of MeetQ drops idle sockets, so reconnect.
  useEffect(() => {
    if (!Number.isFinite(sessionId)) return
    let socket: WebSocket | null = null
    let retry: number | undefined
    let closed = false
    let delay = 1000

    const handle = (event: MessageEvent) => {
      const payload = JSON.parse(event.data as string) as LivePayload
      if (payload.type === 'segment' && payload.segment) {
        const incoming = payload.segment
        setSegments((prev) =>
          prev.some((s) => s.id === incoming.id)
            ? prev
            : [...prev, incoming].sort((a, b) => a.start_sec - b.start_sec),
        )
        setPartial('')
        return
      }
      if (payload.type === 'delta') {
        setPartial((prev) => prev + (payload.text ?? ''))
        return
      }
      if (payload.type === 'questions') {
        setGenerating(false)
        void refresh()
        void refreshLive()
        return
      }
      setSession((prev) => (prev ? { ...prev, ...(payload as Partial<MeetSession>) } : prev))
      if (payload.status === 'done' || payload.status === 'error') {
        // a finished run may have replaced the transcript (the full-text pass after a
        // live recording), and segment ids can be reused, so drop stale highlights
        setHighlighted([])
        void refresh()
      }
    }

    const connect = () => {
      if (closed) return
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      socket = new WebSocket(`${proto}://${window.location.host}/api/ws/sessions/${sessionId}`)
      socket.onmessage = handle
      socket.onopen = () => {
        delay = 1000
      }
      socket.onclose = () => {
        if (closed) return
        retry = window.setTimeout(connect, delay)
        delay = Math.min(delay * 2, 15000)
      }
    }
    connect()

    const timer = setInterval(() => {
      setSession((prev) => {
        if (prev && prev.status !== 'done' && prev.status !== 'error') {
          void refresh()
          if (prev.status === 'recording') void refreshLive()
        }
        return prev
      })
    }, 3000)
    return () => {
      closed = true
      if (retry) window.clearTimeout(retry)
      socket?.close()
      clearInterval(timer)
    }
  }, [sessionId, refresh, refreshLive])

  const visibleQuestions = useMemo(
    () => (filter === 'all' ? questions : questions.filter((q) => q.type === filter)),
    [questions, filter],
  )

  const onQuestionClick = (q: Question) => {
    setHighlighted(q.evidence_segment_ids)
    const first = q.evidence_segment_ids[0]
    const node = first !== undefined ? segmentRefs.current[first] : null
    if (node) node.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  const regenerate = async () => {
    if (!session) return
    setRegenerating(true)
    try {
      await api.regenerate(sessionId, session.mode)
      setTimeout(() => void refresh(), 1500)
    } catch (e) {
      setError((e as Error).message)
      setRegenerating(false)
    }
  }

  const generateNow = async () => {
    setGenerating(true)
    try {
      // the round runs in the background; the WebSocket reports when it lands
      await api.generateNow(sessionId)
      await refreshLive()
    } catch (e) {
      setError((e as Error).message)
      setGenerating(false)
    }
  }

  const stopRecording = async () => {
    try {
      await api.stopRecording(sessionId)
      await refresh()
      await refreshLive()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  if (!session) return <p className="empty">読み込み中…</p>

  const recording = session.status === 'recording'
  const busy = session.status !== 'done' && session.status !== 'error'

  return (
    <div data-testid="session-detail">
      <h1 data-testid="session-title">{session.title || `セッション ${session.id}`}</h1>
      {error && <p className="error">{error}</p>}

      <div className="panel">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div>
            <strong data-testid="session-status">{session.status}</strong>{' '}
            <span className="muted" data-testid="session-stage">
              {session.stage}
            </span>
          </div>
          <div className="row">
            {recording && (
              <>
                <button
                  className="primary"
                  onClick={generateNow}
                  disabled={generating}
                  data-testid="generate-now"
                >
                  {generating ? '生成中…' : '今すぐ生成'}
                </button>
                <button onClick={stopRecording} data-testid="stop-recording">
                  録音停止
                </button>
              </>
            )}
            <button onClick={regenerate} disabled={busy || regenerating} data-testid="regenerate">
              質問を再生成
            </button>
            <a href={`/api/sessions/${session.id}/export?format=txt`} download>
              <button data-testid="export-txt">文字起こし (.txt)</button>
            </a>
            <a href={`/api/sessions/${session.id}/export?format=md`} download>
              <button data-testid="export-md">Markdown</button>
            </a>
            <a href={`/api/sessions/${session.id}/export?format=json`} download>
              <button data-testid="export-json">JSON</button>
            </a>
          </div>
        </div>
        <div className="progress" style={{ marginTop: 10 }}>
          <div style={{ width: `${Math.round(session.progress * 100)}%` }} />
        </div>
        {session.error && (
          <p className="error" data-testid="session-error">
            {session.error}
          </p>
        )}
        {live?.running && (
          <p className="muted" data-testid="live-status" style={{ marginBottom: 0 }}>
            ライブ転写中: 確定 {live.finals} 件 / ローリング生成 {live.rounds} 回
            {live.interval_sec ? `（${live.interval_sec} 秒ごと）` : ''} / 再接続{' '}
            {live.reconnects} 回
          </p>
        )}
        <p className="muted" data-testid="session-usage" style={{ marginBottom: 0 }}>
          利用量: 転写 {session.usage.transcribed_sec} 秒 / STT {session.usage.stt_requests} 回 /
          LLM 入力 {session.usage.llm_input_tokens} tokens ・ 出力{' '}
          {session.usage.llm_output_tokens} tokens（{session.usage.llm_requests} 回）
        </p>
      </div>

      <div style={{ marginBottom: 16 }}>
        <CorrectionsPanel
          sessionId={session.id}
          initialRules={session.corrections}
          disabled={busy}
          onApplied={() => {
            setHighlighted([])
            void refresh()
          }}
        />
      </div>

      {session.summary && (
        <div className="panel">
          <h2>累積サマリ</h2>
          <p data-testid="session-summary" style={{ margin: 0 }}>
            {session.summary}
          </p>
        </div>
      )}

      <div className="split">
        <div className="panel">
          <h2>文字起こし</h2>
          <div className="transcript" data-testid="transcript">
            {segments.length === 0 ? (
              <p className="empty">まだ文字起こしはありません。</p>
            ) : (
              segments.map((s) => (
                <div
                  key={s.id}
                  ref={(node) => {
                    segmentRefs.current[s.id] = node
                  }}
                  className={`segment${highlighted.includes(s.id) ? ' highlight' : ''}`}
                  data-testid={`segment-${s.id}`}
                  data-highlighted={highlighted.includes(s.id) ? 'true' : 'false'}
                >
                  <span className="ts">{formatTime(s.start_sec)}</span>
                  <span>{s.text}</span>
                </div>
              ))
            )}
            {partial && (
              <div className="segment" data-testid="partial-text">
                <span className="ts">…</span>
                <span className="muted">{partial}</span>
              </div>
            )}
          </div>
        </div>

        <div className="panel">
          <h2>質問</h2>
          <div className="row" style={{ marginBottom: 12 }}>
            <select
              data-testid="type-filter"
              value={filter}
              onChange={(e) => setFilter(e.target.value as QuestionType | 'all')}
            >
              <option value="all">すべて</option>
              {TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <span className="muted" data-testid="questions-count">
              {visibleQuestions.length} 件
            </span>
          </div>
          {visibleQuestions.length === 0 ? (
            <p className="empty">質問はまだありません。</p>
          ) : (
            visibleQuestions.map((q) => (
              <div
                key={q.id}
                className="question"
                data-testid={`question-${q.id}`}
                data-round={q.round}
                onClick={() => onQuestionClick(q)}
              >
                <div>
                  <span className="badge">P{q.priority}</span>
                  <span className="badge">{q.type}</span>
                  {q.round > 0 && <span className="badge">round {q.round}</span>}
                </div>
                <div style={{ margin: '6px 0' }}>{q.text}</div>
                <div className="muted">
                  根拠 {formatTime(q.evidence_start_sec)}: 「{q.evidence_quote}」
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
