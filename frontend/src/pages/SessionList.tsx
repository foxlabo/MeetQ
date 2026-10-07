import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, formatTime, type MeetSession } from '../api'

export default function SessionList() {
  const [sessions, setSessions] = useState<MeetSession[] | null>(null)
  const [error, setError] = useState('')

  const load = () => {
    api
      .listSessions()
      .then(setSessions)
      .catch((e: Error) => setError(e.message))
  }

  useEffect(() => {
    load()
    const timer = setInterval(load, 4000)
    return () => clearInterval(timer)
  }, [])

  return (
    <div data-testid="session-list">
      <h1>セッション一覧</h1>
      {error && <p className="error">{error}</p>}
      <div className="panel">
        <div className="row" style={{ marginBottom: 12 }}>
          <Link to="/new">
            <button className="primary" data-testid="new-session-button">
              新規セッション
            </button>
          </Link>
        </div>
        {sessions === null ? (
          <p className="empty">読み込み中…</p>
        ) : sessions.length === 0 ? (
          <p className="empty" data-testid="sessions-empty">
            セッションはまだありません。
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>タイトル</th>
                <th>種別</th>
                <th>状態</th>
                <th>長さ</th>
                <th>質問</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id} data-testid={`session-row-${s.id}`}>
                  <td>
                    <Link to={`/sessions/${s.id}`} data-testid={`session-link-${s.id}`}>
                      {s.title || `セッション ${s.id}`}
                    </Link>
                  </td>
                  <td>{s.source_type === 'record' ? '録音' : 'アップロード'}</td>
                  <td data-testid={`session-status-${s.id}`}>{s.status}</td>
                  <td>{formatTime(s.duration_sec)}</td>
                  <td>{s.question_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
