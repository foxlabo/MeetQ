import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type Devices } from '../api'

export default function NewSession() {
  const navigate = useNavigate()
  const [title, setTitle] = useState('')
  const [mode, setMode] = useState('followup')
  const [glossary, setGlossary] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [devices, setDevices] = useState<Devices | null>(null)
  const [loopbackIndex, setLoopbackIndex] = useState<number | null>(null)
  const [micIndex, setMicIndex] = useState<number | null>(null)
  const [live, setLive] = useState(true)

  useEffect(() => {
    api.settings().then((s) => setGlossary(s.glossary)).catch(() => undefined)
    api
      .devices()
      .then((d) => {
        setDevices(d)
        setLoopbackIndex(d.default_loopback_index)
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  const submitUpload = async () => {
    if (!file) return
    setBusy(true)
    setError('')
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('title', title)
      form.append('mode', mode)
      form.append('glossary', glossary)
      const session = await api.upload(form)
      navigate(`/sessions/${session.id}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const startRecording = async () => {
    setBusy(true)
    setError('')
    try {
      const session = await api.startRecording({
        title,
        mode,
        glossary,
        live,
        loopback_index: loopbackIndex,
        mic_index: micIndex,
      })
      navigate(`/sessions/${session.id}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div data-testid="new-session">
      <h1>新規セッション</h1>
      {error && <p className="error" data-testid="new-session-error">{error}</p>}

      <div className="panel">
        <label>
          <span>タイトル</span>
          <input
            data-testid="title-input"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="定例ミーティング"
          />
        </label>
        <label>
          <span>質問モード</span>
          <select data-testid="mode-select" value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="followup">followup（会議フォローアップ）</option>
            <option value="comprehension">comprehension（理解度チェック）</option>
          </select>
        </label>
        <label>
          <span>用語集（1行に1語）</span>
          <textarea
            data-testid="session-glossary"
            rows={3}
            value={glossary}
            onChange={(e) => setGlossary(e.target.value)}
          />
        </label>
      </div>

      <div className="split">
        <div className="panel">
          <h2>ファイルをアップロード</h2>
          <p className="muted">mp4 / mp3 / wav / m4a</p>
          <input
            type="file"
            data-testid="file-input"
            accept=".mp4,.mp3,.wav,.m4a"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <div className="row" style={{ marginTop: 12 }}>
            <button
              className="primary"
              data-testid="upload-button"
              disabled={!file || busy}
              onClick={submitUpload}
            >
              アップロードして処理開始
            </button>
          </div>
        </div>

        <div className="panel">
          <h2>録音する</h2>
          {devices && !devices.available && (
            <p className="muted" data-testid="devices-unavailable">
              音声デバイスを利用できません: {devices.reason}
            </p>
          )}
          {devices && devices.available && (
            <>
              <label>
                <span>スピーカー出力（loopback / others）</span>
                <select
                  data-testid="loopback-select"
                  value={loopbackIndex ?? ''}
                  onChange={(e) => setLoopbackIndex(Number(e.target.value))}
                >
                  {devices.loopback.map((d) => (
                    <option key={d.index} value={d.index}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>マイク（self / 任意）</span>
                <select
                  data-testid="mic-select"
                  value={micIndex ?? ''}
                  onChange={(e) =>
                    setMicIndex(e.target.value === '' ? null : Number(e.target.value))
                  }
                >
                  <option value="">使用しない</option>
                  {devices.input.map((d) => (
                    <option key={d.index} value={d.index}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="row" style={{ gap: 8 }}>
                <input
                  type="checkbox"
                  data-testid="live-toggle"
                  style={{ width: 'auto' }}
                  checked={live}
                  onChange={(e) => setLive(e.target.checked)}
                />
                <span style={{ margin: 0 }}>
                  ライブ転写する（録音中に確定テキストと質問が出ます）
                </span>
              </label>
              <div className="row">
                <button
                  className="primary"
                  data-testid="start-recording"
                  disabled={busy}
                  onClick={startRecording}
                >
                  録音開始
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
