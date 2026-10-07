import { useState } from 'react'
import { api, formatTime, type CorrectionPreview } from '../api'

type Props = {
  sessionId: number
  initialRules: string
  disabled: boolean
  onApplied: () => void
}

/** Term corrections for one session: preview where each rule fires, then apply. */
export default function CorrectionsPanel({ sessionId, initialRules, disabled, onApplied }: Props) {
  const [open, setOpen] = useState(false)
  const [rules, setRules] = useState(initialRules)
  const [preview, setPreview] = useState<CorrectionPreview | null>(null)
  const [saveGlobal, setSaveGlobal] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const openPanel = async () => {
    setOpen(true)
    if (!initialRules) {
      // start from the saved defaults so a known fix is one click away
      try {
        const settings = await api.settings()
        if (settings.corrections) setRules(settings.corrections)
      } catch {
        /* the textarea simply starts empty */
      }
    }
  }

  const runPreview = async () => {
    setBusy(true)
    setError('')
    setMessage('')
    try {
      setPreview(await api.previewCorrections(sessionId, rules))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const apply = async () => {
    setBusy(true)
    setError('')
    try {
      const result = await api.applyCorrections(sessionId, rules, saveGlobal)
      setMessage(
        result.replacements > 0
          ? `${result.replacements} か所を修正しました（${result.segments_changed} 行 / 質問 ${result.questions_changed} 件）`
          : `ルールに一致する箇所はありませんでした（${result.segments_changed} 行を元に戻しました）`,
      )
      setPreview(null)
      onApplied()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (!open) {
    return (
      <button onClick={openPanel} disabled={disabled} data-testid="corrections-open">
        用語を修正
      </button>
    )
  }

  const canApply = preview !== null && preview.errors.length === 0 && !busy && !disabled

  return (
    <div className="panel" data-testid="corrections-panel">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2 style={{ margin: 0 }}>用語を修正</h2>
        <button onClick={() => setOpen(false)}>閉じる</button>
      </div>
      <p className="muted">
        聞き間違えられた言葉を「誤 =&gt; 正」で1行ずつ書きます（例: 使用 =&gt; 仕様）。
        長い言葉が優先されるので「航空会社 =&gt; 航空会社」のように書くとその言葉は変わりません。
        元の文字起こしは残るので、ルールを消して適用し直せば元に戻ります。
      </p>
      <textarea
        data-testid="corrections-input"
        rows={4}
        value={rules}
        placeholder={'使用 => 仕様\n以降 => 移行'}
        onChange={(e) => {
          setRules(e.target.value)
          setPreview(null)
        }}
      />
      <div className="row" style={{ marginTop: 8 }}>
        <button onClick={runPreview} disabled={busy || disabled} data-testid="corrections-preview">
          プレビュー
        </button>
        <button
          className="primary"
          onClick={apply}
          disabled={!canApply}
          data-testid="corrections-apply"
        >
          このセッションに適用
        </button>
        <label className="row" style={{ gap: 6, margin: 0 }}>
          <input
            type="checkbox"
            style={{ width: 'auto' }}
            checked={saveGlobal}
            onChange={(e) => setSaveGlobal(e.target.checked)}
            data-testid="corrections-save-global"
          />
          <span style={{ margin: 0 }}>設定にも保存（今後の文字起こしに自動で適用）</span>
        </label>
      </div>
      {error && <p className="error">{error}</p>}
      {message && (
        <p className="muted" data-testid="corrections-result">
          {message}
        </p>
      )}
      {preview && (
        <div data-testid="corrections-preview-result" style={{ marginTop: 10 }}>
          {preview.errors.map((e) => (
            <p key={e} className="error">
              {e}
            </p>
          ))}
          {preview.rules.length > 0 && (
            <table>
              <thead>
                <tr>
                  <th>誤 → 正</th>
                  <th>件数</th>
                  <th>該当箇所</th>
                </tr>
              </thead>
              <tbody>
                {preview.rules.map((r) => (
                  <tr key={r.wrong}>
                    <td>
                      {r.wrong} → {r.right}
                      {r.wrong === r.right && <span className="muted">（変更しない）</span>}
                    </td>
                    <td data-testid={`corrections-count-${r.wrong}`}>{r.count}</td>
                    <td>
                      {r.examples.map((ex, i) => (
                        <div key={i} className="muted">
                          {formatTime(ex.start_sec)} …{ex.before}
                          <mark>{ex.match}</mark>
                          {ex.after}…
                        </div>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="muted">合計 {preview.total} か所</p>
        </div>
      )}
    </div>
  )
}
