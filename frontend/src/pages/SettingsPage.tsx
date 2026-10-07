import { useEffect, useState } from 'react'
import { api, type AppSettings } from '../api'

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.settings().then(setSettings).catch((e: Error) => setError(e.message))
  }, [])

  if (!settings) return <p className="empty">読み込み中…</p>

  const update = (patch: Partial<AppSettings>) => {
    setSettings({ ...settings, ...patch })
    setSaved(false)
  }

  const save = () => {
    api
      .saveSettings({
        stt_model: settings.stt_model,
        stt_live_model: settings.stt_live_model,
        llm_model: settings.llm_model,
        language: settings.language,
        glossary: settings.glossary,
        corrections: settings.corrections,
      })
      .then((next) => {
        setSettings(next)
        setSaved(true)
      })
      .catch((e: Error) => setError(e.message))
  }

  return (
    <div data-testid="settings-page">
      <h1>設定</h1>
      {error && <p className="error">{error}</p>}
      <div className="panel">
        <label>
          <span>ファイル転写モデル (MEETQ_STT_MODEL)</span>
          <input
            data-testid="stt-model"
            value={settings.stt_model}
            onChange={(e) => update({ stt_model: e.target.value })}
          />
        </label>
        <label>
          <span>ライブ転写モデル (MEETQ_STT_LIVE_MODEL)</span>
          <input
            data-testid="stt-live-model"
            value={settings.stt_live_model}
            onChange={(e) => update({ stt_live_model: e.target.value })}
          />
        </label>
        <label>
          <span>質問生成モデル (MEETQ_LLM_MODEL)</span>
          <input
            data-testid="llm-model"
            value={settings.llm_model}
            onChange={(e) => update({ llm_model: e.target.value })}
          />
        </label>
        <label>
          <span>既定言語</span>
          <input
            data-testid="language"
            value={settings.language}
            onChange={(e) => update({ language: e.target.value })}
          />
        </label>
        <label>
          <span>用語集（1行に1語。転写の keywords として渡されます）</span>
          <textarea
            data-testid="glossary"
            rows={6}
            value={settings.glossary}
            onChange={(e) => update({ glossary: e.target.value })}
          />
        </label>
        <label>
          <span>
            用語の修正ルール（1行に1つ「誤 =&gt; 正」。新しい文字起こしに自動で適用し、正しい側の言葉は
            文字起こしのヒントにも使います）
          </span>
          <textarea
            data-testid="corrections"
            rows={5}
            placeholder={'使用 => 仕様\n以降 => 移行'}
            value={settings.corrections}
            onChange={(e) => update({ corrections: e.target.value })}
          />
        </label>
        <div className="row">
          <button className="primary" onClick={save} data-testid="save-settings">
            保存
          </button>
          {saved && <span className="muted">保存しました</span>}
        </div>
      </div>
      <div className="panel">
        <h2>API キー</h2>
        <p data-testid="api-key-status">
          OPENAI_API_KEY: {settings.api_key_configured ? '設定済み' : '未設定'}
        </p>
        <p className="muted">キーは .env でのみ設定します。画面には表示されません。</p>
      </div>
    </div>
  )
}
