# MeetQ

会議（Zoom / Teams / Google Meet など種類を問わない）や動画の音声を、**会議にボットを参加させずに**
取り込み、文字起こし → その内容に対する質問を AI が生成するローカル Web アプリです。
OS の音声（WASAPI ループバック）を直接取るため、会議アプリごとの連携は不要です。

文字起こしと質問生成はどちらも OpenAI API を使います。**外部への送信先は OpenAI API だけ**です。

## できること（Phase 1）

- **F1 取り込み**: mp4 / mp3 / wav / m4a のアップロード
- **F2 録音**: デバイス一覧、録音の開始/停止。スピーカー出力（others）とマイク（self）を別チャンネルで保存
- **F3 文字起こし**: ローカルの VAD で無音区切りにチャンク化し、チャンクごとに転写。
  タイムスタンプは転写モデルに依存せずチャンク側の開始秒を使う。
  直前チャンクの末尾テキストを `prompt` として引き継ぎ、用語集を `keywords` として渡す
- **F4 質問生成**: `followup`（会議フォローアップ）/ `comprehension`（理解度チェック）。
  Structured Outputs で取得し、`evidence_quote` が文字起こしの部分文字列であることを検証。
  満たさない質問は破棄して再生成（最大2回）
- **F5 UI**: セッション一覧 / 新規作成 / 進捗表示 / タイムスタンプ付き文字起こし /
  質問パネル（type フィルタ）/ 質問クリックで根拠へスクロール＋ハイライト /
  再生成 / Markdown・JSON エクスポート
- **F6 設定画面**: 各モデル名・言語・用語集。API キーは表示されず、設定済みかどうかだけ分かります
- **用語の修正**: 聞き間違えられた専門用語を「使用 => 仕様」のようなルールで一括修正できます。
  適用前に件数と前後の文脈をプレビューでき、元の文字起こしは残るのでルールを消せば元に戻ります。
  設定に保存したルールは今後の文字起こしに自動で適用され、正しい語は文字起こしのヒントにも使われます
- **文字起こしのダウンロード**: タイムスタンプつきのテキスト（.txt）で、1 文 1 行に組み直して
  保存できます（Markdown の文字起こし部分も同じ）。話者が変わるところでは必ず改行します
- **F7 利用量**: セッションごとの転写秒数と LLM トークン数を記録して表示

## できること（Phase 2 / リアルタイム）

- **F8 ライブ転写**: 録音中の音声を Realtime API（WebSocket）へ流し、確定テキストを
  そのまま画面に追記します。mic と loopback を分ける場合はセッション 2 本。
  切断時は自動で再接続します（再接続回数は画面に出ます）
- **F9 ローリング質問生成**: 直近 N 分の窓＋累積サマリを入力に、既定 90 秒ごとに質問を生成。
  画面の「今すぐ生成」ボタンでも発火します
- **F10 重複・既回答フィルタ**: 既出の質問と似た質問は出さず、会議の中で既に答えが
  出ている質問も除外します
- **録音停止後の全文化**: ライブの文字起こしは速さ優先で数秒ごとに区切るため、切れ目の
  単語が抜けたり化けたりします。録音を止めると、保存した音声を無音の位置で区切って
  もう一度文字起こしし直し（全文の最終版）、質問もその全文から作り直します

## 必要なもの

- Windows 11
- Python 3.11 以上 + [Poetry](https://python-poetry.org/)
- Node.js 20 以上
- クレジットのある OpenAI API キー

## セットアップ

```bash
git clone <this repo>
cd MeetQ
poetry config virtualenvs.in-project true --local   # .venv をプロジェクト直下に作る（scripts\verify.ps1 が前提にしています）
poetry install
```

フロントエンドをビルドします（成果物 `meetq/static/` を FastAPI が配信します）。

```bash
cd frontend
npm install
npm run build
```

UI の E2E テスト（T5 / T6）を走らせる場合は Playwright のブラウザも入れます。
`scripts\verify.ps1` は実行時に自動で入れるので、手動で走らせるときだけ必要です。

```bash
poetry run playwright install chromium
```

## .env

プロジェクト直下に `.env` を作成し、下の内容をコピーして `OPENAI_API_KEY` を自分のキーに
置き換えます（`.env` は `.gitignore` 済み。**コミットしないでください**）。必須なのは
`OPENAI_API_KEY` だけで、ほかは書かなければ既定値が使われます。

```ini
# 必須
OPENAI_API_KEY=sk-...

# 任意（既定値。モデル名はここで自由に差し替えられます）
MEETQ_STT_MODEL=gpt-transcribe
MEETQ_STT_LIVE_MODEL=gpt-live-transcribe
MEETQ_LLM_MODEL=gpt-6-astra
MEETQ_LANGUAGE=ja
MEETQ_HOST=127.0.0.1
MEETQ_PORT=8765
MEETQ_DATA_DIR=./data
MEETQ_STT_CONCURRENCY=3
MEETQ_CHUNK_TARGET_SEC=20
MEETQ_CHUNK_MIN_SEC=10
MEETQ_CHUNK_MAX_SEC=60
OPENAI_TIMEOUT=600

# 任意（Phase 2 / ライブ転写とローリング生成）
MEETQ_LIVE_SAMPLE_RATE=24000   # Realtime API の下限は 24000
MEETQ_LIVE_TURN_MIN_SEC=2      # 無音で確定させる最短ターン長
MEETQ_LIVE_TURN_MAX_SEC=5      # 強制的に確定させる最大ターン長
MEETQ_LIVE_DELAY=low           # minimal|low|medium|high|xhigh
MEETQ_ROLLING_INTERVAL_SEC=90  # ローリング生成の間隔
MEETQ_ROLLING_WINDOW_SEC=300   # 質問生成に使う直近の窓
```

既定値の定義は `meetq/config.py` にあります。
モデル名は UI の設定画面からも変更でき、保存すると `.env` の値より優先されます。
API キーだけは `.env` でのみ設定でき、画面にも API レスポンスにも出ません。

## 起動

```bash
poetry run meetq
```

http://127.0.0.1:8765 を開きます。`--host` / `--port` で変更できます。

録音を使う場合は、取り込みたい音が既定の再生デバイスから鳴っている必要があります
（ループバックは「いまスピーカーに出ている音」を取ります）。

## 検証

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1
```

プリフライト（API キーと最小の推論リクエスト）→ ruff → tsc → フロントビルド → T1〜T6 を実行し、
最後に `VERIFY RESULT: ALL PASS` または `VERIFY RESULT: NOT PASSED` を出力します。
（T6 まで含めてフル実行すると数分かかります）

テストが API に送る音声は `scripts\make_fixture.ps1` が Windows SAPI で生成する
60〜90 秒のフィクスチャだけです（課金を抑えるため）。T3 と T6 は実行中に
PC から実際に音が鳴ります（既定の再生デバイスから流してループバックで録ります）。

個別に走らせる場合:

```bash
poetry run pytest -m t1   # ユニット（OpenAI はモック）
poetry run pytest -m t2   # 転写の実 API 統合
poetry run pytest -m t3   # ループバック録音 E2E（音が鳴ります）
poetry run pytest -m t4   # API E2E（実 LLM）
poetry run pytest -m t5   # UI E2E（Playwright）
poetry run pytest -m t6   # ライブ転写＋ローリング生成 E2E（音が鳴ります）
```

API キーも音声デバイスも使わずに走るのは T1 と `tests/test_t5_corrections_ui.py`
（用語の修正 UI）だけです。GitHub Actions（`.github/workflows/ci.yml`）ではこの 2 つと
ruff・tsc・フロントビルドを実行します。

## 別の PC から使う

Cloudflare Tunnel（`cloudflared`）で、別の PC のブラウザから開けます。

- **録音はクラウドに移せません**。録音されるのは MeetQ が動いている PC の音です
- **MeetQ 自体に認証はありません**。公開する場合は必ず Cloudflare Access で
  アクセス制限をかけてください（かけないと、URL を知っている人が誰でも文字起こしを読めて、
  あなたの OpenAI クレジットを使えます）
- 公開ホスト名で出す場合、MeetQ は `127.0.0.1:8765` のままで、トンネルの転送先を
  `http://127.0.0.1:8765` にします。Cloudflare 経由のアップロードは無料プランで 100 MB まで、
  1 リクエストは 125 秒でタイムアウトします（重い処理はバックグラウンドで動くので通常は問題ありません）
- `--host 0.0.0.0` で LAN に出す構成にする場合は、先に Windows ファイアウォールで
  8765 番の受信をブロックし、同じ LAN の端末から直接見えないようにしてください

## 注意

音声は OpenAI に送信されます。客先の会議で使う場合は、契約上そのデータを外部 API に
送ってよいかを必ず事前に確認してください。

## ライセンス

[MIT](LICENSE)
