# Generates tests/fixtures/meeting_ja.wav from a known script using Windows SAPI.
# Falls back to an English voice + English script when no Japanese voice is installed
# (and to any installed voice reading the English script when there is no English
# voice either, e.g. on a CI image), and prints which one was used. Also writes a manifest with the keywords used by T2/T3.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$fixtureDir = Join-Path $root "tests\fixtures"
$wavPath = Join-Path $fixtureDir "meeting_ja.wav"
$manifestPath = Join-Path $fixtureDir "meeting_ja.json"
New-Item -ItemType Directory -Force -Path $fixtureDir | Out-Null

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

$jaVoice = $synth.GetInstalledVoices() |
    Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like "ja*" } |
    Select-Object -First 1

$jaScript = @(
    "それでは、定例ミーティングを始めます。まず新しいバージョンのリリース日について確認させてください。",
    "現在の計画では、リリース日は来週の金曜を予定しています。ただしテストがまだ終わっていません。",
    "来週の金曜までに結合テストを完了できるかどうか、正直なところ厳しいと感じています。",
    "次に予算の話です。今期の予算はすでに八十パーセントを消化しています。追加の予算申請が必要かもしれません。",
    "予算の承認には二週間かかりますので、今日中に判断したいところです。",
    "それから、移行作業の担当者がまだ決まっていません。担当者未定のままでは作業を開始できません。",
    "先ほどリリース日は来週の金曜と言いましたが、担当者未定であれば再来週にずれる可能性もあります。",
    "リスクとしては、予算超過とスケジュール遅延の二点が大きいと考えています。",
    "最後に、次回までの宿題を整理します。担当者の決定と、追加予算の見積もりをお願いします。",
    "以上で本日のミーティングを終わります。ありがとうございました。"
) -join ""

$enScript = @(
    "Let's start the regular meeting. First I want to confirm the release date for the new version.",
    "The current plan is a release date of next Friday, but integration testing is not finished yet.",
    "Honestly, finishing integration testing by next Friday feels very tight.",
    "Next, the budget. We have already spent eighty percent of this quarter's budget.",
    "Approving an additional budget takes two weeks, so we should decide today.",
    "Also, the owner for the migration work is still undecided. Work cannot start while the owner is undecided.",
    "I said the release date is next Friday, but if the owner stays undecided it may slip by a week.",
    "The two big risks are going over budget and slipping the schedule.",
    "Finally, the action items are deciding the owner and estimating the additional budget.",
    "That is all for today's meeting. Thank you very much."
) -join " "

if ($null -ne $jaVoice) {
    $synth.SelectVoice($jaVoice.VoiceInfo.Name)
    $script = $jaScript
    $language = "ja"
    $keywords = @("リリース日", "予算", "来週の金曜", "担当者")
    Write-Host "FIXTURE VOICE: $($jaVoice.VoiceInfo.Name) (ja)"
} else {
    $enVoice = $synth.GetInstalledVoices() |
        Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like "en*" } |
        Select-Object -First 1
    if ($null -eq $enVoice) {
        # the tests only need speech-like audio with pauses, not a particular language
        $enVoice = $synth.GetInstalledVoices() |
            Where-Object { $_.Enabled } |
            Select-Object -First 1
    }
    if ($null -eq $enVoice) { throw "no usable SAPI voice installed" }
    $synth.SelectVoice($enVoice.VoiceInfo.Name)
    $script = $enScript
    $language = "en"
    $keywords = @("release date", "budget", "next Friday", "owner")
    Write-Host "FIXTURE VOICE: $($enVoice.VoiceInfo.Name) ($($enVoice.VoiceInfo.Culture.Name), en script fallback)"
}

$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

$synth.Rate = 0
$synth.SetOutputToWaveFile($wavPath, $format)
$synth.Speak($script)
$synth.SetOutputToNull()
$synth.Dispose()

$bytes = (Get-Item $wavPath).Length
$seconds = [Math]::Round(($bytes - 44) / 32000.0, 2)

$manifest = [ordered]@{
    language = $language
    keywords = $keywords
    script   = $script
    duration_sec = $seconds
    wav = "meeting_ja.wav"
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding utf8

Write-Host "FIXTURE WAV: $wavPath ($seconds sec, $bytes bytes)"
Write-Host "FIXTURE LANGUAGE: $language"
Write-Host "FIXTURE KEYWORDS: $($keywords -join ', ')"
if ($seconds -lt 55 -or $seconds -gt 95) {
    Write-Warning "fixture duration $seconds s is outside the 60-90 s target"
}
