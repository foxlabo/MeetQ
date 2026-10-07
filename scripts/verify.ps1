# MeetQ verification driver.
# Order: preflight -> ruff -> tsc -> frontend build -> T1..T6
# Prints one "[PASS|FAIL|BLOCKED] Tn name" line per stage and a final
# "VERIFY RESULT: ALL PASS" / "VERIFY RESULT: NOT PASSED". Exit code matches.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# ----------------------------------------------------------------- preflight
& $py (Join-Path $root "scripts\preflight.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "VERIFY RESULT: NOT PASSED"
    exit 1
}

$script:results = @()
function Record($status, $name, $detail) {
    $script:results += [pscustomobject]@{ Status = $status; Name = $name; Detail = $detail }
    if ($detail) { Write-Host "[$status] $name - $detail" } else { Write-Host "[$status] $name" }
}

function Invoke-Stage($name, [scriptblock]$body) {
    & $body
    if ($LASTEXITCODE -eq 0) { Record "PASS" $name "" } else { Record "FAIL" $name "exit $LASTEXITCODE" }
}

# --------------------------------------------------------------------- lint
Invoke-Stage "L1 ruff" { & $py -m ruff check . }

# ------------------------------------------------------- frontend typecheck
$frontend = Join-Path $root "frontend"
if (Test-Path (Join-Path $frontend "package.json")) {
    Push-Location $frontend
    if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
        & npm install --no-audit --no-fund | Out-Null
    }
    Invoke-Stage "L2 tsc" { & npx --no-install tsc --noEmit -p tsconfig.json }
    Invoke-Stage "L3 frontend build" { & npm run build }
    Pop-Location
} else {
    Record "FAIL" "L2 tsc" "frontend/ not present"
    Record "FAIL" "L3 frontend build" "frontend/ not present"
}

# --------------------------------------------- browsers for the UI E2E tests
# Idempotent: a no-op once the browser is present, required on a fresh machine.
& $py -m playwright install chromium 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { Record "PASS" "L4 playwright browser" "" }
else { Record "FAIL" "L4 playwright browser" "playwright install chromium exited $LASTEXITCODE" }

# -------------------------------------------------------------------- tests
function Invoke-Pytest($id, $name, $marker) {
    $log = Join-Path $env:TEMP "meetq_$id.log"
    & $py -m pytest -m $marker -q --no-header -rA 2>&1 | Tee-Object -FilePath $log | Out-Null
    $code = $LASTEXITCODE
    $text = Get-Content $log -Raw
    if ($code -eq 0) {
        Record "PASS" "$id $name" ""
    } elseif ($text -match "MEETQ_BLOCKED:\s*(.+)") {
        Record "BLOCKED" "$id $name" $Matches[1].Trim()
    } else {
        $tail = ($text -split "`n" | Select-Object -Last 6) -join " | "
        Record "FAIL" "$id $name" $tail
    }
    Write-Host ("    log: {0}" -f $log)
}

Invoke-Pytest "T1" "unit (chunker/qgen/schema)" "t1"
Invoke-Pytest "T2" "STT integration (real API)" "t2"
Invoke-Pytest "T3" "loopback E2E (real API)" "t3"
Invoke-Pytest "T4" "API E2E (real LLM)" "t4"
Invoke-Pytest "T5" "UI E2E (Playwright)" "t5"
Invoke-Pytest "T6" "realtime E2E (Phase 2)" "t6"

# ------------------------------------------------------------------ verdict
$failed = @($script:results | Where-Object { $_.Status -ne "PASS" -and $_.Status -ne "BLOCKED" })
if ($failed.Count -eq 0) {
    Write-Host "VERIFY RESULT: ALL PASS"
    exit 0
} else {
    Write-Host "VERIFY RESULT: NOT PASSED"
    exit 1
}
