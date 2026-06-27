# RAG API End-to-End Demo - PowerShell
# Usage:  cd C:\Users\admin\OneDrive\Documents\RAG
#         .\scripts\demo.ps1
#
# Prerequisites:
#   1. pgvector running  -- docker start pgvector-rag
#   2. Server running    -- .venv\Scripts\uvicorn app.main:app --port 8000

param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$DocsDir = "C:\Users\admin\OneDrive\Documents\RAG\data\sample_docs"
)

Set-Location "C:\Users\admin\OneDrive\Documents\RAG"
$env:PYTHONPATH = "C:\Users\admin\OneDrive\Documents\RAG"

# ---- Generate a fresh JWT token ----
$tokenScript = @"
import sys, datetime
sys.path.insert(0, r'C:\Users\admin\OneDrive\Documents\RAG')
from app.config import get_settings
import jose.jwt as j
s = get_settings()
tok = j.encode(
    {'sub': 'rakesh@demo', 'domain': 'all',
     'actions': ['read:public','read:internal','read:confidential','read:restricted'],
     'exp': datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=8)},
    s.jwt_secret, algorithm='HS256'
)
print(tok)
"@
$tokenLines = (& ".venv\Scripts\python.exe" -c $tokenScript 2>&1)
$TOKEN = ($tokenLines | Where-Object { $_ -match "^eyJ" } | Select-Object -Last 1).ToString().Trim()
if (-not $TOKEN) {
    Write-Host "ERROR: Could not generate JWT token. Output: $tokenLines" -ForegroundColor Red
    exit 1
}
Write-Host "Token generated OK ($($TOKEN.Length) chars)" -ForegroundColor DarkGray

$AUTH_HEADER = @{ Authorization = "Bearer $TOKEN"; "Content-Type" = "application/json" }

function Show-Section([string]$title) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Invoke-Api([string]$method, [string]$path, $body = $null, [bool]$needsAuth = $true) {
    $headers = if ($needsAuth) { $AUTH_HEADER } else { @{} }
    $params = @{
        Uri            = "$BaseUrl$path"
        Method         = $method
        Headers        = $headers
        UseBasicParsing = $true
        TimeoutSec     = 90
    }
    if ($body) { $params.Body = ($body | ConvertTo-Json -Compress) }
    try {
        $r = Invoke-WebRequest @params
        return $r.Content | ConvertFrom-Json
    } catch {
        try {
            $stream = $_.Exception.Response.GetResponseStream()
            $msg = [System.IO.StreamReader]::new($stream).ReadToEnd()
            Write-Host "  ERROR $($_.Exception.Response.StatusCode): $msg" -ForegroundColor Red
        } catch {
            Write-Host "  ERROR: $_" -ForegroundColor Red
        }
        return $null
    }
}

# ---- 1. Health ----
Show-Section "1. HEALTH CHECK"
$h = Invoke-Api GET "/health" -needsAuth $false
if ($h) { Write-Host "  Status: $($h.status)  DB: $($h.db)  Env: $($h.env)" }

# ---- 2. Upload documents ----
Show-Section "2. UPLOAD SAMPLE DOCUMENTS"
$files  = @("hr_policy.md","security_policy.md","product_faq.md","incident_response.md")
$jobIds = @{}

foreach ($fname in $files) {
    $fpath    = "$DocsDir\$fname"
    $content  = [System.IO.File]::ReadAllText($fpath)
    $boundary = [System.Guid]::NewGuid().ToString("N")
    $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes(
        "--$boundary`r`nContent-Disposition: form-data; name=`"file`"; filename=`"$fname`"`r`nContent-Type: text/plain`r`n`r`n$content`r`n--$boundary--"
    )
    try {
        $r = Invoke-WebRequest -Uri "$BaseUrl/documents/upload" -Method POST `
             -Headers @{ Authorization = "Bearer $TOKEN" } `
             -Body $bodyBytes `
             -ContentType "multipart/form-data; boundary=$boundary" `
             -UseBasicParsing -TimeoutSec 30
        $job = $r.Content | ConvertFrom-Json
        $jobIds[$fname] = $job.job_id
        Write-Host "  Uploaded $fname  -->  job $($job.job_id)"
    } catch {
        Write-Host "  SKIP $fname (already indexed or error)" -ForegroundColor DarkGray
    }
}

Write-Host "  Waiting for indexing to complete..."
Start-Sleep -Seconds 8
foreach ($fname in $files) {
    if ($jobIds[$fname]) {
        $s = (Invoke-Api GET "/documents/status/$($jobIds[$fname])" -needsAuth $false).status
        Write-Host "  $fname : $s"
    }
}

# ---- 3. List documents ----
Show-Section "3. INDEXED DOCUMENTS"
$docs = Invoke-Api GET "/documents"
if ($docs) {
    $docs | ForEach-Object {
        Write-Host ("  {0,-32} dept={1,-12} level={2,-12} chunks={3}" -f `
            $_.source, $_.department, $_.access_level, $_.chunks_count)
    }
}

# ---- 4. Hybrid search ----
Show-Section "4. HYBRID SEARCH  ('What is the PTO policy?')"
$sr = Invoke-Api POST "/search" @{ question = "What is the PTO policy?" }
if ($sr) {
    Write-Host "  Mode: $($sr.search_mode)   Results: $($sr.results.Count)"
    $sr.results | ForEach-Object {
        Write-Host ("  [{0:F3}]  {1}" -f $_.source.score, $_.source.source)
    }
}

# ---- 5. /ask first call (LLM) ----
Show-Section "5. ASK  (1st call -- LLM generates answer)"
$question = @{
    question = "What is the PTO policy and how many vacation days do employees get?"
    top_k    = 4
}
$t0 = Get-Date
$a1 = Invoke-Api POST "/ask" $question
$t1 = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

if ($a1) {
    Write-Host "  Time: ${t1}s   Cached: $($a1.cached)"
    Write-Host ""
    Write-Host "  ANSWER:" -ForegroundColor Green
    Write-Host "  $($a1.answer)"
    Write-Host ""
    Write-Host "  SOURCES:" -ForegroundColor Green
    $a1.sources | ForEach-Object { Write-Host "    - $($_.source)  [$($_.department)]" }
}

# ---- 6. /ask second call (cache hit) ----
Show-Section "6. ASK AGAIN  (2nd call -- served from cache)"
$t0 = Get-Date
$a2 = Invoke-Api POST "/ask" $question
$t2 = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

if ($a2) {
    Write-Host "  Time: ${t2}s   Cached: $($a2.cached)"
    if ($t2 -lt $t1) {
        Write-Host "  Cache saved approx $([math]::Round($t1 - $t2, 1))s" -ForegroundColor Green
    }
}

# ---- 7. Cache stats ----
Show-Section "7. CACHE STATS"
$cs = Invoke-Api GET "/cache/stats"
if ($cs) {
    Write-Host "  Total entries : $($cs.total_entries)"
    Write-Host "  Cache hits    : $($cs.total_hits)"
    Write-Host "  Live entries  : $($cs.live_entries)"
}

# ---- 8. Submit feedback ----
Show-Section "8. SUBMIT FEEDBACK  (thumbs-up rating=1)"
$fb = Invoke-Api POST "/feedback" @{
    question = "What is the PTO policy?"
    answer   = if ($a1) { $a1.answer } else { "" }
    rating   = 1
    comment  = "Clear and accurate!"
}
if ($fb) { Write-Host "  Feedback recorded: $($fb | ConvertTo-Json -Compress)" }

# ---- 9. Debug chunks ----
Show-Section "9. DEBUG CHUNKS  (first 3 stored chunks)"
$chunks = Invoke-Api GET "/debug/chunks?limit=3"
if ($chunks) {
    $chunks | ForEach-Object {
        Write-Host ("  source={0}  dept={1}  chars={2}" -f $_.source, $_.department, $_.text.Length)
    }
}

# ---- Done ----
Show-Section "DEMO COMPLETE"
Write-Host "  Swagger UI : $BaseUrl/docs" -ForegroundColor Green
Write-Host "  ReDoc      : $BaseUrl/redoc" -ForegroundColor Green
Write-Host ""
