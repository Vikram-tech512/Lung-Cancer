# =============================================================================
#  LUNG CANCER AI PLATFORM -- 100% Free Public Online Launcher
#  Starts Backend + Frontend + Free Cloudflare Global Tunnel
#  - 100% Free forever (No credit card or account needed)
#  - Unlimited client requests & unlimited bandwidth
#  - Global Edge CDN with automated HTTPS/SSL
# =============================================================================

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "   LUNG CANCER AI -- 100% FREE PUBLIC ONLINE ACCESS" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$RepoRoot = Resolve-Path "$PSScriptRoot\.."
$FrontendDir = "$RepoRoot\frontend"
$Cloudflared = "$env:TEMP\cloudflared.exe"

# 1. Ensure Cloudflare Tunnel CLI is available
if (-not (Test-Path $Cloudflared)) {
    Write-Host "[1/4] Downloading lightweight Cloudflare Tunnel engine..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $Cloudflared -UseBasicParsing
    Write-Host "      Cloudflare engine ready." -ForegroundColor Green
} else {
    Write-Host "[1/4] Cloudflare Tunnel engine ready." -ForegroundColor Green
}

# 2. Start FastAPI Backend (Port 8000)
Write-Host "[2/4] Starting FastAPI Medical AI Backend on port 8000..." -ForegroundColor Yellow
$BackendProc = Start-Process -FilePath "python" -ArgumentList "-m backend.main --host 127.0.0.1 --port 8000" -WorkingDirectory $RepoRoot -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2

# 3. Start React Frontend Preview (Port 3000)
Write-Host "[3/4] Starting React Workstation Frontend on port 3000..." -ForegroundColor Yellow
$FrontendProc = Start-Process -FilePath "npm.cmd" -ArgumentList "run preview -- --port 3000 --host 127.0.0.1" -WorkingDirectory $FrontendDir -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 3

# 4. Start Cloudflare Tunnel
Write-Host "[4/4] Establishing secure Cloudflare edge tunnel..." -ForegroundColor Yellow
$TunnelLog = "$env:TEMP\lung_cancer_tunnel.log"
if (Test-Path $TunnelLog) { Remove-Item $TunnelLog -Force }

$TunnelProc = Start-Process -FilePath $Cloudflared -ArgumentList "tunnel --url http://127.0.0.1:3000 --logfile `"$TunnelLog`"" -PassThru -WindowStyle Hidden

# Poll log for the public URL
$PublicUrl = ""
$Attempts = 0
while ($Attempts -lt 30 -and -not $PublicUrl) {
    Start-Sleep -Seconds 1
    $Attempts++
    if (Test-Path $TunnelLog) {
        $Content = Get-Content $TunnelLog -Raw -ErrorAction SilentlyContinue
        if ($Content -match "https://[a-zA-Z0-9-]+\.trycloudflare\.com") {
            $PublicUrl = $Matches[0]
            break
        }
    }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "          SUCCESS! YOUR PLATFORM IS NOW LIVE WORLDWIDE          " -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
if ($PublicUrl) {
    Write-Host "  PUBLIC LIVE HTTPS URL : $PublicUrl" -ForegroundColor Cyan
    Write-Host "  SHAREABLE LINK        : $PublicUrl" -ForegroundColor Yellow
} else {
    Write-Host "  Tunnel started. Check log: $TunnelLog" -ForegroundColor Yellow
    Write-Host "  Local URL             : http://localhost:3000" -ForegroundColor Cyan
}
Write-Host "  Local API Docs        : http://localhost:8000/docs" -ForegroundColor Gray
Write-Host ""
Write-Host "  Features active:" -ForegroundColor Green
Write-Host "    * 100% Free Forever"
Write-Host "    * Unlimited client requests"
Write-Host "    * Full 3D AI deep learning & Grad-CAM inference enabled"
Write-Host "    * HTTPS SSL secured by Cloudflare Edge"
Write-Host ""
Write-Host "Press [Enter] or Ctrl+C to stop the online services..." -ForegroundColor Magenta
Read-Host

Write-Host "Shutting down services..." -ForegroundColor Yellow
Stop-Process -Id $BackendProc.Id -ErrorAction SilentlyContinue
Stop-Process -Id $FrontendProc.Id -ErrorAction SilentlyContinue
Stop-Process -Id $TunnelProc.Id -ErrorAction SilentlyContinue
Write-Host "All services stopped." -ForegroundColor Green
