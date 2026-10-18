# =============================================================================
#  LUNG CANCER AI PLATFORM -- Zoho Catalyst Deployment Script
#  Deploys the application to Zoho Catalyst Cloud (Web Client + AppSail).
#
#  Usage:
#    .\deploy\deploy-zoho.ps1 -BuildPackage
#    .\deploy\deploy-zoho.ps1 -Target client
#    .\deploy\deploy-zoho.ps1 -Target client -BackendUrl "https://appsail-backend.catalystserverless.com"
#    .\deploy\deploy-zoho.ps1 -Target appsail
#    .\deploy\deploy-zoho.ps1 -Target all -BackendUrl "https://appsail-backend.catalystserverless.com"
# =============================================================================
param(
  [ValidateSet("client", "appsail", "all")]
  [string]$Target = "client",
  [string]$BackendUrl = "",
  [switch]$BuildPackage
)

$ErrorActionPreference = "Stop"

function Log($m)  { Write-Host "[$(Get-Date -f 'HH:mm:ss')] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "  [ERROR] $m" -ForegroundColor Red; exit 1 }

$RepoRoot = Resolve-Path "$PSScriptRoot\.."
$FrontendDir = "$RepoRoot\frontend"
$BackendDir = "$RepoRoot\backend"
$ClientZipOutput = "$RepoRoot\dist-zoho-catalyst.zip"
$AppsailZipOutput = "$RepoRoot\dist-zoho-appsail.zip"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  LUNG CANCER AI -- ZOHO CATALYST DEPLOYMENT" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Build Frontend (if Target is client or all, or -BuildPackage)
if ($Target -in @("client", "all") -or $BuildPackage) {
    Log "Building React frontend with production optimizations..."
    Push-Location $FrontendDir
    try {
        if ($BackendUrl) {
            $env:VITE_API_BASE_URL = $BackendUrl
            Log "Injected VITE_API_BASE_URL = $BackendUrl"
        }
        npm run build
        if ($LASTEXITCODE -ne 0) {
            Fail "Frontend build failed ('npm run build')."
        }
    } finally {
        Pop-Location
    }
    Ok "Built frontend into '$FrontendDir\dist'"

    # Ensure client-package.json is in dist directory for Zoho Catalyst SPA routing
    if (Test-Path "$FrontendDir\client-package.json") {
        Copy-Item "$FrontendDir\client-package.json" "$FrontendDir\dist\client-package.json" -Force
        Ok "Copied client-package.json to dist directory."
    }

    # Package Web Client into ZIP
    Log "Packaging Web Client into ZIP ($ClientZipOutput)..."
    if (Test-Path $ClientZipOutput) { Remove-Item $ClientZipOutput -Force }
    Compress-Archive -Path "$FrontendDir\dist\*" -DestinationPath $ClientZipOutput -Force
    Ok "Web Client ZIP created: $ClientZipOutput"
}

# 2. Package AppSail Backend
if ($Target -in @("appsail", "all") -or $BuildPackage) {
    Log "Packaging AppSail Backend into ZIP ($AppsailZipOutput)..."
    if (Test-Path $AppsailZipOutput) { Remove-Item $AppsailZipOutput -Force }

    # Temporary staging directory to exclude __pycache__ and scratch files
    $StagingDir = "$RepoRoot\.appsail_stage"
    if (Test-Path $StagingDir) { Remove-Item $StagingDir -Recurse -Force }
    New-Item -ItemType Directory -Path $StagingDir | Out-Null

    try {
        # Copy backend files except __pycache__
        Get-ChildItem -Path $BackendDir -Recurse | Where-Object {
            $_.FullName -notmatch '__pycache__' -and $_.FullName -notmatch '\.pytest_cache'
        } | ForEach-Object {
            $relative = $_.FullName.Substring($BackendDir.Length).TrimStart('\', '/')
            $dest = Join-Path $StagingDir $relative
            if ($_.PSIsContainer) {
                if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Path $dest | Out-Null }
            } else {
                $parent = Split-Path $dest
                if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
                Copy-Item $_.FullName $dest -Force
            }
        }
        Compress-Archive -Path "$StagingDir\*" -DestinationPath $AppsailZipOutput -Force
        Ok "AppSail Backend ZIP created: $AppsailZipOutput"
    } finally {
        if (Test-Path $StagingDir) { Remove-Item $StagingDir -Recurse -Force }
    }
}

if ($BuildPackage) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  BUILD PACKAGES READY FOR ZOHO CATALYST!" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  1. Web Client Package : $ClientZipOutput" -ForegroundColor Yellow
    Write-Host "  2. AppSail Package    : $AppsailZipOutput" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  How to deploy via Zoho Catalyst Console:"
    Write-Host "    Web Client (Frontend):" -ForegroundColor Cyan
    Write-Host "      a. Go to https://catalyst.zoho.com -> Your Project"
    Write-Host "      b. In left menu, click 'Web Client Hosting'"
    Write-Host "      c. Upload: dist-zoho-catalyst.zip"
    Write-Host "    AppSail (Backend):" -ForegroundColor Cyan
    Write-Host "      a. In left menu, click 'AppSail'"
    Write-Host "      b. Create or select service 'lung-cancer-ai-backend'"
    Write-Host "      c. Upload: dist-zoho-appsail.zip (Stack: Python 3.11, Command: python -u main.py)"
    Write-Host "    Connect Frontend & Backend:" -ForegroundColor Cyan
    Write-Host "      Once AppSail is deployed, copy its URL and run:"
    Write-Host "      .\deploy\deploy-zoho.ps1 -Target client -BackendUrl <APPS_URL>" -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

# 3. Check for Zoho Catalyst CLI
Log "Checking Zoho Catalyst CLI..."
$CatalystCmd = Get-Command "catalyst.cmd" -ErrorAction SilentlyContinue
if (-not $CatalystCmd) {
    $CatalystCmd = Get-Command "catalyst.ps1" -ErrorAction SilentlyContinue
}
if (-not $CatalystCmd) {
    $CatalystCmd = Get-Command "catalyst" -ErrorAction SilentlyContinue
}

if (-not $CatalystCmd) {
    Warn "Zoho Catalyst CLI ('zcatalyst-cli') is not found in PATH."
    Write-Host "  You can deploy instantly by uploading the generated ZIP packages:" -ForegroundColor Yellow
    Write-Host "    Web Client: $ClientZipOutput" -ForegroundColor Green
    Write-Host "    AppSail:    $AppsailZipOutput" -ForegroundColor Green
    Write-Host "  Or install the CLI with: npm install -g zcatalyst-cli" -ForegroundColor DarkYellow
    exit 0
}

# 4. Deploy via CLI
Log "Deploying via Zoho Catalyst CLI ($($CatalystCmd.Source))..."
Push-Location $RepoRoot
try {
    if ($Target -eq "client") {
        & $CatalystCmd deploy --only client
    } elseif ($Target -eq "appsail") {
        & $CatalystCmd deploy --only appsail
    } else {
        & $CatalystCmd deploy
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  ZOHO CATALYST DEPLOYMENT COMPLETED!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Manage in Console: https://catalyst.zoho.com" -ForegroundColor Cyan
Write-Host ""

