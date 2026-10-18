# =============================================================================
#  LUNG CANCER AI PLATFORM -- AWS S3 + CloudFront Deployment Script
#  Deploys the React frontend to an S3 bucket with CloudFront CDN + HTTPS.
#  
#  Usage:
#    .\deploy\deploy-cloudfront.ps1
#    .\deploy\deploy-cloudfront.ps1 -AwsAccessKey "AKIA..." -AwsSecretKey "..." -Region "us-east-1"
#    .\deploy\deploy-cloudfront.ps1 -BackendUrl "http://54.210.12.34:8000"
# =============================================================================
param(
  [string]$AwsAccessKey,
  [string]$AwsSecretKey,
  [string]$AwsSessionToken,
  [string]$Region          = "us-east-1",
  [string]$StackName       = "lung-cancer-ai-frontend",
  [string]$BackendUrl      = "",
  [string]$BucketPrefix    = "lung-cancer-ai-fe"
)

$ErrorActionPreference = "Stop"

function Log($m)  { Write-Host "[$(Get-Date -f 'HH:mm:ss')] $m" -ForegroundColor Cyan }
Write-Host ""
function Ok($m)   { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "  [ERROR] $m" -ForegroundColor Red; exit 1 }

# 1. Locate AWS CLI
$AwsExe = "aws"
$DefaultCliPath = "C:\Users\ashok\AppData\Local\Programs\Amazon\AWSCLIV2\aws.exe"
if (Test-Path $DefaultCliPath) {
    $AwsExe = $DefaultCliPath
} elseif (-not (Get-Command "aws" -ErrorAction SilentlyContinue)) {
    Fail "AWS CLI not found. Please install AWS CLI v2 or ensure it is in PATH."
}

# 2. Inject credentials if passed as parameters
if ($AwsAccessKey) { $env:AWS_ACCESS_KEY_ID = $AwsAccessKey }
if ($AwsSecretKey) { $env:AWS_SECRET_ACCESS_KEY = $AwsSecretKey }
if ($AwsSessionToken) { $env:AWS_SESSION_TOKEN = $AwsSessionToken }
$env:AWS_DEFAULT_REGION = $Region

# 3. Verify AWS authentication
Log "Verifying AWS credentials with region '$Region'..."
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
$callerIdRaw = & $AwsExe sts get-caller-identity --output json 2>&1
$callerExitCode = $LASTEXITCODE
$ErrorActionPreference = $prevEAP

if ($callerExitCode -ne 0) {
    Write-Host ""
    Warn "AWS credentials are not configured or expired."
    Write-Host "  To configure credentials, run:" -ForegroundColor DarkYellow
    Write-Host "    & '$AwsExe' configure" -ForegroundColor Yellow
    Write-Host "  Or pass them directly into this script:" -ForegroundColor DarkYellow
    Write-Host "    .\deploy\deploy-cloudfront.ps1 -AwsAccessKey 'AKIA...' -AwsSecretKey '...'" -ForegroundColor Yellow
    exit 1
}

$callerId = $callerIdRaw | ConvertFrom-Json
$AccountId = $callerId.Account
Ok "Authenticated as Account: $AccountId (Arn: $($callerId.Arn))"

$RepoRoot = Resolve-Path "$PSScriptRoot\.."
$FrontendDir = "$RepoRoot\frontend"
$TemplateFile = "$PSScriptRoot\cloudfront-s3.yaml"

if (-not (Test-Path $TemplateFile)) {
    Fail "CloudFormation template not found at $TemplateFile"
}

# 4. Deploy or update CloudFormation Stack
Log "Deploying S3 + CloudFront CloudFormation stack '$StackName'..."
$cfParams = @("BucketNamePrefix=$BucketPrefix")
if ($BackendUrl) {
    $backendDomain = ($BackendUrl -replace '^https?://', '') -replace ':[0-9]+$', ''
    $cfParams += "BackendDomainName=$backendDomain"
}

& $AwsExe cloudformation deploy `
    --template-file $TemplateFile `
    --stack-name $StackName `
    --parameter-overrides $cfParams `
    --capabilities CAPABILITY_IAM `
    --no-fail-on-empty-changeset

if ($LASTEXITCODE -ne 0) {
    Fail "CloudFormation deployment failed."
}
Ok "CloudFormation stack '$StackName' deployed successfully."

# 5. Retrieve Stack Outputs
Log "Fetching CloudFront distribution details from CloudFormation outputs..."
$outputsJson = & $AwsExe cloudformation describe-stacks `
    --stack-name $StackName `
    --query "Stacks[0].Outputs" `
    --output json | ConvertFrom-Json

$BucketName = ($outputsJson | Where-Object { $_.OutputKey -eq "BucketName" }).OutputValue
$DistributionId = ($outputsJson | Where-Object { $_.OutputKey -eq "DistributionId" }).OutputValue
$DistributionDomain = ($outputsJson | Where-Object { $_.OutputKey -eq "DistributionDomainName" }).OutputValue
$CloudFrontUrl = ($outputsJson | Where-Object { $_.OutputKey -eq "CloudFrontURL" }).OutputValue

Ok "S3 Bucket: $BucketName"
Ok "CloudFront Distribution: $DistributionId"
Ok "Domain: $DistributionDomain"

# 6. Build Frontend
Log "Building React frontend with production configuration..."
Push-Location $FrontendDir
try {
    if ($BackendUrl) {
        $env:VITE_API_BASE_URL = $BackendUrl
        Log "Set VITE_API_BASE_URL = $BackendUrl"
    }
    
    npm run build
    if ($LASTEXITCODE -ne 0) {
        Fail "Frontend build failed ('npm run build')."
    }
} finally {
    Pop-Location
}
Ok "Built static assets at '$FrontendDir\dist'"

# 7. Sync files to S3 bucket
Log "Uploading assets to S3 bucket s3://$BucketName..."
# Sync hashed static assets (js, css, images) with long-term cache
& $AwsExe s3 sync "$FrontendDir\dist\assets" "s3://$BucketName/assets" `
    --delete `
    --cache-control "public, max-age=31536000, immutable"

# Sync html & other root files with no-cache so updates propagate immediately
& $AwsExe s3 sync "$FrontendDir\dist" "s3://$BucketName" `
    --exclude "assets/*" `
    --delete `
    --cache-control "no-cache, no-store, must-revalidate"

if ($LASTEXITCODE -ne 0) {
    Fail "Failed to sync build files to S3."
}
Ok "Frontend synced to S3."

# 8. Invalidate CloudFront Cache
Log "Creating CloudFront cache invalidation for distribution '$DistributionId'..."
$inval = & $AwsExe cloudfront create-invalidation `
    --distribution-id $DistributionId `
    --paths "/*" `
    --output json | ConvertFrom-Json
$invalId = $inval.Invalidation.Id
Ok "Invalidation created: $invalId"

# 9. Save deployment metadata
$deployInfo = @{
    timestamp          = (Get-Date -f "yyyy-MM-dd HH:mm:ss")
    region             = $Region
    stackName          = $StackName
    s3Bucket           = $BucketName
    distributionId     = $DistributionId
    distributionDomain = $DistributionDomain
    cloudFrontUrl      = $CloudFrontUrl
    backendUrl         = $BackendUrl
}
$infoFile = "$PSScriptRoot\cloudfront-deployment-info.json"
$deployInfo | ConvertTo-Json -Depth 4 | Out-File $infoFile -Encoding UTF8
Ok "Deployment info saved to $infoFile"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  LUNG CANCER AI -- CLOUDFRONT DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Frontend HTTPS URL : $CloudFrontUrl" -ForegroundColor Yellow
Write-Host "  CloudFront Domain  : https://$DistributionDomain" -ForegroundColor Yellow
Write-Host "  S3 Storage Bucket  : s3://$BucketName" -ForegroundColor Cyan
Write-Host "  Distribution ID    : $DistributionId" -ForegroundColor Cyan
if ($BackendUrl) {
    Write-Host "  Connected Backend  : $BackendUrl" -ForegroundColor White
}
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Open the Frontend URL in your browser to verify the deployment."
Write-Host ""
