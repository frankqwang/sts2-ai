param(
    [Parameter(Mandatory = $true)]
    [string]$TeacherCheckpoint,
    [string]$Checkpoint = "STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt",
    [ValidateSet("hard_override", "full_replace", "rerank", "replace")]
    [string]$Mode = "hard_override",
    [ValidateSet(20, 50)]
    [int]$SeedCount = 20,
    [string]$Transport = "pipe-binary",
    [int]$Port = 15527,
    [switch]$AutoLaunch,
    [string]$OutputPath = ""
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$pythonExe = "python"

if (-not (Test-Path -LiteralPath $TeacherCheckpoint)) {
    throw "Teacher checkpoint not found: $TeacherCheckpoint"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Base checkpoint not found: $Checkpoint"
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $safeMode = $Mode.Trim().ToLowerInvariant()
    $safeCount = [string]$SeedCount
    $OutputPath = Join-Path $repoRoot "STS2AI\Artifacts\eval\ironclad_teacher_${safeMode}_${safeCount}.json"
}

$outputDir = Split-Path -Parent $OutputPath
if (-not [string]::IsNullOrWhiteSpace($outputDir) -and -not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$argsList = @(
    "$repoRoot\STS2AI\Python\evaluate_ai.py",
    "--checkpoint", $Checkpoint,
    "--combat-teacher-checkpoint", $TeacherCheckpoint,
    "--combat-teacher-mode", $Mode,
    "--character", "IRONCLAD",
    "--ascension", "0",
    "--transport", $Transport,
    "--port", "$Port",
    "--num-games", "$SeedCount",
    "--output", $OutputPath
)

if ($AutoLaunch) {
    $argsList += "--auto-launch"
}

Write-Host "Running IRONCLAD teacher eval:"
Write-Host "  mode       = $Mode"
Write-Host "  seed_count = $SeedCount"
Write-Host "  output     = $OutputPath"

& $pythonExe @argsList
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
