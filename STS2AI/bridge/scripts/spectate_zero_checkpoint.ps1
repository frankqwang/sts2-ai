param(
    [switch]$StopExistingGodot,
    [string]$CheckpointPath,
    [ValidateSet("stateless", "history_transformer", "recurrent_gru")]
    [string]$ModelVariant = "stateless",
    [string]$BuildFile = "",
    [string]$GodotExe = "",
    [string]$PythonExe = "",
    [string]$BaseUrl = "",
    [int]$McpPort = 15526,
    [string]$Resolution = "1600x900",
    [string]$CharacterId = "IRONCLAD",
    [string]$Seed = "",
    [double]$StepDelay = 0.9,
    [int]$MaxSteps = 800,
    [string]$OutputDir = "",
    [string]$AppDataRoot = "",
    [switch]$MuteAudio
)

$ErrorActionPreference = "Stop"

$commonScript = Join-Path $PSScriptRoot "trainer_common.ps1"
. $commonScript

$sts2aiRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$repoRoot = Split-Path -Parent $sts2aiRoot

if ([string]::IsNullOrWhiteSpace($CheckpointPath)) {
    throw "CheckpointPath is required."
}

$resolvedCheckpoint = (Resolve-Path -LiteralPath $CheckpointPath).Path
if (-not (Test-Path -LiteralPath $resolvedCheckpoint)) {
    throw "Checkpoint not found: $CheckpointPath"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    Join-Path $sts2aiRoot ("Artifacts\recording\visible_zero_{0}" -f $stamp)
} else {
    $OutputDir
}
$runRoot = (New-Item -ItemType Directory -Force -Path $runRoot).FullName
$logDir = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$resolvedBuildFile = ""
if (-not [string]::IsNullOrWhiteSpace($BuildFile)) {
    $resolvedBuildFile = (Resolve-Path -LiteralPath $BuildFile).Path
}

$godotExe = Resolve-GodotExe -ExplicitPath $GodotExe
if ($godotExe -match "_console\.exe$") {
    $visibleGodot = $godotExe -replace "_console\.exe$", ".exe"
    if (Test-Path -LiteralPath $visibleGodot) {
        $godotExe = (Resolve-Path -LiteralPath $visibleGodot).Path
    }
}
$pythonExe = Resolve-PythonExe -ExplicitPath $PythonExe
$resolvedBaseUrl = Resolve-BaseUrl -BaseUrl $BaseUrl -McpPort $McpPort
$singleplayerStateUrl = Resolve-SingleplayerStateUrl -ResolvedBaseUrl $resolvedBaseUrl

$resolvedAppDataRoot = if ([string]::IsNullOrWhiteSpace($AppDataRoot)) {
    Join-Path $runRoot "appdata"
} else {
    $AppDataRoot
}
New-Item -ItemType Directory -Force -Path $resolvedAppDataRoot | Out-Null
$null = Initialize-IsolatedEditorDataRoot -AppDataRoot $resolvedAppDataRoot
$null = Set-EditorWindowDefaults -AppDataRoot $resolvedAppDataRoot -Resolution $Resolution -McpPort $McpPort -ForceWindowed
if ($MuteAudio) {
    $null = Set-EditorAudioDefaults -AppDataRoot $resolvedAppDataRoot
}
$editorRunSaveRoot = Join-Path $resolvedAppDataRoot "SlayTheSpire2\editor"

$overlayFile = Join-Path $runRoot "live_overlay.json"
$pythonStdout = Join-Path $logDir "spectate.stdout.log"
$pythonStderr = Join-Path $logDir "spectate.stderr.log"
$manifestPath = Join-Path $runRoot "manifest.json"

$manifest = [ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    checkpoint = $resolvedCheckpoint
    model_variant = $ModelVariant
    build_file = $(if ([string]::IsNullOrWhiteSpace($resolvedBuildFile)) { $null } else { $resolvedBuildFile })
    base_url = $resolvedBaseUrl
    mcp_port = $McpPort
    resolution = $Resolution
    character_id = $CharacterId
    seed = $(if ([string]::IsNullOrWhiteSpace($Seed)) { $null } else { $Seed })
    step_delay = $StepDelay
    max_steps = $MaxSteps
    godot_exe = $godotExe
    python_exe = $pythonExe
    appdata_root = $resolvedAppDataRoot
    overlay_file = $overlayFile
    stdout_log = $pythonStdout
    stderr_log = $pythonStderr
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Assert-CleanSingleplayerPort -StateUrl $singleplayerStateUrl -StopExistingGodot:$StopExistingGodot
$clearedRunSaveFiles = Clear-EditorRunSaves -RootPath $editorRunSaveRoot

$isSteamExe = (Test-Path -LiteralPath ([IO.Path]::ChangeExtension($godotExe, ".pck")))
$godotArgs = @(
    "--verbose",
    "--display-driver", "windows",
    "--rendering-driver", "opengl3",
    "--windowed",
    "--resolution", $Resolution,
    "--mcp-instant",
    "--mcp-port", [string]$McpPort
)
if (-not $isSteamExe) {
    $godotArgs += @("--path", $repoRoot)
}

$env:STS2_ZERO_CHECKPOINT = $resolvedCheckpoint
$env:STS2_ZERO_MODEL_VARIANT = $ModelVariant

$pythonArgs = @(
    "-m", "game_bridge.spectate.cli",
    "--mode", "external",
    "--external-policy", "game_bridge.spectate.zero_external_policy:select_action",
    "--overlay-file", $overlayFile,
    "--base-url", $resolvedBaseUrl,
    "--request-timeout-s", "30",
    "--ready-timeout-s", "90",
    "--character-id", $CharacterId,
    "--max-steps", [string]$MaxSteps,
    "--step-delay", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:0.00}", $StepDelay))
)
if (-not [string]::IsNullOrWhiteSpace($Seed)) {
    $pythonArgs += @("--seed", $Seed)
}
if (-not [string]::IsNullOrWhiteSpace($resolvedBuildFile)) {
    $pythonArgs += @("--build-file", $resolvedBuildFile)
}

$godotEnv = @{ APPDATA = $resolvedAppDataRoot }
$godotProc = $null
$spectateProc = $null

try {
    $godotProc = Start-ProcessWithEnvironment `
        -FilePath $godotExe `
        -ArgumentList $godotArgs `
        -WorkingDirectory $repoRoot `
        -Environment $godotEnv `
        -PassThru

    Write-Host "Launched visible Godot PID=$($godotProc.Id)" -ForegroundColor Green
    $null = Focus-GodotGameWindow -ProcessId $godotProc.Id

    if (-not (Wait-SingleplayerEndpoint -StateUrl $singleplayerStateUrl)) {
        throw "Visible MCP singleplayer endpoint did not become ready in time."
    }
    Start-Sleep -Seconds 8

    $spectateProc = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $pythonArgs `
        -WorkingDirectory (Join-Path $sts2aiRoot "bridge") `
        -RedirectStandardOutput $pythonStdout `
        -RedirectStandardError $pythonStderr `
        -WindowStyle Hidden `
        -PassThru

    Write-Host "Launched spectator PID=$($spectateProc.Id)" -ForegroundColor Green
    Write-Host "RunRoot      : $runRoot"
    Write-Host "Manifest     : $manifestPath"
    Write-Host "Stdout log   : $pythonStdout"
    Write-Host "Stderr log   : $pythonStderr"

    $spectateProc.WaitForExit()
    $exitCode = [int]$spectateProc.ExitCode
    if ($exitCode -ne 0) {
        throw "spectate cli exited with code $exitCode. See $pythonStderr"
    }
}
finally {
    if ($spectateProc -and -not $spectateProc.HasExited) {
        Stop-Process -Id $spectateProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($godotProc -and -not $godotProc.HasExited) {
        Stop-Process -Id $godotProc.Id -Force -ErrorAction SilentlyContinue
    }
}
