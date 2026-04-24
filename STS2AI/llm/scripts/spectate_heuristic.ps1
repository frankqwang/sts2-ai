param(
    [switch]$StopExistingGodot,
    [string]$BuildFile = "",
    [string]$EncounterId = "",
    [string]$GodotExe = "",
    [string]$PythonExe = "",
    [string]$BaseUrl = "",
    [int]$McpPort = 15526,
    [string]$Resolution = "1600x900",
    [string]$CharacterId = "IRONCLAD",
    [string]$Seed = "",
    [int]$Floor = 0,
    [double]$RequestTimeoutSeconds = 120.0,
    [double]$ReadyTimeoutSeconds = 150.0,
    [double]$StepDelay = 0.9,
    [int]$MaxSteps = 800,
    [string]$OutputDir = "",
    [string]$AppDataRoot = "",
    [switch]$MuteAudio
)

# 复用 bridge/scripts/trainer_common.ps1 和 spectate_zero_checkpoint.ps1 的结构，
# 只把 --external-policy 换成启发式老师，去掉 CheckpointPath 依赖。

$ErrorActionPreference = "Stop"

$llmRoot = Split-Path -Parent $PSScriptRoot    # ...\STS2AI\llm
$sts2aiRoot = Split-Path -Parent $llmRoot       # ...\STS2AI
$repoRoot = Split-Path -Parent $sts2aiRoot      # repo root
$bridgeScripts = Join-Path $sts2aiRoot "bridge\scripts"
$commonScript = Join-Path $bridgeScripts "trainer_common.ps1"
. $commonScript

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    Join-Path $sts2aiRoot ("Artifacts\llm\spectate_heuristic\{0}" -f $stamp)
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
$spectatorModSource = Join-Path $sts2aiRoot "ENV\Spectator\SpectatorBridgeMod\bin\Debug\net9.0"
$spectatorModInstallDir = Sync-SpectatorModArtifacts -SourceDir $spectatorModSource -GodotExe $godotExe

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
    policy = "heuristic_teacher"
    build_file = $(if ([string]::IsNullOrWhiteSpace($resolvedBuildFile)) { $null } else { $resolvedBuildFile })
    encounter_id = $(if ([string]::IsNullOrWhiteSpace($EncounterId)) { $null } else { $EncounterId })
    base_url = $resolvedBaseUrl
    mcp_port = $McpPort
    resolution = $Resolution
    character_id = $CharacterId
    seed = $(if ([string]::IsNullOrWhiteSpace($Seed)) { $null } else { $Seed })
    floor = $(if ($Floor -gt 0) { $Floor } else { $null })
    request_timeout_s = $RequestTimeoutSeconds
    ready_timeout_s = $ReadyTimeoutSeconds
    step_delay = $StepDelay
    max_steps = $MaxSteps
    godot_exe = $godotExe
    spectator_mod_dir = $spectatorModInstallDir
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

# 注意：spectate cli 的 WorkingDirectory 是 STS2AI/bridge，heuristic_policy.py
# 自己会 sys.path.insert 到 STS2AI，因此能 `import llm.*`
$pythonArgs = @(
    "-m", "game_bridge.spectate.cli",
    "--mode", "external",
    "--external-policy", "llm.inference.heuristic_policy:select_action",
    "--overlay-file", $overlayFile,
    "--base-url", $resolvedBaseUrl,
    "--request-timeout-s", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:0.##}", $RequestTimeoutSeconds)),
    "--ready-timeout-s", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:0.##}", $ReadyTimeoutSeconds)),
    "--character-id", $CharacterId,
    "--max-steps", [string]$MaxSteps,
    "--step-delay", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:0.00}", $StepDelay))
)
if (-not [string]::IsNullOrWhiteSpace($Seed)) {
    $pythonArgs += @("--seed", $Seed)
}
if ($Floor -gt 0) {
    $pythonArgs += @("--floor", [string]$Floor)
}
if (-not [string]::IsNullOrWhiteSpace($EncounterId)) {
    $pythonArgs += @("--encounter-id", $EncounterId)
}
if (-not [string]::IsNullOrWhiteSpace($resolvedBuildFile)) {
    $pythonArgs += @("--build-file", $resolvedBuildFile)
}

# 把 STS2AI 加进 PYTHONPATH，保证 `import llm.*` 能成（cwd 在 bridge/，
# 只加那层的话只能 import game_bridge.*）
$existingPyPath = if ($env:PYTHONPATH) { $env:PYTHONPATH } else { "" }
if ($existingPyPath -and ($existingPyPath -notlike "*$sts2aiRoot*")) {
    $env:PYTHONPATH = $sts2aiRoot + [IO.Path]::PathSeparator + $existingPyPath
} else {
    $env:PYTHONPATH = $sts2aiRoot
}
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

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
