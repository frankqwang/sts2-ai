param(
    [switch]$StopExistingGodot,
    [string]$Checkpoint = "",
    [string]$CombatCheckpoint = "",
    [string]$GodotExe = "",
    [string]$PythonExe = "",
    [string]$BaseUrl = "",
    [int]$McpPort = 15600,
    [int]$OverlayPort = 8765,
    [ValidateSet("http", "pipe", "pipe-binary")]
    [string]$Transport = "http",
    [string]$Resolution = "1920x1080",
    [string]$CharacterId = "IRONCLAD",
    [string]$Seed = "",
    [double]$StepDelay = 1.0,
    [double]$CombatDelay = 0.45,
    [int]$Episodes = 1,
    [string]$OutputDir = "",
    [string]$DecisionOverlayFile = "",
    [string]$MetricsFile = "",
    [string]$AppDataRoot = "",
    [switch]$MuteAudio,
    [switch]$Greedy,
    [string[]]$PythonExtraArgs = @(),
    [string[]]$GodotExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$commonScript = Join-Path $PSScriptRoot "trainer_common.ps1"
. $commonScript

$sts2aiRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$repoRoot = Split-Path -Parent $sts2aiRoot
$demoScript = Join-Path $repoRoot "STS2AI\Python\demo_play.py"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$recordingRoot = if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    Join-Path $sts2aiRoot ("Artifacts\recording\visible_demo_{0}" -f $stamp)
} else {
    $OutputDir
}
$recordingRoot = (New-Item -ItemType Directory -Force -Path $recordingRoot).FullName
$logDir = Join-Path $recordingRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$defaultHybridCheckpoint = Join-Path $sts2aiRoot "Assets\checkpoints\act1\retrieval_final_iter2175.pt"
$defaultCombatCheckpoint = Join-Path $sts2aiRoot "Assets\checkpoints\act1\_no_default_combat_override.pt"

function Resolve-VisibleGodotExe {
    param(
        [string]$ExplicitPath
    )

    $resolved = Resolve-GodotExe -ExplicitPath $ExplicitPath
    if ($resolved -match "_console\.exe$") {
        $visible = $resolved -replace "_console\.exe$", ".exe"
        if (Test-Path -LiteralPath $visible) {
            return (Resolve-Path -LiteralPath $visible).Path
        }
    }
    return $resolved
}

function Resolve-RequiredRecordingCheckpoint {
    param(
        [string]$ExplicitPath,
        [string]$EnvVar,
        [string]$DefaultPath,
        [string]$Label
    )

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        $candidates += $ExplicitPath
    }
    $envValue = [Environment]::GetEnvironmentVariable($EnvVar)
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        $candidates += $envValue
    }
    $candidates += $DefaultPath

    $checked = @()
    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        $resolvedPath = Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue | Select-Object -First 1
        $resolved = if ($null -ne $resolvedPath) {
            [System.IO.Path]::GetFullPath($resolvedPath.Path)
        } else {
            [System.IO.Path]::GetFullPath($candidate)
        }
        $checked += $resolved
        if (Test-Path -LiteralPath $resolved) {
            return $resolved
        }
    }

    $checkedText = ($checked | ForEach-Object { "  - $_" }) -join [Environment]::NewLine
    throw "$Label checkpoint not found.`nChecked:`n$checkedText`nPass -Checkpoint or set $EnvVar."
}

function Resolve-OptionalRecordingCheckpoint {
    param(
        [string]$ExplicitPath,
        [string]$EnvVar,
        [string]$DefaultPath
    )

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        $candidates += $ExplicitPath
    }
    $envValue = [Environment]::GetEnvironmentVariable($EnvVar)
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        $candidates += $envValue
    }
    $candidates += $DefaultPath

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        $resolvedPath = Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue | Select-Object -First 1
        $resolved = if ($null -ne $resolvedPath) {
            [System.IO.Path]::GetFullPath($resolvedPath.Path)
        } else {
            [System.IO.Path]::GetFullPath($candidate)
        }
        if (Test-Path -LiteralPath $resolved) {
            return $resolved
        }
    }
    return $null
}

$godotExe = Resolve-VisibleGodotExe -ExplicitPath $GodotExe
$pythonExe = Resolve-PythonExe -ExplicitPath $PythonExe
$resolvedBaseUrl = Resolve-BaseUrl -BaseUrl $BaseUrl -McpPort $McpPort
$resolvedMcpPort = Resolve-McpPortFromBaseUrl -ResolvedBaseUrl $resolvedBaseUrl -DefaultPort $McpPort
$singleplayerStateUrl = Resolve-SingleplayerStateUrl -ResolvedBaseUrl $resolvedBaseUrl

$resolvedHybridCheckpoint = Resolve-RequiredRecordingCheckpoint `
    -ExplicitPath $Checkpoint `
    -EnvVar "STS2_DEMO_HYBRID_CHECKPOINT" `
    -DefaultPath $defaultHybridCheckpoint `
    -Label "Hybrid demo"

$resolvedCombatCheckpoint = Resolve-OptionalRecordingCheckpoint `
    -ExplicitPath $CombatCheckpoint `
    -EnvVar "STS2_DEMO_COMBAT_CHECKPOINT" `
    -DefaultPath $defaultCombatCheckpoint

$resolvedOutputDir = Join-Path $recordingRoot "demo_output"
New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null

$resolvedDecisionOverlayFile = if ([string]::IsNullOrWhiteSpace($DecisionOverlayFile)) {
    Join-Path $recordingRoot "live_overlay.json"
} else {
    $DecisionOverlayFile
}
$overlayDir = Split-Path -Parent $resolvedDecisionOverlayFile
if (-not [string]::IsNullOrWhiteSpace($overlayDir)) {
    New-Item -ItemType Directory -Force -Path $overlayDir | Out-Null
}

$resolvedAppDataRoot = if ([string]::IsNullOrWhiteSpace($AppDataRoot)) {
    Join-Path $recordingRoot "appdata"
} else {
    $AppDataRoot
}
New-Item -ItemType Directory -Force -Path $resolvedAppDataRoot | Out-Null
$null = Initialize-IsolatedEditorDataRoot -AppDataRoot $resolvedAppDataRoot
$null = Set-EditorWindowDefaults -AppDataRoot $resolvedAppDataRoot -Resolution $Resolution -McpPort $resolvedMcpPort -ForceWindowed
if ($MuteAudio) {
    $null = Set-EditorAudioDefaults -AppDataRoot $resolvedAppDataRoot
}
$editorRunSaveRoot = Join-Path $resolvedAppDataRoot "SlayTheSpire2\editor"

$godotArgs = @(
    "--verbose",
    "--display-driver", "windows",
    "--rendering-driver", "opengl3",
    "--windowed",
    "--resolution", $Resolution,
    "--mcp-instant",
    "--mcp-port", [string]$resolvedMcpPort,
    "--mcp-decision-overlay-file", $resolvedDecisionOverlayFile
) + $GodotExtraArgs + @("--path", $repoRoot)

$pythonArgs = @(
    $demoScript,
    "--checkpoint", $resolvedHybridCheckpoint,
    "--transport", $Transport,
    "--port", [string]$resolvedMcpPort,
    "--base-url", $resolvedBaseUrl,
    "--overlay-port", [string]$OverlayPort,
    "--decision-overlay-file", $resolvedDecisionOverlayFile,
    "--character-id", $CharacterId,
    "--step-delay", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:0.00}", $StepDelay)),
    "--combat-delay", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:0.00}", $CombatDelay)),
    "--episodes", [string]$Episodes,
    "--output-dir", $resolvedOutputDir
)
if (-not [string]::IsNullOrWhiteSpace($Seed)) {
    $pythonArgs += @("--seed", $Seed)
}
if ($resolvedCombatCheckpoint) {
    $pythonArgs += @("--combat-checkpoint", $resolvedCombatCheckpoint)
}
if (-not [string]::IsNullOrWhiteSpace($MetricsFile)) {
    $pythonArgs += @("--metrics-file", $MetricsFile)
}
if ($Greedy) {
    $pythonArgs += "--greedy"
}
if ($PythonExtraArgs.Count -gt 0) {
    $pythonArgs += $PythonExtraArgs
}

$pythonStdout = Join-Path $logDir "demo_play.stdout.log"
$pythonStderr = Join-Path $logDir "demo_play.stderr.log"
$manifestPath = Join-Path $recordingRoot "recording_manifest.json"

$manifest = [ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    mode = "visible_demo"
    transport = $Transport
    base_url = $resolvedBaseUrl
    mcp_port = $resolvedMcpPort
    overlay_port = $OverlayPort
    resolution = $Resolution
    episodes = $Episodes
    seed = $(if ([string]::IsNullOrWhiteSpace($Seed)) { $null } else { $Seed })
    step_delay = $StepDelay
    combat_delay = $CombatDelay
    godot_exe = $godotExe
    python_exe = $pythonExe
    checkpoint = $resolvedHybridCheckpoint
    combat_checkpoint = $resolvedCombatCheckpoint
    overlay_file = $resolvedDecisionOverlayFile
    output_dir = $resolvedOutputDir
    appdata_root = $resolvedAppDataRoot
    dashboard_overlay_url = "http://localhost:$OverlayPort/"
    recording_overlay_url = "http://localhost:$OverlayPort/?mode=recording"
    python_stdout = $pythonStdout
    python_stderr = $pythonStderr
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host "Starting visible AI recording demo..." -ForegroundColor Cyan
Write-Host "GodotExe           : $godotExe"
Write-Host "PythonExe          : $pythonExe"
Write-Host "Transport          : $Transport"
Write-Host "BaseUrl            : $resolvedBaseUrl"
Write-Host "Resolution         : $Resolution"
Write-Host "Episodes           : $Episodes"
Write-Host "Seed               : $(if ([string]::IsNullOrWhiteSpace($Seed)) { '<random>' } else { $Seed })"
Write-Host "HybridCheckpoint   : $resolvedHybridCheckpoint"
Write-Host "CombatCheckpoint   : $(if ($resolvedCombatCheckpoint) { $resolvedCombatCheckpoint } else { 'embedded / not found' })"
Write-Host "OverlayFile        : $resolvedDecisionOverlayFile"
Write-Host "RecordingOverlay   : http://localhost:$OverlayPort/?mode=recording"
Write-Host "DashboardOverlay   : http://localhost:$OverlayPort/"
Write-Host "ArtifactsRoot      : $recordingRoot"

Assert-CleanSingleplayerPort -StateUrl $singleplayerStateUrl -StopExistingGodot:$StopExistingGodot
$clearedRunSaveFiles = Clear-EditorRunSaves -RootPath $editorRunSaveRoot
if ($clearedRunSaveFiles.Count -gt 0) {
    Write-Host "Cleared run-save files: $($clearedRunSaveFiles.Count)" -ForegroundColor Yellow
}

$godotEnv = @{ APPDATA = $resolvedAppDataRoot }
$godotProc = $null
$demoProc = $null

try {
    $godotProc = Start-ProcessWithEnvironment `
        -FilePath $godotExe `
        -ArgumentList $godotArgs `
        -WorkingDirectory $repoRoot `
        -Environment $godotEnv `
        -PassThru

    Write-Host "Launched Godot process $($godotProc.Id)." -ForegroundColor Green
    $null = Focus-GodotGameWindow -ProcessId $godotProc.Id

    Write-Host "Waiting for visible MCP endpoint..." -ForegroundColor Yellow
    if (-not (Wait-SingleplayerEndpoint -StateUrl $singleplayerStateUrl)) {
        throw "Visible MCP singleplayer endpoint did not become ready in time."
    }
    $null = Focus-GodotGameWindow -ProcessId $godotProc.Id

    $demoProc = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $pythonArgs `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $pythonStdout `
        -RedirectStandardError $pythonStderr `
        -WindowStyle Hidden `
        -PassThru

    Write-Host "Launched demo_play.py process $($demoProc.Id)." -ForegroundColor Green
    $demoProc.WaitForExit()
    $exitCode = [int]$demoProc.ExitCode
    if ($exitCode -ne 0) {
        throw "demo_play.py exited with code $exitCode. See $pythonStderr"
    }
}
finally {
    if ($demoProc -and -not $demoProc.HasExited) {
        Stop-Process -Id $demoProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($godotProc -and -not $godotProc.HasExited) {
        Stop-Process -Id $godotProc.Id -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "Visible demo finished." -ForegroundColor Cyan
Write-Host "Summary           : $(Join-Path $resolvedOutputDir 'demo_summary.json')"
Write-Host "Decision trace    : $(Join-Path $resolvedOutputDir 'decision_trace.jsonl')"
Write-Host "Manifest          : $manifestPath"
Write-Host "Python stdout     : $pythonStdout"
Write-Host "Python stderr     : $pythonStderr"
exit 0
