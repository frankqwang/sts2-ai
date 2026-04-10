param(
    [ValidateSet("Quick", "Full", "Both")]
    [string]$Mode = "Quick",
    [string]$BaselineBackend = "godot-http",
    [string]$CandidateBackend = "headless-pipe",
    [string]$PythonExe = "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe",
    [string]$GodotExe = "D:\dev\Godot_v4.5.1-stable_mono_win64\Godot_v4.5.1-stable_mono_win64_console.exe",
    [string]$HeadlessDll = "",
    [string]$Checkpoint = "",
    [string]$CombatCheckpoint = "",
    [string]$OutputRoot = "",
    [int]$BaselinePort = 15526,
    [int]$CandidatePort = 15527,
    [int]$AuditMaxSteps = 800,
    [int]$DiscoverCount = 240,
    [int]$BossAuditCount = 240,
    [int]$RewardAuditCount = 240,
    [int]$PolicyRolloutSeedCount = 20,
    [int]$PolicyRolloutTraceCount = 3,
    [int]$CombatSaveLoadSeedCount = 5,
    [int]$TrainingIterations = 20,
    [int]$TrainingNumEnvs = 4,
    [int]$TrainingEpisodesPerIter = 4,
    [int]$TrainingMaxEpisodeSteps = 600,
    [int]$PpoMinibatch = 32,
    [switch]$AutoLaunch = $true
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$sts2aiRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$repoRoot = Split-Path -Parent $sts2aiRoot
$defaultCheckpoint = Join-Path $sts2aiRoot "Assets\checkpoints\act1\retrieval_final_iter2175.pt"
$defaultCombatCheckpoint = Join-Path $sts2aiRoot "Assets\checkpoints\act1\retrieval_final_iter2175.pt"
$defaultHeadlessDll = Join-Path $sts2aiRoot "ENV\Sim\Runtime\HeadlessSim\bin\Debug\net9.0\HeadlessSim.dll"
$verificationRoot = if ($OutputRoot) { $OutputRoot } else { Join-Path $sts2aiRoot "Artifacts\verification" }
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = Join-Path $verificationRoot "sim_vs_godot_audit_$timestamp"
$quickDir = Join-Path $runRoot "quick"
$fullDir = Join-Path $runRoot "full"
$logDir = Join-Path $runRoot "logs"
$summaryPath = Join-Path $runRoot "verification_summary.md"
$manifestPath = Join-Path $runRoot "run_manifest.json"

New-Item -ItemType Directory -Force -Path $runRoot, $quickDir, $fullDir, $logDir | Out-Null
$runRoot = (Resolve-Path $runRoot).Path
$quickDir = Join-Path $runRoot "quick"
$fullDir = Join-Path $runRoot "full"
$logDir = Join-Path $runRoot "logs"
$summaryPath = Join-Path $runRoot "verification_summary.md"
$manifestPath = Join-Path $runRoot "run_manifest.json"

$manifest = [ordered]@{
    created_at = (Get-Date).ToString("s")
    mode = $Mode
    repo_root = [string]$repoRoot
    baseline_backend = $BaselineBackend
    candidate_backend = $CandidateBackend
    python_exe = $PythonExe
    godot_exe = $GodotExe
    checkpoint = $checkpointPath
    combat_checkpoint = $combatCheckpointPath
    steps = [ordered]@{}
}

function Save-Manifest {
    ($manifest | ConvertTo-Json -Depth 10) | Set-Content -Path $manifestPath -Encoding UTF8
}

function Resolve-CheckpointPath {
    param(
        [Parameter(Mandatory = $true)][string]$PreferredPath,
        [Parameter(Mandatory = $true)][string[]]$Patterns,
        [switch]$Strict
    )

    if ($PreferredPath -and (Test-Path $PreferredPath)) {
        return (Resolve-Path $PreferredPath).Path
    }
    if ($Strict) {
        throw "Checkpoint not found: $PreferredPath"
    }

    foreach ($pattern in $Patterns) {
        $candidate = Get-ChildItem -Path $repoRoot -Recurse -Filter $pattern -ErrorAction SilentlyContinue |
            Where-Object {
                $_.FullName -notlike "*\STS2AI\Artifacts\verification\*" -and
                $_.FullName -notlike "*\STS2AI\Artifacts\recording\*" -and
                $_.FullName -notlike "*\artifacts\verification\*" -and
                $_.FullName -notlike "*\artifacts\recording\*"
            } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
    }

    throw "Checkpoint not found. Preferred=$PreferredPath Patterns=$($Patterns -join ', ')"
}

function Uses-HeadlessBackend {
    param([Parameter(Mandatory = $true)][string]$Backend)
    return $Backend -like "headless*"
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $false)][string]$WorkingDirectory = $repoRoot
    )

    $stdoutPath = Join-Path $logDir "$Name.stdout.log"
    $stderrPath = Join-Path $logDir "$Name.stderr.log"
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    Write-Host "    $FilePath $($ArgumentList -join ' ')" -ForegroundColor DarkGray
    $proc = Start-Process -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath
    $passed = $proc.ExitCode -eq 0
    $manifest.steps[$Name] = [ordered]@{
        passed = $passed
        exit_code = $proc.ExitCode
        stdout = $stdoutPath
        stderr = $stderrPath
    }
    Save-Manifest
    if ($passed) {
        Write-Host "    PASS" -ForegroundColor Green
    }
    else {
        Write-Host "    FAIL (exit=$($proc.ExitCode))" -ForegroundColor Red
    }
    return $passed
}

if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}
if (-not (Test-Path $GodotExe)) {
    throw "Godot not found: $GodotExe"
}
$headlessDllPath = if ($HeadlessDll) {
    if (-not (Test-Path $HeadlessDll)) {
        throw "Headless DLL not found: $HeadlessDll"
    }
    (Resolve-Path $HeadlessDll).Path
}
else {
    if ((Uses-HeadlessBackend -Backend $BaselineBackend) -or (Uses-HeadlessBackend -Backend $CandidateBackend)) {
        if (-not (Test-Path $defaultHeadlessDll)) {
            throw "Headless DLL not found: $defaultHeadlessDll"
        }
        (Resolve-Path $defaultHeadlessDll).Path
    }
    else {
        $defaultHeadlessDll
    }
}
$checkpointPath = if ($Checkpoint) {
    Resolve-CheckpointPath -PreferredPath $Checkpoint -Patterns @("hybrid_final.pt", "hybrid_*.pt") -Strict
}
else {
    Resolve-CheckpointPath -PreferredPath $defaultCheckpoint -Patterns @("retrieval_final_iter2175.pt", "hybrid_final.pt", "hybrid_*.pt")
}

$combatCheckpointPath = if ($CombatCheckpoint) {
    Resolve-CheckpointPath -PreferredPath $CombatCheckpoint -Patterns @("combat_*.pt") -Strict
}
else {
    Resolve-CheckpointPath -PreferredPath $defaultCombatCheckpoint -Patterns @("retrieval_final_iter2175.pt", "combat_final.pt", "combat_*.pt")
}

$manifest.checkpoint = $checkpointPath
$manifest.combat_checkpoint = $combatCheckpointPath
    $manifest.headless_dll = $headlessDllPath
    $manifest.policy_rollout_seed_count = $PolicyRolloutSeedCount
    $manifest.policy_rollout_trace_count = $PolicyRolloutTraceCount
    $manifest.combat_saveload_seed_count = $CombatSaveLoadSeedCount

Save-Manifest

$overallPassed = $true
$autoLaunchArgs = if ($AutoLaunch) { @("--auto-launch") } else { @() }
$launchContextArgs = @(
    "--repo-root", [string]$repoRoot,
    "--godot-exe", [string]$GodotExe,
    "--headless-dll", [string]$headlessDllPath
)

if ($Mode -in @("Quick", "Both")) {
    $overallPassed = (Invoke-Step -Name "build" -FilePath "dotnet" -ArgumentList @("build", "sts2.csproj", "-c", "Debug")) -and $overallPassed
    $quickTraceSeeds = @("CONSIST_A", "CONSIST_B", "CONSIST_C", "CARD_SCAN_24")
    $quickTraceArgs = @(
        "STS2AI/Python/combat_turn_trace.py",
        "--compare",
        "--backend-a", $BaselineBackend,
        "--backend-b", $CandidateBackend,
        "--port-a", ($BaselinePort + 20),
        "--port-b", ($CandidatePort + 20),
        "--driver-mode", "bidirectional",
        "--max-turns", "12",
        "--max-steps", "120",
        "--report-json", (Join-Path $quickDir "state_trace_parity_report.json"),
        "--seeds"
    ) + $quickTraceSeeds
    $quickTraceArgs += $autoLaunchArgs
    $quickTraceArgs += $launchContextArgs
    $overallPassed = (Invoke-Step -Name "quick_state_trace_parity" -FilePath $PythonExe -ArgumentList $quickTraceArgs) -and $overallPassed

    $quickAuditArgs = @(
        "STS2AI/Python/test_simulator_consistency.py",
        "--test", "audit",
        "--baseline-backend", $BaselineBackend,
        "--candidate-backend", $CandidateBackend,
        "--baseline-port", $BaselinePort,
        "--candidate-port", $CandidatePort,
        "--parity-mode", "bidirectional",
        "--parity-detail", "full",
        "--coverage-enforcement", "advisory",
        "--max-steps", $AuditMaxSteps,
        "--report-json", (Join-Path $quickDir "audit_full_report.json")
    )
    $quickAuditArgs += $autoLaunchArgs
    $quickAuditArgs += $launchContextArgs
    $overallPassed = (Invoke-Step -Name "quick_audit_full" -FilePath $PythonExe -ArgumentList $quickAuditArgs) -and $overallPassed

    $quickSaveLoadArgs = @(
        "STS2AI/Python/verify_save_load.py",
        "--backend", $CandidateBackend,
        "--port", $CandidatePort,
        "--report-json", (Join-Path $quickDir "save_load_report.json")
    )
    $quickSaveLoadArgs += $autoLaunchArgs
    $quickSaveLoadArgs += $launchContextArgs
    $overallPassed = (Invoke-Step -Name "quick_save_load" -FilePath $PythonExe -ArgumentList $quickSaveLoadArgs) -and $overallPassed

    $quickNnArgs = @(
        "STS2AI/Python/nn_backend_parity_audit.py",
        "--checkpoint", $checkpointPath,
        "--combat-checkpoint", $combatCheckpointPath,
        "--baseline-backend", $BaselineBackend,
        "--candidate-backend", $CandidateBackend,
        "--baseline-port", ($BaselinePort + 40),
        "--candidate-port", ($CandidatePort + 40),
        "--include-default-seeds",
        "--max-steps", "120",
        "--report-json", (Join-Path $quickDir "nn_backend_parity_report.json")
    )
    $quickNnArgs += $autoLaunchArgs
    $quickNnArgs += $launchContextArgs
    $overallPassed = (Invoke-Step -Name "quick_nn_backend_parity" -FilePath $PythonExe -ArgumentList $quickNnArgs) -and $overallPassed
}

if ($Mode -in @("Full", "Both")) {
    $fullDisplayArgs = @(
        "STS2AI/Python/test_simulator_consistency.py",
        "--test", "parity",
        "--baseline-backend", $BaselineBackend,
        "--candidate-backend", $CandidateBackend,
        "--baseline-port", ($BaselinePort + 60),
        "--candidate-port", ($CandidatePort + 60),
        "--parity-mode", "bidirectional",
        "--parity-detail", "display",
        "--max-steps", $AuditMaxSteps,
        "--report-json", (Join-Path $fullDir "display_parity_report.json")
    )
    $fullDisplayArgs += $autoLaunchArgs
    $fullDisplayArgs += $launchContextArgs
    [void](Invoke-Step -Name "full_display_parity" -FilePath $PythonExe -ArgumentList $fullDisplayArgs)

    $fullBossArgs = @(
        "STS2AI/Python/diagnostics/boss_outcome_audit.py",
        "--baseline-backend", $BaselineBackend,
        "--candidate-backend", $CandidateBackend,
        "--baseline-port", ($BaselinePort + 80),
        "--candidate-port", ($CandidatePort + 80),
        "--count", $BossAuditCount,
        "--policy", "coverage",
        "--policy", "exit",
        "--policy", "training",
        "--step-cap", "600",
        "--long-cap", "1200",
        "--report-json", (Join-Path $fullDir "boss_outcome_audit.json")
    )
    $fullBossArgs += $autoLaunchArgs
    $fullBossArgs += $launchContextArgs
    $overallPassed = (Invoke-Step -Name "full_boss_outcome_audit" -FilePath $PythonExe -ArgumentList $fullBossArgs) -and $overallPassed

    $fullRewardArgs = @(
        "STS2AI/Python/diagnostics/reward_loop_audit.py",
        "--baseline-backend", $BaselineBackend,
        "--candidate-backend", $CandidateBackend,
        "--baseline-port", ($BaselinePort + 100),
        "--candidate-port", ($CandidatePort + 100),
        "--count", $RewardAuditCount,
        "--policy", "coverage",
        "--policy", "exit",
        "--policy", "training",
        "--max-steps", "600",
        "--report-json", (Join-Path $fullDir "reward_loop_audit.json")
    )
    $fullRewardArgs += $autoLaunchArgs
    $fullRewardArgs += $launchContextArgs
    $overallPassed = (Invoke-Step -Name "full_reward_loop_audit" -FilePath $PythonExe -ArgumentList $fullRewardArgs) -and $overallPassed

    $fullDiscoverArgs = @(
        "STS2AI/Python/test_simulator_consistency.py",
        "--test", "discover",
        "--backend", $CandidateBackend,
        "--port", ($CandidatePort + 120),
        "--discover-count", $DiscoverCount,
        "--max-steps", $AuditMaxSteps,
        "--report-json", (Join-Path $fullDir "discover_report.json")
    )
    $fullDiscoverArgs += $autoLaunchArgs
    $fullDiscoverArgs += $launchContextArgs
    $overallPassed = (Invoke-Step -Name "full_discover" -FilePath $PythonExe -ArgumentList $fullDiscoverArgs) -and $overallPassed

    $fullPolicyRolloutArgs = @(
        "STS2AI/Python/diagnostics/policy_rollout_audit.py",
        "--checkpoint", $checkpointPath,
        "--combat-checkpoint", $combatCheckpointPath,
        "--baseline-backend", $BaselineBackend,
        "--candidate-backend", $CandidateBackend,
        "--baseline-port", ($BaselinePort + 130),
        "--candidate-port", ($CandidatePort + 130),
        "--seed-count", $PolicyRolloutSeedCount,
        "--trace-count", $PolicyRolloutTraceCount,
        "--max-steps", $AuditMaxSteps,
        "--report-json", (Join-Path $fullDir "policy_rollout_audit.json")
    )
    $overallPassed = (Invoke-Step -Name "full_policy_rollout_audit" -FilePath $PythonExe -ArgumentList $fullPolicyRolloutArgs) -and $overallPassed

    $baselineCombatSaveLoadArgs = @(
        "STS2AI/Python/saveload_combat_parity.py",
        "--backend", $BaselineBackend,
        "--port", ($BaselinePort + 170),
        "--seeds", $CombatSaveLoadSeedCount,
        "--auto-launch",
        "--output", (Join-Path $fullDir "baseline_saveload_combat_parity.json")
    )
    $baselineCombatSaveLoadArgs += $launchContextArgs
    [void](Invoke-Step -Name "full_baseline_saveload_combat_parity" -FilePath $PythonExe -ArgumentList $baselineCombatSaveLoadArgs)

    $candidateCombatSaveLoadArgs = @(
        "STS2AI/Python/saveload_combat_parity.py",
        "--backend", $CandidateBackend,
        "--port", ($CandidatePort + 170),
        "--seeds", $CombatSaveLoadSeedCount,
        "--auto-launch",
        "--output", (Join-Path $fullDir "candidate_saveload_combat_parity.json")
    )
    $candidateCombatSaveLoadArgs += $launchContextArgs
    [void](Invoke-Step -Name "full_candidate_saveload_combat_parity" -FilePath $PythonExe -ArgumentList $candidateCombatSaveLoadArgs)

    $trainingAuditArgs = @(
        "STS2AI/Python/diagnostics/training_semantic_audit.py",
        "--checkpoint", $checkpointPath,
        "--combat-checkpoint", $combatCheckpointPath,
        "--baseline-backend", $BaselineBackend,
        "--candidate-backend", $CandidateBackend,
        "--baseline-start-port", ($BaselinePort + 140),
        "--candidate-start-port", ($CandidatePort + 140),
        "--iterations", $TrainingIterations,
        "--num-envs", $TrainingNumEnvs,
        "--episodes-per-iter", $TrainingEpisodesPerIter,
        "--max-episode-steps", $TrainingMaxEpisodeSteps,
        "--ppo-minibatch", $PpoMinibatch,
        "--report-json", (Join-Path $fullDir "training_semantic_audit.json")
    )
    $trainingAuditArgs += $launchContextArgs
    $overallPassed = (Invoke-Step -Name "full_training_semantic_audit" -FilePath $PythonExe -ArgumentList $trainingAuditArgs) -and $overallPassed
}

Invoke-Step -Name "write_summary" -FilePath $PythonExe -ArgumentList @(
    "STS2AI/Python/diagnostics/sim_vs_godot_audit_report.py",
    "--run-root", $runRoot,
    "--output", $summaryPath
) | Out-Null

$manifest.summary_markdown = $summaryPath
$manifest.overall_passed = $overallPassed
Save-Manifest

Write-Host ""
Write-Host "Run root: $runRoot" -ForegroundColor Cyan
Write-Host "Summary : $summaryPath" -ForegroundColor Cyan

if (-not $overallPassed) {
    exit 1
}
