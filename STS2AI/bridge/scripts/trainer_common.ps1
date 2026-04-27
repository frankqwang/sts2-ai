function Resolve-CommandOrPath {
    param(
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    if (Test-Path -LiteralPath $Value) {
        return (Resolve-Path -LiteralPath $Value).Path
    }

    $command = Get-Command -Name $Value -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }

    return $null
}

function Resolve-GodotExe {
    param(
        [string]$ExplicitPath
    )

    $candidates = @(
        $ExplicitPath,
        $env:STS2_GODOT_EXE,
        "godot4",
        "godot",
        "D:\dev\Godot_v4.5.1-stable_mono_win64\Godot_v4.5.1-stable_mono_win64_console.exe",
        "D:\dev\Godot_v4.5.1-stable_mono_win64\Godot_v4.5.1-stable_mono_win64.exe"
    )

    foreach ($candidate in $candidates) {
        $resolved = Resolve-CommandOrPath -Value $candidate
        if ($null -ne $resolved) {
            return $resolved
        }
    }

    throw "Unable to locate a Godot executable. Pass -GodotExe, set STS2_GODOT_EXE, or put Godot on PATH."
}

function Resolve-PythonExe {
    param(
        [string]$ExplicitPath
    )

    $candidates = @(
        $ExplicitPath,
        $env:STS2_PYTHON_EXE,
        "python",
        "python3",
        "py",
        "C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe"
    )

    foreach ($candidate in $candidates) {
        $resolved = Resolve-CommandOrPath -Value $candidate
        if ($null -ne $resolved) {
            return $resolved
        }
    }

    throw "Unable to locate a Python executable. Pass -PythonExe, set STS2_PYTHON_EXE, or put Python on PATH."
}

function Resolve-SpectatorModInstallDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$GodotExe
    )

    $godotDir = Split-Path -Parent $GodotExe
    if ([string]::IsNullOrWhiteSpace($godotDir)) {
        throw "Unable to resolve Godot install directory from: $GodotExe"
    }

    return (Join-Path $godotDir "mods\sts2_mcp_spectator")
}

function Sync-SpectatorModArtifacts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceDir,
        [Parameter(Mandatory = $true)]
        [string]$GodotExe
    )

    $resolvedSource = (Resolve-Path -LiteralPath $SourceDir -ErrorAction Stop).Path
    $installDir = Resolve-SpectatorModInstallDir -GodotExe $GodotExe
    New-Item -ItemType Directory -Force -Path $installDir | Out-Null

    $requiredFiles = @(
        "sts2_mcp_spectator.dll",
        "sts2_mcp_spectator.json"
    )
    foreach ($name in $requiredFiles) {
        $sourcePath = Join-Path $resolvedSource $name
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            throw "Spectator mod artifact missing: $sourcePath"
        }
        Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $installDir $name) -Force
    }

    Get-ChildItem -LiteralPath $resolvedSource -Filter "*.dll" |
        Where-Object { $_.Name -ne "sts2_mcp_spectator.dll" } |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $installDir $_.Name) -Force
            # STS2 loads mod DLLs into the default AssemblyLoadContext. Some
            # dependency probes use the game executable directory, not the mod
            # subdirectory, so mirror runtime dependencies there as well.
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path (Split-Path -Parent $GodotExe) $_.Name) -Force
            $repoManagedDir = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "..\.godot\mono\temp\bin\Debug"
            $repoManagedDir = [System.IO.Path]::GetFullPath($repoManagedDir)
            if (Test-Path -LiteralPath $repoManagedDir) {
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $repoManagedDir $_.Name) -Force
            }
        }

    $protobufDll = Join-Path $resolvedSource "Google.Protobuf.dll"
    if (Test-Path -LiteralPath $protobufDll) {
        # STS2 calls Assembly.GetTypes() before mod initializers run. Since the
        # spectator mod contains generated protobuf types, preload protobuf as a
        # dependency mod so the default AssemblyLoadContext can resolve it.
        $protobufModId = "Google.Protobuf"
        $protobufModDir = Join-Path (Split-Path -Parent $installDir) $protobufModId
        New-Item -ItemType Directory -Force -Path $protobufModDir | Out-Null
        Copy-Item -LiteralPath $protobufDll -Destination (Join-Path $protobufModDir "$protobufModId.dll") -Force
        $protobufManifest = [ordered]@{
            id = $protobufModId
            name = "Google Protobuf"
            author = "Google"
            description = "Runtime dependency for STS2 MCP Spectator."
            version = "3.28.3"
            has_pck = $false
            has_dll = $true
            affects_gameplay = $false
        }
        $protobufManifest |
            ConvertTo-Json -Depth 4 |
            Set-Content -LiteralPath (Join-Path $protobufModDir "$protobufModId.json") -Encoding utf8
    }

    $optionalFiles = @(
        "README.md"
    )
    foreach ($name in $optionalFiles) {
        $sourcePath = Join-Path $resolvedSource $name
        if (Test-Path -LiteralPath $sourcePath) {
            Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $installDir $name) -Force
        }
    }

    $staleFiles = @(
        "sts2_mcp_spectator.pdb",
        "sts2_mcp_spectator.runtimeconfig.json"
    )
    foreach ($name in $staleFiles) {
        $stalePath = Join-Path $installDir $name
        if (Test-Path -LiteralPath $stalePath) {
            Remove-Item -LiteralPath $stalePath -Force
        }
    }

    return $installDir
}

function Resolve-BaseUrl {
    param(
        [string]$BaseUrl,
        [int]$McpPort = 15526
    )

    if (-not [string]::IsNullOrWhiteSpace($BaseUrl)) {
        return ([string]$BaseUrl).TrimEnd("/")
    }

    return "http://127.0.0.1:$McpPort"
}

function Resolve-McpPortFromBaseUrl {
    param(
        [string]$ResolvedBaseUrl,
        [int]$DefaultPort = 15526
    )

    if ([string]::IsNullOrWhiteSpace($ResolvedBaseUrl)) {
        return $DefaultPort
    }

    try {
        $uri = [System.Uri]$ResolvedBaseUrl
        if ($uri.Port -gt 0) {
            return [int]$uri.Port
        }
    }
    catch {
    }

    return $DefaultPort
}

function Resolve-SingleplayerStateUrl {
    param(
        [string]$ResolvedBaseUrl
    )

    $normalized = [string]$ResolvedBaseUrl
    if ([string]::IsNullOrWhiteSpace($normalized)) {
        return "http://127.0.0.1:15526/api/game_bridge/rpc"
    }

    $normalized = $normalized.TrimEnd("/")
    if ($normalized -match "/api/game_bridge/rpc$") {
        return $normalized
    }
    if ($normalized -match "/api/game_bridge$") {
        return "$normalized/rpc"
    }
    return "$normalized/api/game_bridge/rpc"
}

function Start-ProcessWithEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = "",
        [hashtable]$Environment = @{},
        [switch]$CreateNoWindow,
        [switch]$PassThru
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    if ($null -ne $startInfo.ArgumentList) {
        foreach ($arg in $ArgumentList) {
            [void]$startInfo.ArgumentList.Add([string]$arg)
        }
    }
    else {
        $quotedArgs = @(
            $ArgumentList | ForEach-Object {
                $stringArg = [string]$_
                if ($stringArg -match '[\s"]') {
                    '"' + ($stringArg -replace '"', '\"') + '"'
                }
                else {
                    $stringArg
                }
            }
        )
        $startInfo.Arguments = [string]::Join(" ", $quotedArgs)
    }
    if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
        $startInfo.WorkingDirectory = $WorkingDirectory
    }
    $startInfo.UseShellExecute = $false
    if ($CreateNoWindow) {
        $startInfo.CreateNoWindow = $true
    }

    foreach ($entry in $Environment.GetEnumerator()) {
        $name = [string]$entry.Key
        $value = [string]$entry.Value
        if (-not [string]::IsNullOrWhiteSpace($name)) {
            if ($null -ne $startInfo.Environment) {
                $startInfo.Environment[$name] = $value
            }
            else {
                $startInfo.EnvironmentVariables[$name] = $value
            }
        }
    }

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    [void]$process.Start()
    if ($PassThru) {
        return $process
    }
    return $null
}

function Focus-ProcessWindow {
    param(
        [int]$ProcessId,
        [string[]]$WindowTitles = @(),
        [int]$Attempts = 8,
        [int]$DelayMilliseconds = 350
    )

    if ($ProcessId -le 0) {
        return $false
    }

    try {
        $shell = New-Object -ComObject WScript.Shell
    }
    catch {
        return $false
    }

    for ($i = 0; $i -lt $Attempts; $i++) {
        Start-Sleep -Milliseconds $DelayMilliseconds
        try {
            foreach ($title in $WindowTitles | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) {
                if ($shell.AppActivate($title)) {
                    return $true
                }
            }
        }
        catch {
        }

        try {
            if ($shell.AppActivate($ProcessId)) {
                return $true
            }
        }
        catch {
        }
    }

    return $false
}

function Focus-GodotGameWindow {
    param(
        [int]$ProcessId,
        [int]$Attempts = 12,
        [int]$DelayMilliseconds = 350
    )

    $titles = @(
        "Slay the Spire 2 (DEBUG)",
        "Slay the Spire 2",
        "Godot Engine"
    )

    return Focus-ProcessWindow -ProcessId $ProcessId -WindowTitles $titles -Attempts $Attempts -DelayMilliseconds $DelayMilliseconds
}

function Wait-TrainerEndpoint {
    param(
        [string]$StateUrl = "http://127.0.0.1:15526/api/v2/combat_env/state",
        [int]$Attempts = 60,
        [int]$DelayMilliseconds = 250
    )

    for ($i = 0; $i -lt $Attempts; $i++) {
        try {
            $null = Invoke-RestMethod -Uri $StateUrl -Method Get -TimeoutSec 2
            return $true
        }
        catch {
        }
        Start-Sleep -Milliseconds $DelayMilliseconds
    }

    return $false
}

function Get-TrainerState {
    param(
        [string]$StateUrl = "http://127.0.0.1:15526/api/v2/combat_env/state"
    )

    try {
        return Invoke-RestMethod -Uri $StateUrl -Method Get -TimeoutSec 2
    }
    catch {
        return $null
    }
}

function Get-SingleplayerState {
    param(
        [string]$StateUrl = "http://127.0.0.1:15526/api/game_bridge/rpc"
    )

    try {
        $response = Invoke-RestMethod -Uri $StateUrl -Method Post -Body '{"method":"STATE"}' -ContentType "application/json" -TimeoutSec 2
        return $response.state.state
    }
    catch {
        return $null
    }
}

function Stop-GodotProcesses {
    $processes = Get-Process | Where-Object { $_.ProcessName -like 'Godot*' }
    if ($null -eq $processes -or $processes.Count -eq 0) {
        return @()
    }

    $stoppedIds = @()
    foreach ($process in $processes) {
        $stoppedIds += $process.Id
    }

    $processes | Stop-Process -Force
    Start-Sleep -Milliseconds 1200
    return $stoppedIds
}

function Initialize-IsolatedEditorDataRoot {
    param(
        [string]$AppDataRoot
    )

    if ([string]::IsNullOrWhiteSpace($AppDataRoot)) {
        return @()
    }

    $sourceEditorRoot = Join-Path $env:APPDATA "SlayTheSpire2\editor"
    if (-not (Test-Path -LiteralPath $sourceEditorRoot)) {
        return @()
    }

    $targetEditorRoot = Join-Path $AppDataRoot "SlayTheSpire2\editor"
    $copySpecs = @(
        "1\settings.save",
        "1\settings.save.backup",
        "1\profile.save",
        "1\profile.save.backup",
        "1\profile1\saves\prefs.save",
        "1\profile1\saves\prefs.save.backup",
        "1\profile1\saves\progress.save",
        "1\profile1\saves\progress.save.backup",
        "1\modded\profile1\saves\prefs.save",
        "1\modded\profile1\saves\prefs.save.backup",
        "1\modded\profile1\saves\progress.save",
        "1\modded\profile1\saves\progress.save.backup"
    )

    $copied = @()
    foreach ($relativePath in $copySpecs) {
        $sourcePath = Join-Path $sourceEditorRoot $relativePath
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            continue
        }

        $targetPath = Join-Path $targetEditorRoot $relativePath
        if (Test-Path -LiteralPath $targetPath) {
            $shouldReplace = $false
            if ($relativePath -eq "1\settings.save" -or $relativePath -eq "1\settings.save.backup") {
                try {
                    $existingSettings = Get-Content -LiteralPath $targetPath -Raw -ErrorAction Stop
                    if ($existingSettings -match '"mod_settings"\s*:\s*null') {
                        $shouldReplace = $true
                    }
                }
                catch {
                }
            }

            if (-not $shouldReplace) {
                continue
            }
        }

        $targetDir = Split-Path -Parent $targetPath
        if (-not [string]::IsNullOrWhiteSpace($targetDir)) {
            New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
        }

        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
        $copied += $targetPath
    }

    return $copied
}

function Set-EditorWindowDefaults {
    param(
        [string]$AppDataRoot,
        [string]$Resolution = "1280x720",
        [string]$InstanceId = "",
        [int]$McpPort = 15526,
        [switch]$ForceWindowed
    )

    if ([string]::IsNullOrWhiteSpace($AppDataRoot)) {
        return @()
    }

    $parts = ([string]$Resolution).Split("x", 2)
    if ($parts.Count -ne 2) {
        return @()
    }

    $width = 0
    $height = 0
    if (-not [int]::TryParse($parts[0], [ref]$width)) {
        return @()
    }
    if (-not [int]::TryParse($parts[1], [ref]$height)) {
        return @()
    }

    $windowPosition = $null
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop | Out-Null
        $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
        $screenWidth = [int]$bounds.Width
        $screenHeight = [int]$bounds.Height
        $maxX = [Math]::Max(0, $screenWidth - $width)
        $maxY = [Math]::Max(0, $screenHeight - $height)

        $slotSeed = if (-not [string]::IsNullOrWhiteSpace($InstanceId)) {
            $InstanceId
        } else {
            "port-$McpPort"
        }

        $slotNumber = 0
        if ($slotSeed -match '(\d+)$') {
            $slotNumber = [int]$Matches[1]
        }
        else {
            $slotNumber = [Math]::Abs($slotSeed.GetHashCode())
        }

        $slot = $slotNumber % 8
        $column = [Math]::Floor($slot / 4)
        $row = $slot % 4

        # Center the window on screen for single-instance / spectator use.
        # Multi-instance training slots still get a small offset via $slot.
        $columnStep = 80
        $rowStep = 90
        $centeredX = [Math]::Max(0, [Math]::Floor(($screenWidth - $width) / 2))
        $centeredY = [Math]::Max(0, [Math]::Floor(($screenHeight - $height) / 2))

        $positionX = [Math]::Max(0, [Math]::Min($maxX, $centeredX + ($column * $columnStep)))
        $positionY = [Math]::Max(0, [Math]::Min($maxY, $centeredY + ($row * $rowStep)))

        $windowPosition = [pscustomobject]@{
            X = $positionX
            Y = $positionY
        }
    }
    catch {
    }

    $settingsFiles = @(
        (Join-Path $AppDataRoot "SlayTheSpire2\editor\1\settings.save")
        (Join-Path $AppDataRoot "SlayTheSpire2\editor\1\settings.save.backup")
    )

    $updated = @()
    foreach ($settingsPath in $settingsFiles) {
        if (-not (Test-Path -LiteralPath $settingsPath)) {
            continue
        }

        try {
            $json = Get-Content -LiteralPath $settingsPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
            if ($ForceWindowed) {
                $json.fullscreen = $false
            }
            if ($null -eq $json.window_size) {
                $json | Add-Member -NotePropertyName window_size -NotePropertyValue ([pscustomobject]@{ X = $width; Y = $height }) -Force
            }
            else {
                $json.window_size.X = $width
                $json.window_size.Y = $height
            }
            $json.resize_windows = $false
            $json | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $settingsPath -Encoding utf8
            $updated += $settingsPath
        }
        catch {
        }
    }

    return $updated
}

function Set-EditorAudioDefaults {
    param(
        [string]$AppDataRoot,
        [float]$MasterVolume = 0.0,
        [float]$BgmVolume = 0.0,
        [float]$SfxVolume = 0.0,
        [float]$AmbienceVolume = 0.0
    )

    if ([string]::IsNullOrWhiteSpace($AppDataRoot)) {
        return @()
    }

    $settingsFiles = @(
        (Join-Path $AppDataRoot "SlayTheSpire2\editor\1\settings.save")
        (Join-Path $AppDataRoot "SlayTheSpire2\editor\1\settings.save.backup")
    )

    $updated = @()
    foreach ($settingsPath in $settingsFiles) {
        if (-not (Test-Path -LiteralPath $settingsPath)) {
            continue
        }

        try {
            $json = Get-Content -LiteralPath $settingsPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
            $json.volume_master = [float]$MasterVolume
            $json.volume_bgm = [float]$BgmVolume
            $json.volume_sfx = [float]$SfxVolume
            $json.volume_ambience = [float]$AmbienceVolume
            $json | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $settingsPath -Encoding utf8
            $updated += $settingsPath
        }
        catch {
        }
    }

    return $updated
}

function Clear-EditorRunSaves {
    param(
        [string]$RootPath = ""
    )

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($RootPath)) {
        $candidates += $RootPath
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:APPDATA)) {
        $candidates += (Join-Path $env:APPDATA "SlayTheSpire2\\editor")
    }

    $deleted = @()
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if ([string]::IsNullOrWhiteSpace($candidate) -or -not (Test-Path -LiteralPath $candidate)) {
            continue
        }

        $runSaveFiles = Get-ChildItem -LiteralPath $candidate -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "current_run*" }

        foreach ($file in $runSaveFiles) {
            try {
                Remove-Item -LiteralPath $file.FullName -Force -ErrorAction Stop
                $deleted += $file.FullName
            }
            catch {
            }
        }
    }

    return $deleted
}

function Assert-CleanTrainerPort {
    param(
        [string]$StateUrl = "http://127.0.0.1:15526/api/v2/combat_env/state",
        [switch]$StopExistingGodot
    )

    $existingState = Get-TrainerState -StateUrl $StateUrl
    if ($null -eq $existingState) {
        return
    }

    if ($StopExistingGodot) {
        $stoppedIds = Stop-GodotProcesses
        if ($stoppedIds.Count -gt 0) {
            Write-Host "Stopped existing Godot processes: $($stoppedIds -join ', ')" -ForegroundColor Yellow
        }
        return
    }

    $episodeNumber = $existingState.episode_number
    $encounterId = $existingState.encounter_id
    throw "Another trainer is already listening on $StateUrl (episode=$episodeNumber, encounter=$encounterId). Stop the existing Godot process first or re-run with -StopExistingGodot."
}

function Assert-CleanSingleplayerPort {
    param(
        [string]$StateUrl = "http://127.0.0.1:15526/api/game_bridge/rpc",
        [switch]$StopExistingGodot
    )

    $existingState = Get-SingleplayerState -StateUrl $StateUrl
    if ($null -eq $existingState) {
        return
    }

    if ($StopExistingGodot) {
        $stoppedIds = Stop-GodotProcesses
        if ($stoppedIds.Count -gt 0) {
            Write-Host "Stopped existing Godot processes: $($stoppedIds -join ', ')" -ForegroundColor Yellow
        }
        return
    }

    $stateType = $existingState.state_type
    throw "Another MCP singleplayer API is already listening on $StateUrl (state_type=$stateType). Stop the existing Godot process first or re-run with -StopExistingGodot."
}

function Wait-SingleplayerEndpoint {
    param(
        [string]$StateUrl = "http://127.0.0.1:15526/api/game_bridge/rpc",
        [int]$Attempts = 60,
        [int]$DelayMilliseconds = 250
    )

    for ($i = 0; $i -lt $Attempts; $i++) {
        $state = Get-SingleplayerState -StateUrl $StateUrl
        if ($null -ne $state) {
            return $true
        }
        Start-Sleep -Milliseconds $DelayMilliseconds
    }

    return $false
}

function Wait-SingleplayerRunStarted {
    param(
        [string]$StateUrl = "http://127.0.0.1:15526/api/game_bridge/rpc",
        [int]$Attempts = 600,
        [int]$DelayMilliseconds = 250
    )

    for ($i = 0; $i -lt $Attempts; $i++) {
        $state = Get-SingleplayerState -StateUrl $StateUrl
        if ($null -ne $state -and $state.state_type -ne "menu") {
            return $true
        }
        Start-Sleep -Milliseconds $DelayMilliseconds
    }

    return $false
}
