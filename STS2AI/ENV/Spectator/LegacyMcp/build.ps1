<#
.SYNOPSIS
    Builds the STS2_MCP mod for the Slay the Spire 2 source build.

.DESCRIPTION
    Compiles STS2_MCP.dll against the source-build game's assemblies, packs a source-build
    compatible STS2_MCP.pck, and optionally installs both files into the Godot executable's
    mods directory.

.PARAMETER SourceRepoDir
    Path to the Slay the Spire 2 source repo root.
    Falls back to the STS2_SOURCE_REPO environment variable if not specified.

.PARAMETER AssemblyDir
    Direct path to the source-build assembly directory containing sts2.dll.
    Falls back to STS2_ASSEMBLY_DIR if not specified.

.PARAMETER GodotExe
    Path to the Godot Mono executable used to run the source build and pack the .pck.
    Falls back to STS2_GODOT_EXE if not specified.

.PARAMETER Install
    Copy the built DLL and PCK into <godot_exe_dir>\mods after a successful build.

.PARAMETER Configuration
    Build configuration (default: Release).

.EXAMPLE
    .\build.ps1 -SourceRepoDir "C:\Users\Administrator\Desktop\Slay the Spire 2"

.EXAMPLE
    .\build.ps1 -AssemblyDir "C:\Users\Administrator\Desktop\Slay the Spire 2\.godot\mono\temp\bin\Debug"

.EXAMPLE
    .\build.ps1 -GodotExe "C:\dev\game\Godot_v4.5.1-stable_mono_win64\Godot_v4.5.1-stable_mono_win64_console.exe" -Install
#>
param(
    [string]$SourceRepoDir,
    [string]$AssemblyDir,
    [string]$GodotExe,
    [switch]$Install,
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

function Resolve-GodotExe {
    param([string]$ConfiguredPath)

    if ($ConfiguredPath -and (Test-Path $ConfiguredPath)) {
        return $ConfiguredPath
    }

    if ($env:STS2_GODOT_EXE -and (Test-Path $env:STS2_GODOT_EXE)) {
        return $env:STS2_GODOT_EXE
    }

    $defaultGodotDir = "C:\dev\game\Godot_v4.5.1-stable_mono_win64"
    $candidates = @(
        (Join-Path $defaultGodotDir "Godot_v4.5.1-stable_mono_win64_console.exe"),
        (Join-Path $defaultGodotDir "Godot_v4.5.1-stable_mono_win64.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

# --- Resolve assembly directory ---
if (-not $AssemblyDir) {
    $AssemblyDir = $env:STS2_ASSEMBLY_DIR
}
if (-not $AssemblyDir) {
    if (-not $SourceRepoDir) {
        $SourceRepoDir = $env:STS2_SOURCE_REPO
    }
    if (-not $SourceRepoDir) {
        $SourceRepoDir = "C:\Users\Administrator\Desktop\Slay the Spire 2"
    }
    $AssemblyDir = Join-Path $SourceRepoDir ".godot\mono\temp\bin\Debug"
}
if (-not $AssemblyDir) {
    Write-Host @"
ERROR: Assembly directory not specified.

Provide it via parameter or environment variable:
  .\build.ps1 -SourceRepoDir "C:\Users\Administrator\Desktop\Slay the Spire 2"
  .\build.ps1 -AssemblyDir "C:\Users\Administrator\Desktop\Slay the Spire 2\.godot\mono\temp\bin\Debug"

Or set it once in your PowerShell profile:
  `$env:STS2_SOURCE_REPO = "C:\Users\Administrator\Desktop\Slay the Spire 2"
"@ -ForegroundColor Red
    exit 1
}

if (-not (Test-Path (Join-Path $AssemblyDir "sts2.dll"))) {
    Write-Host "ERROR: Could not find sts2.dll in '$AssemblyDir'." -ForegroundColor Red
    Write-Host "Make sure -SourceRepoDir points to the Slay the Spire 2 source repo root, or pass -AssemblyDir directly." -ForegroundColor Red
    exit 1
}

$GodotExe = Resolve-GodotExe -ConfiguredPath $GodotExe
if (-not $GodotExe) {
    Write-Host @"
ERROR: Could not find a Godot executable for packing the source-build mod.

Provide it via:
  .\build.ps1 -GodotExe "C:\path\to\Godot_v4.5.1-stable_mono_win64_console.exe"

Or set it once:
  `$env:STS2_GODOT_EXE = "C:\path\to\Godot_v4.5.1-stable_mono_win64_console.exe"
"@ -ForegroundColor Red
    exit 1
}

# --- Check prerequisites ---
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    Write-Host @"
ERROR: 'dotnet' not found.

Install the .NET 9 SDK from:
  https://dotnet.microsoft.com/download/dotnet/9.0
"@ -ForegroundColor Red
    exit 1
}

# --- Build ---
$scriptDir = $PSScriptRoot
$project = Join-Path $scriptDir "STS2_MCP.csproj"
$outDir = Join-Path (Join-Path $scriptDir "out") "STS2_MCP"
$manifestPath = Join-Path $scriptDir "mod_manifest.json"
$pckPath = Join-Path $outDir "STS2_MCP.pck"
$packerProjectDir = Join-Path (Join-Path $scriptDir "tools") "pck_packer"
$godotModsDir = Join-Path (Split-Path -Path $GodotExe -Parent) "mods"

Write-Host "=== Building STS2_MCP ($Configuration) ===" -ForegroundColor Cyan
Write-Host "Assembly dir   : $AssemblyDir"
Write-Host "Godot exe      : $GodotExe"
Write-Host "Output         : $outDir"
Write-Host ""

dotnet build $project -c $Configuration -o $outDir -p:STS2AssemblyDir="$AssemblyDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Packing STS2_MCP.pck ===" -ForegroundColor Cyan
& $GodotExe --headless --path $packerProjectDir --script (Join-Path $packerProjectDir "pack_mod.gd") -- $manifestPath $pckPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Install) {
    New-Item -ItemType Directory -Force -Path $godotModsDir | Out-Null
    Copy-Item -Force (Join-Path $outDir "STS2_MCP.dll") (Join-Path $godotModsDir "STS2_MCP.dll")
    Copy-Item -Force $pckPath (Join-Path $godotModsDir "STS2_MCP.pck")
}

Write-Host ""
Write-Host "=== Build succeeded ===" -ForegroundColor Green
Write-Host "Source-build mod files:"
Write-Host "  $outDir\STS2_MCP.dll"
Write-Host "  $pckPath"
Write-Host ""
Write-Host "Install target (source build via Godot):"
Write-Host "  $godotModsDir\STS2_MCP.dll"
Write-Host "  $godotModsDir\STS2_MCP.pck"
if (-not $Install) {
    Write-Host ""
    Write-Host "Re-run with -Install to copy both files into the Godot mods directory automatically."
}
