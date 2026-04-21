#!/bin/bash
# One-shot parity test: Sim vs Spectator
# Usage: bash STS2AI/scripts/run_parity_test.sh [SEED] [MAX_STEPS]
set -u

REPO_ROOT="C:/dev/sts2-ai"
GODOT_EXE="C:/dev/game/Godot_v4.5.1-stable_mono_win64/Godot_v4.5.1-stable_mono_win64_console.exe"
SIM_EXE="$REPO_ROOT/STS2AI/ENV/Sim/HeadlessSim/bin/Debug/net9.0/HeadlessSim.exe"
MOD_SRC="$REPO_ROOT/STS2AI/ENV/SpectatorBridgeMod/bin/Debug/net9.0/sts2_mcp_spectator.dll"
# Real mod path: next to Godot exe, NOT repo mods/
MOD_DEST="C:/dev/game/Godot_v4.5.1-stable_mono_win64/mods/sts2_mcp_spectator"
# Use port 15600 for Sim to avoid conflict with Godot's internal sim hosts
SIM_PORT=15600
SPEC_PORT=15526
SEED="${1:-CONSIST_A}"
MAX_STEPS="${2:-300}"

cleanup() {
    echo "=== Cleanup ==="
    taskkill //F //IM HeadlessSim.exe 2>/dev/null || true
    taskkill //F //IM Godot_v4.5.1-stable_mono_win64_console.exe 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Parity Test: seed=$SEED max_steps=$MAX_STEPS ==="

# 1. Kill stale
echo "[1/7] Kill stale..."
cleanup 2>/dev/null
sleep 2

# 2. Clean saves
echo "[2/7] Clean saves..."
find C:/Users/Administrator -name "current_run.save*" -delete 2>/dev/null || true

# 3. Build all
echo "[3/7] Build..."
cd "$REPO_ROOT"
ERRORS=0
dotnet build sts2.csproj -v q 2>&1 | tail -1
dotnet build STS2AI/ENV/Sim/HeadlessSim/HeadlessSim.csproj -v q 2>&1 | tail -1
dotnet build STS2AI/ENV/SpectatorBridgeMod/sts2_mcp_spectator.csproj -v q \
    -p:STS2AssemblyDir="$REPO_ROOT/.godot/mono/temp/bin/Debug" 2>&1 | tail -1

# 4. Deploy mod to real path (next to Godot exe)
echo "[4/7] Deploy mod..."
mkdir -p "$MOD_DEST"
cp "$MOD_SRC" "$MOD_DEST/"
cp "$REPO_ROOT/STS2AI/ENV/SpectatorBridgeMod/sts2_mcp_spectator.json" "$MOD_DEST/" 2>/dev/null || true

# 5. Verify
echo "[5/7] Verify..."
[ -f "$SIM_EXE" ] || { echo "FAIL: Sim binary missing"; exit 1; }
[ -f "$MOD_DEST/sts2_mcp_spectator.dll" ] || { echo "FAIL: Mod DLL missing"; exit 1; }

# 6. Start backends
# Start Godot FIRST — it may spawn internal sim hosts on 15527.
# Our Sim uses a different port (15600) to avoid conflict.
echo "[6/7] Start Spectator (port $SPEC_PORT)..."
"$GODOT_EXE" --headless --fixed-fps 1000 --path "$REPO_ROOT" -- --mcp-port $SPEC_PORT > /dev/null 2>&1 &
SPEC_PID=$!
sleep 25

# Verify Spectator
curl -s --max-time 5 http://127.0.0.1:$SPEC_PORT/api/v2/full_run_env/state | \
    python -c "import sys,json; print('  Spectator:', json.load(sys.stdin).get('state_type'))" 2>/dev/null \
    || { echo "  FAIL: Spectator not ready"; exit 1; }

# Now start Sim on separate port (after Godot's internal sims are done starting)
echo "  Start Sim (port $SIM_PORT)..."
"$SIM_EXE" --port $SIM_PORT > /dev/null 2>&1 &
SIM_PID=$!
sleep 8

# Check sim is alive
kill -0 $SIM_PID 2>/dev/null || { echo "  FAIL: Sim process died"; exit 1; }
echo "  Sim: running (pid $SIM_PID)"

# 7. Run parity test (no --auto-launch, we manage processes ourselves)
echo "[7/7] Parity test..."
cd "$REPO_ROOT/STS2AI/Python"
python -m networkV2.s7_diagnostics.test_simulator_consistency \
    --test parity \
    --baseline-backend godot-http --baseline-port $SPEC_PORT \
    --candidate-backend headless-pipe --candidate-port $SIM_PORT \
    --seeds $SEED --max-steps $MAX_STEPS \
    --parity-detail summary --parity-mode forward \
    --report-json "$REPO_ROOT/STS2AI/Artifacts/parity_report_latest.json"

echo "Exit code: $?"
