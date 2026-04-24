#!/bin/bash
# One-shot schema parity test: pipe protobuf sim vs HTTP protobuf-JSON spectator.
# Usage: bash STS2AI/bridge/scripts/run_parity_test.sh [SEED] [MAX_STEPS]
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_PORT="${SIM_PORT:-15600}"
SPEC_PORT="${SPEC_PORT:-15526}"
SEED="${1:-CONSIST_A}"
MAX_STEPS="${2:-300}"
REPORT_PATH="$REPO_ROOT/STS2AI/Artifacts/parity/game_bridge/parity_latest.json"

cd "$REPO_ROOT/STS2AI/bridge" || exit 1
python -m game_bridge.parity \
    --real-base-url "http://127.0.0.1:$SPEC_PORT" \
    --sim-port "$SIM_PORT" \
    --seed "$SEED" \
    --max-steps "$MAX_STEPS" \
    --output-path "$REPORT_PATH"
