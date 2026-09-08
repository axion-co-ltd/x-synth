#!/usr/bin/env bash
# Run the pipeline on an existing work/ tree and check it against
# data/expected.json.
#
#   ./scripts/reproduce.sh
#
# Expects the labelling runs to be present under work/ already; README.md
# documents how to generate them. Takes tens of minutes, machine depending.
# EPOCHS can be lowered for a smoke run, but the thresholds assume the default.
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-30}
LOG=${LOG:-work/reproduce.log}
RUNS=(work/run_r0_train work/run_r0_test work/run_r0_cross)

python3 -c 'import torch' 2>/dev/null || {
  echo "torch is required; see \"0. Prerequisites\" in README.md" >&2; exit 1; }

missing=0
for d in "${RUNS[@]}"; do
  [ -d "$d" ] || { echo "missing $d" >&2; missing=1; }
done
[ -f work/tiles_r0.jsonl ] || { echo "missing work/tiles_r0.jsonl" >&2; missing=1; }
if [ "$missing" -ne 0 ]; then
  echo >&2
  echo "The labelling runs are not present. They are derived output and are not" >&2
  echo "shipped; see \"2. Generate the data\" in README.md for how to produce them." >&2
  exit 1
fi

# Extra arguments are passed through, so e.g. `./scripts/reproduce.sh --device cuda`
# reaches run_xsynth.py.
python3 scripts/run_xsynth.py "${RUNS[@]}" \
    --preset r0 --tiles work/tiles_r0.jsonl \
    --epochs "$EPOCHS" "$@" 2>&1 | tee "$LOG"

echo
python3 scripts/verify_repro.py "$LOG"
