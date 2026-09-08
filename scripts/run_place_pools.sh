#!/usr/bin/env bash
# Placement-only sweep: candidate pool sizes for all 180 library cells (no routing).
#
#   ./scripts/run_place_pools.sh          # in the tmux session xsynth-pools
#
# OPTIONAL. Not needed to reproduce the published numbers: the `r0_*` presets
# already name their cells. This sweep is how they were chosen.
#
# Runs three configurations over everything. The digit in the name is the
# `RELAXATION` value:
#   R0 (`place_only.style`)      minimum width only; the upstream default
#   R1 (`place_only_rx1.style`)  up to width + 1
#   R2 (`place_only_rx2.style`)  up to width + 2
#
# Why placement only: each label costs one router invocation, but "how many
# candidates does this cell produce" is answered by placement alone. A cell with
# a single candidate has nothing to rank, so cell selection starts from here.
#
# Measured: R0 3 min, R1 20 min, R2 70 min at 3-way parallelism. Most cells
# finish in under a second; `SDFLx3` alone takes 45 min at R2.
#
# Results accumulate per cell in JSONL, so an interrupted run keeps what it had
# and a rerun skips the cells already done.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="work/place_pools"
SESSION="${XSYNTH_SESSION:-xsynth-pools}"
JOBS="${XSYNTH_JOBS:-3}"

if [ "${XSYNTH_INNER:-0}" != "1" ]; then
  command -v tmux >/dev/null || {
    echo "tmux not found.  sudo apt-get install -y tmux" >&2; exit 1; }
  tmux has-session -t "$SESSION" 2>/dev/null && {
    echo "session already exists: tmux kill-session -t $SESSION" >&2; exit 1; }
  mkdir -p "$OUT"
  tmux new-session -d -s "$SESSION" \
    "XSYNTH_INNER=1 '$ROOT/scripts/run_place_pools.sh' 2>&1 | tee '$ROOT/$OUT/run.log'"
  echo "started tmux session: $SESSION"
  echo "   tmux attach -t $SESSION"
  echo "   results: $OUT/all_rel0.jsonl (one JSON object per cell)"
  exit 0
fi

run() {                       # run <tag> <style> <output file> <per-cell cap>
  echo; echo "=== $1 · $(basename "$2") ==="
  python3 scripts/place_pools.py --min-tr 0 --style "$2" \
      --out "$OUT/$3" --jobs "$JOBS" --timeout "$4"
}

run R0 configs/place_only.style     all_rel0.jsonl 1800
run R1 configs/place_only_rx1.style all_rx1.jsonl  3600
run R2 configs/place_only_rx2.style all_rx2.jsonl  3600

echo; echo "=== all done ==="
wc -l "$OUT"/all_rel0.jsonl "$OUT"/all_rx1.jsonl "$OUT"/all_rx2.jsonl
