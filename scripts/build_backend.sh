#!/usr/bin/env bash
# Build the AutoCellGen placer + router backend.
#
#   ./scripts/build_backend.sh
#
# Needs cmake, g++ and curl. If cmake is missing (this is the only sudo step):
#   sudo apt-get install -y cmake
#
# The submodule is pinned to an upstream commit and every local change lives in
# patches/autocellgen/backend.patch, applied here with `git apply`. That fails
# loudly if upstream moves, instead of silently building unpatched sources.
# patches/autocellgen/xsynth_route.cpp is a new file this repository owns and is
# copied in rather than patched.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/third_party/autocellgen"
BACKEND="$SUB/MAKE/PLACE_ROUTE/csyn_fp"
PATCH="$ROOT/patches/autocellgen/backend.patch"
XSYNTH_SRC="$ROOT/patches/autocellgen/xsynth_route.cpp"
XSYNTH_DST="$BACKEND/src/xsynth_route.cpp"
Z3_PREBUILT="$ROOT/third_party/z3-prebuilt"
Z3_VER="4.8.11"
Z3_DIST="z3-${Z3_VER}-x64-glibc-2.31"
JOBS="$(nproc)"

for tool in cmake g++ curl; do
  command -v "$tool" >/dev/null || {
    echo "$tool not found.  sudo apt-get install -y cmake g++ curl" >&2; exit 1; }
done

# --- 1. submodule ----------------------------------------------------------
if [ ! -f "$BACKEND/CMakeLists.txt" ]; then
  echo "==> initializing the AutoCellGen submodule"
  git -C "$ROOT" submodule update --init third_party/autocellgen
fi

# --- 2. Z3 release binaries ------------------------------------------------
# Avoids a 20-40 minute source build and the gcc 13 compatibility problems.
if [ ! -f "$Z3_PREBUILT/$Z3_DIST/bin/libz3.so" ]; then
  echo "==> downloading the Z3 $Z3_VER release"
  mkdir -p "$Z3_PREBUILT"
  curl -sSL -o "$Z3_PREBUILT/z3.zip" \
    "https://github.com/Z3Prover/z3/releases/download/z3-${Z3_VER}/${Z3_DIST}.zip"
  python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$Z3_PREBUILT/z3.zip" "$Z3_PREBUILT"
fi
[ -f "$Z3_PREBUILT/Z3Config.cmake" ] || { echo "Z3Config.cmake missing" >&2; exit 1; }

# --- 3. apply the backend patch --------------------------------------------
# `git apply --reverse --check` succeeding means the patch is already in place,
# which makes re-running this script a no-op.
if git -C "$SUB" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "==> backend patch already applied"
else
  echo "==> applying backend.patch"
  git -C "$SUB" apply --check "$PATCH" || {
    echo "backend.patch does not apply to $(git -C "$SUB" rev-parse --short HEAD)." >&2
    echo "The submodule is pinned; run 'git submodule update --init' to reset it." >&2
    exit 1; }
  git -C "$SUB" apply "$PATCH"
fi

if ! cmp -s "$XSYNTH_SRC" "$XSYNTH_DST" 2>/dev/null; then
  echo "==> installing xsynth_route.cpp"
  cp "$XSYNTH_SRC" "$XSYNTH_DST"
fi

# --- 4. build ---------------------------------------------------------------
# Upstream hardcodes CMAKE_BUILD_TYPE=Debug. The loop is dominated by Z3 calls,
# so Release is forced here: it directly determines the data-generation cost.
echo "==> building (Release, -j$JOBS)"
cmake -S "$BACKEND" -B "$BACKEND/build" \
  -DZ3_DIR="$Z3_PREBUILT" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  >"$BACKEND/build-configure.log" 2>&1 || {
    echo "configure failed; see $BACKEND/build-configure.log" >&2
    tail -25 "$BACKEND/build-configure.log" >&2; exit 1; }

cmake --build "$BACKEND/build" -j"$JOBS" \
  >"$BACKEND/build-compile.log" 2>&1 || {
    echo "compile failed; see $BACKEND/build-compile.log" >&2
    grep -iE 'error' "$BACKEND/build-compile.log" | head -25 >&2; exit 1; }

BIN="$BACKEND/build/placement"
[ -x "$BIN" ] || { echo "no binary produced: $BIN" >&2; exit 1; }

echo
echo "build complete: $BIN"
echo "   libz3:  $Z3_PREBUILT/$Z3_DIST/bin/libz3.so"
echo
echo "next: LD_LIBRARY_PATH=$Z3_PREBUILT/$Z3_DIST/bin $BIN -i <netlist> -d <style> -o <out>"
