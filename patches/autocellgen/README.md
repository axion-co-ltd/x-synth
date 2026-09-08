# Backend patches

The AutoCellGen submodule is pinned to an upstream commit and is never modified
in place in this repository. Everything we change lives here.

| File | What it is |
|---|---|
| `backend.patch` | `git diff` against the pinned upstream commit |
| `xsynth_route.cpp` | A new file this repository owns; copied into `src/` |

`scripts/build_backend.sh` applies both. `git apply --check` runs first, so if
upstream moves the build stops with an error instead of silently compiling
unpatched sources.

## What `backend.patch` changes

| | Change | Why |
|---|---|---|
| B1 | `cmake_minimum_required` 2.8 -> 3.10 | CMake 3.27+ rejects 2.8 |
| B3 | `set(Z3_DIR ...)` made conditional | so `-DZ3_DIR` is respected |
| P1 | `main.cpp` calls `xsynth_route_main()` first; it returns -1 unless `--xsynth-route` or `--xsynth-tiles` is given | the entry point for routing a chosen subset of candidates, and for tile hints |
| P2 | Remove the early break in the routing loop, behind `XSYNTH_LABEL_ALL` | upstream stops at the first M1-routable solution; a dataset needs a label for every candidate |
| P3 | Z3 timeout from `XSYNTH_Z3_TIMEOUT_MS` | upstream fixes it at 60 minutes per candidate |
| P4 | Expose `m1_usage` | the Pareto ranking needs M1 as a number, not a category |
| P6 | IO-error `exit(0)` -> a flag, behind `XSYNTH_NO_EXIT` | a missing access point is normal when routing a narrow tile window |
| P7 | Record `solver_status` | to tell `unsat` apart from `unknown` |
| P8 | Boundary-trace `exit(0)` -> early return | the same failure mode inside `generate_ascii()` |

Every environment variable defaults to upstream behavior, so an unset
environment builds and runs exactly as upstream does.

## Reverting

```bash
git -C third_party/autocellgen apply --reverse patches/autocellgen/backend.patch
rm third_party/autocellgen/MAKE/PLACE_ROUTE/csyn_fp/src/xsynth_route.cpp
```
