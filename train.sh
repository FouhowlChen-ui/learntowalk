#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYEXE="${EXAM_PYTHON:-python}"
export PYTHONIOENCODING=utf-8

# Training does NOT render; on headless Linux default MUJOCO_GL=disable so
# mujoco skips all GL backend imports (no libEGL / libOSMesa needed).
# Override via `MUJOCO_GL=egl bash train.sh` etc. only if you really need
# real-time rendering (which also requires --flag_rendering and a display).
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    export MUJOCO_GL="${MUJOCO_GL:-disable}"
fi
export MPLBACKEND="${MPLBACKEND:-Agg}"

"$PYEXE" -u -m walk.run_train "$@"
