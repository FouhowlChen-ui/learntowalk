#!/usr/bin/env bash
# EXAM · 一键对单次训练会话跑评估（walk.run_eval）
# Linux / macOS / Git Bash。
#
#   cd EXAM && bash eval_one.sh
#
# 无显示器时：
# - 若未导出 EXAM_SKIP_REPLAY：会询问是否生成 replay.mp4；选「是」或 EXAM_SKIP_REPLAY=0 时，脚本在 MUJOCO_GL 仍为 disable
#   会自动改为 osmesa 再调用 Python。
# - 仍需系统装好 OSMesa（如 Ubuntu: apt install -y libosmesa6），或在有 EGL 的机器上事先 export MUJOCO_GL=egl。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYEXE="${EXAM_PYTHON:-python}"
export PYTHONIOENCODING=utf-8

HEADLESS=0
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  HEADLESS=1
fi

if [ "$HEADLESS" = "1" ]; then
  if [ -z "${MUJOCO_GL:-}" ]; then
    export MUJOCO_GL=disable
  fi
  case "${MUJOCO_GL}" in
    egl)
      export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
      ;;
    osmesa)
      export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
      ;;
    disable)
      unset PYOPENGL_PLATFORM 2>/dev/null || true
      ;;
  esac
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "=== EXAM · run_eval（最佳 checkpoint） ==="

echo "请输入训练会话目录（内需 session_config.json、trained_models、best_checkpoint.json 等）。"
echo "示例：…/results/train_session_20260515-181646"
echo "留空表示当前目录。"
read -rp "会话目录路径: " RAW

RAW="${RAW:-.}"
RAW="${RAW#\"}"
RAW="${RAW%\"}"
RAW="${RAW#\'}"
RAW="${RAW%\'}"

if [ ! -d "$RAW" ]; then
  echo "[错误] 目录不存在: $RAW"
  exit 1
fi

SESSION_DIR="$(cd "$RAW" && pwd)"

if [ ! -f "${SESSION_DIR}/session_config.json" ]; then
  echo "[错误] 未找到 session_config.json：${SESSION_DIR}"
  exit 1
fi

if [ ! -f "${SESSION_DIR}/best_checkpoint.json" ]; then
  echo "[错误] 未找到 best_checkpoint.json（使用 --best-eval 时需要）：${SESSION_DIR}"
  exit 1
fi

# ---------- 回放视频（replay.mp4）：含 EXAM_SKIP_REPLAY / MUJOCO_GL 协调 ----------
SKIP_REPLAY=0
if [ "$HEADLESS" = "1" ]; then
  if [ ! -z "${EXAM_SKIP_REPLAY+x}" ]; then
    # 已由环境导出时尊重用户
    if [ "${EXAM_SKIP_REPLAY}" = "1" ] || [ "${EXAM_SKIP_REPLAY,,}" = "true" ]; then
      SKIP_REPLAY=1
    fi
    echo ""
    echo "(已使用环境变量 EXAM_SKIP_REPLAY=${EXAM_SKIP_REPLAY})"
  else
    echo ""
    echo "当前无显示器，MUJOCO_GL=${MUJOCO_GL}"
    echo "  - replay.mp4 需要离屏渲染：推荐系统安装 OSMesa（如 Ubuntu: sudo apt install -y libosmesa6）；"
    echo "  - 或使用 GPU EGL: 先 export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl（若曾因 EGL 报错 import，请不要用 egl）。"
    read -rp "是否生成回放视频 replay.mp4？[y/N] " WANT_REPLAY
    case "${WANT_REPLAY}" in
      [yY]|[yY][eE][sS]) SKIP_REPLAY=0 ;;
      *) SKIP_REPLAY=1 ;;
    esac
  fi
else
  # 有桌面：默认生成视频（除非明确要求跳过）
  if [ ! -z "${EXAM_SKIP_REPLAY+x}" ] && { [ "${EXAM_SKIP_REPLAY}" = "1" ] || [ "${EXAM_SKIP_REPLAY,,}" = "true" ]; }; then
    SKIP_REPLAY=1
  fi
fi

# disable 无法在 MuJoCo 里产出帧录像：只要不跳过回放就从 disable 切换到 osmesa
if [ "$SKIP_REPLAY" = "0" ] && [ "${MUJOCO_GL}" = "disable" ]; then
  echo "[eval_one] 为生成 replay.mp4，已将 MUJOCO_GL 设为 osmesa（需 libOSMesa，如 Ubuntu: libosmesa6）。"
  export MUJOCO_GL=osmesa
  export PYOPENGL_PLATFORM=osmesa
fi

echo ""
echo "会话目录 : ${SESSION_DIR}"
echo "模式    : --best-eval（最优 checkpoint）"
if [ "$SKIP_REPLAY" = "1" ]; then
  echo "回放视频: 跳过（--skip_replay）"
else
  echo "回放视频: 生成（MUJOCO_GL=${MUJOCO_GL} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-<unset>})"
fi
echo ""

EVAL_ARGS=( -u -m walk.run_eval "${SESSION_DIR}" --best-eval )
if [ "$SKIP_REPLAY" = "1" ]; then
  EVAL_ARGS+=( --skip_replay )
fi

"$PYEXE" "${EVAL_ARGS[@]}"

echo ""
echo "完成。分析结果一般在：${SESSION_DIR}/trained_models/analyze_results_<步数>_00/"
