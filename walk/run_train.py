import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if os.getcwd() != str(_PROJECT_ROOT):
    os.chdir(str(_PROJECT_ROOT))

import walk  # noqa: F401  E402

from walk.envs.env_handler import EnvironmentHandler  # noqa: E402
from walk.train.config import ImitationTrainSessionConfig  # noqa: E402
from walk.train.ppo_train import (  # noqa: E402
    ppo_evaluate_with_rendering,
    ppo_train_with_parameters,
)
from walk.utils import train_log_handler  # noqa: E402
from walk.utils.checkpoint_data import ImitationTrainCheckpointData  # noqa: E402


DEFAULT_RESULTS_ROOT = str(_PROJECT_ROOT / "results")
_CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.zip$")


def _build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        default=str(_PROJECT_ROOT / "configs" / "leg26_baseline.json"),
    )
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--segment_timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--flag_auto_best",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--flag_early_stop",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.02)
    parser.add_argument(
        "--flag_lr_decay_on_plateau",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lr_decay_patience", type=int, default=2)
    parser.add_argument("--lr_decay_factor", type=float, default=0.7)
    parser.add_argument("--min_lr_ratio", type=float, default=0.2)

    parser.add_argument(
        "--flag_rendering",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--flag_realtime_evaluate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    resume_group = parser.add_argument_group("resume")
    resume_group.add_argument(
        "--auto_resume", action=argparse.BooleanOptionalAction, default=False
    )
    resume_group.add_argument("--resume_from_dir", type=str, default=None)
    resume_group.add_argument("--resume_checkpoint_step", type=int, default=None)

    return parser


def _find_latest_train_session(results_root: str):
    if not os.path.isdir(results_root):
        return None
    candidates = []
    for name in os.listdir(results_root):
        path = os.path.join(results_root, name)
        if name.startswith("train_session_") and os.path.isdir(path):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _list_checkpoints(session_dir: str):
    model_dir = os.path.join(session_dir, "trained_models")
    if not os.path.isdir(model_dir):
        return []
    found = []
    for fname in os.listdir(model_dir):
        match = _CHECKPOINT_PATTERN.match(fname)
        if not match:
            continue
        step = int(match.group(1))
        found.append((step, os.path.join(model_dir, fname)))
    found.sort(key=lambda x: x[0])
    return found


def _resolve_resume_checkpoint(session_dir: str, requested_step):
    found = _list_checkpoints(session_dir)
    if not found:
        raise FileNotFoundError(
            f"No model_<step>.zip in {session_dir}/trained_models/"
        )
    if requested_step is not None:
        for step, path in found:
            if step == requested_step:
                return path, step
        raise FileNotFoundError(
            f"model_{requested_step}.zip not found. Available: {[s for s, _ in found]}"
        )
    return found[-1][1], found[-1][0]


def _restore_train_log_into_handler(handler):
    if not os.path.exists(handler.log_path):
        return
    try:
        handler.load_log_data(ImitationTrainCheckpointData)
        print(
            f"[Resume] loaded {len(handler.log_datas)} log entries from "
            f"{handler.log_path}"
        )
    except Exception as e:
        print(f"[Resume] failed to load train_log.json: {e}")


def _print_run_banner(config, args):
    ep = config.env_params
    terrain_type = getattr(ep, "terrain_type", "flat") or "flat"
    terrain_params = getattr(ep, "terrain_params", "") or ""
    resample = bool(getattr(ep, "terrain_resample_per_reset", False))

    if terrain_type == "flat":
        terrain_desc = "flat (default)"
    elif terrain_type == "slope":
        terrain_desc = f"slope (params='{terrain_params}')"
    elif terrain_type == "slope_random":
        terrain_desc = (
            f"slope_random (params='{terrain_params}', "
            f"resample_per_reset={resample})"
        )
    elif terrain_type == "random":
        terrain_desc = f"random (amplitude='{terrain_params}')"
    elif terrain_type == "harmonic_sinusoidal":
        terrain_desc = f"harmonic_sinusoidal (params='{terrain_params}')"
    else:
        terrain_desc = f"{terrain_type} (params='{terrain_params}')"

    print("=" * 70)
    print(f"  config_file        : {args.config_file_path}")
    print(f"  total_timesteps    : {config.total_timesteps:,}")
    print(f"  num_envs           : {config.env_params.num_envs}")
    print(f"  device             : {config.ppo_params.device}")
    print(f"  terrain            : {terrain_desc}")
    print(f"  results_root       : {DEFAULT_RESULTS_ROOT}")
    print("=" * 70)


def main():
    args = _build_parser().parse_args()

    from walk.utils.headless import is_headless

    if is_headless() and (args.flag_rendering or args.flag_realtime_evaluate):
        print(
            "[run_train] WARN: headless server detected; "
            "--flag_rendering / --flag_realtime_evaluate require a display. "
            "Both flags forced to False."
        )
        args.flag_rendering = False
        args.flag_realtime_evaluate = False

    config = EnvironmentHandler.get_session_config_from_path(
        args.config_file_path, ImitationTrainSessionConfig
    )

    config.total_timesteps = int(config.total_timesteps)
    config.env_params.num_envs = int(config.env_params.num_envs)

    if args.total_timesteps is not None:
        config.total_timesteps = int(args.total_timesteps)
    if args.num_envs is not None:
        config.env_params.num_envs = int(args.num_envs)

    _print_run_banner(config, args)

    if args.flag_realtime_evaluate:
        ppo_evaluate_with_rendering(config)
        return

    resume_session_dir = None
    if args.resume_from_dir:
        resume_session_dir = os.path.abspath(args.resume_from_dir)
        if not os.path.isdir(resume_session_dir):
            raise FileNotFoundError(
                f"--resume_from_dir not found: {resume_session_dir}"
            )
    elif args.auto_resume:
        latest = _find_latest_train_session(DEFAULT_RESULTS_ROOT)
        if latest is None:
            raise FileNotFoundError(
                f"--auto_resume but no train_session_* under {DEFAULT_RESULTS_ROOT}"
            )
        resume_session_dir = os.path.abspath(latest)
        print(f"[Resume] auto_resume picked: {resume_session_dir}")

    if resume_session_dir is not None:
        ckpt_path, ckpt_step = _resolve_resume_checkpoint(
            resume_session_dir, args.resume_checkpoint_step
        )
        log_dir = resume_session_dir
        print(f"[Resume] log_dir: {log_dir}, checkpoint: {ckpt_path}")
        resume_from_checkpoint_path = ckpt_path
    else:
        log_dir = os.path.join(
            DEFAULT_RESULTS_ROOT,
            f"train_session_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        resume_from_checkpoint_path = None

    os.makedirs(log_dir, exist_ok=True)
    handler = train_log_handler.TrainLogHandler(log_dir)
    if resume_from_checkpoint_path is not None:
        _restore_train_log_into_handler(handler)

    ppo_train_with_parameters(
        config,
        train_time_step=config.total_timesteps,
        is_rendering_on=args.flag_rendering,
        train_log_handler=handler,
        log_dir=log_dir,
        segment_timesteps=args.segment_timesteps,
        enable_auto_best=args.flag_auto_best,
        enable_early_stop=args.flag_early_stop,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        enable_lr_decay_on_plateau=args.flag_lr_decay_on_plateau,
        lr_decay_patience=args.lr_decay_patience,
        lr_decay_factor=args.lr_decay_factor,
        min_lr_ratio=args.min_lr_ratio,
        resume_from_checkpoint_path=resume_from_checkpoint_path,
    )


if __name__ == "__main__":
    main()
