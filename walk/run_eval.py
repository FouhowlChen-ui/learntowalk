"""EXAM eval entry (modeled after rl_train_leg26_3d.run_eval_leg26).

Stage 1 only (human baseline); no exo / SMAT / extension-peak modules.

Per-checkpoint outputs land in ``trained_models/analyze_results_<step>_<idx>/``.
The following outputs are intentionally disabled in this build:
    - publication_muscle_gait/
    - mee_metrics.json
    - muscle_activation_summary.{csv,png}
    - segmented_muscle_data_r.png
    - segmented_muscle_data_mean_std_r.png
    - segmented_muscle_data_mean_std_ctrl_r.png

Headless servers: ``walk/__init__.py`` defaults ``MUJOCO_GL=disable`` on Linux when
no display is detected so ``import mujoco`` skips EGL. Set ``MUJOCO_GL=osmesa``
or ``egl`` before import when offscreen replay is required.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if os.getcwd() != str(_PROJECT_ROOT):
    os.chdir(str(_PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401
import numpy as np  # noqa: E402

import walk  # noqa: F401, E402

from walk.analyzer.gait_analyze import GaitAnalyzer  # noqa: E402
# from walk.analyzer.gait_analyze import generate_publication_muscle_gait_activation_only  # disabled
from walk.analyzer.gait_data import GaitData  # noqa: E402
from walk.analyzer.gait_evaluate import ImitationGaitEvaluator  # noqa: E402
# from walk.analyzer.leg26_mee_eval import compute_mee_metrics_26  # disabled
# from walk.analyzer.leg26_muscle_summary import analyze_single_muscle_activation_26  # disabled
from walk.analyzer.train_log_analyzer import TrainLogAnalyzer  # noqa: E402
from walk.envs.leg_base import MyoAssistLegBase  # noqa: E402
from walk.train.config import ImitationTrainSessionConfig  # noqa: E402
from walk.utils.checkpoint_data import ImitationTrainCheckpointData  # noqa: E402
from walk.utils.data_types import DictionableDataclass  # noqa: E402
from walk.utils.headless import is_headless  # noqa: E402
from walk.utils.train_log_handler import TrainLogHandler  # noqa: E402


SHOW_PLOT = False
_CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.zip$")


def _print_render_status() -> None:
    headless = is_headless()
    print(
        f"[render] DISPLAY={os.environ.get('DISPLAY', '<unset>')} "
        f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '<unset>')} "
        f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')} "
        f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', '<unset>')}"
    )
    if headless:
        print(
            "[render] headless server: routing to MUJOCO_GL offscreen renderer. "
            "If EGL is unavailable, export MUJOCO_GL=osmesa or use --skip_replay."
        )


def discover_checkpoint_steps_from_disk(log_dir: str) -> List[int]:
    model_dir = os.path.join(log_dir, "trained_models")
    if not os.path.isdir(model_dir):
        return []
    steps: List[int] = []
    for fname in os.listdir(model_dir):
        m = _CHECKPOINT_PATTERN.match(fname)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def _apply_checkpoint_filters(
    steps: List[int],
    lo: Optional[int],
    hi: Optional[int],
    stride: int,
) -> List[int]:
    if lo is not None or hi is not None:
        before = len(steps)
        steps = [s for s in steps if (lo is None or s >= lo) and (hi is None or s <= hi)]
        print(f"[info] checkpoint range [{lo}, {hi}]: {before} -> {len(steps)}")
    st = max(1, int(stride))
    if st > 1 and steps:
        before = len(steps)
        steps = steps[::st]
        print(f"[info] checkpoint stride={st}: {before} -> {len(steps)}")
    return steps


def _resolve_best_eval_step_and_model(log_dir: str) -> Tuple[int, str]:
    bp = os.path.join(log_dir, "best_checkpoint.json")
    if not os.path.isfile(bp):
        raise FileNotFoundError(f"missing best_checkpoint.json: {bp}")
    with open(bp, "r", encoding="utf-8") as f:
        data = json.load(f)
    step = int(data["best_checkpoint"])
    primary = os.path.join(log_dir, "trained_models", f"model_{step}.zip")
    if os.path.isfile(primary):
        return step, primary
    fallback = os.path.join(log_dir, "trained_models", "best_model.zip")
    if os.path.isfile(fallback):
        print(
            f"[info] model_{step}.zip not found; falling back to "
            "trained_models/best_model.zip"
        )
        return step, fallback
    raise FileNotFoundError(
        f"no zip for best step={step} and best_model.zip not found"
    )


def _run_single_evaluation(
    log_handler: TrainLogHandler,
    config,
    evaluate_param: Dict[str, Any],
    ckpt_step: int,
    save_dir: str,
    model_file_path: str,
) -> Optional[str]:
    evaluator: Optional[ImitationGaitEvaluator] = None
    try:
        evaluator = ImitationGaitEvaluator(log_handler, config)
        evaluator.load_reference_data()
        evaluator.initialize_env(enable_offscreen_renderer=False)
        gait_data_path = evaluator.evaluate(
            result_dir=save_dir,
            file_name="gait_evaluated_data.json",
            velocity_mode=MyoAssistLegBase.VelocityMode[
                evaluate_param.get("velocity_mode", "UNIFORM")
            ],
            target_velocity_period=evaluate_param.get("target_velocity_period", 2.0),
            min_target_velocity=evaluate_param["min_target_velocity"],
            max_target_velocity=evaluate_param["max_target_velocity"],
            terminate_when_done=True,
            max_timestep=int(evaluate_param.get("num_timesteps", 600)),
            num_episodes=int(evaluate_param.get("evaluation_episodes", 1)),
            num_timesteps=int(ckpt_step),
            trained_model_path_override=os.path.abspath(model_file_path),
        )
        return gait_data_path
    except Exception as e:
        print(f"   [evaluate] failed: {e}")
        traceback.print_exc()
        return None
    finally:
        if evaluator is not None and evaluator.env is not None:
            try:
                evaluator.env.close()
            except Exception:
                pass


def _eval_stage1_human_only(
    log_handler: TrainLogHandler,
    config,
    evaluate_param: Dict[str, Any],
    ckpt_step: int,
    eval_idx: int,
    model_file_path: str,
    analyze_result_dir: str,
    num_eval_steps: int,
) -> Optional[str]:
    print(f"--- [Leg26 stage1 human baseline] eval checkpoint {ckpt_step} ---")

    gait_data_path = _run_single_evaluation(
        log_handler, config, evaluate_param, ckpt_step,
        analyze_result_dir, model_file_path,
    )

    # --- DISABLED: mee_metrics.json -------------------------------------------
    # try:
    #     compute_mee_metrics_26(
    #         config, model_file_path, analyze_result_dir,
    #         num_steps=num_eval_steps, effective_stage=1, ckpt_step=ckpt_step,
    #     )
    # except Exception as e:
    #     print(f"[warn] Leg26 MEE metrics failed: {e}")
    #     traceback.print_exc()

    # --- DISABLED: muscle_activation_summary.csv / .png -----------------------
    # if gait_data_path and os.path.exists(gait_data_path):
    #     dname = f"analyze_results_{ckpt_step}_{eval_idx:02d}"
    #     try:
    #         analyze_single_muscle_activation_26(
    #             gait_data_path, analyze_result_dir, dname
    #         )
    #     except Exception as e:
    #         print(f"[warn] Leg26 muscle summary failed: {e}")
    #         traceback.print_exc()

    # --- DISABLED: publication_muscle_gait/ -----------------------------------
    # if gait_data_path and os.path.exists(gait_data_path):
    #     try:
    #         gait_pub = GaitData()
    #         gait_pub.read_json_data(gait_data_path)
    #         generate_publication_muscle_gait_activation_only(
    #             analyze_result_dir, gait_pub
    #         )
    #     except Exception as e:
    #         print(f"[warn] publication muscle gait (activation only) failed: {e}")
    #         traceback.print_exc()

    return gait_data_path


def _run_gait_analyzer_plots_stage1(
    gait_analyzer: GaitAnalyzer,
    result_dir: str,
    segmented_ref_data: Dict[str, Any],
    ckpt_step: int,
    eval_idx: int,
) -> None:
    """Stage 1 GaitAnalyzer plots: kinematics, velocity-coloured, sim-ref."""
    gait_analyzer.plot_entire_result(result_dir=result_dir, is_right_foot_based=True)
    gait_analyzer.plot_segmented_kinematics_result(result_dir=result_dir)
    gait_analyzer.plot_left_right_comparison(result_dir=result_dir)

    # --- DISABLED: segmented_muscle_data_r.png + mean_std_*.png ---------------
    # gait_analyzer.plot_segmented_muscle_data(result_dir=result_dir, is_plot_right=True)

    gait_analyzer.joint_angle_by_velocity(result_dir=result_dir)

    if segmented_ref_data:
        try:
            gait_analyzer.plot_right_ref_comparison(result_dir=result_dir)
            gait_analyzer.save_sim_ref_metrics_csv(
                result_dir=result_dir,
                metadata={"checkpoint": ckpt_step, "eval_idx": eval_idx},
            )
        except Exception as e:
            print(f"   [warn] sim-ref comparison plot/metrics failed: {e}")
            traceback.print_exc()


def evaluate_log_dir(
    log_dir: str,
    *,
    checkpoint_steps: Optional[List[int]] = None,
    enable_replay: bool = True,
    checkpoint_min: Optional[int] = None,
    checkpoint_max: Optional[int] = None,
    checkpoint_stride: int = 1,
    checkpoints_from_disk: bool = False,
    best_eval: bool = False,
):
    project_root = Path(__file__).resolve().parents[1]
    log_dir = os.path.abspath(log_dir)
    if os.getcwd() != str(project_root):
        os.chdir(str(project_root))
        print(f"[eval] chdir to project root: {project_root}")

    _print_render_status()

    session_config_path = os.path.join(log_dir, "session_config.json")
    if not os.path.isfile(session_config_path):
        raise SystemExit(f"[error] missing session_config.json: {session_config_path}")

    with open(session_config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)

    for attr in ("reference_data_path", "model_path"):
        val = getattr(config.env_params, attr, None)
        if not val:
            continue
        if os.path.isabs(val) and os.path.exists(val):
            continue
        if not os.path.isabs(val) and os.path.exists(val):
            continue
        candidate = str(project_root / val)
        if os.path.exists(candidate):
            setattr(config.env_params, attr, candidate)

    log_handler = TrainLogHandler(log_dir)
    log_handler.load_log_data(ImitationTrainCheckpointData)

    explicit_model_paths: Optional[Dict[int, str]] = None
    if best_eval:
        if checkpoint_steps:
            raise SystemExit("[error] --best-eval and --checkpoint_steps are mutually exclusive")
        step, zm = _resolve_best_eval_step_and_model(log_dir)
        checkpoint_steps = [step]
        explicit_model_paths = {step: zm}
        src = "best_checkpoint.json (+ best_model.zip fallback)"
    elif checkpoint_steps:
        checkpoint_steps = sorted(int(x) for x in checkpoint_steps)
        src = "explicit --checkpoint_steps"
    elif checkpoints_from_disk:
        checkpoint_steps = discover_checkpoint_steps_from_disk(log_dir)
        src = "disk: trained_models/model_*.zip"
        if not checkpoint_steps:
            print("[warn] no model_*.zip on disk; falling back to train_log.json")
            checkpoint_steps = sorted(
                ld.num_timesteps for ld in log_handler.log_datas
            )
            src = "train_log.json (disk empty fallback)"
    else:
        checkpoint_steps = sorted(ld.num_timesteps for ld in log_handler.log_datas)
        src = "train_log.json"

    checkpoint_steps = _apply_checkpoint_filters(
        checkpoint_steps, checkpoint_min, checkpoint_max, checkpoint_stride
    )

    print(f"[Leg26 info] checkpoint source: {src}")
    print("Will evaluate checkpoints in order:")
    print(checkpoint_steps)

    train_log_analyzer = TrainLogAnalyzer(log_handler)

    segmented_path = project_root / "reference_data" / "segmented.npz"
    segmented_ref_data: Dict[str, Any] = {}
    if os.path.exists(segmented_path):
        segmented_npz = np.load(segmented_path, allow_pickle=True)
        segmented_ref_data = {k: segmented_npz[k] for k in segmented_npz.files}
    else:
        print(f"[warn] segmented reference not found: {segmented_path} (skip sim-ref plots)")

    for ckpt_step in checkpoint_steps:
        print(f"\n========== [Leg26] eval checkpoint {ckpt_step} ==========")

        if explicit_model_paths and ckpt_step in explicit_model_paths:
            model_file_path = explicit_model_paths[ckpt_step]
        else:
            model_file_path = os.path.join(
                log_dir, "trained_models", f"model_{ckpt_step}.zip"
            )
        if not os.path.exists(model_file_path):
            alt = os.path.join(log_dir, "trained_models", f"model_{ckpt_step}")
            if os.path.exists(alt) or os.path.exists(alt + ".zip"):
                model_file_path = alt
            else:
                print(f"[skip] model file not found: {model_file_path}")
                continue

        for eval_idx, evaluate_param in enumerate(config.evaluate_param_list):
            analyze_result_dir = os.path.join(
                log_dir, "trained_models",
                f"analyze_results_{ckpt_step}_{eval_idx:02d}",
            )
            os.makedirs(analyze_result_dir, exist_ok=True)
            num_eval_steps = int(evaluate_param.get("num_timesteps", 600))

            gait_data_path = _eval_stage1_human_only(
                log_handler, config, evaluate_param,
                ckpt_step, eval_idx, model_file_path,
                analyze_result_dir, num_eval_steps,
            )

            try:
                train_log_analyzer.plot_reward(
                    result_dir=analyze_result_dir, show_plot=SHOW_PLOT,
                )
            except Exception as e:
                print(f"[warn] reward curve plot failed: {e}")
                traceback.print_exc()

            if gait_data_path and os.path.exists(gait_data_path):
                gait_evaluator: Optional[ImitationGaitEvaluator] = None
                try:
                    gc.collect()
                    try:
                        plt.close("all")
                    except Exception:
                        pass

                    gait_evaluator = ImitationGaitEvaluator(log_handler, config)
                    gait_evaluator.load_reference_data()
                    gait_evaluator.initialize_env(
                        enable_offscreen_renderer=enable_replay,
                    )

                    if enable_replay:
                        try:
                            gait_evaluator.replay(
                                gait_data_path,
                                os.path.join(analyze_result_dir, "replay.mp4"),
                                cam_distance=evaluate_param.get("cam_distance", 2.5),
                                use_activation_visualization=evaluate_param.get(
                                    "visualize_activation", False
                                ),
                                cam_type=evaluate_param.get("cam_type", "follow"),
                                video_fps=int(config.env_params.control_framerate),
                            )
                        except Exception as e:
                            print(
                                "[warn] replay failed (export MUJOCO_GL=osmesa|egl "
                                "before Python for offscreen MP4, or use "
                                f"--skip_replay): {e}"
                            )
                            traceback.print_exc()

                    gait_data = GaitData()
                    gait_data.read_json_data(gait_data_path)
                    gait_analyzer = GaitAnalyzer(
                        gait_data, segmented_ref_data, SHOW_PLOT
                    )
                    if len(gait_analyzer.get_gait_segment_index(is_right_foot_based=True)) >= 1:
                        _run_gait_analyzer_plots_stage1(
                            gait_analyzer, analyze_result_dir, segmented_ref_data,
                            ckpt_step, eval_idx,
                        )
                    else:
                        print(
                            f"[warn] checkpoint {ckpt_step}: not enough gait cycles to plot."
                        )
                except Exception as e:
                    print(f"[error] replay / gait analysis failed: {e}")
                    traceback.print_exc()
                finally:
                    if gait_evaluator is not None and gait_evaluator.env is not None:
                        try:
                            gait_evaluator.env.close()
                        except Exception:
                            pass

        print(f"========== [Leg26] checkpoint {ckpt_step} done ==========")

    print("\n=== Leg26 batch evaluation finished ===")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "EXAM eval entry (modeled after run_eval_leg26; stage 1 human only). "
            "Headless ready; default imports skip EGL (`MUJOCO_GL=disable`). "
            "Video needs `--skip_replay` or working `egl`/`osmesa`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "log_dir", nargs="?", default=None,
        help="train_session directory (containing session_config.json)",
    )
    parser.add_argument(
        "--checkpoint_steps", type=int, nargs="*", default=None,
        help="evaluate only these steps (mutually exclusive with --best-eval)",
    )
    parser.add_argument(
        "--checkpoint-min", dest="checkpoint_min", type=int, default=None,
    )
    parser.add_argument(
        "--checkpoint-max", dest="checkpoint_max", type=int, default=None,
    )
    parser.add_argument(
        "--checkpoint-stride", dest="checkpoint_stride", type=int, default=1,
    )
    parser.add_argument(
        "--checkpoints-from-disk", dest="checkpoints_from_disk",
        action="store_true",
    )
    parser.add_argument("--skip_replay", action="store_true")
    parser.add_argument(
        "--best-eval", dest="best_eval", action="store_true",
        help="evaluate the best checkpoint listed in best_checkpoint.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = input("Enter train_session log dir: ").strip()
    log_dir = os.path.abspath(os.path.expanduser(log_dir))

    evaluate_log_dir(
        log_dir,
        checkpoint_steps=args.checkpoint_steps,
        enable_replay=not args.skip_replay,
        checkpoint_min=args.checkpoint_min,
        checkpoint_max=args.checkpoint_max,
        checkpoint_stride=args.checkpoint_stride,
        checkpoints_from_disk=args.checkpoints_from_disk,
        best_eval=args.best_eval,
    )


if __name__ == "__main__":
    main()
