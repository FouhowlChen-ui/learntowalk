""""""

import json
import os
from typing import Optional

import numpy as np

from walk.envs.env_handler import EnvironmentHandler
from walk.envs.leg26_estimator import MetabolicEnergyEstimator26


def _apply_eval_policy_setup(model, effective_stage):
    """"""
    return None


def _apply_eval_schedule_progress(config, env, effective_stage, ckpt_step):
    """"""
    return None


def compute_mee_metrics_26(
    config,
    model_path: str,
    save_dir: str,
    num_steps: int = 500,
    effective_stage: Optional[int] = None,
    ckpt_step: Optional[int] = None,
) -> Optional[dict]:
    """"""
    original_num_envs = config.env_params.num_envs
    try:
        config.env_params.num_envs = 1
        env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    except Exception as e:
        print(f"   [MEE26] env creation failed: {e}")
        config.env_params.num_envs = original_num_envs
        return None

    try:
        model = EnvironmentHandler.get_stable_baselines3_model(config, env)
        model = model.load(model_path, env=env)
    except Exception as e:
        print(f"   [MEE26] model load failed: {e}")
        env.close()
        config.env_params.num_envs = original_num_envs
        return None

    if effective_stage is not None:
        _apply_eval_policy_setup(model, effective_stage)
        if ckpt_step is not None:
            _apply_eval_schedule_progress(config, env, effective_stage, ckpt_step)

    env_inner = env.unwrapped if hasattr(env, "unwrapped") else env
    nu = int(env_inner.sim.model.nu)
    na = int(env_inner.sim.model.na)
    #
    mee_muscles = min(nu, 26)

    mee_alpha = float(getattr(config.env_params, "mee_alpha", 1.5))
    mee_beta = float(getattr(config.env_params, "mee_beta", 1.0))
    estimator = MetabolicEnergyEstimator26(
        num_muscles=mee_muscles, alpha=mee_alpha, beta=mee_beta
    )

    dt = (
        env_inner.dt
        if hasattr(env_inner, "dt")
        else 1.0 / float(getattr(config.env_params, "control_framerate", 30))
    )
    body_mass = float(np.sum(env_inner.sim.model.body_mass))

    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    mee_rates = []
    #
    positive_powers = []
    negative_powers = []
    total_distance = 0.0
    prev_x = 0.0
    episode_count = 0

    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        if len(step_result) == 5:
            new_obs, _reward, terminated, truncated, _info = step_result
            done = terminated or truncated
        else:
            new_obs, _reward, done, _info = step_result

        take = min(na, mee_muscles)
        muscle_act = env_inner.sim.data.act[:take].copy()
        if take < mee_muscles:
            muscle_act = np.pad(muscle_act, (0, mee_muscles - take))
        rate = estimator.compute_instantaneous_rate(muscle_act, dt)
        mee_rates.append(rate)

        #
        positive_powers.append(0.0)
        negative_powers.append(0.0)

        try:
            cur_x = float(env_inner.sim.data.joint("pelvis_tx").qpos[0])
            total_distance += abs(cur_x - prev_x)
            prev_x = cur_x
        except Exception:
            pass

        obs = new_obs

        is_done_real = (
            isinstance(done, (np.ndarray, list))
            and len(done) > 0
            and bool(done[0])
        ) or (
            not isinstance(done, (np.ndarray, list)) and bool(done)
        )
        if is_done_real:
            episode_count += 1
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            prev_x = 0.0

    env.close()
    config.env_params.num_envs = original_num_envs

    cot = (
        estimator.compute_cot(total_distance, body_mass)
        if total_distance > 0.01
        else 0.0
    )
    metrics = {
        "num_muscles": mee_muscles,
        "num_actuators": nu,
        "mee_cumulative": float(estimator.cumulative_mee),
        "mee_rate_mean": float(np.mean(mee_rates)) if mee_rates else 0.0,
        "mee_rate_std": float(np.std(mee_rates)) if mee_rates else 0.0,
        "cot": float(cot),
        "total_distance": float(total_distance),
        "body_mass": float(body_mass),
        "num_steps": int(num_steps),
        "num_episodes": int(episode_count),
    }

    if effective_stage == 1:
        metrics["schema_note"] = "leg26_stage1_human_only (no exo mechanical power)"
    else:
        #
        metrics["mpp_mean"] = 0.0
        metrics["mnp_mean"] = 0.0
        metrics["schema_note"] = (
            f"leg26_stage{effective_stage}_exo_torque_unavailable_in_baseline"
        )

    metrics_path = os.path.join(save_dir, "mee_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(
        f"   [MEE26] CoT={cot:.4f}, MEE_cum={estimator.cumulative_mee:.2f}, "
        f"mee_muscles={mee_muscles} nu={nu} -> {metrics_path}"
    )
    return metrics
