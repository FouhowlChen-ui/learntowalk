import json
import os
import shutil
from datetime import datetime

import numpy as np
import torch

from walk.envs.env_handler import EnvironmentHandler
from walk.utils.data_types import DictionableDataclass


def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def _set_model_learning_rate(model, new_lr):
    new_lr = float(new_lr)
    model.learning_rate = new_lr
    model.lr_schedule = lambda _: new_lr
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = new_lr


def _progress_bar_available() -> bool:
    """SB3's ProgressBarCallback needs both tqdm and rich. Detect once."""
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _get_checkpoint_train_metrics(train_log_handler, ckpt_step):
    selected = None
    for ld in train_log_handler.log_datas:
        if int(ld.num_timesteps) == int(ckpt_step):
            selected = ld
            break
        if int(ld.num_timesteps) <= int(ckpt_step):
            selected = ld
    if selected is None:
        return {}
    return {
        "log_num_timesteps": int(getattr(selected, "num_timesteps", -1)),
        "average_reward_per_episode": float(
            getattr(selected, "average_reward_per_episode", 0.0)
        ),
        "std": float(getattr(selected, "std", 0.0)),
        "value_loss": float(getattr(selected, "value_loss", 0.0)),
    }


def ppo_evaluate_with_rendering(config):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(
        config, is_rendering_on=True, is_evaluate_mode=True
    )
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    obs, info = env.reset()
    for _ in range(config.evaluate_param_list[0]["num_timesteps"]):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        if truncated:
            obs, info = env.reset()

    env.close()


def ppo_train_with_parameters(
    config,
    train_time_step,
    is_rendering_on,
    train_log_handler,
    log_dir,
    segment_timesteps=1_000_000,
    enable_auto_best=True,
    enable_early_stop=False,
    early_stop_patience=3,
    early_stop_min_delta=0.02,
    enable_lr_decay_on_plateau=True,
    lr_decay_patience=2,
    lr_decay_factor=0.7,
    min_lr_ratio=0.2,
    resume_from_checkpoint_path=None,
):
    seed = 1234
    np.random.seed(seed)

    is_resuming = bool(resume_from_checkpoint_path)

    if hasattr(config, "total_timesteps"):
        config.total_timesteps = int(train_time_step)

    env = EnvironmentHandler.create_environment(config, is_rendering_on)
    if is_resuming:
        if not os.path.exists(resume_from_checkpoint_path):
            raise FileNotFoundError(
                f"resume_from_checkpoint_path not found: {resume_from_checkpoint_path}"
            )
        print(f"[Resume] Loading checkpoint: {resume_from_checkpoint_path}")
        model = EnvironmentHandler.get_stable_baselines3_model(
            config, env, trained_model_path=resume_from_checkpoint_path
        )
    else:
        model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)
    session_config_path = os.path.join(log_dir, "session_config.json")
    if is_resuming and os.path.exists(session_config_path):
        snapshot_name = (
            f"session_config_resume_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        )
        with open(
            os.path.join(log_dir, snapshot_name), "w", encoding="utf-8"
        ) as f:
            json.dump(session_config_dict, f, ensure_ascii=False, indent=4)
    else:
        with open(session_config_path, "w", encoding="utf-8") as f:
            json.dump(session_config_dict, f, ensure_ascii=False, indent=4)

    custom_callback = EnvironmentHandler.get_callback(config, train_log_handler)
    initial_lr = float(config.ppo_params.learning_rate)

    best_state = {
        "best_checkpoint": None,
        "best_score": -1e9,
        "no_improve_segments": 0,
        "history": [],
    }
    best_state_path = os.path.join(log_dir, "best_checkpoint.json")

    if is_resuming and os.path.exists(best_state_path):
        try:
            with open(best_state_path, "r", encoding="utf-8") as f:
                prev_best = json.load(f)
            for key in (
                "best_checkpoint",
                "best_score",
                "no_improve_segments",
                "history",
            ):
                if key in prev_best:
                    best_state[key] = prev_best[key]
        except Exception as e:
            print(f"[Resume] Failed to load best_checkpoint.json: {e}")

    trained_timesteps = int(model.num_timesteps) if is_resuming else 0
    ckpt_path = None
    ckpt_step = None
    if is_resuming:
        print(
            f"[Resume] from trained_timesteps={trained_timesteps:,}, "
            f"target={int(train_time_step):,}"
        )

    use_progress_bar = _progress_bar_available()
    if not use_progress_bar:
        print(
            "[Train] note: tqdm/rich not both installed; progress bar disabled. "
            "Run `pip install rich` to enable it."
        )

    while trained_timesteps < train_time_step:
        current_chunk = min(segment_timesteps, train_time_step - trained_timesteps)
        print(
            f"\n[Train] segment: {trained_timesteps} -> "
            f"{trained_timesteps + current_chunk}"
        )
        model.learn(
            reset_num_timesteps=False,
            total_timesteps=current_chunk,
            log_interval=1,
            callback=custom_callback,
            progress_bar=use_progress_bar,
        )
        trained_timesteps = int(model.num_timesteps)

        ckpt_step = int(model.num_timesteps)
        ckpt_path = train_log_handler.get_path2save_model(ckpt_step)
        model.save(ckpt_path)
        print(f"[Train] Saved checkpoint: {ckpt_path}.zip")

        if enable_auto_best:
            train_metrics = _get_checkpoint_train_metrics(
                train_log_handler, ckpt_step
            )
            if not train_metrics:
                continue
            score = train_metrics.get("average_reward_per_episode", 0.0)
            improved = score > (best_state["best_score"] + early_stop_min_delta)
            best_state["history"].append(
                {
                    "checkpoint": int(ckpt_step),
                    "score": float(score),
                    **train_metrics,
                    "improved": bool(improved),
                }
            )

            if improved:
                best_state["best_score"] = float(score)
                best_state["best_checkpoint"] = int(ckpt_step)
                best_state["no_improve_segments"] = 0
                src = ckpt_path + ".zip"
                dst = os.path.join(log_dir, "trained_models", "best_model.zip")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(
                        f"[Select] best updated: ckpt={ckpt_step}, "
                        f"score={score:.4f}"
                    )
            else:
                best_state["no_improve_segments"] += 1
                print(
                    f"[Select] no improve. score={score:.4f}, "
                    f"best={best_state['best_score']:.4f}"
                )
                if (
                    enable_lr_decay_on_plateau
                    and lr_decay_patience > 0
                    and best_state["no_improve_segments"] % lr_decay_patience == 0
                ):
                    current_lr = float(
                        model.policy.optimizer.param_groups[0]["lr"]
                    )
                    min_lr = float(initial_lr * min_lr_ratio)
                    new_lr = max(min_lr, current_lr * lr_decay_factor)
                    if new_lr < current_lr:
                        _set_model_learning_rate(model, new_lr)
                        print(
                            f"[LR Decay] {current_lr:.8f} -> {new_lr:.8f}"
                        )

            with open(best_state_path, "w", encoding="utf-8") as f:
                json.dump(best_state, f, ensure_ascii=False, indent=2)

            if (
                enable_early_stop
                and best_state["no_improve_segments"] >= early_stop_patience
            ):
                print(
                    f"[Early Stop] no_improve_segments="
                    f"{best_state['no_improve_segments']}"
                )
                break

    env.close()

    best_path = os.path.join(log_dir, "trained_models", "best_model.zip")
    if not os.path.exists(best_path) and ckpt_path is not None:
        last_ckpt = ckpt_path + ".zip"
        if os.path.exists(last_ckpt):
            shutil.copy2(last_ckpt, best_path)

    del model
    torch.cuda.empty_cache()

    print("Training finished.")
    return best_path
