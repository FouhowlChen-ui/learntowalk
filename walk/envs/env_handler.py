import json
import os

import numpy as np
import stable_baselines3
from myosuite.utils import gym
from stable_baselines3.common.vec_env import SubprocVecEnv

from walk.train.config import ImitationTrainSessionConfig, TrainSessionConfigBase
from walk.utils.data_types import DictionableDataclass


class EnvironmentHandler:
    @staticmethod
    def create_environment(
        config,
        is_rendering_on: bool,
        is_evaluate_mode: bool = False,
        extra_gym_kwargs: dict | None = None,
    ):
        if is_rendering_on:
            from walk.utils.headless import is_headless

            if is_headless():
                print(
                    "[env_handler] WARN: headless server detected (no DISPLAY); "
                    "interactive on-screen rendering disabled. "
                    "Use offscreen replay (run_eval.py) for video output."
                )
                is_rendering_on = False

        ref_data_dict = EnvironmentHandler.load_reference_data(config)

        gym_make_args = {
            "seed": config.env_params.seed,
            "model_path": config.env_params.model_path,
            "env_params": config.env_params,
            "is_evaluate_mode": is_evaluate_mode,
        }
        if extra_gym_kwargs:
            gym_make_args.update(extra_gym_kwargs)

        if ref_data_dict is not None:
            gym_make_args["reference_data"] = ref_data_dict

        def _make_env(env_id, make_args):
            def _init():
                return gym.make(env_id, **make_args).unwrapped

            return _init

        if is_rendering_on or config.env_params.num_envs == 1:
            env = gym.make(
                config.env_params.env_id, **gym_make_args
            ).unwrapped
            if is_rendering_on:
                env.mujoco_render_frames = True
            config.env_params.num_envs = 1
            config.ppo_params.n_steps = config.ppo_params.batch_size
        else:
            env = SubprocVecEnv(
                [
                    _make_env(config.env_params.env_id, gym_make_args)
                    for _ in range(config.env_params.num_envs)
                ]
            )
        return env

    @staticmethod
    def load_reference_data(config):
        print("=" * 60)
        if not hasattr(config.env_params, "reference_data_path"):
            print("No reference data path provided.")
            print("=" * 60)
            return None

        if not config.env_params.reference_data_path:
            print("No reference data path provided.")
            print("=" * 60)
            return None
        print(f"Loading reference data from {config.env_params.reference_data_path}")
        print("=" * 60)
        if config.env_params.reference_data_path.endswith(".npz"):
            ref_data_npz = np.load(
                config.env_params.reference_data_path, allow_pickle=True
            )
            ref_data_dict = {
                key: ref_data_npz[key].item() for key in ref_data_npz.files
            }
        elif config.env_params.reference_data_path.endswith(".json"):
            with open(
                config.env_params.reference_data_path, "r", encoding="utf-8"
            ) as f:
                ref_data_dict = json.load(f)
        else:
            raise ValueError(
                "Unsupported file format. Use .npz or .json."
            )

        if "resampled_series_data" not in ref_data_dict:
            ref_data_dict["resampled_series_data"] = {}
            for key in ref_data_dict["series_data"].keys():
                original_data_length = len(ref_data_dict["series_data"][key])
                original_sample_rate = ref_data_dict["metadata"]["sample_rate"]
                original_x = np.linspace(
                    0, original_data_length - 1, original_data_length
                )

                new_sample_rate = config.env_params.control_framerate
                new_length = int(
                    original_data_length * new_sample_rate / original_sample_rate
                )
                new_x = np.linspace(0, original_data_length - 1, new_length)
                ref_data_dict["series_data"][key] = np.interp(
                    new_x, original_x, ref_data_dict["series_data"][key]
                )
                ref_data_dict["metadata"]["resampled_data_length"] = new_length
                ref_data_dict["metadata"]["resampled_sample_rate"] = new_sample_rate

        return ref_data_dict

    @staticmethod
    def get_session_config_from_path(config_path, class_type):
        print(f"Loading config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
            session_config = DictionableDataclass.create(class_type, config_dict)
        return session_config

    @staticmethod
    def get_callback(config, train_log_handler):
        from walk.utils import learning_callback

        enable_eval = getattr(config.logger_params, "enable_evaluation", False)
        if isinstance(config, ImitationTrainSessionConfig):
            return learning_callback.ImitationCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
                original_reward_weights=config.env_params.reward_keys_and_weights,
                auto_reward_adjust_params=config.auto_reward_adjust_params,
                enable_evaluation=enable_eval,
            )
        return learning_callback.BaseCustomLearningCallback(
            log_rollout_freq=config.logger_params.logging_frequency,
            evaluate_freq=config.logger_params.evaluate_frequency,
            log_handler=train_log_handler,
            enable_evaluation=enable_eval,
        )

    @staticmethod
    def get_stable_baselines3_model(
        config: TrainSessionConfigBase, env, trained_model_path: str | None = None
    ):
        from walk.train.policies import HumanActorCriticPolicy

        policy_class = HumanActorCriticPolicy
        if trained_model_path is not None:
            print(f"Loading trained model from {trained_model_path}")
            model = stable_baselines3.PPO.load(
                trained_model_path,
                env=env,
                custom_objects={"policy_class": policy_class},
            )
        elif config.env_params.prev_trained_policy_path:
            print(
                f"Loading previous policy from {config.env_params.prev_trained_policy_path}"
            )
            model = stable_baselines3.PPO.load(
                config.env_params.prev_trained_policy_path,
                env=env,
                custom_objects={"policy_class": policy_class},
                verbose=2,
                **DictionableDataclass.to_dict(config.ppo_params),
            )
            model.policy.reset_network(
                reset_shared_net=config.policy_params.custom_policy_params.reset_shared_net_after_load,
                reset_policy_net=config.policy_params.custom_policy_params.reset_policy_net_after_load,
                reset_value_net=config.policy_params.custom_policy_params.reset_value_net_after_load,
            )
        else:
            model = stable_baselines3.PPO(
                policy=policy_class,
                env=env,
                policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                verbose=2,
                **DictionableDataclass.to_dict(config.ppo_params),
            )
        return model
