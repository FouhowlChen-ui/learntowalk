from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from walk.utils import train_log_handler
from walk.utils.checkpoint_data import (
    ImitationTrainCheckpointData,
    TrainCheckpointData,
)
from walk.utils.data_types import DictionableDataclass


class BaseCustomLearningCallback(BaseCallback):
    def __init__(
        self,
        *,
        log_rollout_freq: int,
        evaluate_freq: int,
        log_handler: train_log_handler.TrainLogHandler,
        enable_evaluation: bool = False,
        verbose=1,
    ):
        super().__init__(verbose)
        self.log_rollout_freq = log_rollout_freq
        self.evaluate_freq = evaluate_freq
        self.enable_evaluation = enable_evaluation
        self.train_log_handler = log_handler
        self.log_count = 0

    def _init_callback(self):
        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.current_episode_rewards = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)
        self.current_episode_length_counts = np.zeros(self.training_env.num_envs)

        self.current_reward_dict_sum = [
            {} for _ in range(self.training_env.num_envs)
        ]
        self.episode_reward_dict_sum = [
            {} for _ in range(self.training_env.num_envs)
        ]

        self.prev_logging_timestep = 0

    def _on_step(self) -> bool:
        self.current_episode_rewards += self.locals["rewards"]
        for idx, done in enumerate(self.locals["dones"]):
            self.current_episode_length_counts[idx] += 1
            info_dict = None
            try:
                info_dict = self.locals["infos"][idx]
            except (IndexError, KeyError):
                info_dict = None

            if (
                info_dict
                and isinstance(info_dict, dict)
                and "rwd_dict" in info_dict
            ):
                for key, val in info_dict["rwd_dict"].items():
                    if key not in self.current_reward_dict_sum[idx]:
                        self.current_reward_dict_sum[idx][key] = 0
                    self.current_reward_dict_sum[idx][key] += val
            if done:
                self.rewards_sum[idx] += self.current_episode_rewards[idx]
                self.episode_counts[idx] += 1
                self.current_episode_rewards[idx] = 0.0
                self.episode_length_counts[idx] += self.current_episode_length_counts[idx]
                self.current_episode_length_counts[idx] = 0

                for key, val in self.current_reward_dict_sum[idx].items():
                    if key not in self.episode_reward_dict_sum[idx]:
                        self.episode_reward_dict_sum[idx][key] = 0
                    self.episode_reward_dict_sum[idx][key] += val
                self.current_reward_dict_sum[idx] = {}

        return True

    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()

    def _on_rollout_end(self, write_log: bool = True):
        super()._on_rollout_end()
        self.prev_logging_timestep = self.num_timesteps

        log_data = None
        if self.log_count % self.log_rollout_freq == 0:
            model_path = self.train_log_handler.get_path2save_model(
                self.model.num_timesteps
            )
            self.model.save(model_path)

            def get_logger_value(key, default=-1):
                value = self.logger.name_to_value.get(key, default)
                return value.item() if hasattr(value, "item") else value

            all_keys = set()
            for idx in range(self.training_env.num_envs):
                all_keys.update(self.episode_reward_dict_sum[idx].keys())
            for idx in range(self.training_env.num_envs):
                for key in all_keys:
                    if key not in self.episode_reward_dict_sum[idx]:
                        self.episode_reward_dict_sum[idx][key] = 0

            total_episodes = sum(self.episode_counts)
            if total_episodes == 0 or not all_keys:
                average_reward_dict_per_episode = {}
            else:
                average_reward_dict_per_episode = {
                    key: (
                        sum(
                            self.episode_reward_dict_sum[idx][key]
                            for idx in range(self.training_env.num_envs)
                        )
                        / total_episodes
                    )
                    for key in all_keys
                }

            log_data = TrainCheckpointData(
                approx_kl=get_logger_value("train/approx_kl"),
                clip_fraction=get_logger_value("train/clip_fraction"),
                clip_range=get_logger_value("train/clip_range"),
                clip_range_vf=get_logger_value("train/clip_range_vf"),
                entropy_loss=get_logger_value("train/entropy_loss"),
                explained_variance=get_logger_value("train/explained_variance"),
                learning_rate=get_logger_value("train/learning_rate"),
                loss=get_logger_value("train/loss"),
                n_updates=get_logger_value("train/n_updates"),
                policy_gradient_loss=get_logger_value("train/policy_gradient_loss"),
                std=get_logger_value("train/std"),
                value_loss=get_logger_value("train/value_loss"),
                num_timesteps=self.model.num_timesteps,
                average_num_timestep=(
                    np.sum(self.episode_length_counts) / np.sum(self.episode_counts)
                    if np.sum(self.episode_counts) != 0
                    else np.array(0)
                ).item(),
                average_reward_per_episode=(
                    np.sum(self.rewards_sum) / np.sum(self.episode_counts)
                    if np.sum(self.episode_counts) != 0
                    else np.array(0)
                ).item(),
                average_reward_dict_per_episode=average_reward_dict_per_episode,
                time=f"{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}",
            )
            if write_log:
                self.train_log_handler.add_log_data(log_data)
                self.train_log_handler.write_json_file()

            self.rewards_sum = np.zeros(self.training_env.num_envs)
            self.episode_counts = np.zeros(self.training_env.num_envs)
            self.episode_length_counts = np.zeros(self.training_env.num_envs)
            self.episode_reward_dict_sum = [
                {} for _ in range(self.training_env.num_envs)
            ]
            self.current_reward_dict_sum = [
                {} for _ in range(self.training_env.num_envs)
            ]

        self.log_count += 1
        return log_data


class ImitationCustomLearningCallback(BaseCustomLearningCallback):
    def __init__(
        self,
        *,
        log_rollout_freq: int,
        evaluate_freq: int,
        log_handler,
        original_reward_weights,
        auto_reward_adjust_params,
        enable_evaluation: bool = False,
        verbose=1,
    ):
        super().__init__(
            log_rollout_freq=log_rollout_freq,
            evaluate_freq=evaluate_freq,
            log_handler=log_handler,
            enable_evaluation=enable_evaluation,
            verbose=verbose,
        )
        self._reward_weights = original_reward_weights
        self._auto_reward_adjust_params = auto_reward_adjust_params

    def _init_callback(self):
        super()._init_callback()
        from walk.train.config import ImitationTrainSessionConfig

        self.reward_accumulate = DictionableDataclass.create(
            ImitationTrainSessionConfig.EnvParams.RewardWeights
        )
        self.reward_accumulate = DictionableDataclass.to_dict(self.reward_accumulate)
        for key in self.reward_accumulate.keys():
            self.reward_accumulate[key] = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            rwd = info.get("rwd_dict") if isinstance(info, dict) else None
            if rwd is not None:
                for key in self.reward_accumulate.keys():
                    if key in rwd:
                        self.reward_accumulate[key] += rwd[key]

        super()._on_step()
        return True

    def _on_rollout_end(self, write_log: bool = True):
        log_data_base = super()._on_rollout_end(write_log=False)
        if log_data_base is None:
            return None
        log_data = ImitationTrainCheckpointData(
            **log_data_base.__dict__,
            reward_weights=DictionableDataclass.to_dict(self._reward_weights),
            reward_accumulate=self.reward_accumulate.copy(),
        )
        if write_log:
            self.train_log_handler.add_log_data(log_data)
            self.train_log_handler.write_json_file()

        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)
