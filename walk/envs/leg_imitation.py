import collections

import numpy as np

from walk.envs.leg_base import MyoAssistLegBase
from walk.train.config import ImitationTrainSessionConfig, TrainSessionConfigBase
from walk.utils.data_types import DictionableDataclass


class MyoAssistLegImitation(MyoAssistLegBase):
    def _setup(
        self,
        *,
        env_params: ImitationTrainSessionConfig.EnvParams,
        reference_data: dict | None = None,
        loop_reference_data: bool = False,
        **kwargs,
    ):
        self._flag_random_ref_index = env_params.flag_random_ref_index
        self._out_of_trajectory_threshold = env_params.out_of_trajectory_threshold
        self.reference_data_keys = env_params.reference_data_keys
        self._loop_reference_data = loop_reference_data
        self._reward_keys_and_weights = env_params.reward_keys_and_weights
        self._qpos_corridor_tolerance = getattr(
            env_params, "qpos_corridor_tolerance", {}
        )
        self._qvel_corridor_tolerance = getattr(
            env_params, "qvel_corridor_tolerance", {}
        )

        self.setup_reference_data(data=reference_data)

        super()._setup(env_params=env_params, **kwargs)

    def set_reward_weights(self, reward_keys_and_weights):
        self._reward_keys_and_weights = reward_keys_and_weights
        updated_reward_dict = DictionableDataclass.to_dict(reward_keys_and_weights)
        for key, value in updated_reward_dict.items():
            if isinstance(value, dict):
                self.rwd_keys_wt[key] = sum(value.values())
            else:
                self.rwd_keys_wt[key] = value

    def _corridor_reward(
        self, diff: float, tolerance: float, k: float = 8.0
    ) -> float:
        excess = max(0.0, abs(float(diff)) - float(tolerance))
        return self.dt * np.exp(-k * np.square(excess))

    def get_obs_dict(self, sim):
        return super().get_obs_dict(sim)

    def _get_qpos_diff(self) -> dict:
        ref_idx = 0 if self._imitation_index is None else int(self._imitation_index)

        def get_qpos_diff_one(key: str):
            sim_val = float(
                np.asarray(
                    self.sim.data.joint(f"{key}").qpos[0].copy()
                ).reshape(-1)[0]
            )
            ref_val = float(
                np.asarray(
                    self._reference_data["series_data"][f"q_{key}"][ref_idx]
                ).reshape(-1)[0]
            )
            return sim_val - ref_val

        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qpos_imitation_rewards:
            name_diff_dict[q_key] = get_qpos_diff_one(q_key)
        return name_diff_dict

    def _get_qvel_diff(self):
        ref_idx = 0 if self._imitation_index is None else int(self._imitation_index)
        ref_dq = float(
            np.asarray(
                self._reference_data["series_data"]["dq_pelvis_tx"][ref_idx]
            ).reshape(-1)[0]
        )
        speed_ratio_to_target_velocity = (
            self._target_velocity / max(abs(ref_dq), 1e-6)
        ) * (1.0 if ref_dq >= 0 else -1.0)

        def get_qvel_diff_one(key: str):
            sim_val = float(
                np.asarray(
                    self.sim.data.joint(f"{key}").qvel[0].copy()
                ).reshape(-1)[0]
            )
            ref_val = float(
                np.asarray(
                    self._reference_data["series_data"][f"dq_{key}"][ref_idx]
                ).reshape(-1)[0]
            )
            return sim_val - ref_val * speed_ratio_to_target_velocity

        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qvel_imitation_rewards:
            name_diff_dict[q_key] = get_qvel_diff_one(q_key)
        return name_diff_dict

    def _get_qpos_diff_nparray(self):
        return np.array([diff for diff in self._get_qpos_diff().values()])

    def _get_end_effector_diff(self):
        return np.array([0])

    def _calculate_imitation_rewards(self, obs_dict):
        base_reward, base_info = super()._calculate_base_reward(obs_dict)

        q_diff_dict = self._get_qpos_diff()
        dq_diff_dict = self._get_qvel_diff()
        anchor_diff_array = self._get_end_effector_diff()

        q_reward_dict = {}
        for joint_name, diff in q_diff_dict.items():
            tolerance = self._qpos_corridor_tolerance.get(joint_name, 0.0)
            q_reward_dict[joint_name] = self._corridor_reward(diff, tolerance, k=8.0)

        dq_reward_dict = {}
        for joint_name, diff in dq_diff_dict.items():
            tolerance = self._qvel_corridor_tolerance.get(joint_name, 0.0)
            dq_reward_dict[joint_name] = self._corridor_reward(diff, tolerance, k=8.0)

        anchor_reward = self.dt * np.mean(np.exp(-5 * np.square(anchor_diff_array)))

        qpos_imitation_rewards = np.sum(
            [
                q_reward_dict[key]
                * self._reward_keys_and_weights.qpos_imitation_rewards[key]
                for key in q_reward_dict.keys()
            ]
        )
        qvel_imitation_rewards = np.sum(
            [
                dq_reward_dict[key]
                * self._reward_keys_and_weights.qvel_imitation_rewards[key]
                for key in dq_reward_dict.keys()
            ]
        )

        base_reward.update(
            {
                "qpos_imitation_rewards": qpos_imitation_rewards,
                "qvel_imitation_rewards": qvel_imitation_rewards,
                "end_effector_imitation_reward": anchor_reward,
            }
        )

        imitation_rewards = base_reward
        info = base_info
        return imitation_rewards, info

    def get_reward_dict(self, obs_dict):
        imitation_rewards, info = self._calculate_imitation_rewards(obs_dict)

        rwd_dict = collections.OrderedDict(
            (key, imitation_rewards[key]) for key in imitation_rewards
        )
        rwd_dict.update(
            {
                "sparse": 0,
                "solved": False,
                "done": self._get_done(),
            }
        )
        rwd_dict["dense"] = np.sum(
            [
                wt * rwd_dict[key]
                for key, wt in self.rwd_keys_wt.items()
                if key in rwd_dict
            ],
            axis=0,
        )
        return rwd_dict

    def _follow_reference_motion(self, is_x_follow: bool):
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qpos = self._reference_data["series_data"][
                f"q_{key}"
            ][self._imitation_index]
            if not is_x_follow and key == "pelvis_tx":
                self.sim.data.joint(f"{key}").qpos = 0
        speed_ratio_to_target_velocity = (
            self._target_velocity
            / self._reference_data["series_data"]["dq_pelvis_tx"][self._imitation_index]
        )
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qvel = (
                self._reference_data["series_data"][f"dq_{key}"][self._imitation_index]
                * speed_ratio_to_target_velocity
            )

    def imitation_step(self, is_x_follow: bool, specific_index: int | None = None):
        if specific_index is None:
            self._imitation_index += 1
            if self._imitation_index >= self._reference_data_length:
                self._imitation_index = 0
        else:
            self._imitation_index = specific_index
        self._follow_reference_motion(is_x_follow)
        self.forward()
        return self._imitation_index

    def step(self, a, **kwargs):
        if self._imitation_index is not None:
            self._imitation_index += 1
            if self._imitation_index < self._reference_data_length:
                is_out_of_index = False
            else:
                if self._loop_reference_data:
                    self._imitation_index = 0
                    is_out_of_index = False
                else:
                    is_out_of_index = True
                    self._imitation_index = self._reference_data_length - 1
        else:
            is_out_of_index = True

        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        if is_out_of_index:
            reward = 0
            truncated = True
        else:
            q_diff_nparray = self._get_qpos_diff_nparray()
            is_out_of_trajectory = np.any(
                np.abs(q_diff_nparray) > self._out_of_trajectory_threshold
            )
            terminated = terminated or is_out_of_trajectory

        return (next_obs, reward, terminated, truncated, info)

    def setup_reference_data(self, data: dict | None):
        self._reference_data = data
        self._imitation_index = None
        if data is not None:
            self._reference_data_length = self._reference_data["metadata"][
                "resampled_data_length"
            ]
        else:
            raise ValueError("Reference data is not set")

    def reset(self, **kwargs):
        if self._flag_random_ref_index:
            self._imitation_index = self.np_random.integers(
                0, int(self._reference_data_length * 0.8)
            )
        else:
            self._imitation_index = 0
        self._follow_reference_motion(False)

        obs = super().reset(
            reset_qpos=self.sim.data.qpos,
            reset_qvel=self.sim.data.qvel,
            **kwargs,
        )
        return obs

    def _initialize_pose(self):
        super()._initialize_pose()
