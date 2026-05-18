import collections
import os
import random
from enum import Enum

import mujoco
import numpy as np

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from myosuite.envs import env_base
from myosuite.utils import gym

from walk.train.config import TrainSessionConfigBase
from walk.utils.data_types import DictionableDataclass
from walk.utils.hfield_manager import HfieldManager


class MetabolicEnergyEstimator:
    MUSCLE_MASSES_22 = np.array(
        [
            0.263, 0.080, 0.040, 0.030, 0.546, 0.170, 0.193, 0.510, 0.280, 0.400, 0.110,
            0.263, 0.080, 0.040, 0.030, 0.546, 0.170, 0.193, 0.510, 0.280, 0.400, 0.110,
        ],
        dtype=np.float64,
    )

    def __init__(self, num_muscles=22, alpha=1.5, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.muscle_masses = self.MUSCLE_MASSES_22[:num_muscles].copy()
        self._mass_alpha = self.muscle_masses ** self.alpha
        self.reset()

    def reset(self):
        self._cumulative_mee = 0.0
        self._step_count = 0
        self._cumulative_distance = 0.0

    def compute_instantaneous_rate(self, activations, dt):
        act = np.clip(activations[: len(self._mass_alpha)], 0.0, 1.0)
        rate = float(np.sum(self._mass_alpha * (act ** self.beta)))
        self._cumulative_mee += rate * dt
        self._step_count += 1
        return rate

    def compute_cot(self, distance_traveled, body_mass):
        if distance_traveled > 0.01:
            return self._cumulative_mee / (distance_traveled * body_mass * 9.81)
        return 0.0

    @property
    def cumulative_mee(self):
        return self._cumulative_mee


class MyoAssistLegBase(env_base.MujocoEnv):
    MYO_CREDIT = """ExamWalk LegBase"""

    class VelocityMode(Enum):
        UNIFORM = 0
        SINUSOIDAL = 1
        STEP = 2

    DEFAULT_OBS_KEYS = [
        "qpos",
        "qvel",
        "act",
        "sensor",
        "target_velocity",
    ]

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        print(f"=== env seed: {seed} ===")
        print(f"=== env model_path: {model_path} ===")
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )
        self._setup(**kwargs)

    def _setup(self, *, env_params: TrainSessionConfigBase.EnvParams, **kwargs):
        self.is_evaluate_mode = kwargs.pop("is_evaluate_mode", False)

        self.sim.model.opt.timestep = 1 / env_params.physics_sim_framerate
        self._safe_height = env_params.safe_height

        self._min_target_velocity = env_params.min_target_velocity
        self._max_target_velocity = env_params.max_target_velocity
        self._min_target_velocity_period = env_params.min_target_velocity_period
        self._max_target_velocity_period = env_params.max_target_velocity_period
        self._change_mode_and_target_velocity_randomly()

        self._step_count_per_episode = 0
        self.CUSTOM_MAX_EPISODE_STEPS = env_params.custom_max_episode_steps
        self._prev_muscle_activations_for_reward = None

        num_muscles = 22
        mee_alpha = getattr(env_params, "mee_alpha", 1.5)
        mee_beta = getattr(env_params, "mee_beta", 1.0)
        self._mee_reference_rate = getattr(env_params, "mee_reference_rate", 1.35)
        self._over_activation_threshold = getattr(env_params, "over_activation_threshold", 0.8)
        self._training_stage = getattr(env_params, "training_stage", 1)
        self._mee_estimator = MetabolicEnergyEstimator(
            num_muscles=num_muscles, alpha=mee_alpha, beta=mee_beta
        )

        self._enable_lumbar_joint = env_params.enable_lumbar_joint
        self._lumbar_joint_fixed_angle = env_params.lumbar_joint_fixed_angle
        self._lumbar_joint_damping_value = env_params.lumbar_joint_damping_value

        self.observation_joint_pos_keys = env_params.observation_joint_pos_keys
        self.observation_joint_vel_keys = env_params.observation_joint_vel_keys
        self.observation_sensor_keys = env_params.observation_sensor_keys
        self.joint_limit_sensor_keys = env_params.joint_limit_sensor_keys

        try:
            lumbar_joint_id = self.sim.model.joint("lumbar_extension").id
            has_lumbar_extension = True
        except (KeyError, ValueError, TypeError):
            has_lumbar_extension = False

        if not self._enable_lumbar_joint:
            if "lumbar_extension" in self.observation_joint_pos_keys:
                self.observation_joint_pos_keys.remove("lumbar_extension")
            if "lumbar_extension" in self.observation_joint_vel_keys:
                self.observation_joint_vel_keys.remove("lumbar_extension")
            if has_lumbar_extension:
                self.sim.data.joint("lumbar_extension").qpos[0] = self._lumbar_joint_fixed_angle
                self.sim.model.jnt_range[lumbar_joint_id] = [
                    self._lumbar_joint_fixed_angle,
                    self._lumbar_joint_fixed_angle + 1e-6,
                ]
                dof_adr = self.sim.model.jnt_dofadr[lumbar_joint_id]
                joint_type = self.sim.model.jnt_type[lumbar_joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    dof_count = 6
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    dof_count = 3
                elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    dof_count = 1
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    dof_count = 1
                else:
                    dof_count = 0
                self.sim.model.dof_damping[dof_adr] = self._lumbar_joint_damping_value
            else:
                self.sim.model.body("torso").quat = [1, 0, 0, self._lumbar_joint_fixed_angle]

        frame_skip = env_params.physics_sim_framerate // env_params.control_framerate
        original_reward_dict = DictionableDataclass.to_dict(env_params.reward_keys_and_weights)
        self.rwd_keys_wt = {}
        for key, value in original_reward_dict.items():
            if isinstance(value, dict):
                self.rwd_keys_wt[key] = sum(value.values())
            else:
                self.rwd_keys_wt[key] = value

        self._initialize_pose()
        self._reset_heel_strike_buffer()
        self._reset_reward_per_step()
        self._reset_properties_per_step()

        super()._setup(
            obs_keys=self.DEFAULT_OBS_KEYS,
            weighted_reward_keys=self.rwd_keys_wt,
            frame_skip=frame_skip,
            **kwargs,
        )

        obs_dict_keys = list(self.get_obs_dict(self.sim).keys())
        assert set(self.DEFAULT_OBS_KEYS + ["time"]) == set(obs_dict_keys)

        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = self.sim.model.key_qvel[0]

        geom_1_indices = np.where(self.sim.model.geom_group == 1)
        self.sim.model.geom_rgba[geom_1_indices, 3] = 0

        self._terrain_type = env_params.terrain_type
        self._terrain_params = env_params.terrain_params
        self._terrain_resample_per_reset = bool(
            getattr(env_params, "terrain_resample_per_reset", False)
        )
        self._hfield_manager = HfieldManager(self.sim, "terrain", self.np_random)
        self._hfield_manager.set_hfield(self._terrain_type, self._terrain_params)

        observation, _reward, done, *_, _info = self.step(np.zeros(self.sim.model.nu))
        for _ in range(30):
            super().step(a=np.zeros(self.sim.model.nu))

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["time"] = np.array([sim.data.time])

        qpos = []
        for key in self.observation_joint_pos_keys:
            qpos.append(sim.data.joint(f"{key}").qpos[0].copy())
        qvel = []
        for key in self.observation_joint_vel_keys:
            qvel.append(sim.data.joint(f"{key}").qvel[0].copy())
        obs_dict["qpos"] = np.array(qpos)
        obs_dict["qvel"] = np.array(qvel)
        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()
        obs_dict["sensor"] = []
        for key in self.observation_sensor_keys:
            sensor_data = sim.data.sensor(f"{key}").data.copy()
            if "foot" in key or "toes" in key:
                model_mass = np.sum(self.sim.model.body_mass)
                sensor_data = sensor_data / (model_mass * 9.81)
            obs_dict["sensor"].extend(sensor_data)
        obs_dict["sensor"] = np.array(obs_dict["sensor"])

        obs_dict["target_velocity"] = np.array([self._target_velocity])
        return obs_dict

    def _calculate_reward_per_step(self, obs_dict, muscle_activations):
        reward_per_steps = {
            "muscle_activation_penalty_per_step": 0.0,
            "average_velocity_per_step": 0.0,
            "footstep_delta_time": 0.0,
        }
        info = {}
        return reward_per_steps, info

    def _calculate_base_reward(self, obs_dict):
        model_mass = np.sum(self.sim.model.body_mass)
        model_weight = model_mass * 9.81

        forward_reward = self.dt * np.exp(
            -5
            * np.square(
                self.sim.data.joint("pelvis_tx").qvel[0].copy() - self._target_velocity
            )
        )

        muscle_activations = self._get_muscle_activation()
        muscle_activation_penalty = -self.dt * np.mean(np.square(muscle_activations))

        over_act = np.maximum(0.0, muscle_activations - self._over_activation_threshold)
        over_activation_penalty = -self.dt * np.mean(np.square(over_act))

        self._current_muscle_activations = muscle_activations

        mee_rate = self._mee_estimator.compute_instantaneous_rate(
            muscle_activations, self.dt
        )
        normalized_mee = np.clip(
            mee_rate / max(self._mee_reference_rate, 1e-6), 0.0, 2.5
        )
        metabolic_energy_penalty = -self.dt * normalized_mee

        joint_constraint_force_penalty = (
            -self.dt * self._get_max_joint_constraint_force() / model_weight
        )

        reward_per_steps, info = self._calculate_reward_per_step(
            obs_dict, muscle_activations
        )

        if self._prev_muscle_activations_for_reward is not None:
            muscle_activation_diff_penalty = self.dt * np.mean(
                np.exp(
                    -4
                    * np.square(
                        self._prev_muscle_activations_for_reward - muscle_activations
                    )
                )
            )
        else:
            muscle_activation_diff_penalty = 0
        self._prev_muscle_activations_for_reward = muscle_activations

        normalized_foot_force_sum = (
            np.abs(self._get_foot_force("r")) + np.abs(self._get_foot_force("l"))
        ) / model_weight
        foot_force_penalty = -self.dt * max(0, normalized_foot_force_sum - 1.2)

        base_reward = {
            "forward_reward": forward_reward,
            "muscle_activation_penalty": muscle_activation_penalty,
            "over_activation_penalty": over_activation_penalty,
            "muscle_activation_diff_penalty": muscle_activation_diff_penalty,
            "foot_force_penalty": foot_force_penalty,
            "joint_constraint_force_penalty": joint_constraint_force_penalty,
            "metabolic_energy_penalty": metabolic_energy_penalty,
        }
        base_reward.update(reward_per_steps)

        pelvis_x = self.sim.data.joint("pelvis_tx").qpos[0].copy()
        distance_traveled = max(0.0, pelvis_x)
        cot = self._mee_estimator.compute_cot(distance_traveled, model_mass)

        info = {
            "muscle_activations": muscle_activations,
            "mee_rate": mee_rate,
            "mee_cumulative": self._mee_estimator.cumulative_mee,
            "cot": cot,
        }
        return base_reward, info

    def get_reward_dict(self, obs_dict):
        base_reward, info = self._calculate_base_reward(obs_dict)
        rwd_dict = collections.OrderedDict(
            (key, base_reward[key]) for key in base_reward
        )
        rwd_dict.update(
            {
                "sparse": 0,
                "solved": False,
                "done": self._get_done(),
            }
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()],
            axis=0,
        )
        return rwd_dict

    def step(self, a, **kwargs):
        self._modulate_target_velocity()
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        self._step_count_per_episode += 1
        is_over_time_limit = (
            self._step_count_per_episode >= self.CUSTOM_MAX_EPISODE_STEPS
        )
        return (next_obs, reward, terminated, truncated or is_over_time_limit, info)

    def just_forward(self):
        self.sim.forward()

    def set_target_velocity_mode_manually(
        self,
        mode,
        starting_phase: float,
        initial_target_velocity: float,
        min_target_velocity: float,
        max_target_velocity: float,
        target_velocity_period: float = None,
    ):
        self._velocity_mode_for_this_episode = mode
        self._starting_phase = starting_phase
        if (
            mode == MyoAssistLegBase.VelocityMode.SINUSOIDAL
            and target_velocity_period is None
        ):
            raise ValueError("target_velocity_period required for sinusoidal mode")
        self._target_velocity_period = target_velocity_period
        self._target_velocity = initial_target_velocity
        self._prev_step_changed_time = self.sim.data.time
        self._min_target_velocity = min_target_velocity
        self._max_target_velocity = max_target_velocity

    def _change_mode_and_target_velocity_randomly(self):
        velocity_mode = random.choice(list(MyoAssistLegBase.VelocityMode))
        starting_phase = random.uniform(0, 2 * np.pi)
        target_velocity_period = random.uniform(
            self._min_target_velocity_period, self._max_target_velocity_period
        )
        self.set_target_velocity_mode_manually(
            velocity_mode,
            starting_phase,
            self._min_target_velocity,
            self._min_target_velocity,
            self._max_target_velocity,
            target_velocity_period,
        )
        if self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.UNIFORM:
            self._target_velocity = random.uniform(
                self._min_target_velocity, self._max_target_velocity
            )
        elif (
            self._velocity_mode_for_this_episode
            == MyoAssistLegBase.VelocityMode.SINUSOIDAL
        ):
            self._target_velocity = self._calc_sinusoidal_target_velocity(
                self._starting_phase,
                self._target_velocity_period,
                self._min_target_velocity,
                self._max_target_velocity,
            )
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.STEP:
            self._target_velocity = np.random.uniform(
                self._min_target_velocity, self._max_target_velocity
            )

    def _calc_sinusoidal_target_velocity(
        self, phase: float, period: float, min_velocity: float, max_velocity: float
    ):
        return min_velocity + (max_velocity - min_velocity) * (
            np.sin(phase + 2 * np.pi * self.sim.data.time / period) + 1
        ) / 2

    def _modulate_target_velocity(self):
        if self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.UNIFORM:
            pass
        elif (
            self._velocity_mode_for_this_episode
            == MyoAssistLegBase.VelocityMode.SINUSOIDAL
        ):
            self._target_velocity = self._calc_sinusoidal_target_velocity(
                self._starting_phase,
                self._target_velocity_period,
                self._min_target_velocity,
                self._max_target_velocity,
            )
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.STEP:
            if (
                self.sim.data.time - self._prev_step_changed_time
                > self._target_velocity_period
            ):
                self._target_velocity = np.random.uniform(
                    self._min_target_velocity, self._max_target_velocity
                )
                self._prev_step_changed_time = self.sim.data.time

    def reset(self, **kwargs):
        self._step_count_per_episode = 0
        if not self.is_evaluate_mode:
            self._change_mode_and_target_velocity_randomly()
        self.sim.data.joint("pelvis_tx").qvel[0] = self._target_velocity

        if (
            self._terrain_resample_per_reset
            and getattr(self, "_hfield_manager", None) is not None
        ):
            try:
                self._hfield_manager.resample()
            except Exception as e:
                print(f"[hfield] resample failed, keep previous terrain: {e}")

        self.sim.forward()
        self.robot.sync_sims(self.sim, self.sim_obsd)

        self._reset_heel_strike_buffer()
        self._reset_reward_per_step()
        self._reset_properties_per_step()
        self._mee_estimator.reset()

        obs = super().reset(**kwargs)
        return obs

    def _get_done(self):
        pelvis_height = self.sim.data.joint("pelvis_ty").qpos[0].copy()
        if pelvis_height < self._safe_height:
            return True
        return False

    def _get_muscle_activation(self):
        if not self._enable_lumbar_joint:
            return self.sim.data.act[:].copy()
        muscle_activations_with_lumbar = np.concatenate(
            (
                self.sim.data.act[:].copy(),
                np.array(
                    [
                        self.sim.data.actuator(
                            "lumbar_extension_motor"
                        ).ctrl[0].copy()
                    ]
                ).reshape(1,),
            )
        )
        return muscle_activations_with_lumbar

    def _get_max_joint_constraint_force(self):
        max_constraint_force = 0
        for sensor_name in self.joint_limit_sensor_keys:
            sensor_data = self.sim.data.sensor(sensor_name).data[0].copy()
            max_constraint_force = max(max_constraint_force, np.max(np.abs(sensor_data)))
        return max_constraint_force

    def _reset_properties_per_step(self):
        self._prev_pelvis_tx_pos = self.sim.data.body("pelvis").xpos[0]
        self._prev_step_time = self.sim.data.time
        self._activation_square_sum = 0
        self._footstep_delta_time = 0
        self._delta_velocity_sum = 0

    def _reset_reward_per_step(self):
        self.reward_muscle_activation_penalty_per_step = 0
        self.reward_average_velocity_per_step = 0
        self.reward_footstep_delta_time = 0

    def _reset_heel_strike_buffer(self):
        self._r_heel_striking_value_buffer = []
        self._l_heel_striking_value_buffer = []
        self._last_heel_strike_foot = ""

    def _get_foot_force(self, foot_side_alphabet: str):
        foot_force = (
            self.sim.data.sensor(f"{foot_side_alphabet}_foot").data.copy()[0]
            + self.sim.data.sensor(f"{foot_side_alphabet}_toes").data.copy()[0]
        )
        return foot_force

    def _initialize_pose(self):
        self.sim.data.qpos[:] = self.sim.model.key_qpos[0][:]
        self.just_forward()
