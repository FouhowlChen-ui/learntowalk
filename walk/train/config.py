from dataclasses import dataclass, field


JOINT_LIMIT_SENSOR_NAMES = [
    "r_knee_sensor",
    "l_knee_sensor",
    "r_hip_sensor",
    "l_hip_sensor",
    "r_ankle_sensor",
    "l_ankle_sensor",
    "r_mtp_sensor",
    "l_mtp_sensor",
]


@dataclass
class TrainSessionConfigBase:
    total_timesteps: int = 1000

    @dataclass
    class LoggerParams:
        logging_frequency: int = int(1)
        evaluate_frequency: int = int(64)
        enable_evaluation: bool = False

    logger_params: LoggerParams = field(default_factory=LoggerParams)

    @dataclass
    class EnvParams:
        @dataclass
        class RewardWeights:
            forward_reward: float = 0.01
            muscle_activation_penalty: float = 0.1
            muscle_activation_diff_penalty: float = 0.0

            footstep_delta_time: float = 0.0
            average_velocity_per_step: float = 0.0
            muscle_activation_penalty_per_step: float = 0.0

            joint_constraint_force_penalty: float = 0.0
            foot_force_penalty: float = 0.0

            over_activation_penalty: float = 0.0
            metabolic_energy_penalty: float = 0.0

        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)

        env_id: str = ""
        num_envs: int = 1
        seed: int = 0
        safe_height: float = 0.65
        control_framerate: int = 30
        physics_sim_framerate: int = 1200

        min_target_velocity: float = 0.5
        max_target_velocity: float = 3.0
        min_target_velocity_period: float = 3
        max_target_velocity_period: float = 5

        custom_max_episode_steps: int = 500
        model_path: str = None
        prev_trained_policy_path: str = None
        reference_data_path: str = ""

        enable_lumbar_joint: bool = False
        lumbar_joint_fixed_angle: float = 0.0
        lumbar_joint_damping_value: float = 0.05

        observation_joint_pos_keys: list = field(default_factory=list)
        observation_joint_vel_keys: list = field(default_factory=list)
        observation_sensor_keys: list = field(default_factory=list)

        joint_limit_sensor_keys: list = field(
            default_factory=lambda: list(JOINT_LIMIT_SENSOR_NAMES)
        )

        terrain_type: str = "flat"
        terrain_params: str = ""
        terrain_resample_per_reset: bool = False

        mee_alpha: float = 1.5
        mee_beta: float = 1.0
        mee_k_energy: float = 0.2
        mee_reference_rate: float = 1.35
        over_activation_threshold: float = 0.8

        qpos_corridor_tolerance: dict = field(default_factory=dict)
        qvel_corridor_tolerance: dict = field(default_factory=dict)

        training_stage: int = 1

    env_params: EnvParams = field(default_factory=EnvParams)

    evaluate_param_list: list = field(default_factory=list)

    @dataclass
    class PolicyParams:
        @dataclass
        class CustomPolicyParams:
            reset_shared_net_after_load: bool = False
            reset_policy_net_after_load: bool = False
            reset_value_net_after_load: bool = False

            net_arch: dict = field(default_factory=dict)
            log_std_init: float = field(default=-2.0)
            net_indexing_info: dict = field(default_factory=dict)

        custom_policy_params: CustomPolicyParams = field(
            default_factory=CustomPolicyParams
        )

    policy_params: PolicyParams = field(default_factory=PolicyParams)

    @dataclass
    class PPOParams:
        learning_rate: float = 3e-4
        n_steps: int = 4096
        batch_size: int = 2048
        n_epochs: int = 10
        gamma: float = 0.99
        gae_lambda: float = 0.95
        clip_range: float = 0.2
        clip_range_vf: float = 0.2
        ent_coef: float = 0.01
        vf_coef: float = 0.5
        max_grad_norm: float = 0.5
        use_sde: bool = False
        sde_sample_freq: int = -1
        target_kl: float = None
        device: str = "cpu"

    ppo_params: PPOParams = field(default_factory=PPOParams)


@dataclass
class ImitationTrainSessionConfig(TrainSessionConfigBase):
    @dataclass
    class AutoRewardAdjustParams:
        learning_rate: float = 0.0

    auto_reward_adjust_params: AutoRewardAdjustParams = field(
        default_factory=AutoRewardAdjustParams
    )

    @dataclass
    class EnvParams(TrainSessionConfigBase.EnvParams):
        @dataclass
        class RewardWeights(TrainSessionConfigBase.EnvParams.RewardWeights):
            qpos_imitation_rewards: dict = field(default_factory=dict)
            qvel_imitation_rewards: dict = field(default_factory=dict)
            end_effector_imitation_reward: float = 0.0

        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)

        flag_random_ref_index: bool = False
        out_of_trajectory_threshold: float = 1.0
        reference_data_path: str = ""
        reference_data_keys: list = field(default_factory=list)

    env_params: EnvParams = field(default_factory=EnvParams)
