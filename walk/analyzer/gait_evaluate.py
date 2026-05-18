import json
import os
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from walk.analyzer.gait_data import GaitData
from walk.envs.env_handler import EnvironmentHandler
from walk.envs.leg_base import MyoAssistLegBase


class GaitEvaluatorBase:
    def __init__(self, train_log_handler, session_config):
        self.train_log_handler = train_log_handler
        self.session_config = session_config
        self.env = None
        self._offscreen_ready = False

    def initialize_env(
        self,
        *,
        convex_hull_flag: bool = False,
        enable_offscreen_renderer: bool = True,
    ):
        session_config = self.session_config
        session_config.env_params.num_envs = 1
        session_config.env_params.custom_max_episode_steps = 1000000000
        session_config.env_params.out_of_trajectory_threshold = 1000000

        self.free_cam = mujoco.MjvCamera()
        self.env = EnvironmentHandler.create_environment(
            session_config, is_rendering_on=False, is_evaluate_mode=True
        )
        self._offscreen_ready = False
        if enable_offscreen_renderer:
            try:
                _ = self.env.unwrapped.sim.renderer.render_offscreen(
                    camera_id=self.free_cam, width=960, height=540
                )
                self._offscreen_ready = True
            except Exception as e:
                self._offscreen_ready = False
                print(f"[WARN] Offscreen init failed: {e}")

        if self._offscreen_ready:
            self.env.sim.renderer._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_CONVEXHULL
            ] = (0 if not convex_hull_flag else 1)

    def evaluate(
        self,
        result_dir: str,
        file_name: str,
        *,
        velocity_mode,
        target_velocity_period: float,
        min_target_velocity: float,
        max_target_velocity: float,
        terminate_when_done: bool,
        max_timestep: int = 600,
        num_episodes: int = 1,
        num_timesteps: int = None,
        trained_model_path_override: str | None = None,
    ):
        if num_timesteps is None:
            if not self.train_log_handler.log_datas:
                raise RuntimeError(
                    "No checkpoint info, please pass num_timesteps explicitly"
                )
            num_timesteps = self.train_log_handler.log_datas[-1].num_timesteps

        if trained_model_path_override:
            trained_model_path = trained_model_path_override
        else:
            trained_model_path = self.train_log_handler.get_path2save_model(
                num_timesteps
            )
        print(f"[GaitEvaluator] Using model at {num_timesteps}: {trained_model_path}")

        model = EnvironmentHandler.get_stable_baselines3_model(
            self.session_config, self.env, trained_model_path=trained_model_path
        )

        self.env.mujoco_render_frames = False
        env_myoassist: MyoAssistLegBase = self.env.unwrapped
        env_myoassist.set_target_velocity_mode_manually(
            velocity_mode,
            0,
            (min_target_velocity + max_target_velocity) / 2,
            min_target_velocity,
            max_target_velocity,
            target_velocity_period=target_velocity_period,
        )

        gait_data = GaitData()
        if num_episodes is None or num_episodes < 1:
            num_episodes = 1

        for episode_idx in range(num_episodes):
            obs, info = self.env.reset()
            for time_step in range(max_timestep):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, truncated, info = self.env.step(action)
                gait_data.add_data(
                    mj_model=self.env.sim.model,
                    mj_data=self.env.sim.data,
                    target_velocity=env_myoassist._target_velocity,
                )
                if done or truncated:
                    if terminate_when_done:
                        break
                    obs, info = self.env.reset()

        gait_data_path = os.path.join(result_dir, file_name)
        gait_data.save_json_data(gait_data_path)
        return gait_data_path

    def replay(
        self,
        input_gait_data_path: str,
        output_video_path: str,
        *,
        cam_distance: float = 2.5,
        use_activation_visualization: bool = False,
        cam_type: str = "follow",
        video_fps: int = 30,
    ):
        gait_data = GaitData()
        gait_data.read_json_data(input_gait_data_path)

        if not self._offscreen_ready:
            try:
                _ = self.env.unwrapped.sim.renderer.render_offscreen(
                    camera_id=self.free_cam, width=960, height=540
                )
                self._offscreen_ready = True
            except Exception as e:
                raise RuntimeError(f"Offscreen renderer unavailable: {e}")

        max_timestep = gait_data.metadata["data_length"]
        frames = []

        self.env.sim.renderer._scene_option.flags[
            mujoco.mjtVisFlag.mjVIS_ACTUATOR
        ] = (1 if use_activation_visualization else 0)

        cam_pos_range = (
            gait_data.series_data["joint_data"]["pelvis_tx"]["qpos"][0][0],
            gait_data.series_data["joint_data"]["pelvis_tx"]["qpos"][max_timestep - 1][
                0
            ],
        )

        def cam_move(time_step: int):
            if cam_type == "follow":
                cam_target_pos = self.env.unwrapped.sim.data.body("pelvis").xpos.copy()
                cam_target_pos[2] = 0.8
            elif cam_type == "average_speed":
                cam_target_pos = self.env.unwrapped.sim.data.body("pelvis").xpos.copy()
                cam_target_pos[2] = 0.8
                cam_target_pos[0] = (
                    cam_pos_range[0]
                    + (cam_pos_range[1] - cam_pos_range[0]) * time_step / max_timestep
                )
            else:
                raise ValueError(f"Invalid cam_type: {cam_type}")
            self.free_cam.distance = cam_distance
            self.free_cam.azimuth = 90
            self.free_cam.elevation = 0
            self.free_cam.lookat = cam_target_pos

        for time_step in range(max_timestep):
            gait_data.apply_to_env(
                time_index=time_step,
                mj_model=self.env.sim.model,
                mj_data=self.env.sim.data,
            )
            self.env.just_forward()
            cam_move(time_step)
            frame = self.env.unwrapped.sim.renderer.render_offscreen(
                camera_id=self.free_cam, width=960, height=540
            )
            frames.append(frame)

        writer = imageio.get_writer(
            output_video_path, fps=video_fps, codec="libx264", macro_block_size=None
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return frames

    def __del__(self):
        if getattr(self, "env", None) is not None:
            try:
                self.env.close()
            except Exception:
                pass


class ImitationGaitEvaluator(GaitEvaluatorBase):
    def __init__(self, train_log_handler, session_config):
        super().__init__(train_log_handler, session_config)

    def load_reference_data(self):
        print("=== reference data loading ===")
        if not self.session_config.env_params.reference_data_path:
            self.ref_data_dict = None
            return
        ref_path = self.session_config.env_params.reference_data_path
        if not os.path.isabs(ref_path) and not os.path.exists(ref_path):
            project_root = Path(__file__).resolve().parents[2]
            ref_path = str(project_root / ref_path)
        ref_path = str(ref_path)

        if ref_path.endswith(".npz"):
            ref_data_npz = np.load(ref_path, allow_pickle=True)
            ref_data_dict = {
                key: ref_data_npz[key].item() for key in ref_data_npz.files
            }
        elif ref_path.endswith(".json"):
            with open(ref_path, "r", encoding="utf-8") as f:
                ref_data_dict = json.load(f)
        else:
            self.ref_data_dict = None
            return

        if ref_data_dict and "series_data" in ref_data_dict:
            ref_data_dict["resampled_series_data"] = {}
            for key in ref_data_dict["series_data"].keys():
                original_data_length = len(ref_data_dict["series_data"][key])
                original_sample_rate = ref_data_dict["metadata"]["sample_rate"]
                original_x = np.linspace(
                    0, original_data_length - 1, original_data_length
                )
                new_sample_rate = self.session_config.env_params.control_framerate
                new_length = int(
                    original_data_length * new_sample_rate / original_sample_rate
                )
                new_x = np.linspace(0, original_data_length - 1, new_length)
                ref_data_dict["series_data"][key] = np.interp(
                    new_x, original_x, ref_data_dict["series_data"][key]
                )
                ref_data_dict["metadata"]["resampled_data_length"] = new_length
                ref_data_dict["metadata"]["resampled_sample_rate"] = new_sample_rate

        self.ref_data_dict = ref_data_dict

    def initialize_env(self, *, convex_hull_flag=False, enable_offscreen_renderer=True):
        if self.ref_data_dict is not None:
            self.session_config.env_params.reference_data = self.ref_data_dict
        super().initialize_env(
            convex_hull_flag=convex_hull_flag,
            enable_offscreen_renderer=enable_offscreen_renderer,
        )
