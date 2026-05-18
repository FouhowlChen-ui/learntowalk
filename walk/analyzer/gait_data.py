from __future__ import annotations

import json

import numpy as np

from walk.utils import numpy_utils


class GaitData:
    def __init__(self):
        self.series_data = {
            "joint_data": {},
            "actuator_data": {},
            "sensor_data": {},
            "physics_data": {"contacts": {"data": []}},
            "target_data": {"target_velocity": []},
        }
        self.metadata = {"data_length": 0}

    def add_data(self, *, mj_model, mj_data, target_velocity: float, printing=False):
        import mujoco

        self.metadata["data_length"] += 1

        for idx in range(mj_model.nu):
            actuator_name = mj_model.actuator(idx).name
            actuator_data = mj_data.actuator(actuator_name)
            actuator_dict = self.series_data["actuator_data"].setdefault(
                f"{actuator_name}", {"force": [], "velocity": [], "ctrl": []}
            )
            actuator_dict["force"].append(
                numpy_utils.numpy2array(actuator_data.force.copy())
            )
            actuator_dict["velocity"].append(
                numpy_utils.numpy2array(actuator_data.velocity.copy())
            )
            actuator_dict["ctrl"].append(
                numpy_utils.numpy2array(actuator_data.ctrl.copy())
            )
        for idx in range(mj_model.njnt):
            joint_name = mj_model.joint(idx).name
            joint_data = mj_data.joint(joint_name)
            joint_dict = self.series_data["joint_data"].setdefault(
                f"{joint_name}", {"qpos": [], "qvel": []}
            )
            joint_dict["qpos"].append(numpy_utils.numpy2array(joint_data.qpos.copy()))
            joint_dict["qvel"].append(numpy_utils.numpy2array(joint_data.qvel.copy()))
        for idx in range(mj_model.nsensor):
            sensor_name = mj_model.sensor(idx).name
            sensor_data = mj_data.sensor(sensor_name)
            sensor_dict = self.series_data["sensor_data"].setdefault(
                f"{sensor_name}", {"data": []}
            )
            sensor_dict["data"].append(numpy_utils.numpy2array(sensor_data.data.copy()))

        contacts = []
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(mj_model.ptr, mj_data.ptr, i, force)
            contacts.append(
                {
                    "pos": contact.pos.copy().tolist(),
                    "force": force[:3].tolist(),
                    "torque": force[3:].tolist(),
                    "geom1": mj_model.id2name(contact.geom1, "geom"),
                    "geom2": mj_model.id2name(contact.geom2, "geom"),
                }
            )
        self.series_data["physics_data"]["contacts"]["data"].append(contacts)
        self.series_data["target_data"]["target_velocity"].append([target_velocity])

    def apply_to_env(self, *, time_index: int, mj_model, mj_data):
        for idx in range(mj_model.nu):
            actuator_name = mj_model.actuator(idx).name
            actuator_data = mj_data.actuator(actuator_name)
            actuator_data.force = self.series_data["actuator_data"][actuator_name][
                "force"
            ][time_index]
            actuator_data.velocity = self.series_data["actuator_data"][actuator_name][
                "velocity"
            ][time_index]
            actuator_data.ctrl = self.series_data["actuator_data"][actuator_name][
                "ctrl"
            ][time_index]
        for idx in range(mj_model.njnt):
            joint_name = mj_model.joint(idx).name
            joint_data = mj_data.joint(joint_name)
            joint_data.qpos = self.series_data["joint_data"][joint_name]["qpos"][
                time_index
            ]
            joint_data.qvel = self.series_data["joint_data"][joint_name]["qvel"][
                time_index
            ]

    def save_json_data(self, path):
        data = {"series_data": self.series_data, "metadata": self.metadata}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def read_json_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data_loaded = json.load(f)
        self.series_data = data_loaded["series_data"]
        self.metadata = data_loaded["metadata"]
