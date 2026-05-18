import json
import os

from walk.utils.checkpoint_data import TrainCheckpointData
from walk.utils.data_types import DictionableDataclass


class TrainLogHandler:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, "trained_models")
        self.log_path = os.path.join(log_dir, "train_log.json")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.log_datas: list = []

    def get_path2save_model(self, num_timesteps: int):
        return os.path.join(self.model_dir, f"model_{num_timesteps}")

    def add_log_data(self, log_data: TrainCheckpointData):
        self.log_datas.append(log_data)

    def write_json_file(self):
        data = {
            "log_datas": [
                DictionableDataclass.to_dict(log_data) for log_data in self.log_datas
            ],
        }
        with open(self.log_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def load_log_data(self, checkpoint_data_type: type):
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        for log_data_json in json_data.get("log_datas", []):
            checkpoint_data = checkpoint_data_type(**log_data_json)
            self.log_datas.append(checkpoint_data)
