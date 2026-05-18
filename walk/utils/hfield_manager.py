import math

import numpy as np


class HfieldManager:
    def __init__(self, sim, hfield_name: str, np_random):
        self._sim = sim
        self._model_hfield_geom = sim.model.geom(hfield_name)
        self._hfield = sim.model.hfield(hfield_name)
        self._hfield_pos = sim.model.geom(hfield_name).pos
        self._hfield_size = sim.model.geom(hfield_name).size
        self.np_random = np_random

        self._model_geom_ground_plane = sim.model.geom("ground-plane")
        self._data_geom_ground_plane = sim.data.geom("ground-plane")

        self._last_type = "flat"
        self._last_params = ""

    def set_hfield(self, type: str = "flat", params: str = ""):
        self._last_type = type
        self._last_params = params

        if params == "":
            params_float_list = []
        else:
            params_float_list = list(map(float, params.split(" ")))
        self._model_hfield_geom.rgba = [1, 1, 1, 1]
        self._model_hfield_geom.pos[2] = 0.0
        self._model_geom_ground_plane.rgba = [1, 1, 1, 0]

        if type == "flat":
            self._hfield.data[:] = 0.0
        elif type == "random":
            self._create_random_hfield(params_float_list)
            print(f"[hfield] random terrain enabled (amplitude={params_float_list})")
        elif type == "harmonic_sinusoidal":
            self._create_harmonic_sinusoidal_hfield(params_float_list)
            print(f"[hfield] harmonic_sinusoidal terrain enabled (params={params_float_list})")
        elif type == "slope":
            self._create_slope_hfield(params_float_list)
            print(f"[hfield] slope terrain enabled (slope={params_float_list})")
        elif type == "slope_random":
            self._create_slope_random_hfield(params_float_list)
            print(
                f"[hfield] slope_random enabled "
                f"(range={params_float_list}, current_slope={self._current_random_slope:+.4f})"
            )
        else:
            raise ValueError(f"Invalid terrain type: {type}")

    def resample(self):
        self.set_hfield(self._last_type, self._last_params)

    def _make_safe_zone(self, hfield_data):
        nrow, ncol = int(self._hfield.nrow), int(self._hfield.ncol)

        safezone_radius = 3.0
        tile_size_row = 2 * self._hfield.size[1] / nrow
        tile_size_col = 2 * self._hfield.size[0] / ncol
        center_index = int(
            (self._hfield_size[0] - self._hfield_pos[0]) / tile_size_col
        )
        tile_num_safezone = math.ceil(safezone_radius / tile_size_row)
        tile_num_safezone_col = math.ceil(safezone_radius / tile_size_col)

        center_row = nrow // 2
        center_col = center_index
        row_start = center_row - tile_num_safezone
        row_end = center_row + tile_num_safezone
        col_start = center_col - tile_num_safezone_col
        col_end = center_col + tile_num_safezone_col

        safezone_rows = np.arange(row_start, row_end)
        safezone_cols = np.arange(col_start, col_end)
        safezone_row_grid, safezone_col_grid = np.meshgrid(
            safezone_rows, safezone_cols, indexing="ij"
        )

        dist_from_center = np.sqrt(
            ((safezone_row_grid - center_row) * tile_size_row) ** 4
            + ((safezone_col_grid - center_col) * tile_size_col) ** 4
        )

        mask = np.clip(dist_from_center / safezone_radius, 0, 1)
        hfield_data[row_start:row_end, col_start:col_end] *= mask
        return hfield_data

    def _create_random_hfield(self, params: list):
        amplitude = params[0] if params else 0.05
        nrow, ncol = int(self._hfield.nrow), int(self._hfield.ncol)
        self._hfield.data[:] = self._make_safe_zone(
            self.np_random.uniform(low=0, high=amplitude, size=(nrow, ncol))
        )

    def _create_harmonic_sinusoidal_hfield(self, params: list):
        row_params = []
        col_params = []
        for idx in range(0, len(params), 4):
            amplitude_row = params[idx]
            row_period = params[idx + 1]
            amplitude_col = params[idx + 2]
            col_period = params[idx + 3]
            row_params.append((amplitude_row, row_period))
            col_params.append((amplitude_col, col_period))

        nrow, ncol = int(self._hfield.nrow), int(self._hfield.ncol)
        row_idx = np.arange(nrow)
        col_idx = np.arange(ncol)
        row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing="ij")

        tile_size_col = 2 * self._hfield.size[0] / ncol
        center_index = int(
            (self._hfield_size[0] - self._hfield_pos[0]) / tile_size_col
        )

        hfield_data = np.zeros_like(row_grid, dtype=np.float32)
        for amplitude, period in row_params:
            freq_row = 2 * np.pi / period
            hfield_data += amplitude * np.sin(freq_row * row_grid)
        for amplitude, period in col_params:
            freq_col = 2 * np.pi / period
            hfield_data += amplitude * np.sin(
                freq_col * col_grid - 2 * np.pi * center_index - 2 * np.pi / 4
            )
        min_val = np.min(hfield_data)
        if min_val < 0:
            hfield_data = hfield_data - min_val

        self._hfield.data[:] = self._make_safe_zone(hfield_data)

    def _create_slope_hfield(self, params: list):
        slope = params[0] if params else 0.05
        self._build_slope_data(slope)

    def _create_slope_random_hfield(self, params: list):
        if len(params) >= 2:
            slope_min = float(params[0])
            slope_max = float(params[1])
        elif len(params) == 1:
            mag = abs(float(params[0]))
            slope_min, slope_max = -mag, mag
        else:
            slope_min, slope_max = -0.10, 0.10

        slope = float(self.np_random.uniform(slope_min, slope_max))
        self._current_random_slope = slope
        self._build_slope_data(slope)

    def _build_slope_data(self, slope: float):
        nrow = int(np.asarray(self._hfield.nrow).item())
        ncol = int(np.asarray(self._hfield.ncol).item())
        tile_size_row = 2 * float(self._hfield.size[1]) / nrow
        tile_size_col = 2 * float(self._hfield.size[0]) / ncol
        center_index = int(
            (float(self._hfield_size[0]) - float(self._hfield_pos[0])) / tile_size_col
        ) + 5

        row_idx = np.arange(nrow)
        col_idx = np.arange(ncol)
        row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing="ij")

        hfield_data = np.zeros_like(row_grid, dtype=np.float32)
        offset = (col_grid - center_index) * slope * tile_size_row
        if slope >= 0:
            hfield_data += np.where(col_grid > center_index, offset, 0)
        else:
            hfield_data += np.where(col_grid > center_index, offset, 0)
            min_v = float(np.min(hfield_data))
            if min_v < 0:
                hfield_data = hfield_data - min_v

        self._hfield.data[:] = hfield_data
