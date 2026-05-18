import matplotlib
import platform

if platform.system() == "Darwin":
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
from walk.analyzer.gait_data import GaitData
import os
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import butter, filtfilt
import csv
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
#
# ============================================================

#
PUBLICATION_LR_COLOR_RIGHT = "#B22222"   #
PUBLICATION_LR_COLOR_LEFT = "#4682B4"    #
PUBLICATION_SIM_REF_SIM_COLOR = PUBLICATION_LR_COLOR_RIGHT
PUBLICATION_SIM_REF_REF_COLOR = PUBLICATION_LR_COLOR_LEFT
#
PUBLICATION_SIM_REF_EXO_MATCH_SIM = "#A52A2A"
PUBLICATION_SIM_REF_EXO_MATCH_REF = "#4682B4"
PUBLICATION_LR_LEGEND_RIGHT = "Right (Stance Start)"
PUBLICATION_LR_LEGEND_LEFT = "Left (Stance Start)"
PUBLICATION_LR_RIBBON_ALPHA = 0.2

#
PUBLICATION_EXO_MEAN_TOE_LINE_BLUE = "#A7C0DE"

#
PUBLICATION_AXIS_SPINE_LW = 0.9
PUBLICATION_AXIS_TICK_LENGTH = 4.0
PUBLICATION_AXIS_TICK_WIDTH = 0.8

#
PUBLICATION_LABEL_FONTSIZE = 12
PUBLICATION_SUBPLOT_TITLE_FONTSIZE = 13
PUBLICATION_TICK_FONTSIZE = 11
PUBLICATION_LEGEND_FONTSIZE = 11
PUBLICATION_SUPTITLE_FONTSIZE = 16
#
PUBLICATION_MUSCLE_SINGLE_LEGEND_FONTSIZE = 9
#
PUBLICATION_RENDER_DPI = 200
PUBLICATION_SAVEFIG_DPI = 200

#
MUSCLE_PUB_ON_LINE = PUBLICATION_LR_COLOR_RIGHT     #
MUSCLE_PUB_OFF_LINE = PUBLICATION_LR_COLOR_LEFT      #
MUSCLE_PUB_SINGLE_LINE = PUBLICATION_LR_COLOR_LEFT   #
MUSCLE_PUB_RIBBON_ALPHA = 0.22
MUSCLE_PUB_Y_MAX = 100.0

#
MUSCLE_FUNCTIONAL_WINDOWS = {
    "glutmax":    (0, 30),
    "hamstrings": (85, 115),
    "iliopsoas":  (35, 75),
    "soleus":     (20, 50),
    "gastroc":    (20, 50),
    "tibant":     (60, 100),
    "edl":        (60, 100),
    "fdl":        (20, 50),
    "rectfem":    (50, 75),
    "vasti":      (0, 30),
    "bifemsh":    (75, 100),
}


def style_publication_axes(ax, *, labelbottom=True, labelleft=True):
    """"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(PUBLICATION_AXIS_SPINE_LW)
        ax.spines[side].set_color("black")
    ax.tick_params(
        axis="both", which="major", direction="in",
        length=PUBLICATION_AXIS_TICK_LENGTH, width=PUBLICATION_AXIS_TICK_WIDTH,
        bottom=True, top=False, left=True, right=False,
        labelbottom=labelbottom, labelleft=labelleft,
        labelsize=plt.rcParams.get("xtick.labelsize", PUBLICATION_TICK_FONTSIZE),
    )


def apply_publication_rcparams():
    """"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": PUBLICATION_LABEL_FONTSIZE,
        "axes.titlesize": PUBLICATION_SUBPLOT_TITLE_FONTSIZE,
        "xtick.labelsize": PUBLICATION_TICK_FONTSIZE,
        "ytick.labelsize": PUBLICATION_TICK_FONTSIZE,
        "legend.fontsize": PUBLICATION_LEGEND_FONTSIZE,
        "axes.linewidth": PUBLICATION_AXIS_SPINE_LW,
        "xtick.major.width": PUBLICATION_AXIS_TICK_WIDTH,
        "ytick.major.width": PUBLICATION_AXIS_TICK_WIDTH,
        "xtick.major.size": PUBLICATION_AXIS_TICK_LENGTH,
        "ytick.major.size": PUBLICATION_AXIS_TICK_LENGTH,
        "xtick.direction": "in", "ytick.direction": "in",
        "lines.linewidth": 1.35, "axes.grid": False,
        "figure.dpi": PUBLICATION_RENDER_DPI,
        "savefig.dpi": PUBLICATION_SAVEFIG_DPI,
        "axes.unicode_minus": False,
    })


# ============================================================
#
# ============================================================

#
PUBLICATION_JOINT_Y_AXES: List[Tuple[Tuple[float, float], np.ndarray]] = [
    ((-40, 40), np.arange(-40, 41, 20)),
    ((-80, 20), np.arange(-80, 21, 20)),
    ((-30, 30), np.arange(-30, 31, 15)),
]
PUBLICATION_JOINT_YLABELS_EN = ["Hip Angle (deg)", "Knee Angle (deg)", "Ankle Angle (deg)"]
#
PUBLICATION_LR_JOINT_KEYS = [
    ("hip_flexion_r", "hip_flexion_l"),
    ("knee_angle_r", "knee_angle_l"),
    ("ankle_angle_r", "ankle_angle_l"),
]
#
MUSCLE_PUB_GRID_ROWS, MUSCLE_PUB_GRID_COLS = 3, 4
#
SEGMENTED_JOINT_GRID_POS = {
    "hip_flexion_l": (0, 0), "hip_flexion_r": (0, 1),
    "knee_angle_l": (1, 0), "knee_angle_r": (1, 1),
    "ankle_angle_l": (2, 0), "ankle_angle_r": (2, 1),
}


def _pub_standard_gait_x_axis(ax, *, show_xlabel: bool = False) -> None:
    """"""
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(False)
    if show_xlabel:
        ax.set_xlabel("Gait cycle (%)")


def _pub_zero_reference_hline(ax) -> None:
    """"""
    ax.axhline(0.0, color="0.75", linewidth=0.85, linestyle="-", zorder=1)


def _pub_toe_off_line_neutral(ax, toe_pct: float) -> None:
    """"""
    ax.axvline(
        float(np.clip(toe_pct, 0.0, 100.0)),
        color="0.15", linewidth=0.95, linestyle=":", zorder=2,
    )


def _muscle_subplot_frame(ax, i: int, toe_off: float, nrows: int, ncols: int) -> None:
    """"""
    _pub_zero_reference_hline(ax)
    _pub_toe_off_line_neutral(ax, toe_off)
    _pub_standard_gait_x_axis(ax, show_xlabel=(i // ncols == nrows - 1))
    style_publication_axes(ax, labelbottom=True, labelleft=True)


def style_lr_joint_axes_3x2(
    axes,
    toe_off_l: float,
    toe_off_r: float,
    color_l: str,
    color_r: str,
) -> None:
    """"""
    for row, ((ylim, yticks), ylabel) in enumerate(
        zip(PUBLICATION_JOINT_Y_AXES, PUBLICATION_JOINT_YLABELS_EN)
    ):
        for col in (0, 1):
            axes[row][col].set_ylim(*ylim)
            axes[row][col].set_yticks(yticks)
        axes[row][0].axvline(toe_off_l, color=color_l, linestyle=":", linewidth=1.2, alpha=0.8, zorder=2)
        axes[row][1].axvline(toe_off_r, color=color_r, linestyle=":", linewidth=1.2, alpha=0.8, zorder=2)
        axes[row][0].set_ylabel(ylabel)
    for row in range(3):
        for col in (0, 1):
            ax = axes[row][col]
            _pub_standard_gait_x_axis(ax, show_xlabel=(row == 2))
            _pub_zero_reference_hline(ax)
            style_publication_axes(ax, labelbottom=True, labelleft=True)


def add_figure_lr_column_titles(fig, axes_3x2, titles: Tuple[str, str] = ("Left leg", "Right leg")) -> None:
    """"""
    p0 = axes_3x2[0][0].get_position()
    p1 = axes_3x2[0][1].get_position()
    fs = plt.rcParams.get("axes.titlesize", 11)
    y = max(p0.y1, p1.y1) + 0.018
    fig.text(p0.x0 + 0.5 * p0.width, y, titles[0], ha="center", va="bottom", fontsize=fs, transform=fig.transFigure)
    fig.text(p1.x0 + 0.5 * p1.width, y, titles[1], ha="center", va="bottom", fontsize=fs, transform=fig.transFigure)


def _auto_ylim_pad(lo: float, hi: float, *, nonnegative_floor: bool = False) -> Tuple[float, float]:
    """"""
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = float(max(abs(lo), abs(hi), 0.0)) * 0.2
        if not np.isfinite(span) or span <= 0:
            span = 1.0
    pad = 0.08 * max(span, 1e-9)
    y_lo, y_hi = lo - pad, hi + pad
    if nonnegative_floor and np.isfinite(lo) and lo >= 0:
        y_lo = 0.0
    return y_lo, y_hi


def _muscle_grid_figsize(fig_size_multiplier: float) -> Tuple[float, float]:
    """"""
    w = MUSCLE_PUB_GRID_COLS * 3.15 * fig_size_multiplier
    h = MUSCLE_PUB_GRID_ROWS * 2.55 * fig_size_multiplier
    return w, h


def _figure_legend_mean_sd(fig, line_color: str, *, transform_axes=None) -> None:
    """"""
    fig.legend(
        handles=[Line2D([0], [0], color=line_color, linestyle="-", label="Mean ± SD")],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.948),
        bbox_transform=fig.transFigure if transform_axes is None else transform_axes,
        ncol=1,
        frameon=False,
        fontsize=PUBLICATION_LEGEND_FONTSIZE,
        handlelength=1.55,
        handletextpad=0.45,
        borderaxespad=0.2,
    )


def _plot_lr_mean_sd_on_ax(
    ax,
    x: np.ndarray,
    mean_r: np.ndarray,
    std_r: np.ndarray,
    mean_l: np.ndarray,
    std_l: np.ndarray,
    color_r: str,
    color_l: str,
    alpha_fill: float,
    *,
    legend_right: Optional[str] = None,
    legend_left: Optional[str] = None,
) -> None:
    """"""
    ax.plot(x, mean_r, color=color_r, label=legend_right)
    ax.fill_between(x, mean_r - std_r, mean_r + std_r, color=color_r, alpha=alpha_fill)
    ax.plot(x, mean_l, color=color_l, linestyle="--", dashes=(5, 3), label=legend_left)
    ax.fill_between(x, mean_l - std_l, mean_l + std_l, color=color_l, alpha=alpha_fill)


# ============================================================
#
# ============================================================

def _compute_functional_mean(muscle_name: str, mean_curve_101: np.ndarray) -> float:
    """"""
    base = muscle_name.rsplit("_", 1)[0]
    window = MUSCLE_FUNCTIONAL_WINDOWS.get(base)
    if window is None:
        return float(np.mean(mean_curve_101))
    start_pct, end_pct = window
    if end_pct <= 100:
        return float(np.mean(mean_curve_101[start_pct:end_pct + 1]))
    part1 = mean_curve_101[start_pct:101]
    part2 = mean_curve_101[0:end_pct - 100 + 1]
    return float(np.mean(np.concatenate([part1, part2])))


def _get_functional_window_label(muscle_name: str) -> str:
    """"""
    base = muscle_name.rsplit("_", 1)[0]
    window = MUSCLE_FUNCTIONAL_WINDOWS.get(base)
    if window is None:
        return "0-100%"
    if window[1] > 100:
        return f"{window[0]}-100%+0-{window[1]-100}%GC"
    return f"{window[0]}-{window[1]}%GC"


def short_muscle_name(full: str) -> str:
    """"""
    return full.replace("_r", "").replace("_l", "").replace("_R", "").replace("_L", "")


def leg_label(is_right: bool) -> str:
    """"""
    return "Right leg" if is_right else "Left leg"


def _pearson_r(a, b):
    """Pearson correlation between two 1-D arrays; NaN if constant or too short."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class GaitAnalyzer:
    JOINT_NAMES = {
        'HIP': "hip",
        'KNEE': "knee",
        'ANKLE': "ankle",
        'LEFT_HIP': "left hip",
        'RIGHT_HIP': "right hip",
        'LEFT_KNEE': "left knee",
        'RIGHT_KNEE': "right knee",
        'LEFT_ANKLE': "left ankle",
        'RIGHT_ANKLE': "right ankle"
    }
    # Modified: Increased HIP range for better visibility of flexion
    JOINT_LIMIT = {
        "HIP": (-40, 40),  #
        "KNEE": (-70, 10),
        "ANKLE": (-30, 25),
    }

    def __init__(self, gait_data: GaitData, segmented_ref_data: dict, show_plot: bool):
        self.gait_data = gait_data
        self.segmented_ref_data = segmented_ref_data
        self.show_plot = show_plot
        self.fig_size_multiplier = 1
        self.dpi = 300

        self.toe_off_color = "#000000"
        self.toe_off_linestyle = "--"
        self.toe_off_linewidth = 1
        self.toe_off_alpha = 0.6
        plt.ioff()  # Turn off interactive mode

    def get_gait_segment_index(self, *, is_right_foot_based: bool):
        """"""
        primary_char = "r" if is_right_foot_based else "l"
        secondary_char = "l" if is_right_foot_based else "r"

        result_strike_to_toe_off = []
        sensor_data = self.gait_data.series_data["sensor_data"]
        foot_threshold = 0.1

        primary_stance_ing = False
        primary_stance_start_idx = None
        for idx, (primary_foot, primary_toes) in enumerate(
                zip([v[0] for v in sensor_data[f"{primary_char}_foot"]["data"]],
                    [v[0] for v in sensor_data[f"{primary_char}_toes"]["data"]])):

            if idx > 0:
                prev_primary_combined = sensor_data[f"{primary_char}_foot"]["data"][idx - 1][0] + \
                                        sensor_data[f"{primary_char}_toes"]["data"][idx - 1][0]
                curr_primary_combined = sensor_data[f"{primary_char}_foot"]["data"][idx][0] + \
                                        sensor_data[f"{primary_char}_toes"]["data"][idx][0]

                prev_primary_foot = sensor_data[f"{primary_char}_foot"]["data"][idx - 1][0]
                curr_primary_foot = sensor_data[f"{primary_char}_foot"]["data"][idx][0]
                prev_primary_toes = sensor_data[f"{primary_char}_toes"]["data"][idx - 1][0]
                curr_primary_toes = sensor_data[f"{primary_char}_toes"]["data"][idx][0]

                is_primary_foot_down = prev_primary_foot < foot_threshold and curr_primary_foot >= foot_threshold
                is_primary_toe_off = prev_primary_toes > foot_threshold and curr_primary_toes <= foot_threshold
                is_primary_rising_edge = prev_primary_combined < foot_threshold and curr_primary_combined >= foot_threshold
                is_primary_falling_edge = prev_primary_combined >= foot_threshold and curr_primary_combined < foot_threshold

                if is_primary_rising_edge and is_primary_foot_down and primary_stance_start_idx is None:
                    primary_stance_ing = True
                    primary_stance_start_idx = idx
                elif is_primary_falling_edge and is_primary_toe_off and primary_stance_ing:
                    primary_stance_ing = False
                    if primary_stance_start_idx is not None:
                        result_strike_to_toe_off.append((primary_stance_start_idx, idx))
                    primary_stance_start_idx = None

        result_strike_to_strike = []
        for idx in range(len(result_strike_to_toe_off) - 1):
            result_strike_to_strike.append((result_strike_to_toe_off[idx][0], result_strike_to_toe_off[idx][1],
                                            result_strike_to_toe_off[idx + 1][0]))
        return result_strike_to_strike[1:]

    def get_toe_off_average(self, *, is_right_foot_based: bool):
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=is_right_foot_based)
        toe_off_cycles = []
        for start_idx, toe_off_idx, end_idx in gait_segment_index:
            toe_off_cycle = (toe_off_idx - start_idx) / (end_idx - start_idx)
            toe_off_cycles.append(toe_off_cycle)
        return np.mean(toe_off_cycles)

    def plot_entire_result(self, *, result_dir, is_right_foot_based: bool) -> None:
        fig, axes = plt.subplots(6, 2, figsize=(12 * self.fig_size_multiplier, 6 * self.fig_size_multiplier),
                                 dpi=self.dpi)

        joint_data = self.gait_data.series_data["joint_data"]

        axes[0][0].set_title(self.JOINT_NAMES['LEFT_HIP'])
        axes[0][0].plot(np.rad2deg([v[0] for v in joint_data["hip_flexion_l"]["qpos"]]))

        axes[1][0].set_title(self.JOINT_NAMES['LEFT_KNEE'])
        axes[1][0].plot(np.rad2deg([v[0] for v in joint_data["knee_angle_l"]["qpos"]]))

        axes[2][0].set_title(self.JOINT_NAMES['LEFT_ANKLE'])
        axes[2][0].plot(np.rad2deg([v[0] for v in joint_data["ankle_angle_l"]["qpos"]]))

        axes[0][1].set_title(self.JOINT_NAMES['RIGHT_HIP'])
        axes[0][1].plot(np.rad2deg([v[0] for v in joint_data["hip_flexion_r"]["qpos"]]))

        axes[1][1].set_title(self.JOINT_NAMES['RIGHT_KNEE'])
        axes[1][1].plot(np.rad2deg([v[0] for v in joint_data["knee_angle_r"]["qpos"]]))

        axes[2][1].set_title(self.JOINT_NAMES['RIGHT_ANKLE'])
        axes[2][1].plot(np.rad2deg([v[0] for v in joint_data["ankle_angle_r"]["qpos"]]))

        axes[3][0].set_title("pelvis x position")
        axes[3][0].plot([v[0] for v in joint_data["pelvis_tx"]["qpos"]])
        axes[3][1].set_title("pelvis y position")
        axes[3][1].plot([v[0] for v in joint_data["pelvis_ty"]["qpos"]])

        axes[4][0].set_title("pelvis x velocity")
        axes[4][0].plot([v[0] for v in joint_data["pelvis_tx"]["qvel"]])
        if "target_data" in self.gait_data.series_data and "target_velocity" in self.gait_data.series_data[
            "target_data"]:
            axes[4][0].plot([v[0] for v in self.gait_data.series_data["target_data"]["target_velocity"]])

        sensor_data = self.gait_data.series_data["sensor_data"]

        axes[5][0].set_title("left sensor")
        axes[5][0].plot([v[0] for v in sensor_data["l_foot"]["data"]], label="l_foot")
        axes[5][0].plot([v[0] for v in sensor_data["l_toes"]["data"]], label="l_toes")

        axes[5][1].set_title("right sensor")
        axes[5][1].plot([v[0] for v in sensor_data["r_foot"]["data"]], label="r_foot")
        axes[5][1].plot([v[0] for v in sensor_data["r_toes"]["data"]], label="r_toes")
        axes[5][1].plot([v[0] for v in sensor_data["l_foot"]["data"]], label="l_foot", alpha=0.3)
        axes[5][1].plot([v[0] for v in sensor_data["l_toes"]["data"]], label="l_toes", alpha=0.3)

        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=is_right_foot_based)
        for ax_row in axes:
            for ax in ax_row:
                for start_idx, toe_off_idx, end_idx in gait_segment_index:
                    ax.axvspan(start_idx, toe_off_idx, color='#00ff00', alpha=0.1)

        axes[0][0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1][0].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2][0].set_ylim(*self.JOINT_LIMIT['ANKLE'])
        axes[0][1].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1][1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2][1].set_ylim(*self.JOINT_LIMIT['ANKLE'])

        postfix = "_right_based" if is_right_foot_based else "_left_based"
        fig.tight_layout()
        fig.savefig(
            os.path.join(result_dir, f"kinematics_data{postfix}.png"),
            bbox_inches="tight", dpi=PUBLICATION_SAVEFIG_DPI, pad_inches=0.03,
        )

        if self.show_plot:
            plt.show()
        plt.close()

    def joint_angle_by_velocity(self, *, result_dir: str):
        """"""
        self._apply_publication_rcparams()
        #
        _axis_title_fs = PUBLICATION_LABEL_FONTSIZE + 3
        ref_line_color = "#555555"
        ref_line_style = "--"
        joint_data = self.gait_data.series_data["joint_data"]

        ylabels = ["Hip Angle (deg)", "Knee Angle (deg)", "Ankle Angle (deg)"]
        lims = [(-40, 40), (-80, 20), (-30, 30)]
        ytk = [np.arange(-40, 41, 20), np.arange(-80, 21, 20), np.arange(-30, 31, 15)]
        toe_off_pct = self.get_toe_off_average(is_right_foot_based=True) * 100
        color_sim = PUBLICATION_SIM_REF_SIM_COLOR

        fig, axes = plt.subplots(
            3, 1, figsize=(5 * self.fig_size_multiplier, 9 * self.fig_size_multiplier),
            dpi=PUBLICATION_RENDER_DPI, sharex=True,
        )
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=True)

        #
        axes[0].plot(
            np.rad2deg(self.segmented_ref_data["q_hip_flexion_r"]),
            label="Reference",
            color=ref_line_color,
            linestyle=ref_line_style,
            zorder=2,
        )
        axes[1].plot(
            np.rad2deg(self.segmented_ref_data["q_knee_angle_r"]),
            label="Reference",
            color=ref_line_color,
            linestyle=ref_line_style,
            zorder=2,
        )
        axes[2].plot(
            np.rad2deg(self.segmented_ref_data["q_ankle_angle_r"]),
            label="Reference",
            color=ref_line_color,
            linestyle=ref_line_style,
            zorder=2,
        )

        def lowpass_filter(data, cutoff=2.0, fs=100.0, order=2):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data)
            return y

        actual_speed = [v[0] for v in joint_data["pelvis_tx"]["qvel"]]
        if len(actual_speed) > 100:
            actual_speed_smooth = lowpass_filter(np.array(actual_speed), cutoff=1.0, fs=30.0, order=2)
        else:
            actual_speed_smooth = np.array(actual_speed)

        vel_min = np.floor(np.min(actual_speed_smooth) * 10) / 10
        vel_max = np.ceil(np.max(actual_speed_smooth) * 10) / 10
        vel_range = (vel_min, vel_max)
        norm = mcolors.Normalize(vmin=vel_range[0], vmax=vel_range[1])
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "blue_purple_red", ["#0000ff", "#eeeeee", "#ff0000"]
        )

        def interp_array(x, y, num_points):
            x_new = np.linspace(x[0], x[-1], num_points)
            y_new = np.interp(x_new, x, y)
            return x_new, y_new

        def plot_colored_line(ax, x, y, c, label=None):
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.array(c))
            lc.set_linewidth(plt.rcParams["lines.linewidth"])
            lc.set_zorder(3)
            ax.add_collection(lc)
            if label:
                ax.plot([], [], color=cmap(norm(np.mean(c))), label=label)

        interp_points = 400
        #
        for irow, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index):
            segment_length = end_idx - start_idx
            x_normalized = np.linspace(0, 100, segment_length)

            hip_flexion_r_data = np.rad2deg(
                [v[0] for v in joint_data["hip_flexion_r"]["qpos"]][start_idx:end_idx]
            )
            knee_angle_r_data = np.rad2deg(
                [v[0] for v in joint_data["knee_angle_r"]["qpos"]][start_idx:end_idx]
            )
            ankle_angle_r_data = np.rad2deg(
                [v[0] for v in joint_data["ankle_angle_r"]["qpos"]][start_idx:end_idx]
            )
            actual_speed_smooth_segment = actual_speed_smooth[start_idx:end_idx]

            x_hip_r, hip_flexion_r_data_interp = interp_array(
                x_normalized, hip_flexion_r_data, interp_points
            )
            x_knee_r, knee_angle_r_data_interp = interp_array(
                x_normalized, knee_angle_r_data, interp_points
            )
            x_ankle_r, ankle_angle_r_data_interp = interp_array(
                x_normalized, ankle_angle_r_data, interp_points
            )
            _, actual_speed_interp = interp_array(
                x_normalized, actual_speed_smooth_segment, interp_points
            )

            plot_colored_line(
                axes[0],
                x_hip_r,
                hip_flexion_r_data_interp,
                actual_speed_interp,
                label="Hip Flexion R" if irow == 0 else None,
            )
            plot_colored_line(
                axes[1],
                x_knee_r,
                knee_angle_r_data_interp,
                actual_speed_interp,
                label="Knee Angle R" if irow == 0 else None,
            )
            plot_colored_line(
                axes[2],
                x_ankle_r,
                ankle_angle_r_data_interp,
                actual_speed_interp,
                label="Ankle Angle R" if irow == 0 else None,
            )

        for i, ax in enumerate(axes):
            ax.set_ylim(*lims[i])
            ax.set_yticks(ytk[i])
            ax.set_xlim(0, 100)
            ax.set_ylabel(ylabels[i], fontsize=_axis_title_fs)
            ax.axhline(0.0, color="0.75", linewidth=0.85, zorder=1)
            ax.axvline(
                toe_off_pct,
                color=color_sim,
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
                zorder=1,
            )
            ax.grid(False)
            style_publication_axes(
                ax,
                labelbottom=True,
                labelleft=True,
            )
            ax.tick_params(axis="both", which="major", labelsize=PUBLICATION_TICK_FONTSIZE + 1)

        axes[2].set_xlabel("Gait cycle (%)", fontsize=_axis_title_fs)

        #
        plt.subplots_adjust(left=0.14, right=0.96, bottom=0.11, top=0.84, hspace=0.22)

        pos0 = axes[0].get_position()
        _cb_scale = 0.7
        _cb_w = min(pos0.width * 0.62, pos0.width) * _cb_scale
        #
        _cb_h = max(pos0.height * 0.13, 0.022) * _cb_scale
        _cax_left = pos0.x1 - _cb_w
        #
        _v_gap = 0.06
        _cax_bottom = pos0.y1 + _v_gap
        _cb_label_fs = PUBLICATION_TICK_FONTSIZE
        _cb_tick_fs = PUBLICATION_TICK_FONTSIZE - 1
        cax = fig.add_axes([_cax_left, _cax_bottom, _cb_w, _cb_h])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
        )
        cb.set_label("Speed (m/s)", fontsize=_cb_label_fs, labelpad=5)
        cb.ax.tick_params(labelsize=_cb_tick_fs, direction="in", pad=2)

        fig.savefig(
            os.path.join(result_dir, "joint_angle_cmap_by_velocity.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
        )

        if self.show_plot:
            plt.show()
        plt.close()

    def plot_exo_segmented_data(self, *, result_dir) -> None:
        #
        self._apply_publication_rcparams()
        _exo_seg_title_fs = PUBLICATION_SUBPLOT_TITLE_FONTSIZE
        _toe_blue = PUBLICATION_EXO_MEAN_TOE_LINE_BLUE
        _cmap_seg = plt.cm.get_cmap("tab10")

        exo_data_l = self.gait_data.series_data["actuator_data"]["Exo_L"]
        exo_data_r = self.gait_data.series_data["actuator_data"]["Exo_R"]
        right_gait_segment_index = self.get_gait_segment_index(is_right_foot_based=True)
        left_gait_segment_index = self.get_gait_segment_index(is_right_foot_based=False)
        x_mapped = np.linspace(0, 100, num=100)
        exo_data_mapped = {
            "Exo_L": [],
            "Exo_R": [],
        }

        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100

        #
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8.5 * self.fig_size_multiplier, 3.4 * self.fig_size_multiplier),
            dpi=PUBLICATION_RENDER_DPI,
        )

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(right_gait_segment_index):
            exo_r_data = [v[0] for v in exo_data_r["force"][start_idx:end_idx]]
            exo_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(exo_r_data)), exo_r_data)
            exo_data_mapped["Exo_R"].append(exo_r_mapped)
            c = _cmap_seg(idx % 10)
            axes[1].plot(x_mapped, exo_r_mapped, color=c, linewidth=1.35)

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(left_gait_segment_index):
            exo_l_data = [v[0] for v in exo_data_l["force"][start_idx:end_idx]]
            exo_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(exo_l_data)), exo_l_data)
            exo_data_mapped["Exo_L"].append(exo_l_mapped)
            c = _cmap_seg(idx % 10)
            axes[0].plot(x_mapped, exo_l_mapped, color=c, linewidth=1.35)

        axes[0].set_title("Hip Exo L", fontsize=_exo_seg_title_fs)
        axes[1].set_title("Hip Exo R", fontsize=_exo_seg_title_fs)

        for i, ax in enumerate(axes):
            ax.set_ylabel("Exo Torque (Nm)")
            ax.set_xlabel("Gait cycle (%)")
            ax.set_ylim(-12, 12)
            ax.set_yticks(np.array([-12, 0, 12]))
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.axhline(0.0, color="0.75", linewidth=0.85, linestyle="-", zorder=1)
            _toe = toe_off_l if i == 0 else toe_off_r
            ax.axvline(
                _toe,
                color=_toe_blue,
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )
            ax.grid(False)
            style_publication_axes(ax, labelbottom=True, labelleft=True)

        fig.tight_layout(pad=1.2, w_pad=2.0)
        fig.savefig(
            os.path.join(result_dir, "exo_segmented_force.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
        )
        if self.show_plot:
            plt.show()
        plt.close(fig)

        #
        _exo_ms_title_fs = PUBLICATION_SUBPLOT_TITLE_FONTSIZE
        _exo_blue = PUBLICATION_LR_COLOR_LEFT
        _exo_alpha = PUBLICATION_LR_RIBBON_ALPHA
        _toe_blue = PUBLICATION_EXO_MEAN_TOE_LINE_BLUE

        fig_mean_std, axes_mean_std = plt.subplots(
            1,
            2,
            figsize=(8.5 * self.fig_size_multiplier, 3.4 * self.fig_size_multiplier),
            dpi=PUBLICATION_RENDER_DPI,
        )

        for exo, data in exo_data_mapped.items():
            data = np.array(data)
            mean_data = np.mean(data, axis=0)
            std_data = np.std(data, axis=0)

            if exo == "Exo_L":
                ax = axes_mean_std[0]
            else:
                ax = axes_mean_std[1]

            ax.plot(x_mapped, mean_data, color=_exo_blue)
            ax.fill_between(
                x_mapped,
                mean_data - std_data,
                mean_data + std_data,
                color=_exo_blue,
                alpha=_exo_alpha,
            )

        axes_mean_std[0].set_title("Hip Exo L", fontsize=_exo_ms_title_fs)
        axes_mean_std[1].set_title("Hip Exo R", fontsize=_exo_ms_title_fs)

        for ax in axes_mean_std:
            ax.set_ylabel("Exo Torque (Nm)")
            ax.set_xlabel("Gait cycle (%)")
            ax.set_ylim(-12, 12)
            ax.set_yticks(np.array([-12, 0, 12]))
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.axhline(0.0, color="0.75", linewidth=0.85, linestyle="-", zorder=1)
            ax.grid(False)
            style_publication_axes(ax, labelbottom=True, labelleft=True)

        axes_mean_std[0].axvline(
            toe_off_l, color=_toe_blue, linestyle=":", linewidth=1.0, alpha=0.9, zorder=2
        )
        axes_mean_std[1].axvline(
            toe_off_r, color=_toe_blue, linestyle=":", linewidth=1.0, alpha=0.9, zorder=2
        )

        fig_mean_std.tight_layout(pad=1.2, w_pad=2.0)
        fig_mean_std.savefig(
            os.path.join(result_dir, "exo_mean_std_data.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
        )
        if self.show_plot:
            plt.show()
        plt.close(fig_mean_std)

    def plot_segmented_kinematics_result(self, *, result_dir) -> None:
        """"""
        self._apply_publication_rcparams()
        color_l = PUBLICATION_LR_COLOR_LEFT
        color_r = PUBLICATION_LR_COLOR_RIGHT
        alpha_fill = PUBLICATION_LR_RIBBON_ALPHA

        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100

        joint_data = self.gait_data.series_data["joint_data"]
        gait_segment_index_r = self.get_gait_segment_index(is_right_foot_based=True)
        gait_segment_index_l = self.get_gait_segment_index(is_right_foot_based=False)
        x_mapped = np.linspace(0, 100, num=100)
        joint_data_mapped_degree = {
            "hip_flexion_l": [], "hip_flexion_r": [],
            "knee_angle_l": [], "knee_angle_r": [],
            "ankle_angle_l": [], "ankle_angle_r": [],
        }

        n_seg_max = max(len(gait_segment_index_r), len(gait_segment_index_l), 1)
        _cmap_name = "tab20" if n_seg_max > 10 else "tab10"
        _cmap_seg = plt.cm.get_cmap(_cmap_name)
        _n_c = 20 if n_seg_max > 10 else 10

        fig, axes = plt.subplots(
            3, 2,
            figsize=(11 * self.fig_size_multiplier, 9 * self.fig_size_multiplier),
            dpi=PUBLICATION_RENDER_DPI,
        )

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index_r):
            hip_flexion_r_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_r"]["qpos"][start_idx:end_idx]])
            knee_angle_r_data = np.rad2deg([v[0] for v in joint_data["knee_angle_r"]["qpos"][start_idx:end_idx]])
            ankle_angle_r_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_r"]["qpos"][start_idx:end_idx]])

            joint_data_mapped_degree["hip_flexion_r"].append(
                np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_r_data)), hip_flexion_r_data))
            joint_data_mapped_degree["knee_angle_r"].append(
                np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_r_data)), knee_angle_r_data))
            joint_data_mapped_degree["ankle_angle_r"].append(
                np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_r_data)), ankle_angle_r_data))

            c = _cmap_seg(idx % _n_c)
            lw_seg = 1.1
            axes[0][1].plot(x_mapped, joint_data_mapped_degree["hip_flexion_r"][-1], color=c, linewidth=lw_seg, alpha=0.92)
            axes[1][1].plot(x_mapped, joint_data_mapped_degree["knee_angle_r"][-1], color=c, linewidth=lw_seg, alpha=0.92)
            axes[2][1].plot(x_mapped, joint_data_mapped_degree["ankle_angle_r"][-1], color=c, linewidth=lw_seg, alpha=0.92)

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index_l):
            hip_flexion_l_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_l"]["qpos"][start_idx:end_idx]])
            knee_angle_l_data = np.rad2deg([v[0] for v in joint_data["knee_angle_l"]["qpos"][start_idx:end_idx]])
            ankle_angle_l_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_l"]["qpos"][start_idx:end_idx]])

            joint_data_mapped_degree["hip_flexion_l"].append(
                np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_l_data)), hip_flexion_l_data))
            joint_data_mapped_degree["knee_angle_l"].append(
                np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_l_data)), knee_angle_l_data))
            joint_data_mapped_degree["ankle_angle_l"].append(
                np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_l_data)), ankle_angle_l_data))

            c = _cmap_seg(idx % _n_c)
            lw_seg = 1.1
            axes[0][0].plot(x_mapped, joint_data_mapped_degree["hip_flexion_l"][-1], color=c, linewidth=lw_seg, alpha=0.92)
            axes[1][0].plot(x_mapped, joint_data_mapped_degree["knee_angle_l"][-1], color=c, linewidth=lw_seg, alpha=0.92)
            axes[2][0].plot(x_mapped, joint_data_mapped_degree["ankle_angle_l"][-1], color=c, linewidth=lw_seg, alpha=0.92)

        style_lr_joint_axes_3x2(axes, toe_off_l, toe_off_r, color_l, color_r)

        fig.tight_layout(pad=1.0, w_pad=1.4, h_pad=1.0, rect=[0.0, 0.0, 1.0, 0.93])
        add_figure_lr_column_titles(fig, axes)
        fig.savefig(
            os.path.join(result_dir, "segmented_joint_data.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
        )
        if self.show_plot:
            plt.show()
        plt.close(fig)

        #
        fig2, axes2 = plt.subplots(
            3, 2,
            figsize=(11 * self.fig_size_multiplier, 9 * self.fig_size_multiplier),
            dpi=PUBLICATION_RENDER_DPI,
        )

        lw_mean = plt.rcParams["lines.linewidth"]
        for joint, data in joint_data_mapped_degree.items():
            if joint not in SEGMENTED_JOINT_GRID_POS:
                continue
            row, col = SEGMENTED_JOINT_GRID_POS[joint]
            ax = axes2[row][col]
            lc = color_l if col == 0 else color_r
            data = np.array(data)
            mean_data_degree = np.mean(data, axis=0)
            std_data_degree = np.std(data, axis=0)
            ax.plot(x_mapped, mean_data_degree, color=lc, linewidth=lw_mean)
            ax.fill_between(
                x_mapped,
                mean_data_degree - std_data_degree,
                mean_data_degree + std_data_degree,
                color=lc,
                alpha=alpha_fill,
            )

        style_lr_joint_axes_3x2(axes2, toe_off_l, toe_off_r, color_l, color_r)

        fig2.tight_layout(pad=1.0, w_pad=1.4, h_pad=1.0, rect=[0.0, 0.0, 1.0, 0.93])
        add_figure_lr_column_titles(fig2, axes2)
        fig2.savefig(
            os.path.join(result_dir, "segmented_joint_data_avg.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
        )
        if self.show_plot:
            plt.figure(fig2.number)
            plt.show()
        plt.close(fig2)

    def plot_left_right_comparison(self, *, result_dir) -> None:
        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100

        joint_data = self.gait_data.series_data["joint_data"]
        gait_segment_index_r = self.get_gait_segment_index(is_right_foot_based=True)
        gait_segment_index_l = self.get_gait_segment_index(is_right_foot_based=False)
        x_mapped = np.linspace(0, 100, num=100)

        joint_data_mapped = {k: [] for k in
                             ["hip_flexion_l", "hip_flexion_r", "knee_angle_l", "knee_angle_r", "ankle_angle_l",
                              "ankle_angle_r"]}

        for idx, (s, t, e) in enumerate(gait_segment_index_r):
            joint_data_mapped["hip_flexion_r"].append(np.interp(x_mapped, np.linspace(0, 100, e - s), np.rad2deg(
                [v[0] for v in joint_data["hip_flexion_r"]["qpos"][s:e]])))
            joint_data_mapped["knee_angle_r"].append(np.interp(x_mapped, np.linspace(0, 100, e - s), np.rad2deg(
                [v[0] for v in joint_data["knee_angle_r"]["qpos"][s:e]])))
            joint_data_mapped["ankle_angle_r"].append(np.interp(x_mapped, np.linspace(0, 100, e - s), np.rad2deg(
                [v[0] for v in joint_data["ankle_angle_r"]["qpos"][s:e]])))

        for idx, (s, t, e) in enumerate(gait_segment_index_l):
            joint_data_mapped["hip_flexion_l"].append(np.interp(x_mapped, np.linspace(0, 100, e - s), np.rad2deg(
                [v[0] for v in joint_data["hip_flexion_l"]["qpos"][s:e]])))
            joint_data_mapped["knee_angle_l"].append(np.interp(x_mapped, np.linspace(0, 100, e - s), np.rad2deg(
                [v[0] for v in joint_data["knee_angle_l"]["qpos"][s:e]])))
            joint_data_mapped["ankle_angle_l"].append(np.interp(x_mapped, np.linspace(0, 100, e - s), np.rad2deg(
                [v[0] for v in joint_data["ankle_angle_l"]["qpos"][s:e]])))

        self._apply_publication_rcparams()
        color_r = PUBLICATION_LR_COLOR_RIGHT
        color_l = PUBLICATION_LR_COLOR_LEFT
        alpha_fill = PUBLICATION_LR_RIBBON_ALPHA

        fig2, axes2 = plt.subplots(
            3,
            1,
            figsize=(5 * self.fig_size_multiplier, 9 * self.fig_size_multiplier),
            dpi=PUBLICATION_RENDER_DPI,
            sharex=True,
        )

        row_spec = list(
            zip(
                PUBLICATION_LR_JOINT_KEYS,
                PUBLICATION_JOINT_YLABELS_EN,
                PUBLICATION_JOINT_Y_AXES,
            )
        )
        for row_i, ((jk_r, jk_l), ylabel, (ylim_pair, yticks)) in enumerate(row_spec):
            ax = axes2[row_i]
            arr_r = np.array(joint_data_mapped[jk_r])
            arr_l = np.array(joint_data_mapped[jk_l])
            mean_r = np.mean(arr_r, axis=0)
            std_r = np.std(arr_r, axis=0)
            mean_l = np.mean(arr_l, axis=0)
            std_l = np.std(arr_l, axis=0)

            lb_r = PUBLICATION_LR_LEGEND_RIGHT if row_i == 0 else None
            lb_l = PUBLICATION_LR_LEGEND_LEFT if row_i == 0 else None
            ax.plot(x_mapped, mean_r, color=color_r, label=lb_r)
            ax.fill_between(x_mapped, mean_r - std_r, mean_r + std_r, color=color_r, alpha=alpha_fill)
            ax.plot(
                x_mapped,
                mean_l,
                color=color_l,
                linestyle="--",
                dashes=(5, 3),
                label=lb_l,
            )
            ax.fill_between(x_mapped, mean_l - std_l, mean_l + std_l, color=color_l, alpha=alpha_fill)

            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim_pair)
            ax.set_yticks(yticks)
            ax.set_xlim(0, 100)
            ax.axhline(0.0, color="0.75", linewidth=0.85, linestyle="-", zorder=1)
            ax.axvline(toe_off_r, color=color_r, linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)
            ax.axvline(toe_off_l, color=color_l, linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)
            ax.grid(False)
            style_publication_axes(
                ax,
                labelbottom=(row_i == 2),
                labelleft=True,
            )

        axes2[2].set_xlabel("Gait cycle (%)")
        axes2[0].legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),
            ncol=2,
            fontsize=PUBLICATION_LEGEND_FONTSIZE,
        )

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.15, top=0.90)
        fig2.savefig(
            os.path.join(result_dir, "left_right_comparison_avg.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
        )
        if self.show_plot:
            plt.show()
        plt.close(fig2)

    def _prepare_right_sim_ref_data(self):
        """Right-leg Sim mean/std (deg) and Ref (deg) for hip/knee/ankle.
        Returns (triples, toe_off_pct) or None.
        triples: [(mean_deg, std_deg, ref_deg, xp), ...] for hip, knee, ankle.
        """
        if not self.segmented_ref_data:
            return None
        gait_seg = self.get_gait_segment_index(is_right_foot_based=True)
        if not gait_seg:
            return None
        toe_off = self.get_toe_off_average(is_right_foot_based=True) * 100
        jd = self.gait_data.series_data["joint_data"]
        xp = np.linspace(0, 100, num=100)
        jkeys = ("hip_flexion_r", "knee_angle_r", "ankle_angle_r")
        rkeys = ("q_hip_flexion_r", "q_knee_angle_r", "q_ankle_angle_r")
        mapped = {jk: [] for jk in jkeys}
        for s, _t, e in gait_seg:
            for jk in jkeys:
                mapped[jk].append(
                    np.interp(xp, np.linspace(0, 100, e - s),
                              [v[0] for v in jd[jk]["qpos"][s:e]])
                )
        triples = []
        for jk, rk in zip(jkeys, rkeys):
            if rk not in self.segmented_ref_data or not mapped[jk]:
                return None
            arr = np.array(mapped[jk])
            mean_deg = np.rad2deg(np.mean(arr, axis=0))
            std_deg = np.rad2deg(np.std(arr, axis=0))
            ref_deg = np.rad2deg(
                np.asarray(self.segmented_ref_data[rk], dtype=np.float64).ravel()
            )
            n = min(mean_deg.size, ref_deg.size)
            if n < 2:
                return None
            triples.append((mean_deg[:n], std_deg[:n], ref_deg[:n], xp[:n]))
        return triples, toe_off

    def _apply_publication_rcparams(self):
        """"""
        apply_publication_rcparams()

    def plot_right_ref_comparison(self, *, result_dir):
        """Right-leg hip/knee/ankle Simulation mean +/- std vs Reference.
        Publication style; output: sim_ref_joints_comparison_with_shade.png
        """
        pack = self._prepare_right_sim_ref_data()
        if pack is None:
            return
        triples, toe_off = pack

        self._apply_publication_rcparams()

        color_sim = PUBLICATION_SIM_REF_EXO_MATCH_SIM
        color_ref = PUBLICATION_SIM_REF_EXO_MATCH_REF
        alpha_fill = PUBLICATION_LR_RIBBON_ALPHA

        fig, axes = plt.subplots(3, 1, figsize=(5, 9), dpi=PUBLICATION_RENDER_DPI, sharex=True)
        for i, (sim_mean, sim_std, ref_curve, xp) in enumerate(triples):
            ax = axes[i]
            lb_s = "Simulation" if i == 0 else None
            lb_r = "Reference" if i == 0 else None
            #
            ax.fill_between(
                xp,
                sim_mean - sim_std,
                sim_mean + sim_std,
                color=color_sim,
                alpha=alpha_fill,
                zorder=1,
            )
            ax.plot(xp, sim_mean, color=color_sim, label=lb_s, zorder=3)
            ax.plot(
                xp,
                ref_curve,
                color=color_ref,
                linestyle="--",
                dashes=(5, 3),
                label=lb_r,
                zorder=3,
            )
            ylim_pair, yticks = PUBLICATION_JOINT_Y_AXES[i]
            ax.set_ylabel(PUBLICATION_JOINT_YLABELS_EN[i])
            ax.set_ylim(*ylim_pair)
            ax.set_yticks(yticks)
            #
            _pub_standard_gait_x_axis(ax, show_xlabel=(i == 2))
            _pub_zero_reference_hline(ax)
            ax.axvline(toe_off, color=color_sim, linestyle=":", linewidth=1.2, alpha=0.8)
            style_publication_axes(
                ax,
                labelbottom=True,
                labelleft=True,
            )

        axes[0].legend(
            frameon=False, loc="upper center",
            bbox_to_anchor=(0.5, 1.25), ncol=2,
            fontsize=PUBLICATION_LEGEND_FONTSIZE,
        )

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.15)
        fig.savefig(
            os.path.join(result_dir, "sim_ref_joints_comparison_with_shade.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
        )
        if self.show_plot:
            plt.show()
        plt.close(fig)

    def save_sim_ref_metrics_csv(self, *, result_dir, metadata=None):
        """Right-leg Sim vs Ref per-joint + summary metrics -> CSV.
        Output: sim_ref_joints_right_metrics.csv
        """
        pack = self._prepare_right_sim_ref_data()
        if pack is None:
            return
        triples, toe_off = pack
        metadata = metadata or {}
        jlabels = ("hip", "knee", "ankle")
        fieldnames = [
            "checkpoint", "eval_idx", "joint", "pearson_r", "nmse",
            "rmse_deg", "pooled_rmse_deg", "toe_off_pct", "note",
        ]
        rows = []
        sq_all = []
        for i, (sim_mean, _sim_std, ref_curve, _xp) in enumerate(triples):
            pr = _pearson_r(sim_mean, ref_curve)
            d = sim_mean - ref_curve
            v = np.var(ref_curve)
            nmse = float(np.mean(d * d) / v) if v >= 1e-12 else float("nan")
            rmse = float(np.sqrt(np.mean(d * d)))
            sq_all.append((d * d).ravel())
            rows.append({
                "checkpoint": metadata.get("checkpoint", ""),
                "eval_idx": metadata.get("eval_idx", ""),
                "joint": jlabels[i],
                "pearson_r": pr,
                "nmse": nmse,
                "rmse_deg": rmse,
                "pooled_rmse_deg": "",
                "toe_off_pct": toe_off,
                "note": "",
            })
        valid_r = [r["pearson_r"] for r in rows if not np.isnan(r["pearson_r"])]
        valid_nmse = [r["nmse"] for r in rows if not np.isnan(r["nmse"])]
        rows.append({
            "checkpoint": metadata.get("checkpoint", ""),
            "eval_idx": metadata.get("eval_idx", ""),
            "joint": "all_joints_summary",
            "pearson_r": float(np.mean(valid_r)) if valid_r else float("nan"),
            "nmse": float(np.mean(valid_nmse)) if valid_nmse else float("nan"),
            "rmse_deg": float(np.mean([r["rmse_deg"] for r in rows])),
            "pooled_rmse_deg": float(np.sqrt(np.mean(np.concatenate(sq_all)))) if sq_all else float("nan"),
            "toe_off_pct": toe_off,
            "note": "",
        })
        out_path = os.path.join(result_dir, "sim_ref_joints_right_metrics.csv")
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({fn: row.get(fn, "") for fn in fieldnames})

    def plot_contact_data(self, *, result_dir, geom_pairs: list[tuple[str, str]] = [("calcn_l_geom_1", "terrain"),
                                                                                    ("calcn_r_geom_1", "terrain")]):
        plot_data = {
            geom_name1: self.gait_data.get_contact_data(geom_name1=geom_name1, geom_name2=geom_name2)
            for geom_name1, geom_name2 in geom_pairs
        }
        fig, axes = plt.subplots(len(geom_pairs), 1,
                                 figsize=(4 * self.fig_size_multiplier, 3 * self.fig_size_multiplier), dpi=self.dpi)
        for idx, (geom_name1, geom_name2) in enumerate(geom_pairs):
            ax = axes[idx]
            ax.plot([force[0] for force in plot_data[geom_name1]], label=f"{geom_name1} force", color="#000000",
                    linestyle="-")
            ax.set_title(f"{geom_name1} and {geom_name2} contact force")
        fig.tight_layout()
        fig.savefig(
            os.path.join(result_dir, "contact_data.png"),
            bbox_inches="tight", dpi=PUBLICATION_SAVEFIG_DPI, pad_inches=0.03,
        )
        if self.show_plot:
            plt.show()
        plt.close()

    def plot_segmented_muscle_data(self, *, result_dir, is_plot_right: bool):
        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100
        toe_off = toe_off_r if is_plot_right else toe_off_l

        post_fix = ['_r', '_R'] if is_plot_right else ['_l', '_L']
        file_name_post_fix = '_r' if is_plot_right else '_l'

        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=is_plot_right)
        x_mapped = np.linspace(0, 100, num=101)

        # Filter to remove "Exo" data from muscle plots
        actuator_names = [actuator_name for actuator_name in self.gait_data.series_data["actuator_data"].keys() if
                          actuator_name[-2:] in post_fix and "Exo" not in actuator_name]
        actuator_names.sort()

        muscle_data_mapped = {actuator_name: [] for actuator_name in actuator_names}
        actuator_num = len(actuator_names)

        for actuator_name in actuator_names:
            actuator_data = self.gait_data.series_data["actuator_data"][actuator_name]
            for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index):
                muscle_data = {
                    "force": -np.interp(x_mapped,
                                        np.linspace(0, 100, num=len(actuator_data["force"][start_idx:end_idx])),
                                        [v[0] for v in actuator_data["force"][start_idx:end_idx]]),
                    "ctrl": np.abs(
                        np.interp(x_mapped, np.linspace(0, 100, num=len(actuator_data["ctrl"][start_idx:end_idx])),
                                  [v[0] for v in actuator_data["ctrl"][start_idx:end_idx]])),
                    "velocity": np.interp(x_mapped,
                                          np.linspace(0, 100, num=len(actuator_data["velocity"][start_idx:end_idx])),
                                          [v[0] for v in actuator_data["velocity"][start_idx:end_idx]])
                }
                muscle_data_mapped[actuator_name].append(muscle_data)

        #
        self._apply_publication_rcparams()
        n_seg_plot = len(gait_segment_index)
        _cmap_name_seg = "tab20" if n_seg_plot > 10 else "tab10"
        _cmap_seg_f = plt.cm.get_cmap(_cmap_name_seg)
        _n_c_seg = 20 if n_seg_plot > 10 else 10

        nrows_seg, ncols_seg = MUSCLE_PUB_GRID_ROWS, MUSCLE_PUB_GRID_COLS
        n_cells_seg = nrows_seg * ncols_seg
        n_plot_seg = min(actuator_num, n_cells_seg)
        if actuator_num > n_cells_seg:
            print(
                f"[WARN] segmented_muscle_data: {actuator_num} muscles > {n_cells_seg}, "
                f"plotting first {n_cells_seg} only."
            )

        fig_w_seg, fig_h_seg = _muscle_grid_figsize(self.fig_size_multiplier)
        fig, axes_seg = plt.subplots(
            nrows_seg, ncols_seg, figsize=(fig_w_seg, fig_h_seg), squeeze=False, dpi=PUBLICATION_RENDER_DPI
        )
        axes_flat_seg = axes_seg.flatten()

        for i in range(n_plot_seg):
            ax = axes_flat_seg[i]
            actuator_name = actuator_names[i]
            short_title = short_muscle_name(actuator_name)
            for seg_idx, muscle_data in enumerate(muscle_data_mapped[actuator_name]):
                c = _cmap_seg_f(seg_idx % _n_c_seg)
                ax.plot(
                    x_mapped,
                    muscle_data["force"],
                    color=c,
                    linewidth=1.1,
                    alpha=0.92,
                )

            all_f = np.array([m["force"] for m in muscle_data_mapped[actuator_name]])
            y_lo_d = float(np.nanmin(all_f))
            y_hi_d = float(np.nanmax(all_f))
            y_lim_lo, y_lim_hi = _auto_ylim_pad(
                y_lo_d,
                y_hi_d,
                nonnegative_floor=(np.isfinite(y_lo_d) and y_lo_d >= 0),
            )

            _muscle_subplot_frame(ax, i, toe_off, nrows_seg, ncols_seg)
            ax.set_ylim(y_lim_lo, y_lim_hi)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            if i % ncols_seg == 0:
                ax.set_ylabel("Muscle force (N)")
            ax.set_title(short_title, pad=6, fontsize=PUBLICATION_SUBPLOT_TITLE_FONTSIZE)

        for j in range(n_plot_seg, n_cells_seg):
            axes_flat_seg[j].set_visible(False)

        _leg_seg = leg_label(is_plot_right)
        fig.suptitle(
            f"{_leg_seg} — muscle force (all gait cycles)",
            fontsize=PUBLICATION_SUPTITLE_FONTSIZE,
            y=0.952,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        fig.savefig(
            os.path.join(result_dir, f"segmented_muscle_data{file_name_post_fix}.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
        )
        if self.show_plot:
            plt.show()
        plt.close(fig)

        #
        self._apply_publication_rcparams()
        nrows_ms, ncols_ms = MUSCLE_PUB_GRID_ROWS, MUSCLE_PUB_GRID_COLS
        n_cells_ms = nrows_ms * ncols_ms
        n_plot_ms = min(actuator_num, n_cells_ms)
        if actuator_num > n_cells_ms:
            print(
                f"[WARN] segmented_muscle_data_mean_std: {actuator_num} muscles > {n_cells_ms}, "
                f"plotting first {n_cells_ms} only."
            )

        _force_line = PUBLICATION_LR_COLOR_RIGHT
        _force_fill_alpha = 0.22
        fig_w_ms, fig_h_ms = _muscle_grid_figsize(self.fig_size_multiplier)
        fig_mean_std, axes_ms = plt.subplots(
            nrows_ms, ncols_ms, figsize=(fig_w_ms, fig_h_ms), squeeze=False, dpi=PUBLICATION_RENDER_DPI
        )
        axes_flat_ms = axes_ms.flatten()

        for i in range(n_plot_ms):
            ax = axes_flat_ms[i]
            actuator_name = actuator_names[i]
            force_data = np.array([m["force"] for m in muscle_data_mapped[actuator_name]])
            mean_force = np.mean(force_data, axis=0)
            std_force = np.std(force_data, axis=0)

            short_title = short_muscle_name(actuator_name)
            lo_b = float(np.nanmin(mean_force - std_force))
            hi_b = float(np.nanmax(mean_force + std_force))
            y_lo, y_hi = _auto_ylim_pad(
                lo_b,
                hi_b,
                nonnegative_floor=(np.isfinite(lo_b) and lo_b >= 0),
            )

            ax.fill_between(
                x_mapped,
                mean_force - std_force,
                mean_force + std_force,
                color=_force_line,
                alpha=_force_fill_alpha,
                linewidth=0,
                zorder=3,
            )
            ax.plot(
                x_mapped,
                mean_force,
                color=_force_line,
                linestyle="-",
                linewidth=plt.rcParams["lines.linewidth"],
                zorder=5,
            )
            _muscle_subplot_frame(ax, i, toe_off, nrows_ms, ncols_ms)
            ax.set_ylim(y_lo, y_hi)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            if i % ncols_ms == 0:
                ax.set_ylabel("Muscle force (N)")
            ax.set_title(short_title, pad=6, fontsize=PUBLICATION_SUBPLOT_TITLE_FONTSIZE)

        for j in range(n_plot_ms, n_cells_ms):
            axes_flat_ms[j].set_visible(False)

        _leg_ms = leg_label(is_plot_right)
        fig_mean_std.suptitle(
            f"{_leg_ms} — muscle force (mean ± SD)",
            fontsize=PUBLICATION_SUPTITLE_FONTSIZE,
            y=0.952,
        )
        _figure_legend_mean_sd(fig_mean_std, _force_line)
        fig_mean_std.tight_layout(rect=[0, 0, 1, 0.955])
        fig_mean_std.savefig(
            os.path.join(result_dir, f"segmented_muscle_data_mean_std{file_name_post_fix}.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
        )
        if self.show_plot:
            plt.show()
        plt.close(fig_mean_std)

        #
        self._apply_publication_rcparams()
        nrows, ncols = MUSCLE_PUB_GRID_ROWS, MUSCLE_PUB_GRID_COLS
        n_cells = nrows * ncols
        n_m = actuator_num
        n_plot = min(n_m, n_cells)
        if n_m > n_cells:
            print(
                f"[WARN] segmented_muscle_data_mean_std_ctrl: {n_m} muscles > {n_cells}, "
                f"plotting first {n_cells} only."
            )

        _ctrl_line = PUBLICATION_LR_COLOR_RIGHT
        _ctrl_fill_alpha = 0.22

        fig_w, fig_h = _muscle_grid_figsize(self.fig_size_multiplier)
        fig_mean_std_ctrl, axes_msc = plt.subplots(
            nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, dpi=PUBLICATION_RENDER_DPI
        )
        axes_flat = axes_msc.flatten()

        for i in range(n_plot):
            ax = axes_flat[i]
            actuator_name = actuator_names[i]
            ctrl_data = 100 * np.array([m["ctrl"] for m in muscle_data_mapped[actuator_name]])
            mean_ctrl = np.mean(ctrl_data, axis=0)
            std_ctrl = np.std(ctrl_data, axis=0)

            short_title = short_muscle_name(actuator_name)
            ax.fill_between(
                x_mapped,
                mean_ctrl - std_ctrl,
                mean_ctrl + std_ctrl,
                color=_ctrl_line,
                alpha=_ctrl_fill_alpha,
                linewidth=0,
                zorder=3,
            )
            ax.plot(
                x_mapped,
                mean_ctrl,
                color=_ctrl_line,
                linestyle="-",
                linewidth=plt.rcParams["lines.linewidth"],
                zorder=5,
            )
            _muscle_subplot_frame(ax, i, toe_off, nrows, ncols)
            ax.set_ylim(0, 100)
            ax.set_yticks(np.arange(0, 101, 20))
            if i % ncols == 0:
                ax.set_ylabel("Muscle activation (%)")
            ax.set_title(short_title, pad=6, fontsize=PUBLICATION_SUBPLOT_TITLE_FONTSIZE)

        for j in range(n_plot, n_cells):
            axes_flat[j].set_visible(False)

        _leg_en = leg_label(is_plot_right)
        fig_mean_std_ctrl.suptitle(
            f"{_leg_en} — muscle activation (mean ± SD, ctrl)",
            fontsize=PUBLICATION_SUPTITLE_FONTSIZE,
            y=0.952,
        )
        _figure_legend_mean_sd(fig_mean_std_ctrl, _ctrl_line)
        fig_mean_std_ctrl.tight_layout(rect=[0, 0, 1, 0.955])
        fig_mean_std_ctrl.savefig(
            os.path.join(result_dir, f"segmented_muscle_data_mean_std_ctrl{file_name_post_fix}.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
        )
        if self.show_plot:
            plt.show()
        plt.close(fig_mean_std_ctrl)

    def get_publication_mean_curves(self):
        """"""
        joint_data = self.gait_data.series_data["joint_data"]
        gait_idx_r = self.get_gait_segment_index(is_right_foot_based=True)
        gait_idx_l = self.get_gait_segment_index(is_right_foot_based=False)
        x_mapped = np.linspace(0, 100, num=100)

        def process_segments(segment_indices, raw_data_key, rad2deg=True):
            mapped_list = []
            raw_series = joint_data[raw_data_key]["qpos"]
            for s, _, e in segment_indices:
                segment = [v[0] for v in raw_series[s:e]]
                if rad2deg:
                    segment = np.rad2deg(segment)
                mapped = np.interp(x_mapped, np.linspace(0, 100, len(segment)), segment)
                mapped_list.append(mapped)
            arr = np.array(mapped_list)
            return np.mean(arr, axis=0), np.std(arr, axis=0)

        def process_exo(segment_indices, actuator_name):
            mapped_list = []
            raw_series = self.gait_data.series_data["actuator_data"][actuator_name]["force"]
            for s, _, e in segment_indices:
                segment = [v[0] for v in raw_series[s:e]]
                mapped = np.interp(x_mapped, np.linspace(0, 100, len(segment)), segment)
                mapped_list.append(mapped)
            arr = np.array(mapped_list)
            return np.mean(arr, axis=0), np.std(arr, axis=0)

        h_r, h_sr = process_segments(gait_idx_r, "hip_flexion_r")
        h_l, h_sl = process_segments(gait_idx_l, "hip_flexion_l")
        k_r, k_sr = process_segments(gait_idx_r, "knee_angle_r")
        k_l, k_sl = process_segments(gait_idx_l, "knee_angle_l")
        a_r, a_sr = process_segments(gait_idx_r, "ankle_angle_r")
        a_l, a_sl = process_segments(gait_idx_l, "ankle_angle_l")
        e_r, e_sr = process_exo(gait_idx_r, "Exo_R")
        e_l, e_sl = process_exo(gait_idx_l, "Exo_L")

        return {
            "x_pct": x_mapped,
            "hip": {"mean_r": h_r, "mean_l": h_l, "std_r": h_sr, "std_l": h_sl},
            "knee": {"mean_r": k_r, "mean_l": k_l, "std_r": k_sr, "std_l": k_sl},
            "ankle": {"mean_r": a_r, "mean_l": a_l, "std_r": a_sr, "std_l": a_sl},
            "exo": {"mean_r": e_r, "mean_l": e_l, "std_r": e_sr, "std_l": e_sl},
        }

    @staticmethod
    def _pairwise_lr_metrics(mean_r: np.ndarray, mean_l: np.ndarray, unit: str) -> dict:
        """"""
        mean_r = np.asarray(mean_r, dtype=float).ravel()
        mean_l = np.asarray(mean_l, dtype=float).ravel()
        n = mean_r.size
        if mean_l.size != n or n < 2:
            return {"error": "length_mismatch_or_too_short", "unit": unit}

        diff = mean_r - mean_l
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        max_abs = float(np.max(np.abs(diff)))

        r_mat = np.corrcoef(mean_r, mean_l)
        pearson_r = float(r_mat[0, 1]) if np.isfinite(r_mat[0, 1]) else float("nan")

        nr = np.linalg.norm(mean_r)
        nl = np.linalg.norm(mean_l)
        if nr > 1e-15 and nl > 1e-15:
            cos_sim = float(np.dot(mean_r, mean_l) / (nr * nl))
        else:
            cos_sim = float("nan")

        vmin = float(min(mean_r.min(), mean_l.min()))
        vmax = float(max(mean_r.max(), mean_l.max()))
        rng = vmax - vmin
        nrmse = float(rmse / rng) if rng > 1e-12 else float("nan")

        return {
            "n_points": n,
            "pearson_r": pearson_r,
            "cosine_similarity": cos_sim,
            "rmse": rmse,
            "mae": mae,
            "max_abs_error": max_abs,
            "nrmse_by_pooled_range": nrmse,
            "unit": unit,
        }

    def compute_left_right_publication_metrics(self) -> dict:
        """"""
        c = self.get_publication_mean_curves()
        return {
            "hip_angle": self._pairwise_lr_metrics(c["hip"]["mean_r"], c["hip"]["mean_l"], "deg"),
            "knee_angle": self._pairwise_lr_metrics(c["knee"]["mean_r"], c["knee"]["mean_l"], "deg"),
            "ankle_angle": self._pairwise_lr_metrics(c["ankle"]["mean_r"], c["ankle"]["mean_l"], "deg"),
            "exo_torque": self._pairwise_lr_metrics(c["exo"]["mean_r"], c["exo"]["mean_l"], "Nm"),
        }

    def plot_publication_ready_combined(self, *, result_dir):
        """
        Generates a publication-quality combined plot of Joint Kinematics and Exo Torques.
        Layout: 4 Rows (Hip, Knee, Ankle, Exo Torque), combining Left and Right legs in each subplot.
        """
        self._apply_publication_rcparams()

        color_r = PUBLICATION_LR_COLOR_RIGHT
        color_l = PUBLICATION_LR_COLOR_LEFT
        alpha_fill = PUBLICATION_LR_RIBBON_ALPHA

        #
        fig, axes = plt.subplots(4, 1, figsize=(5, 10), dpi=PUBLICATION_RENDER_DPI, sharex=True)

        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100

        c = self.get_publication_mean_curves()
        x_mapped = c["x_pct"]

        for i, jk in enumerate(("hip", "knee", "ankle")):
            mean_r, std_r = c[jk]["mean_r"], c[jk]["std_r"]
            mean_l, std_l = c[jk]["mean_l"], c[jk]["std_l"]
            _plot_lr_mean_sd_on_ax(
                axes[i],
                x_mapped,
                mean_r,
                std_r,
                mean_l,
                std_l,
                color_r,
                color_l,
                alpha_fill,
                legend_right=PUBLICATION_LR_LEGEND_RIGHT if i == 0 else None,
                legend_left=PUBLICATION_LR_LEGEND_LEFT if i == 0 else None,
            )
            ylim_pair, yticks = PUBLICATION_JOINT_Y_AXES[i]
            axes[i].set_ylabel(PUBLICATION_JOINT_YLABELS_EN[i])
            axes[i].set_ylim(*ylim_pair)
            axes[i].set_yticks(yticks)

        # 4. Exo Torque
        mean_r, std_r = c["exo"]["mean_r"], c["exo"]["std_r"]
        mean_l, std_l = c["exo"]["mean_l"], c["exo"]["std_l"]

        axes[3].plot(x_mapped, mean_r, color=color_r)
        axes[3].fill_between(x_mapped, mean_r - std_r, mean_r + std_r, color=color_r, alpha=alpha_fill)
        axes[3].plot(x_mapped, mean_l, color=color_l, linestyle="--", dashes=(5, 3))
        axes[3].fill_between(x_mapped, mean_l - std_l, mean_l + std_l, color=color_l, alpha=alpha_fill)
        axes[3].set_ylabel("Exo Torque (Nm)")
        axes[3].set_xlabel("Gait cycle (%)")
        axes[3].set_ylim(-12, 12)  # Match your XML limit

        #
        for ax in axes:
            _pub_standard_gait_x_axis(ax, show_xlabel=False)
            _pub_zero_reference_hline(ax)
            ax.axvline(toe_off_r, color=color_r, linestyle=":", linewidth=1.2, alpha=0.8)
            ax.axvline(toe_off_l, color=color_l, linestyle=":", linewidth=1.2, alpha=0.8)
            style_publication_axes(ax, labelbottom=True, labelleft=True)

        #
        axes[0].legend(
            frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2,
            fontsize=PUBLICATION_LEGEND_FONTSIZE,
        )

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.15)  # Reduce space between subplots

        save_path = os.path.join(result_dir, "Publication_Combined_Kinematics_Exo.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=PUBLICATION_SAVEFIG_DPI, pad_inches=0.03)
        print(f"Saved publication ready plot to: {save_path}")

        if self.show_plot:
            plt.show()
        plt.close()

    def plot_publication_ready_combined_hip_exo_only(self, *, result_dir):
        """"""
        self._apply_publication_rcparams()
        #
        label_fs = PUBLICATION_LABEL_FONTSIZE + 3
        tick_fs = PUBLICATION_TICK_FONTSIZE + 2
        legend_fs = PUBLICATION_LEGEND_FONTSIZE + 3
        plt.rcParams.update({
            "axes.labelsize": label_fs,
            "xtick.labelsize": tick_fs,
            "ytick.labelsize": tick_fs,
            "legend.fontsize": legend_fs,
        })

        color_r = PUBLICATION_LR_COLOR_RIGHT
        color_l = PUBLICATION_LR_COLOR_LEFT
        alpha_fill = PUBLICATION_LR_RIBBON_ALPHA

        fig, axes = plt.subplots(2, 1, figsize=(5, 6.5), dpi=PUBLICATION_RENDER_DPI, sharex=True)
        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100

        c = self.get_publication_mean_curves()
        x_mapped = c["x_pct"]

        mean_r, std_r = c["hip"]["mean_r"], c["hip"]["std_r"]
        mean_l, std_l = c["hip"]["mean_l"], c["hip"]["std_l"]
        axes[0].plot(x_mapped, mean_r, color=color_r, label=PUBLICATION_LR_LEGEND_RIGHT)
        axes[0].fill_between(x_mapped, mean_r - std_r, mean_r + std_r, color=color_r, alpha=alpha_fill)
        axes[0].plot(
            x_mapped,
            mean_l,
            color=color_l,
            linestyle="--",
            dashes=(5, 3),
            label=PUBLICATION_LR_LEGEND_LEFT,
        )
        axes[0].fill_between(x_mapped, mean_l - std_l, mean_l + std_l, color=color_l, alpha=alpha_fill)
        axes[0].set_ylabel("Hip Angle (deg)", fontsize=label_fs)
        axes[0].set_ylim(-40, 40)
        axes[0].set_yticks(np.arange(-40, 41, 20))

        mean_r, std_r = c["exo"]["mean_r"], c["exo"]["std_r"]
        mean_l, std_l = c["exo"]["mean_l"], c["exo"]["std_l"]
        axes[1].plot(x_mapped, mean_r, color=color_r)
        axes[1].fill_between(x_mapped, mean_r - std_r, mean_r + std_r, color=color_r, alpha=alpha_fill)
        axes[1].plot(x_mapped, mean_l, color=color_l, linestyle="--", dashes=(5, 3))
        axes[1].fill_between(x_mapped, mean_l - std_l, mean_l + std_l, color=color_l, alpha=alpha_fill)
        axes[1].set_ylabel("Exo Torque (Nm)", fontsize=label_fs)
        axes[1].set_xlabel("Gait cycle (%)", fontsize=label_fs)
        axes[1].set_ylim(-12, 12)

        for i, ax in enumerate(axes):
            ax.set_xlim(0, 100)
            ax.axhline(0.0, color="0.75", linewidth=0.85, linestyle="-", zorder=1)
            ax.axvline(toe_off_r, color=color_r, linestyle=":", linewidth=1.2, alpha=0.8)
            ax.axvline(toe_off_l, color=color_l, linestyle=":", linewidth=1.2, alpha=0.8)
            ax.grid(False)
            style_publication_axes(
                ax,
                labelbottom=True,
                labelleft=True,
            )
            ax.tick_params(axis="both", which="major", labelsize=tick_fs)

        axes[0].legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),
            ncol=2,
            fontsize=legend_fs,
        )

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.15)

        save_path = os.path.join(result_dir, "Publication_Combined_Kinematics_Exo_hip_exo_only.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=PUBLICATION_SAVEFIG_DPI, pad_inches=0.03)
        print(f"Saved hip+exo publication plot to: {save_path}")

        if self.show_plot:
            plt.show()
        plt.close()

    def __del__(self):
        plt.close()


# ============================================================
#
# ============================================================

def extract_muscle_activation_stats(
    gait_data: GaitData, is_right: bool = True
) -> Optional[Dict[str, Any]]:
    """"""
    analyzer = GaitAnalyzer(gait_data, None, False)
    segments = analyzer.get_gait_segment_index(is_right_foot_based=is_right)
    if len(segments) < 1:
        return None

    #
    stance_pcts = []
    for (start, toe_off, end) in segments:
        if end > start:
            stance_pcts.append((toe_off - start) / (end - start) * 100)
    mean_stance_pct = float(np.mean(stance_pcts)) if stance_pcts else 60.0
    stance_idx = max(1, min(int(round(mean_stance_pct)), 100))

    post_fix = ['_r', '_R'] if is_right else ['_l', '_L']
    actuator_store = gait_data.series_data["actuator_data"]
    muscle_names = sorted(
        [n for n in actuator_store if n[-2:] in post_fix and "Exo" not in n]
    )

    x_mapped = np.linspace(0, 100, num=101)
    per_muscle: Dict[str, Any] = {}

    for name in muscle_names:
        act = actuator_store[name]
        cycles = []
        for (start, toe_off, end) in segments:
            ctrl_seg = [v[0] for v in act["ctrl"][start:end]]
            if len(ctrl_seg) < 2:
                continue
            interp = np.abs(
                np.interp(x_mapped, np.linspace(0, 100, num=len(ctrl_seg)), ctrl_seg)
            )
            cycles.append(interp)
        if not cycles:
            continue
        cycles_arr = np.array(cycles)
        mean_curve = np.mean(cycles_arr, axis=0)
        std_curve = np.std(cycles_arr, axis=0)
        per_muscle[name] = {
            "mean_activation": float(np.mean(mean_curve)),
            "std_activation": float(np.std(mean_curve)),
            "mean_activation_stance": float(np.mean(mean_curve[:stance_idx])),
            "mean_activation_swing": float(np.mean(mean_curve[stance_idx:])),
            "mean_activation_functional": _compute_functional_mean(name, mean_curve),
            "functional_window": _get_functional_window_label(name),
            "mean_curve": mean_curve,
            "std_curve": std_curve,
        }

    overall_means = [v["mean_activation"] for v in per_muscle.values()]
    return {
        "muscle_names": muscle_names,
        "per_muscle": per_muscle,
        "overall_mean": float(np.mean(overall_means)) if overall_means else 0.0,
        "n_cycles": len(segments),
        "mean_stance_pct": mean_stance_pct,
    }


# ============================================================
#
# ============================================================

def draw_gait_activation_on_off(
    ax,
    x_pct: np.ndarray,
    m_on: dict, m_off: dict,
    stance_pct: float,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_xtick_labels: Optional[bool] = None,
    show_ytick_labels: Optional[bool] = None,
    title: Optional[str] = None,
    title_fontsize: float = PUBLICATION_SUBPLOT_TITLE_FONTSIZE,
) -> None:
    """"""
    mean_on = m_on["mean_curve"] * 100.0
    std_on = m_on["std_curve"] * 100.0
    mean_off = m_off["mean_curve"] * 100.0
    std_off = m_off["std_curve"] * 100.0

    ax.grid(False)
    ax.axhline(0.0, color="0.75", linewidth=0.85, linestyle="-", zorder=1)
    ax.axvline(
        float(np.clip(stance_pct, 0.0, 100.0)),
        color="0.15", linewidth=0.95, linestyle=":", zorder=2,
    )

    ax.fill_between(
        x_pct, mean_off - std_off, mean_off + std_off,
        color=MUSCLE_PUB_OFF_LINE, alpha=MUSCLE_PUB_RIBBON_ALPHA, linewidth=0, zorder=3,
    )
    ax.plot(
        x_pct, mean_off, color=MUSCLE_PUB_OFF_LINE,
        linestyle="--", dashes=(5, 3), zorder=5,
    )

    ax.fill_between(
        x_pct, mean_on - std_on, mean_on + std_on,
        color=MUSCLE_PUB_ON_LINE, alpha=MUSCLE_PUB_RIBBON_ALPHA, linewidth=0, zorder=3,
    )
    ax.plot(x_pct, mean_on, color=MUSCLE_PUB_ON_LINE, linestyle="-", zorder=5)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, MUSCLE_PUB_Y_MAX)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    if show_xlabel:
        ax.set_xlabel("Gait cycle (%)")
    if show_ylabel:
        ax.set_ylabel("Muscle activation (%)")
    if title is not None:
        ax.set_title(title, pad=6, fontsize=title_fontsize)

    lb = show_xtick_labels if show_xtick_labels is not None else show_xlabel
    ll = show_ytick_labels if show_ytick_labels is not None else show_ylabel
    style_publication_axes(ax, labelbottom=lb, labelleft=ll)


def draw_gait_activation_single(
    ax,
    x_pct: np.ndarray,
    m: dict,
    stance_pct: float,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_xtick_labels: Optional[bool] = None,
    show_ytick_labels: Optional[bool] = None,
    title: Optional[str] = None,
    title_fontsize: float = PUBLICATION_SUBPLOT_TITLE_FONTSIZE,
) -> None:
    """"""
    mean = m["mean_curve"] * 100.0
    std = m["std_curve"] * 100.0

    ax.grid(False)
    ax.axhline(0.0, color="0.75", linewidth=0.85, linestyle="-", zorder=1)
    ax.axvline(
        float(np.clip(stance_pct, 0.0, 100.0)),
        color="0.15", linewidth=0.95, linestyle=":", zorder=2,
    )
    ax.fill_between(
        x_pct, mean - std, mean + std,
        color=MUSCLE_PUB_SINGLE_LINE, alpha=MUSCLE_PUB_RIBBON_ALPHA,
        linewidth=0, zorder=3,
    )
    ax.plot(x_pct, mean, color=MUSCLE_PUB_SINGLE_LINE, linestyle="-",
            linewidth=1.35, zorder=5)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, MUSCLE_PUB_Y_MAX)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    if show_xlabel:
        ax.set_xlabel("Gait cycle (%)")
    if show_ylabel:
        ax.set_ylabel("Muscle activation (%)")
    if title is not None:
        ax.set_title(title, pad=6, fontsize=title_fontsize)

    lb = show_xtick_labels if show_xtick_labels is not None else show_xlabel
    ll = show_ytick_labels if show_ytick_labels is not None else show_ylabel
    style_publication_axes(ax, labelbottom=lb, labelleft=ll)


# ============================================================
#
# ============================================================

def make_figure_combined_exo_on_off(
    stats_on: Dict[str, Any],
    stats_off: Dict[str, Any],
    common_muscles: List[str],
    side_label: str,
):
    """"""
    nrows, ncols = 3, 4
    n_cells = nrows * ncols
    muscles = list(common_muscles)
    if len(muscles) > n_cells:
        print(f"   [WARN] combined exo on/off: {len(muscles)} muscles > {n_cells}, plotting first {n_cells} only")
        muscles = muscles[:n_cells]
    n_plot = len(muscles)

    apply_publication_rcparams()
    stance_pct = (
        float(stats_on.get("mean_stance_pct", 60.0))
        + float(stats_off.get("mean_stance_pct", 60.0))
    ) / 2.0
    x_pct = np.linspace(0, 100, 101)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.15, nrows * 2.55), squeeze=False, dpi=PUBLICATION_RENDER_DPI,
    )
    axes_flat = axes.flatten()

    for i, name in enumerate(muscles):
        draw_gait_activation_on_off(
            axes_flat[i], x_pct,
            stats_on["per_muscle"][name], stats_off["per_muscle"][name],
            stance_pct,
            show_xlabel=(i // ncols == nrows - 1),
            show_ylabel=(i % ncols == 0),
            show_xtick_labels=True, show_ytick_labels=True,
            title=short_muscle_name(name), title_fontsize=PUBLICATION_SUBPLOT_TITLE_FONTSIZE,
        )

    for j in range(n_plot, n_cells):
        axes_flat[j].set_visible(False)

    leg_title = leg_label(side_label == "right")
    fig.suptitle(
        f"{leg_title} — muscle activation (Exo ON vs Exo OFF)",
        fontsize=PUBLICATION_SUPTITLE_FONTSIZE, y=0.952,
    )
    handles = [
        Line2D([0], [0], color=MUSCLE_PUB_ON_LINE, linestyle="-", label="Exo ON"),
        Line2D([0], [0], color=MUSCLE_PUB_OFF_LINE, linestyle="--", dashes=(5, 3), label="Exo OFF"),
    ]
    fig.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.948),
        bbox_transform=fig.transFigure, ncol=2, frameon=False,
        fontsize=PUBLICATION_LEGEND_FONTSIZE,
        handlelength=1.55, handletextpad=0.45, columnspacing=0.75, borderaxespad=0.2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return fig


def make_figure_combined_activation_only(
    stats: Dict[str, Any],
    common_muscles: List[str],
    side_label: str,
):
    """"""
    nrows, ncols = 3, 4
    n_cells = nrows * ncols
    muscles = list(common_muscles)
    if len(muscles) > n_cells:
        print(f"   [WARN] combined activation-only: {len(muscles)} muscles > {n_cells}, plotting first {n_cells} only")
        muscles = muscles[:n_cells]
    n_plot = len(muscles)

    apply_publication_rcparams()
    stance_pct = float(stats.get("mean_stance_pct", 60.0))
    x_pct = np.linspace(0, 100, 101)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.15, nrows * 2.55), squeeze=False, dpi=PUBLICATION_RENDER_DPI,
    )
    axes_flat = axes.flatten()

    for i, name in enumerate(muscles):
        draw_gait_activation_single(
            axes_flat[i], x_pct, stats["per_muscle"][name], stance_pct,
            show_xlabel=(i // ncols == nrows - 1),
            show_ylabel=(i % ncols == 0),
            show_xtick_labels=True, show_ytick_labels=True,
            title=short_muscle_name(name), title_fontsize=PUBLICATION_SUBPLOT_TITLE_FONTSIZE,
        )

    for j in range(n_plot, n_cells):
        axes_flat[j].set_visible(False)

    leg_title = leg_label(side_label == "right")
    fig.suptitle(f"{leg_title} — muscle activation", fontsize=PUBLICATION_SUPTITLE_FONTSIZE, y=0.952)
    handles = [
        Line2D([0], [0], color=MUSCLE_PUB_SINGLE_LINE, linestyle="-", label="Mean ± SD"),
    ]
    fig.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.948),
        bbox_transform=fig.transFigure, ncol=1, frameon=False,
        fontsize=PUBLICATION_LEGEND_FONTSIZE,
        handlelength=1.55, handletextpad=0.45, borderaxespad=0.2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return fig


# ============================================================
#
# ============================================================

PUBLICATION_MUSCLE_GAIT_SUBDIR = "publication_muscle_gait"


def generate_publication_muscle_gait_exo_on_off(
    analyze_result_dir: str,
    gait_on: GaitData,
    gait_off: GaitData,
) -> None:
    """"""
    out_dir = os.path.join(analyze_result_dir, PUBLICATION_MUSCLE_GAIT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    x_pct = np.linspace(0, 100, 101)
    fig_w, fig_h = 3.55, 2.92

    for is_right in (True, False):
        stats_on = extract_muscle_activation_stats(gait_on, is_right=is_right)
        stats_off = extract_muscle_activation_stats(gait_off, is_right=is_right)
        if stats_on is None or stats_off is None:
            print(f"   [WARN] publication Exo ON/OFF: cannot extract {'right' if is_right else 'left'} leg gait segments, skip.")
            continue

        common = sorted(set(stats_on["per_muscle"]) & set(stats_off["per_muscle"]))
        if not common:
            print(f"   [WARN] publication Exo ON/OFF: no common muscles for {'right' if is_right else 'left'} leg, skip.")
            continue

        stance_pct = (stats_on["mean_stance_pct"] + stats_off["mean_stance_pct"]) / 2.0
        leg_str = leg_label(is_right)

        apply_publication_rcparams()
        for name in common:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PUBLICATION_RENDER_DPI)
            draw_gait_activation_on_off(
                ax, x_pct, stats_on["per_muscle"][name], stats_off["per_muscle"][name],
                stance_pct, show_xlabel=True, show_ylabel=True,
                title=f"{short_muscle_name(name)} — {leg_str}",
                title_fontsize=PUBLICATION_SUBPLOT_TITLE_FONTSIZE,
            )
            handles = [
                Line2D([0], [0], color=MUSCLE_PUB_ON_LINE, linestyle="-", label="Exo ON"),
                Line2D([0], [0], color=MUSCLE_PUB_OFF_LINE, linestyle="--", dashes=(5, 3), label="Exo OFF"),
            ]
            leg_h = ax.legend(
                handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.915),
                bbox_transform=fig.transFigure, fontsize=PUBLICATION_MUSCLE_SINGLE_LEGEND_FONTSIZE, frameon=False,
                borderpad=0.15, labelspacing=0.2, handlelength=1.55,
                handletextpad=0.4, columnspacing=0.65, ncol=2,
            )
            leg_h.set_zorder(20)
            fig.tight_layout(rect=[0, 0, 1, 0.90])
            safe = short_muscle_name(name) + ("_r" if is_right else "_l")
            fp = os.path.join(out_dir, f"muscle_{safe}_exo_on_off.png")
            fig.savefig(fp, bbox_inches="tight", dpi=PUBLICATION_SAVEFIG_DPI, pad_inches=0.03)
            plt.close(fig)
            print(f"   saved publication: {fp}")

        #
        side_label_str = "right" if is_right else "left"
        out_name = f"muscle_{side_label_str}_leg_all_exo_on_off.png"
        fig_c = make_figure_combined_exo_on_off(stats_on, stats_off, common, side_label_str)
        fp_c = os.path.join(out_dir, out_name)
        fig_c.savefig(fp_c, bbox_inches="tight", dpi=PUBLICATION_SAVEFIG_DPI, pad_inches=0.03)
        plt.close(fig_c)
        print(f"   saved publication: {fp_c}")


def generate_publication_muscle_gait_activation_only(
    analyze_result_dir: str,
    gait_data: GaitData,
) -> None:
    """"""
    out_dir = os.path.join(analyze_result_dir, PUBLICATION_MUSCLE_GAIT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    x_pct = np.linspace(0, 100, 101)
    fig_w, fig_h = 3.55, 2.92

    for is_right in (True, False):
        stats = extract_muscle_activation_stats(gait_data, is_right=is_right)
        if stats is None:
            print(f"   [WARN] publication activation-only: cannot extract {'right' if is_right else 'left'} leg gait segments, skip.")
            continue

        common = sorted(stats["per_muscle"].keys())
        if not common:
            continue

        stance_pct = float(stats["mean_stance_pct"])
        leg_str = leg_label(is_right)

        apply_publication_rcparams()
        for name in common:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PUBLICATION_RENDER_DPI)
            draw_gait_activation_single(
                ax, x_pct, stats["per_muscle"][name], stance_pct,
                show_xlabel=True, show_ylabel=True,
                title=f"{short_muscle_name(name)} — {leg_str}",
                title_fontsize=PUBLICATION_SUBPLOT_TITLE_FONTSIZE,
            )
            handles = [
                Line2D([0], [0], color=MUSCLE_PUB_SINGLE_LINE, linestyle="-", label="Mean ± SD"),
            ]
            leg_h = ax.legend(
                handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.915),
                bbox_transform=fig.transFigure, fontsize=PUBLICATION_MUSCLE_SINGLE_LEGEND_FONTSIZE, frameon=False,
                borderpad=0.15, labelspacing=0.2, handlelength=1.55,
                handletextpad=0.4, ncol=1,
            )
            leg_h.set_zorder(20)
            fig.tight_layout(rect=[0, 0, 1, 0.90])
            safe = short_muscle_name(name) + ("_r" if is_right else "_l")
            fp = os.path.join(out_dir, f"muscle_{safe}_activation_only.png")
            fig.savefig(fp, bbox_inches="tight", dpi=PUBLICATION_SAVEFIG_DPI, pad_inches=0.03)
            plt.close(fig)
            print(f"   saved publication: {fp}")

        side_label_str = "right" if is_right else "left"
        out_name = f"muscle_{side_label_str}_leg_all_activation.png"
        fig_c = make_figure_combined_activation_only(stats, common, side_label_str)
        fp_c = os.path.join(out_dir, out_name)
        fig_c.savefig(fp_c, bbox_inches="tight", dpi=PUBLICATION_SAVEFIG_DPI, pad_inches=0.03)
        plt.close(fig_c)
        print(f"   saved publication: {fp_c}")


# ============================================================
#
# ============================================================

def plot_gait_cycle_comparison(
    stats_on, stats_off, common_muscles: List[str],
    save_dir: str, *, side_label: str = "left",
) -> None:
    """"""
    fig = make_figure_combined_exo_on_off(stats_on, stats_off, common_muscles, side_label)
    fig.savefig(
        os.path.join(save_dir, "muscle_activation_comparison_gait_cycle.png"),
        bbox_inches="tight", dpi=PUBLICATION_SAVEFIG_DPI, pad_inches=0.03,
    )
    plt.close(fig)


# ============================================================
#
# ============================================================

def plot_synchronized_results(csv_path: str, save_dir: str) -> None:
    """"""
    try:
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        if len(df) < 10:
            return
        df['l_phase_pct'] = (df['l_phase'] * 100).astype(int)
        df['r_phase_pct'] = (df['r_phase'] * 100).astype(int)
        df_l = df[['l_phase_pct', 'l_exo']].copy()
        df_l.columns = ['phase', 'exo']
        df_l['leg'] = 'Left'
        df_r = df[['r_phase_pct', 'r_exo']].copy()
        df_r.columns = ['phase', 'exo']
        df_r['leg'] = 'Right'
        df_plot = pd.concat([df_l, df_r])

        apply_publication_rcparams()
        _fs = PUBLICATION_LABEL_FONTSIZE
        _ff = "Times New Roman"
        palette = {'Left': 'blue', 'Right': 'orange'}
        stats = (
            df_plot.groupby(['phase', 'leg'], as_index=False)['exo']
            .agg(mean='mean', std='std')
        )
        stats['std'] = stats['std'].fillna(0.0)

        fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.2), dpi=PUBLICATION_RENDER_DPI)
        for leg_name in ('Left', 'Right'):
            sub = stats[stats['leg'] == leg_name].sort_values('phase')
            if sub.empty:
                continue
            ph = sub['phase'].to_numpy(dtype=float)
            m = sub['mean'].to_numpy(dtype=float)
            s = sub['std'].to_numpy(dtype=float)
            c = palette[leg_name]
            ax.fill_between(ph, m - s, m + s, color=c, alpha=0.22, linewidth=0, zorder=1)
            ax.plot(ph, m, color=c, linewidth=1.35, label=leg_name, zorder=2)

        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(-1.0, 1.0)
        ax.margins(x=0, y=0)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(False)

        for _side in ("top", "right", "left", "bottom"):
            ax.spines[_side].set_visible(True)
            ax.spines[_side].set_linewidth(PUBLICATION_AXIS_SPINE_LW)
            ax.spines[_side].set_color("black")
        ax.tick_params(
            axis="both", which="major", direction="in",
            length=PUBLICATION_AXIS_TICK_LENGTH, width=PUBLICATION_AXIS_TICK_WIDTH,
            bottom=True, top=False, left=True, right=False,
            labelbottom=True, labelleft=True, labeltop=False, labelright=False,
            labelsize=_fs,
        )
        for _lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            _lab.set_fontfamily(_ff)

        ax.set_title(
            'Exoskeleton Activation: Left vs Right Leg (Normalized Phase)',
            fontsize=PUBLICATION_SUBPLOT_TITLE_FONTSIZE, fontfamily=_ff, pad=8,
        )
        ax.set_xlabel("Gait Cycle (%)", fontsize=_fs, fontfamily=_ff)
        ax.set_ylabel("Motor Activation", fontsize=_fs, fontfamily=_ff)
        leg_h = ax.legend(
            title="Leg", loc="upper right",
            frameon=True, fancybox=False, edgecolor="0.3",
        )
        leg_h.get_frame().set_linewidth(PUBLICATION_AXIS_SPINE_LW)
        for _t in leg_h.get_texts():
            _t.set_fontfamily(_ff)
            _t.set_fontsize(_fs)
        if leg_h.get_title() is not None:
            leg_h.get_title().set_fontfamily(_ff)
            leg_h.get_title().set_fontsize(_fs)

        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, "synchronized_gait_comparison.png"),
            bbox_inches="tight", dpi=PUBLICATION_SAVEFIG_DPI, pad_inches=0.03,
        )
        plt.close(fig)
    except Exception as e:
        print(f"   [Plot Error] {e}")