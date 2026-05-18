import json
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HIP_ABDUCTORS = ["abd"]
HIP_ADDUCTORS = ["add"]
HIP_FLEXORS = ["iliopsoas", "rectfem"]
HIP_EXTENSORS = ["glutmax", "hamstrings", "bifemsh"]
KNEE_ANKLE = ["vasti", "gastroc", "soleus", "tibant", "edl", "fdl"]

GROUP_DISPLAY_ORDER = [
    "HIP_ABDUCTORS",
    "HIP_ADDUCTORS",
    "HIP_FLEXORS",
    "HIP_EXTENSORS",
    "KNEE_ANKLE",
]

GROUP_COLORS = {
    "HIP_ABDUCTORS": "#9b59b6",
    "HIP_ADDUCTORS": "#e67e22",
    "HIP_FLEXORS": "#2ecc71",
    "HIP_EXTENSORS": "#3498db",
    "KNEE_ANKLE": "#95a5a6",
    "OTHER": "#bdc3c7",
}

GROUP_LABELS_EN = {
    "HIP_ABDUCTORS": "Hip Abductors",
    "HIP_ADDUCTORS": "Hip Adductors",
    "HIP_FLEXORS": "Hip Flexors",
    "HIP_EXTENSORS": "Hip Extensors",
    "KNEE_ANKLE": "Knee/Ankle",
    "OTHER": "Other",
}


def _base(name: str) -> str:
    return name.rsplit("_", 1)[0] if "_" in name else name


def _group_of(name: str) -> str:
    base = _base(name)
    for g in GROUP_DISPLAY_ORDER:
        if base in globals()[g]:
            return g
    return "OTHER"


def plot_reward_curve(train_log_handler, save_path: str):
    if not train_log_handler.log_datas:
        print("[plot] no log data, skip return curve")
        return
    time_steps = [
        log_data.num_timesteps for log_data in train_log_handler.log_datas
    ]
    rewards = [
        log_data.average_reward_per_episode for log_data in train_log_handler.log_datas
    ]
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.4), dpi=120)
    ax.plot(np.array(time_steps) / 1e6, rewards, color="#4682B4", linewidth=1.5)
    ax.set_xlabel("Timesteps (x 1e6)")
    ax.set_ylabel("Average Episode Return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[plot] return curve saved to {save_path}")


def plot_reward_components(train_log_handler, save_path: str):
    if not train_log_handler.log_datas:
        return
    log_datas = train_log_handler.log_datas[1:]
    if not log_datas:
        return
    if not log_datas[0].average_reward_dict_per_episode:
        return
    time_steps = [ld.num_timesteps for ld in log_datas]
    keys = [
        k
        for k in log_datas[0].average_reward_dict_per_episode.keys()
        if k not in ("sparse", "solved", "done", "dense")
    ]
    if not keys:
        return
    data = {
        k: [ld.average_reward_dict_per_episode.get(k, 0.0) for ld in log_datas]
        for k in keys
    }

    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
    color_map = plt.get_cmap("tab20")
    for idx, (key, vals) in enumerate(data.items()):
        ax.plot(
            np.array(time_steps) / 1e6,
            vals,
            label=key,
            color=color_map(idx / max(len(keys), 1)),
            linewidth=1.0,
        )
    ax.set_xlabel("Timesteps (x 1e6)")
    ax.set_ylabel("Reward Component")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[plot] reward components saved to {save_path}")


def analyze_muscle_activation(json_path: str, save_dir: str, dname: str = ""):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[muscle] load failed {json_path}: {e}")
        return {}

    actuator_data = data.get("series_data", {}).get("actuator_data", {})
    metadata = data.get("metadata", {})
    n_steps = metadata.get("data_length", 0)

    muscle_names: List[str] = sorted(
        [k for k in actuator_data.keys() if "Exo" not in k]
    )
    if not muscle_names:
        print("[muscle] no actuator found, skip.")
        return {}

    results = {}
    for name in muscle_names:
        ctrl = actuator_data[name]["ctrl"]
        ctrl = np.array([v[0] if isinstance(v, list) else v for v in ctrl])
        ctrl = np.abs(ctrl)
        if len(ctrl) == 0:
            continue
        results[name] = {
            "mean": float(np.mean(ctrl)),
            "std": float(np.std(ctrl)),
            "max": float(np.max(ctrl)),
            "integral": float(np.sum(ctrl)),
            "n_steps": int(len(ctrl)),
            "group": _group_of(name),
            "base": _base(name),
        }

    print("=" * 60)
    print(f"  Muscle activation summary: {dname}")
    print(f"  n_steps={n_steps}  num_muscles={len(results)}")
    print("=" * 60)
    for g in GROUP_DISPLAY_ORDER:
        names_in_group = [n for n, r in results.items() if r["group"] == g]
        if not names_in_group:
            continue
        print(f"\n[{GROUP_LABELS_EN[g]}]")
        for n in sorted(names_in_group):
            r = results[n]
            print(
                f"  {n:18s} mean={r['mean']*100:5.2f}%  "
                f"max={r['max']*100:5.2f}%  integral={r['integral']:.1f}"
            )

    os.makedirs(save_dir, exist_ok=True)
    plot_names = sorted(
        results.keys(),
        key=lambda n: (
            GROUP_DISPLAY_ORDER.index(results[n]["group"])
            if results[n]["group"] in GROUP_DISPLAY_ORDER
            else len(GROUP_DISPLAY_ORDER),
            n,
        ),
    )
    if plot_names:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
        means = [results[n]["mean"] * 100 for n in plot_names]
        colors = [GROUP_COLORS[results[n]["group"]] for n in plot_names]
        ax.barh(plot_names, means, color=colors)
        ax.set_xlabel("Mean Activation (%)")
        ax.set_title(f"Muscle Activation Summary - {dname}")
        ax.set_xlim(0, 100)

        seen, handles = [], []
        for n in plot_names:
            g = results[n]["group"]
            if g not in seen:
                seen.append(g)
                handles.append(
                    plt.Rectangle(
                        (0, 0), 1, 1, color=GROUP_COLORS[g], label=GROUP_LABELS_EN[g]
                    )
                )
        if handles:
            ax.legend(handles=handles, loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, "muscle_activation_summary.png"),
            bbox_inches="tight",
            dpi=160,
        )
        plt.close(fig)

    csv_path = os.path.join(save_dir, "muscle_activation_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("muscle,base,group,mean_activation,max_activation,integral,n_steps\n")
        for name in plot_names:
            r = results[name]
            f.write(
                f"{name},{r['base']},{r['group']},"
                f"{r['mean']},{r['max']},{r['integral']},{r['n_steps']}\n"
            )
    print(f"[muscle] saved CSV {csv_path}")
    return results


def plot_joint_kinematics(json_path: str, save_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    joint_data = data["series_data"]["joint_data"]

    targets = [
        ("hip_flexion_r", "hip_flexion_l", "Hip flexion (rad)"),
        ("knee_angle_r", "knee_angle_l", "Knee angle (rad)"),
        ("ankle_angle_r", "ankle_angle_l", "Ankle angle (rad)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), dpi=120, sharex=True)
    for ax, (rkey, lkey, ylabel) in zip(axes, targets):
        if rkey in joint_data:
            r = np.array(
                [v[0] if isinstance(v, list) else v for v in joint_data[rkey]["qpos"]]
            )
            ax.plot(r, color="#B22222", label=f"{rkey} (R)", linewidth=1.0)
        if lkey in joint_data:
            l = np.array(
                [v[0] if isinstance(v, list) else v for v in joint_data[lkey]["qpos"]]
            )
            ax.plot(l, color="#4682B4", linestyle="--", label=f"{lkey} (L)",
                    linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Step (control framerate)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[plot] joint kinematics saved to {save_path}")
