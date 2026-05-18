""""""

import json
import os
from typing import Dict, List

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from walk.analyzer.gait_analyze import (
    PUBLICATION_RENDER_DPI,
    PUBLICATION_SAVEFIG_DPI,
)

from walk.analyzer.leg26_muscle_groups import (
    GROUP_LABELS,
    MUSCLE_GROUP_DISPLAY_ORDER,
    base_name,
    group_color,
    group_label_en,
    group_of,
)


def _extract_ctrl_series(actuator_data: dict, name: str) -> np.ndarray | None:
    if name not in actuator_data:
        return None
    ctrl = actuator_data[name]["ctrl"]
    return np.array([v[0] if isinstance(v, list) else v for v in ctrl])


def analyze_single_muscle_activation_26(
    json_path: str,
    save_dir: str,
    dname: str = "",
) -> Dict[str, dict]:
    """"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"   [Leg26 muscle summary] failed to load {json_path}: {e}")
        return {}

    actuator_data = data.get("series_data", {}).get("actuator_data", {})
    metadata = data.get("metadata", {})
    n_steps = metadata.get("data_length", 0)

    #
    muscle_names: List[str] = sorted(
        [k for k in actuator_data.keys() if "Exo" not in k]
    )

    if not muscle_names:
        print("   [Leg26 muscle summary] no non-Exo actuator found, skip.")
        return {}

    has_exo_keys = any("Exo" in k for k in actuator_data.keys())
    if has_exo_keys:
        print(
            "   [Leg26 muscle summary] WARN: Exo_* actuator detected in JSON; "
            "26m baseline does not expect such fields, skipping them."
        )

    results: Dict[str, dict] = {}
    for name in muscle_names:
        ctrl = _extract_ctrl_series(actuator_data, name)
        if ctrl is None or len(ctrl) == 0:
            continue
        ctrl = np.abs(ctrl)
        results[name] = {
            "mean": float(np.mean(ctrl)),
            "std": float(np.std(ctrl)),
            "max": float(np.max(ctrl)),
            "integral": float(np.sum(ctrl)),
            "n_steps": int(len(ctrl)),
            "group": group_of(name),
            "base": base_name(name),
        }

    print("\n" + "=" * 72)
    print(f"  Leg26 muscle activation summary: {dname}")
    print(f"  total steps: {n_steps}    muscles: {len(results)}")
    print("=" * 72)

    for group_key in MUSCLE_GROUP_DISPLAY_ORDER:
        names_in_group = [n for n, r in results.items() if r["group"] == group_key]
        if not names_in_group:
            continue
        print(f"\n[{GROUP_LABELS[group_key]}]")
        for n in sorted(names_in_group):
            r = results[n]
            print(
                f"  {n:20s}  mean: {r['mean']*100:5.2f}%  "
                f"max: {r['max']*100:5.2f}%  integral: {r['integral']:.1f}"
            )

    other_names = [n for n, r in results.items() if r["group"] == "OTHER"]
    if other_names:
        print(f"\n[Other (ungrouped)]")
        for n in sorted(other_names):
            r = results[n]
            print(
                f"  {n:20s}  mean: {r['mean']*100:5.2f}%  "
                f"max: {r['max']*100:5.2f}%  integral: {r['integral']:.1f}"
            )

    #
    if "Exo_R" in actuator_data or "Exo_L" in actuator_data:
        print(
            "   [Leg26 muscle summary] WARN: Exo_R/Exo_L found in JSON; "
            "ensure 22m 2D gait_evaluated_data.json was not fed into leg26 eval."
        )

    os.makedirs(save_dir, exist_ok=True)
    plot_names = sorted(
        results.keys(),
        key=lambda n: (
            MUSCLE_GROUP_DISPLAY_ORDER.index(results[n]["group"])
            if results[n]["group"] in MUSCLE_GROUP_DISPLAY_ORDER
            else len(MUSCLE_GROUP_DISPLAY_ORDER),
            n,
        ),
    )
    if plot_names:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=PUBLICATION_RENDER_DPI)
        means = [results[n]["mean"] * 100 for n in plot_names]
        colors = [group_color(n) for n in plot_names]
        ax.barh(plot_names, means, color=colors)
        ax.set_xlabel("Mean Activation (%)")
        ax.set_title(f"Leg26 Muscle Activation Summary - {dname}")
        ax.set_xlim(0, 100)

        #
        seen = []
        handles = []
        for n in plot_names:
            g = results[n]["group"]
            if g not in seen:
                seen.append(g)
                handles.append(
                    plt.Rectangle(
                        (0, 0), 1, 1, color=group_color(n), label=group_label_en(g)
                    )
                )
        if handles:
            ax.legend(handles=handles, loc="lower right", fontsize=8)

        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, "muscle_activation_summary.png"),
            bbox_inches="tight",
            dpi=PUBLICATION_SAVEFIG_DPI,
            pad_inches=0.03,
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
    print(f"   [Leg26 muscle summary] saved {csv_path}")
    return results
