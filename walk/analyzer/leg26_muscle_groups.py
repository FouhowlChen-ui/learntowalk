""""""

#
HIP_ABDUCTORS = ["abd"]
HIP_ADDUCTORS = ["add"]
HIP_FLEXORS = ["iliopsoas", "rectfem"]
#
HIP_EXTENSORS = ["glutmax", "hamstrings", "bifemsh"]
KNEE_ANKLE = ["vasti", "gastroc", "soleus", "tibant", "edl", "fdl"]

#
GROUP_LABELS = {
    "HIP_ABDUCTORS": "Hip Abductors (abd)",
    "HIP_ADDUCTORS": "Hip Adductors (add)",
    "HIP_FLEXORS": "Hip Flexors",
    "HIP_EXTENSORS": "Hip Extensors",
    "KNEE_ANKLE": "Knee / Ankle",
}

#
GROUP_LABELS_EN = {
    "HIP_ABDUCTORS": "Hip Abductors (abd)",
    "HIP_ADDUCTORS": "Hip Adductors (add)",
    "HIP_FLEXORS": "Hip Flexors",
    "HIP_EXTENSORS": "Hip Extensors",
    "KNEE_ANKLE": "Knee / Ankle",
    "OTHER": "Other",
}

#
GROUP_COLORS = {
    "HIP_ABDUCTORS": "#9b59b6",   #
    "HIP_ADDUCTORS": "#e67e22",   #
    "HIP_FLEXORS": "#2ecc71",     #
    "HIP_EXTENSORS": "#3498db",   #
    "KNEE_ANKLE": "#95a5a6",      #
    "OTHER": "#bdc3c7",           #
}

#
MUSCLE_GROUP_DISPLAY_ORDER = [
    "HIP_ABDUCTORS",
    "HIP_ADDUCTORS",
    "HIP_FLEXORS",
    "HIP_EXTENSORS",
    "KNEE_ANKLE",
]

#
_BASE_TO_GROUP = {}
for _g in MUSCLE_GROUP_DISPLAY_ORDER:
    for _b in globals()[_g]:
        _BASE_TO_GROUP[_b] = _g


def base_name(actuator_name: str) -> str:
    """"""
    return actuator_name.rsplit("_", 1)[0] if "_" in actuator_name else actuator_name


def group_of(actuator_name: str) -> str:
    """"""
    return _BASE_TO_GROUP.get(base_name(actuator_name), "OTHER")


def group_color(actuator_name: str) -> str:
    """"""
    return GROUP_COLORS[group_of(actuator_name)]


def group_label(group_key: str) -> str:
    return GROUP_LABELS.get(group_key, group_key)


def group_label_en(group_key: str) -> str:
    """"""
    return GROUP_LABELS_EN.get(group_key, group_key)
