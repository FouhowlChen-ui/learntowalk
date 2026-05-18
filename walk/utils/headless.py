"""Headless / display-less server bootstrapping.

This module makes the EXAM project usable on a server with no display
(no X / Wayland) without changing behaviour on a normal desktop:

- Force matplotlib to the non-interactive ``Agg`` backend.
- On Linux, when no ``DISPLAY`` is available, default ``MUJOCO_GL`` to
  ``disable`` so ``import mujoco`` does not pull in PyOpenGL/EGL (many
  headless nodes lack EGL and crash at import time). For ``replay.mp4``,
  set ``MUJOCO_GL=osmesa`` or ``egl`` **before** starting Python when the
  drivers/libs are installed.
- Honour user overrides: if ``MUJOCO_GL`` / ``MPLBACKEND`` /
  ``PYOPENGL_PLATFORM`` is already set, leave it untouched.
- Provide ``is_headless()`` so other modules can guard against
  interactive rendering paths.

Call :func:`setup_headless_environment` exactly once, as early as
possible. ``walk/__init__.py`` does this before any MuJoCo / myosuite
import so that downstream code automatically gets a working renderer.
"""

import os
import sys


_VALID_BACKENDS = ("egl", "osmesa", "glfw", "disable")


def is_headless() -> bool:
    """Best-effort detection of a display-less environment.

    Linux: no ``DISPLAY`` and no ``WAYLAND_DISPLAY`` env var.
    Windows / macOS: always returns ``False`` (display assumed available).
    """
    if sys.platform.startswith("linux"):
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False


def setup_headless_environment(
    *,
    force: bool = False,
    prefer_backend: str = "disable",
    verbose: bool = True,
) -> dict:
    """Configure env vars so renderer + matplotlib work without a display.

    Idempotent. Safe to call from multiple entry points.

    Args:
        force: Apply headless defaults even when a display is detected
            (useful when starting via ``ssh -X`` but you still want
            offscreen rendering).
        prefer_backend: ``disable`` (no GL import; default on headless),
            ``osmesa`` (CPU offscreen, needs Mesa), or ``egl`` (GPU
            offscreen; needs working EGL). Ignored on non-Linux platforms.
        verbose: Print a one-line summary of the chosen backend.

    Returns:
        Resolved settings as a dict.
    """
    if prefer_backend not in _VALID_BACKENDS:
        raise ValueError(
            f"prefer_backend must be one of {_VALID_BACKENDS}, got {prefer_backend!r}"
        )

    headless = force or is_headless()
    settings = {"headless": headless, "platform": sys.platform}

    if "MPLBACKEND" not in os.environ:
        os.environ["MPLBACKEND"] = "Agg"
    settings["MPLBACKEND"] = os.environ["MPLBACKEND"]

    if headless and sys.platform.startswith("linux"):
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = prefer_backend
        chosen = os.environ.get("MUJOCO_GL", prefer_backend)
        if chosen == "egl" and "PYOPENGL_PLATFORM" not in os.environ:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
        elif chosen == "osmesa" and "PYOPENGL_PLATFORM" not in os.environ:
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        elif chosen == "disable":
            # Do not force PyOpenGL to load EGL/OSMesa when no GL is needed.
            os.environ.pop("PYOPENGL_PLATFORM", None)

    settings["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "<unset>")
    settings["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "<unset>")

    if verbose and headless:
        print(
            f"[headless] no display detected; "
            f"MUJOCO_GL={settings['MUJOCO_GL']}, "
            f"PYOPENGL_PLATFORM={settings['PYOPENGL_PLATFORM']}, "
            f"MPLBACKEND={settings['MPLBACKEND']}"
        )

    return settings
