"""Shared LIBERO baseline protocol constants and helpers.

This module intentionally has no heavyweight dependencies so contract tests can
exercise the OpenVLA / WorldVLA alignment rules without a LIBERO install.
"""

from __future__ import annotations

from math import sqrt
from typing import Sequence


DEFAULT_CAMERA_SIZE = 224
DEFAULT_EVAL_EPISODES = 50
DEFAULT_NUM_STEPS_WAIT = 10
NOOP_ACTION_EPS = 1e-4

DUMMY_ACTION: tuple[float, float, float, float, float, float, float] = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0,
)

SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def get_suite_max_steps(suite: str) -> int:
    """Return the WorldVLA/OpenVLA LIBERO raw-step cap for a suite."""
    try:
        return SUITE_MAX_STEPS[suite]
    except KeyError as exc:
        raise ValueError(
            f"Unknown LIBERO suite {suite!r}; expected one of "
            f"{sorted(SUITE_MAX_STEPS)}."
        ) from exc


def is_noop_action(
    action: Sequence[float],
    previous_kept_action: Sequence[float] | None,
    *,
    threshold: float = NOOP_ACTION_EPS,
) -> bool:
    """Return True when an action should be filtered as an OpenVLA no-op.

    OpenVLA's regenerate flow drops actions whose first six controller
    dimensions are effectively zero and whose gripper command did not change
    relative to the previous kept action. For the first action, there is no
    previous gripper command, so stationarity of the first six dims is enough.
    """
    if len(action) < 7:
        raise ValueError(f"Expected 7D LIBERO action, got length {len(action)}")

    motion_norm = sqrt(sum(float(x) * float(x) for x in action[:6]))
    if motion_norm > threshold:
        return False
    if previous_kept_action is None:
        return True
    if len(previous_kept_action) < 7:
        raise ValueError(
            "Expected previous_kept_action to be 7D when provided, got "
            f"length {len(previous_kept_action)}"
        )
    return abs(float(action[6]) - float(previous_kept_action[6])) <= threshold


def rotate_image_180(image):
    """Rotate a HWC image 180 degrees with a defensive copy.

    For numpy arrays this is equivalent to ``image[::-1, ::-1].copy()``. A
    list fallback keeps the helper easy to test in lightweight environments.
    """
    try:
        rotated = image[::-1, ::-1]
    except TypeError:
        return [list(row[::-1]) for row in image[::-1]]
    if hasattr(rotated, "copy"):
        return rotated.copy()
    return rotated
