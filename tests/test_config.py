"""Tests for hairguard.config validation.

Exercises the pure validation function directly so the user's real
~/.hairguard/config.yaml is never read or written.
"""
from __future__ import annotations

from hairguard.config import DEFAULTS, _deep_merge, _validate_and_fix


def _validated(overrides: dict) -> dict:
    """Validate a config built from DEFAULTS plus overrides (no missing fields)."""
    return _validate_and_fix(_deep_merge(DEFAULTS, overrides))


def test_camera_auto_is_accepted() -> None:
    assert _validated({"camera": "auto"})["camera"] == "auto"


def test_explicit_camera_index_is_accepted() -> None:
    assert _validated({"camera": 3})["camera"] == 3


def test_unknown_camera_string_falls_back_to_default() -> None:
    assert _validated({"camera": "banana"})["camera"] == DEFAULTS["camera"]


def test_out_of_range_camera_index_falls_back_to_default() -> None:
    assert _validated({"camera": 999})["camera"] == DEFAULTS["camera"]


def test_default_camera_setting_is_auto() -> None:
    """Fresh installs should detect their camera instead of assuming index 0."""
    assert DEFAULTS["camera"] == "auto"


def test_alert_sound_is_on_by_default() -> None:
    """The overlay only reaches a user who is looking at the screen."""
    assert DEFAULTS["alert"]["sound"] is True


def test_alert_sound_can_be_turned_off() -> None:
    assert _validated({"alert": {"sound": False}})["alert"]["sound"] is False


def test_a_non_boolean_sound_setting_falls_back() -> None:
    assert _validated({"alert": {"sound": "yes please"}})["alert"]["sound"] is True


# ---------------------------------------------------------------------------
# Capture resolution
# ---------------------------------------------------------------------------

def test_capture_resolution_defaults_to_720p() -> None:
    """Detail on the hand is the binding constraint for the dataset."""
    assert DEFAULTS["camera_width"] == 1280
    assert DEFAULTS["camera_height"] == 720


def test_capture_resolution_can_be_overridden() -> None:
    validated = _validated({"camera_width": 1920, "camera_height": 1080})
    assert validated["camera_width"] == 1920
    assert validated["camera_height"] == 1080


def test_absurd_capture_width_falls_back_to_default() -> None:
    assert _validated({"camera_width": 99999})["camera_width"] == DEFAULTS["camera_width"]


def test_detectors_see_the_full_capture_width_by_default() -> None:
    """Measured cost per tick was identical from 640 to 1280, so nothing is thrown away."""
    assert DEFAULTS["detection"]["process_width"] == DEFAULTS["camera_width"]


def test_process_width_is_validated() -> None:
    validated = _validated({"detection": {"process_width": 10}})
    assert validated["detection"]["process_width"] == DEFAULTS["detection"]["process_width"]
