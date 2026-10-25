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


# ---------------------------------------------------------------------------
# The trained detector
# ---------------------------------------------------------------------------

def test_the_model_is_used_by_default() -> None:
    """It catches roughly three times as many episodes as the rule does."""
    assert DEFAULTS["detection"]["model"]["enabled"] is True


def test_the_model_can_be_switched_off_without_deleting_it() -> None:
    """The instant way back, needed because the model is new."""
    validated = _validated({"detection": {"model": {"enabled": False}}})
    assert validated["detection"]["model"]["enabled"] is False


def test_a_non_boolean_enabled_falls_back_to_on() -> None:
    validated = _validated({"detection": {"model": {"enabled": "sure"}}})
    assert validated["detection"]["model"]["enabled"] is True


def test_the_threshold_defaults_to_the_one_training_chose() -> None:
    """Training picked it against a measured false-alarm ceiling; config should not guess."""
    assert DEFAULTS["detection"]["model"]["threshold"] == "auto"


def test_an_explicit_threshold_is_accepted() -> None:
    validated = _validated({"detection": {"model": {"threshold": 0.8}}})
    assert validated["detection"]["model"]["threshold"] == 0.8


def test_a_threshold_outside_the_probability_range_falls_back_to_auto() -> None:
    validated = _validated({"detection": {"model": {"threshold": 1.4}}})
    assert validated["detection"]["model"]["threshold"] == "auto"


def test_a_nonsense_threshold_falls_back_to_auto() -> None:
    validated = _validated({"detection": {"model": {"threshold": "low"}}})
    assert validated["detection"]["model"]["threshold"] == "auto"


def test_the_model_trigger_delay_is_shorter_than_the_rules() -> None:
    """Each model verdict already covers a 2 s window; stacking 0.8 s on top
    would put the alert nearly four seconds after the hand arrived."""
    detection = DEFAULTS["detection"]
    assert detection["model"]["trigger_delay_sec"] < detection["trigger_delay_sec"]


def test_a_negative_model_trigger_delay_falls_back() -> None:
    validated = _validated({"detection": {"model": {"trigger_delay_sec": -1}}})
    assert validated["detection"]["model"]["trigger_delay_sec"] == 0.3


def test_the_model_is_asked_well_before_the_window_fills() -> None:
    """A full 20-frame window is ~2.8 s at the 6.75 fps this loop achieves,
    and the app exists to interrupt a behaviour before it completes."""
    assert DEFAULTS["detection"]["model"]["min_frames"] == 3


def test_min_frames_can_be_raised_to_trade_speed_for_recall() -> None:
    validated = _validated({"detection": {"model": {"min_frames": 20}}})
    assert validated["detection"]["model"]["min_frames"] == 20


def test_a_zero_min_frames_falls_back() -> None:
    validated = _validated({"detection": {"model": {"min_frames": 0}}})
    assert validated["detection"]["model"]["min_frames"] == 3


def test_a_nonsense_min_frames_falls_back() -> None:
    validated = _validated({"detection": {"model": {"min_frames": "soon"}}})
    assert validated["detection"]["model"]["min_frames"] == 3
