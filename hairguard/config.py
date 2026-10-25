import os
import sys
from pathlib import Path
from typing import Any

import yaml

HAIRGUARD_DIR = Path.home() / ".hairguard"
CONFIG_PATH = HAIRGUARD_DIR / "config.yaml"

DEFAULTS: dict[str, Any] = {
    "camera": "auto",
    # Capture resolution. Detail on the hand limits what a model can learn, so
    # frames are grabbed at 720p even though detection runs on a downscaled
    # copy (see detection.process_width) and costs the same either way.
    "camera_width": 1280,
    "camera_height": 720,
    "detection": {
        "trigger_delay_sec": 0.8,
        "dismiss_delay_sec": 2.5,
        "fps": 10,
        # Width the detectors see. MediaPipe rescales to its own model input
        # anyway, and measurement showed the same cost per tick from 640 to
        # 1280, so the frame is passed through at capture width by default.
        # Lower it only if a weaker machine needs the headroom.
        "process_width": 1280,
        "min_confidence": 0.5,
        "combined_threshold": 0.55,
        "rotation_threshold": 0.10,
        "persist_frames": 5,
        "min_presence_score": 0.7,
        "hair_zone_offset": 0.4,
        # The trained detector. Replaces the thresholds above as the verdict
        # when the artefacts in ~/.hairguard/trained are present and match
        # the current feature extractor; the app falls back to the rule and
        # says so otherwise.
        "model": {
            # Set false to pin the app to the old rule without deleting
            # anything — the instant way back if the model misbehaves.
            "enabled": True,
            # "auto" takes the operating point chosen during training, which
            # is the lowest threshold holding the false-alarm rate on "hand
            # resting on head" under 5%. A number here overrides it; higher
            # is quieter and misses more, at roughly ten points of recall per
            # five points of false-alarm rate.
            "threshold": "auto",
            # Sustained model detection before the overlay appears. Much
            # shorter than detection.trigger_delay_sec: each verdict already
            # integrates a run of frames, so this only has to reject a lone
            # window rather than establish persistence from scratch. See the
            # analyzer docstring.
            "trigger_delay_sec": 0.3,
            # In-zone frames before the model is asked at all. The window is
            # 20 frames, which is ~2.8 s at the 6.75 fps this loop achieves,
            # and waiting that long to react defeats the point of the app.
            # `hairlab.truncation` measured the cost of asking earlier,
            # cross-validated by capture day: at 3 frames recall is 78.0% and
            # the false-alarm rate on a resting hand 3.2%, against 86.7% and
            # 3.6% at the full 20 — still far ahead of the 53.9% rule on both
            # axes, for a ninth of the wait. Raise it towards 20 to trade
            # reaction time back for recall.
            "min_frames": 3,
        },
    },
    "alert": {
        "opacity": 0.7,
        "text_color": "#ffffff",
        # The overlay only works if you are looking at the screen.
        "sound": True,
    },
    "pause": {
        "auto_resume_minutes": 15,
    },
}

# Fields that also accept a fixed set of keyword values besides their type.
_LITERALS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("camera",): ("auto",),
    ("detection", "model", "threshold"): ("auto",),
}

_FIELD_RULES: list[tuple[tuple[str, ...], type, tuple | None]] = [
    (("camera",), int, (0, 99)),
    (("camera_width",), int, (160, 4096)),
    (("camera_height",), int, (120, 2160)),
    (("detection", "trigger_delay_sec"), float, (0.0, None)),
    (("detection", "dismiss_delay_sec"), float, (0.0, None)),
    (("detection", "fps"), (int, float), (1, None)),
    (("detection", "process_width"), int, (160, 4096)),
    (("detection", "min_confidence"), float, (0.0, 1.0)),
    (("detection", "combined_threshold"), float, (0.1, 2.0)),
    (("detection", "rotation_threshold"), float, (0.01, 0.5)),
    (("detection", "persist_frames"), (int, float), (1, 30)),
    (("detection", "min_presence_score"), float, (0.0, 1.0)),
    (("detection", "hair_zone_offset"), float, (0.0, 2.0)),
    (("detection", "model", "enabled"), bool, None),
    (("detection", "model", "threshold"), (int, float), (0.0, 1.0)),
    (("detection", "model", "trigger_delay_sec"), (int, float), (0.0, None)),
    # Upper bound is generous rather than the window length: the window comes
    # from the model file, not from here, and a value above it simply means
    # "wait for a full window".
    (("detection", "model", "min_frames"), int, (1, 120)),
    (("alert", "opacity"), float, (0.0, 1.0)),
    (("alert", "text_color"), str, None),
    (("alert", "sound"), bool, None),
    (("pause", "auto_resume_minutes"), (int, float), (1, None)),
]


def _get(data: dict, keys: tuple[str, ...]) -> Any:
    node = data
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _set(data: dict, keys: tuple[str, ...], value: Any) -> None:
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _default(keys: tuple[str, ...]) -> Any:
    node: Any = DEFAULTS
    for k in keys:
        node = node[k]
    return node


def _validate_and_fix(data: dict) -> dict:
    for keys, expected_type, range_ in _FIELD_RULES:
        value = _get(data, keys)
        field_path = ".".join(keys)

        if value is None:
            print(f"[config] Missing field '{field_path}', using default.")
            _set(data, keys, _default(keys))
            continue

        literals = _LITERALS.get(keys)
        if literals is not None and value in literals:
            continue

        if not isinstance(value, expected_type):
            print(
                f"[config] Field '{field_path}' has wrong type "
                f"(expected {expected_type}, got {type(value).__name__}), "
                "using default."
            )
            _set(data, keys, _default(keys))
            continue

        if range_ is not None:
            lo, hi = range_
            if lo is not None and value < lo:
                print(
                    f"[config] Field '{field_path}' value {value!r} is below "
                    f"minimum {lo}, using default."
                )
                _set(data, keys, _default(keys))
                continue
            if hi is not None and value > hi:
                print(
                    f"[config] Field '{field_path}' value {value!r} exceeds "
                    f"maximum {hi}, using default."
                )
                _set(data, keys, _default(keys))
                continue

    return data


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config:
    def __init__(self) -> None:
        HAIRGUARD_DIR.mkdir(parents=True, exist_ok=True)
        if not CONFIG_PATH.exists():
            self._write_defaults()
        self._data = self._load()

    def _write_defaults(self) -> None:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULTS, f, default_flow_style=False, allow_unicode=True)

    def _load(self) -> dict:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        merged = _deep_merge(DEFAULTS, raw)
        return _validate_and_fix(merged)

    def get(self, *keys: str) -> Any:
        return _get(self._data, keys)

    @property
    def data(self) -> dict:
        return self._data

    def show(self) -> None:
        print(yaml.dump(self._data, default_flow_style=False, allow_unicode=True), end="")

    def open_editor(self) -> None:
        editor = os.environ.get("EDITOR")
        if not editor:
            editor = "notepad" if sys.platform == "win32" else "nano"
        import subprocess
        subprocess.run([editor, str(CONFIG_PATH)])
