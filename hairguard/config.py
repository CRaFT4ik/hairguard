import os
import sys
from pathlib import Path
from typing import Any

import yaml

HAIRGUARD_DIR = Path.home() / ".hairguard"
CONFIG_PATH = HAIRGUARD_DIR / "config.yaml"

DEFAULTS: dict[str, Any] = {
    "camera": 0,
    "detection": {
        "trigger_delay_sec": 0.8,
        "dismiss_delay_sec": 2.5,
        "fps": 10,
        "min_confidence": 0.5,
        "pinch_threshold": 0.40,
        "spread_threshold": 0.075,
        "rotation_threshold": 0.10,
        "persist_frames": 5,
        "min_presence_score": 0.7,
        "hair_zone_offset": 0.4,
    },
    "alert": {
        "opacity": 0.7,
        "text_color": "#ffffff",
    },
    "pause": {
        "auto_resume_minutes": 15,
    },
}

_FIELD_RULES: list[tuple[tuple[str, ...], type, tuple | None]] = [
    (("camera",), int, (0, 99)),
    (("detection", "trigger_delay_sec"), float, (0.0, None)),
    (("detection", "dismiss_delay_sec"), float, (0.0, None)),
    (("detection", "fps"), (int, float), (1, None)),
    (("detection", "min_confidence"), float, (0.0, 1.0)),
    (("detection", "pinch_threshold"), float, (0.05, 1.0)),
    (("detection", "spread_threshold"), float, (0.01, 0.5)),
    (("detection", "rotation_threshold"), float, (0.01, 0.5)),
    (("detection", "persist_frames"), (int, float), (1, 30)),
    (("detection", "min_presence_score"), float, (0.0, 1.0)),
    (("detection", "hair_zone_offset"), float, (0.0, 2.0)),
    (("alert", "opacity"), float, (0.0, 1.0)),
    (("alert", "text_color"), str, None),
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
