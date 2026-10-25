"""Tests for ``hairguard doctor`` — what the packaged app can say about itself.

The constraint this answers: when the app fails for someone with no
terminal, something must still be able to say why. The log says what
happened; this says what is *there*. Can the model be loaded, is the feature
extractor it is checked against importable, did the landmark models arrive
with the bundle or will they be downloaded, which detector will actually
decide.

It is also what the build runs against the finished artefact, with the home
directory pointed somewhere empty. Pure Python modules live inside the
bundle's archive rather than as files on disk, so no amount of looking at
the directory tree can establish that ``hairlab.features`` made it in —
whereas asking the built binary establishes it exactly.

One thing it must never do is touch the camera. It is most likely to be run
while the app is running and holding the device, and on Windows a second
open returns black frames rather than an error.
"""
from __future__ import annotations

import sys

import pytest

from hairguard import cli


def _run(capsys, argv=("doctor",)) -> tuple[int, str]:
    code = cli.main(list(argv))
    return int(code or 0), capsys.readouterr().out


# ---------------------------------------------------------------------------
# On this machine, where everything is present
# ---------------------------------------------------------------------------


def test_doctor_reports_a_healthy_install(capsys) -> None:
    code, out = _run(capsys)

    assert code == 0, out
    assert "hairlab.features" in out
    assert "onnxruntime" in out
    assert "Детектор:" in out


def test_doctor_names_the_log_so_it_can_be_found(capsys) -> None:
    """The next question after "what is wrong" is always "where do I look"."""
    from hairguard import diagnostics

    _, out = _run(capsys)

    assert str(diagnostics.log_path()) in out


def test_doctor_says_whether_this_is_a_bundle(capsys) -> None:
    _, out = _run(capsys)

    assert sys.executable in out


def test_doctor_reports_the_feature_width(capsys) -> None:
    """96 here. A model trained against a different number is refused at
    load time, and this is where the two can be compared."""
    from hairlab.features import FEATURE_DIM

    _, out = _run(capsys)

    assert str(FEATURE_DIM) in out


def test_doctor_lists_every_landmark_model_and_where_it_came_from(capsys) -> None:
    """Three files, 21 MB. A bundle that ships two of them downloads the
    third on first run, silently, before the tray icon exists."""
    _, out = _run(capsys)

    for name in ("hand_landmarker.task", "face_landmarker.task",
                 "pose_landmarker_full.task"):
        assert name in out


def test_doctor_says_which_detector_will_decide(capsys) -> None:
    """The one number this app lives by: the rule catches 53.9% of episodes
    and the model 86.7%, and nothing on screen tells them apart."""
    _, out = _run(capsys)

    detector = next(line for line in out.splitlines() if line.startswith("Детектор:"))
    assert "модель" in detector or "правило" in detector


# ---------------------------------------------------------------------------
# Whether the training stack came along
# ---------------------------------------------------------------------------


def test_doctor_reports_which_training_modules_are_reachable(capsys) -> None:
    """From a source checkout they are installed — that is how the model got
    trained. The report says so without complaining."""
    import json

    code, out = _run(capsys, ["doctor", "--json"])

    report = json.loads(out)
    assert code == 0
    assert "torch" in report["training_stack"]
    assert report["ok"] is True


def test_a_bundle_carrying_torch_is_a_failure(capsys, monkeypatch) -> None:
    """The definitive check, and the reason it is asked from inside the app
    rather than by walking the artefact: torch is mostly compiled and shows
    up as files, but ``pytorch_lightning`` is pure Python and would live
    entirely inside the bundle's archive, invisible to any file listing.
    ``find_spec`` sees both.
    """
    import json

    from hairguard import diagnostics

    monkeypatch.setattr(diagnostics, "is_frozen", lambda: True)
    monkeypatch.setattr(cli, "_probe_training_stack", lambda: ["pytorch_lightning"])

    code, out = _run(capsys, ["doctor", "--json"])

    report = json.loads(out)
    assert code != 0
    assert any("pytorch_lightning" in problem for problem in report["problems"])
    assert any("2.8" in problem for problem in report["problems"])


def test_a_clean_bundle_passes(capsys, monkeypatch) -> None:
    import json

    from hairguard import diagnostics

    monkeypatch.setattr(diagnostics, "is_frozen", lambda: True)
    monkeypatch.setattr(cli, "_probe_training_stack", lambda: [])

    code, out = _run(capsys, ["doctor", "--json"])

    assert code == 0
    assert json.loads(out)["training_stack"] == []


def test_the_probe_finds_what_is_importable() -> None:
    """Exercised for real rather than stubbed, so the list of names is
    checked against something."""
    found = cli._probe_training_stack()

    assert "torch" in found
    assert all(name in cli._TRAINING_ONLY for name in found)


# ---------------------------------------------------------------------------
# What it must not do
# ---------------------------------------------------------------------------


def test_doctor_never_opens_the_camera(capsys, monkeypatch) -> None:
    """It is most useful while the app is running, which is exactly when the
    camera is held — and a second open on Windows succeeds and returns black
    frames rather than failing."""
    import cv2

    def forbidden(*args, **kwargs):
        raise AssertionError("doctor opened a capture device")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(
        "hairguard.camera.find_working_camera",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("doctor probed cameras")),
    )

    code, out = _run(capsys)

    assert code == 0
    assert "Камера" in out


def test_doctor_does_not_touch_the_pid_file(capsys, monkeypatch, tmp_path) -> None:
    """Removing it has caused a real incident here."""
    from hairguard.pid import PidFile

    path = tmp_path / "hairguard.pid"
    path.write_text("99999999", encoding="utf-8")
    monkeypatch.setattr(cli, "_pid_file", lambda: PidFile(path))

    _run(capsys)

    assert path.exists()


# ---------------------------------------------------------------------------
# When something is missing
# ---------------------------------------------------------------------------


def test_doctor_fails_when_the_feature_extractor_is_gone(capsys, monkeypatch) -> None:
    """Which is what a bundle that excluded hairlab would look like: the app
    starts, says one line, and detects half as much."""
    monkeypatch.setattr(
        cli, "_probe_features", lambda: (False, "No module named 'hairlab.features'")
    )

    code, out = _run(capsys)

    assert code != 0
    assert "hairlab.features" in out
    assert "правил" in out, "it did not say what the consequence is"


def test_doctor_fails_when_onnxruntime_is_missing(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        cli, "_probe_onnxruntime", lambda: (False, "No module named 'onnxruntime'")
    )

    code, out = _run(capsys)

    assert code != 0
    assert "onnxruntime" in out


def test_doctor_fails_when_the_model_cannot_be_loaded(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "hairguard.twistnet.load_configured",
        lambda config, directory=None: (None, "правило — модель недоступна: нет файла"),
    )

    code, out = _run(capsys)

    assert code != 0
    assert "недоступна" in out


def test_a_deliberately_disabled_model_is_not_a_failure(capsys, monkeypatch) -> None:
    """``detection.model.enabled: false`` is the documented way back to the
    old rule if the model misbehaves. Reporting that as broken would be
    wrong, and would make this command useless for the person who did it."""
    monkeypatch.setattr(
        "hairguard.twistnet.load_configured",
        lambda config, directory=None: (None, "правило (модель отключена в настройках)"),
    )

    code, out = _run(capsys)

    assert code == 0
    assert "отключена" in out


def test_a_missing_landmark_model_is_reported_but_survivable(
    capsys, monkeypatch, tmp_path
) -> None:
    """They download at runtime. Worth saying, not worth failing over."""
    from hairguard import assets

    monkeypatch.setattr(assets, "user_dir", lambda: tmp_path / "empty")
    monkeypatch.setattr(assets, "bundled_dir", lambda: tmp_path / "also-empty")

    code, out = _run(capsys)

    assert code == 0
    assert "скачается" in out or "скачаются" in out


def test_problems_are_summarised_at_the_end(capsys, monkeypatch) -> None:
    """So that a user can copy one line into a message rather than a screen."""
    monkeypatch.setattr(cli, "_probe_features", lambda: (False, "нет модуля"))
    monkeypatch.setattr(cli, "_probe_onnxruntime", lambda: (False, "нет модуля"))

    code, out = _run(capsys)

    assert code != 0
    tail = out[out.rindex("Проблем"):]
    assert "hairlab.features" in tail
    assert "onnxruntime" in tail


def test_doctor_survives_a_broken_config(capsys, monkeypatch) -> None:
    """A doctor that cannot run when something is wrong is no use at all."""

    def explode():
        raise RuntimeError("config.yaml не читается")

    monkeypatch.setattr(cli, "_load_config", explode)

    code, out = _run(capsys)

    assert code != 0
    assert "config.yaml не читается" in out
    # And it still answered the parts that do not need a config.
    assert "hairlab.features" in out


@pytest.mark.parametrize("flag", [[], ["--json"]])
def test_doctor_accepts_a_machine_readable_form(capsys, flag) -> None:
    """The build runs this against the finished bundle and checks the
    detector field, which is more robust against a prose change than
    grepping for a Russian word."""
    code, out = _run(capsys, ["doctor", *flag])

    assert code == 0
    if flag:
        import json

        report = json.loads(out)
        assert report["detector"]
        assert report["ok"] is True
        assert report["frozen"] is False
        assert report["feature_dim"]


def test_the_machine_readable_form_is_pure_ascii(capsys) -> None:
    """Every message in this report is Russian, and a Windows console gives
    a child process a single-byte code page. The build's first attempt at
    reading this back decoded cp1251 bytes as UTF-8 and concluded the bundle
    had picked the wrong detector. Escapes survive any code page."""
    _, out = _run(capsys, ["doctor", "--json"])

    out.encode("ascii")  # raises if anything slipped through
    assert "\\u" in out, "there should be escaped Cyrillic in there"


def test_the_json_form_reports_problems_with_a_failing_exit(capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_probe_onnxruntime", lambda: (False, "нет модуля"))

    code, out = _run(capsys, ["doctor", "--json"])

    import json

    report = json.loads(out)
    assert code != 0
    assert report["ok"] is False
    assert report["onnxruntime"] is None
    assert any("onnxruntime" in problem for problem in report["problems"])
