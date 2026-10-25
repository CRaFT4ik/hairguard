"""Tests for what the diagnostic window actually draws.

Both of the things asserted here were found by rendering a frame and looking
at it, which is the only way either of them could have been found.

``cv2.putText`` draws with Hershey vector fonts, which contain no Cyrillic:
every Russian character it is given comes out as ``?``. The first version of
this window captioned itself in Russian like the rest of the app, and the
result was a status line reading ``???????? ??????``. The panel it sits
under has always been in English for this reason, and now so is everything
else the window writes.

The panel is also a fixed 150 pixels tall. On a 1280-wide camera frame that
is a sixth of the height; on the 256-pixel clips the dataset is made of it
covers the picture and the two columns of text overlap each other. Small
frames are enlarged before anything is drawn on them.
"""
from __future__ import annotations

import numpy as np
import pytest

from hairguard import debugviewer
from hairguard.analyzer import AnalysisResult
from hairguard.debugwire import Snapshot
from hairguard.detector import Landmark, PoseResult
from hairguard.face_detector import FOREHEAD_INDICES, FaceLandmark, FaceResult
from hairguard.hand_detector import HandLandmark, HandResult


def make_snapshot(width=960, height=540, **overrides) -> Snapshot:
    rng = np.random.default_rng(3)
    pose = PoseResult(
        landmarks={
            name: Landmark(x=x, y=y, confidence=0.9)
            for name, (x, y) in {
                "nose": (0.50, 0.38),
                "left_ear": (0.42, 0.40),
                "right_ear": (0.58, 0.40),
                "left_shoulder": (0.35, 0.72),
                "right_shoulder": (0.65, 0.72),
            }.items()
        },
        raw_landmarks=None,
        detected=True,
    )
    hand = HandResult(
        landmarks={
            idx: HandLandmark(x=0.45 + idx * 0.004, y=0.30 + idx * 0.003, z=-0.01 * idx)
            for idx in range(21)
        },
        handedness="Right",
        score=0.9,
    )
    face = FaceResult(
        landmarks={
            idx: FaceLandmark(x=0.38 + 0.02 * position, y=0.30)
            for position, idx in enumerate(FOREHEAD_INDICES)
        },
        detected=True,
    )
    fields = dict(
        frame=rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
        pose=pose,
        hands=[hand],
        face=face,
        result=AnalysisResult(
            should_alert=True,
            hand_near_head=True,
            state="ALERTING",
            episode_started=False,
            episode_ended=False,
            episode_duration=0.0,
            fingertips_in_zone=True,
            consecutive_frames=5,
            model_score=0.979,
            detector="model",
        ),
        fps=6.8,
        combined_threshold=0.55,
        hair_zone_offset=0.4,
        model_threshold=0.966,
        detector_status="модель twistnet (порог 0.966)",
        tick=17,
    )
    fields.update(overrides)
    return Snapshot(**fields)


# ---------------------------------------------------------------------------
# Text the font can actually draw
# ---------------------------------------------------------------------------


def test_everything_the_window_writes_is_drawable():
    """``cv2.putText`` uses Hershey fonts, which have no Cyrillic glyphs: any
    Russian it is handed renders as a row of question marks. Found by
    rendering a frame and looking at it."""
    lines = debugviewer._footer_lines(make_snapshot(), age=0.1, received=42)

    assert lines
    for line in lines:
        line.encode("ascii")  # raises on anything the font cannot draw


def test_the_waiting_message_is_drawable_too():
    for line in debugviewer._footer_lines(None, age=None, received=0):
        line.encode("ascii")


def test_the_stale_warning_is_drawable_too():
    lines = debugviewer._footer_lines(make_snapshot(), age=30.0, received=42)

    for line in lines:
        line.encode("ascii")


def test_the_footer_names_the_detector_that_is_deciding():
    """Which of the two is answering is the first thing to establish when
    the app is behaving oddly: they differ by thirty points of recall."""
    lines = debugviewer._footer_lines(make_snapshot(), age=0.1, received=42)

    joined = " ".join(lines).lower()
    assert "model" in joined
    assert "0.97" in joined  # the threshold it is being compared against


def test_the_footer_says_rule_when_the_rule_is_deciding():
    snapshot = make_snapshot(model_threshold=None)
    snapshot.result.detector = "rule"

    joined = " ".join(
        debugviewer._footer_lines(snapshot, age=0.1, received=0)
    ).lower()

    assert "rule" in joined
    assert "model" not in joined


def test_a_stalled_feed_is_called_out_rather_than_shown_as_a_fresh_frame():
    """The app pauses, or loses the camera, and the window keeps displaying
    the last frame it got. Without this it reads as a live picture of a
    person sitting perfectly still."""
    lines = debugviewer._footer_lines(make_snapshot(), age=30.0, received=42)

    joined = " ".join(lines).lower()
    assert "no frames" in joined or "stalled" in joined
    assert "30" in joined


# ---------------------------------------------------------------------------
# Fitting the picture
# ---------------------------------------------------------------------------


def test_a_small_frame_is_enlarged_so_the_panel_fits():
    """The status panel is a fixed 150 px tall and two columns wide. The
    dataset's clips are 256x256, where it covers the picture and the columns
    write over each other."""
    rendered = debugviewer.render(make_snapshot(width=256, height=256))

    assert rendered.shape[1] >= debugviewer.MIN_DISPLAY_WIDTH
    assert rendered.shape[0] > debugviewer.PANEL_HEIGHT * 2


def test_enlarging_keeps_the_aspect_ratio():
    rendered = debugviewer.render(make_snapshot(width=320, height=240))

    assert rendered.shape[1] / rendered.shape[0] == pytest.approx(320 / 240, abs=0.02)


def test_a_normal_camera_frame_is_left_alone():
    rendered = debugviewer.render(make_snapshot(width=960, height=540))

    assert rendered.shape[:2] == (540, 960)


# ---------------------------------------------------------------------------
# Panel layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    ["State:", "Alert:", "Hand near head:", "Fingertips in zone:", "Persist:",
     "FPS:", "Score(p*2+s):", "Rotation:", "Model(>0.97):"],
)
def test_a_panel_value_is_placed_clear_of_its_label(label) -> None:
    """The offsets this replaces were hand-tuned pixel counts, and the longer
    labels had drifted into their own values: the panel read
    ``Hand near hea[YES]`` and ``Persist:33`` for a count of 3."""
    from hairguard import debugdraw

    image = np.zeros((200, 900, 3), dtype=np.uint8)

    value_x = debugdraw.label_value(
        image, label, "YES", (10, 50), debugdraw.COLOR_WHITE
    )

    assert value_x >= 10 + debugdraw.text_width(label)


def test_the_persist_count_is_drawn_once() -> None:
    """It used to be written into the label *and* over the top of it, so a
    persistence of 3 read as 33."""
    import inspect

    from hairguard import debugdraw

    source = inspect.getsource(debugdraw.draw_status_panel)

    assert "Persist: {" not in source


# ---------------------------------------------------------------------------
# Rendering at all
# ---------------------------------------------------------------------------


def test_rendering_draws_something_over_the_frame():
    snapshot = make_snapshot()
    before = snapshot.frame.copy()

    rendered = debugviewer.render(snapshot)

    assert not np.array_equal(rendered[: debugviewer.PANEL_HEIGHT], before[: debugviewer.PANEL_HEIGHT])


def test_rendering_does_not_scribble_on_the_frame_it_was_given():
    """The app hands the viewer process its own frames; in the degenerate
    case where those are ever shared, drawing on them would corrupt the
    detector's input."""
    snapshot = make_snapshot()
    original = snapshot.frame.copy()

    debugviewer.render(snapshot)

    assert np.array_equal(snapshot.frame, original)


def test_rendering_survives_nothing_being_detected():
    """The most informative frame there is: it is what the user sees when
    the detector is not seeing them."""
    snapshot = make_snapshot(
        pose=PoseResult(landmarks={}, raw_landmarks=None, detected=False),
        hands=[],
        face=FaceResult(landmarks={}, detected=False),
    )
    snapshot.result.state = "IDLE"
    snapshot.result.should_alert = False
    snapshot.result.model_score = None

    rendered = debugviewer.render(snapshot)

    assert rendered.shape[2] == 3
