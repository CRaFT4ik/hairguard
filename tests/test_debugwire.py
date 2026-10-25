"""Tests for the debug-view wire format.

The debug window runs in its own process and never touches a camera, so
everything it draws arrives over a pipe from the detection tick. This module
is that pipe's grammar: whatever the running detector decided has to survive
the trip unchanged, because the entire value of the window is that it shows
what the detector saw rather than an approximation of it.

The two properties worth guarding are therefore fidelity (the hair zone
recomputed on the far side must be the zone the analyzer gated on) and
robustness (a truncated or foreign message must be refused, not half-read).
"""
from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest

from hairguard import debugwire
from hairguard.analyzer import AnalysisResult
from hairguard.detector import Landmark, PoseResult
from hairguard.face_detector import FOREHEAD_INDICES, FaceLandmark, FaceResult
from hairguard.hand_detector import HandLandmark, HandResult
from hairguard.head_zone import HairZone


# ---------------------------------------------------------------------------
# Fixtures: a plausible frame of detection output
# ---------------------------------------------------------------------------


def make_frame(width: int = 320, height: int = 240) -> np.ndarray:
    """A noisy BGR frame — JPEG of a flat colour would hide size problems."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def make_pose(detected: bool = True) -> PoseResult:
    if not detected:
        return PoseResult(landmarks={}, raw_landmarks=None, detected=False)
    names = {
        "nose": (0.50, 0.38),
        "left_ear": (0.42, 0.40),
        "right_ear": (0.58, 0.40),
        "left_shoulder": (0.35, 0.72),
        "right_shoulder": (0.65, 0.72),
        "left_wrist": (0.30, 0.90),
        "right_wrist": (0.70, 0.90),
    }
    return PoseResult(
        landmarks={
            name: Landmark(x=x, y=y, confidence=0.93) for name, (x, y) in names.items()
        },
        raw_landmarks=None,
        detected=True,
    )


def make_hand(offset: float = 0.0, handedness: str = "Right") -> HandResult:
    landmarks = {
        idx: HandLandmark(
            x=0.45 + offset + idx * 0.004,
            y=0.30 + idx * 0.003,
            z=-0.01 * idx,
            presence=0.9,
        )
        for idx in range(21)
    }
    return HandResult(landmarks=landmarks, handedness=handedness, score=0.88)


def make_face(detected: bool = True) -> FaceResult:
    if not detected:
        return FaceResult(landmarks={}, detected=False)
    # A wide, slightly curved arc, plus landmarks the viewer has no use for.
    landmarks = {
        idx: FaceLandmark(x=0.38 + 0.02 * position, y=0.30 + 0.001 * position)
        for position, idx in enumerate(FOREHEAD_INDICES)
    }
    landmarks[1] = FaceLandmark(x=0.5, y=0.5)  # nose tip — never drawn
    landmarks[152] = FaceLandmark(x=0.5, y=0.7)  # chin — never drawn
    return FaceResult(landmarks=landmarks, detected=True)


def make_result(**overrides) -> AnalysisResult:
    fields = dict(
        should_alert=True,
        hand_near_head=True,
        state="ALERTING",
        episode_started=False,
        episode_ended=False,
        episode_duration=0.0,
        pinch_ratio_raw=0.21,
        pinch_ratio_smooth=0.24,
        spread_index_pinky=0.31,
        hand_rotation=0.07,
        combined_score=0.49,
        fingertips_in_zone=True,
        consecutive_frames=6,
        model_score=0.977,
        detector="model",
        in_zone_hand_index=0,
    )
    fields.update(overrides)
    return AnalysisResult(**fields)


def make_snapshot(**overrides) -> debugwire.Snapshot:
    fields = dict(
        frame=make_frame(),
        pose=make_pose(),
        hands=[make_hand(), make_hand(offset=0.1, handedness="Left")],
        face=make_face(),
        result=make_result(),
        fps=6.75,
        combined_threshold=0.55,
        hair_zone_offset=0.4,
        model_threshold=0.966,
        detector_status="модель twistnet (порог 0.966)",
        tick=1234,
    )
    fields.update(overrides)
    return debugwire.Snapshot(**fields)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_the_verdict():
    """The numbers the panel prints are the numbers the analyzer produced."""
    snapshot = make_snapshot()

    restored = debugwire.decode(debugwire.encode(snapshot))

    assert restored.result.state == "ALERTING"
    assert restored.result.should_alert is True
    assert restored.result.detector == "model"
    assert restored.result.model_score == pytest.approx(0.977)
    assert restored.result.combined_score == pytest.approx(0.49)
    assert restored.result.hand_rotation == pytest.approx(0.07)
    assert restored.result.consecutive_frames == 6
    assert restored.result.fingertips_in_zone is True
    assert restored.result.in_zone_hand_index == 0


def test_round_trip_preserves_the_scalars_the_panel_needs():
    snapshot = make_snapshot()

    restored = debugwire.decode(debugwire.encode(snapshot))

    assert restored.fps == pytest.approx(6.75)
    assert restored.combined_threshold == pytest.approx(0.55)
    assert restored.hair_zone_offset == pytest.approx(0.4)
    assert restored.model_threshold == pytest.approx(0.966)
    assert restored.detector_status == "модель twistnet (порог 0.966)"
    assert restored.tick == 1234


def test_round_trip_preserves_hand_landmarks_and_derived_geometry():
    """pinch_ratio and friends are methods, so the landmarks must be exact."""
    snapshot = make_snapshot()
    original = snapshot.hands[0]

    restored = debugwire.decode(debugwire.encode(snapshot)).hands[0]

    assert len(restored.landmarks) == 21
    assert restored.handedness == original.handedness
    assert restored.score == pytest.approx(original.score)
    for idx, landmark in original.landmarks.items():
        assert restored.landmarks[idx].x == pytest.approx(landmark.x)
        assert restored.landmarks[idx].y == pytest.approx(landmark.y)
        assert restored.landmarks[idx].z == pytest.approx(landmark.z)
        assert restored.landmarks[idx].presence == pytest.approx(landmark.presence)
    assert restored.pinch_ratio() == pytest.approx(original.pinch_ratio())
    assert restored.hand_scale() == pytest.approx(original.hand_scale())
    assert restored.is_gripping() == original.is_gripping()


def test_hand_landmark_keys_survive_as_integers():
    """JSON turns dict keys into strings; MediaPipe indices must come back ints."""
    restored = debugwire.decode(debugwire.encode(make_snapshot()))

    assert set(restored.hands[0].landmarks) == set(range(21))


def test_round_trip_reproduces_the_zone_the_analyzer_gated_on():
    """Fidelity, not resemblance: the far side recomputes the same ellipse."""
    snapshot = make_snapshot()
    expected = HairZone(snapshot.face, snapshot.pose, snapshot.hair_zone_offset)

    restored = debugwire.decode(debugwire.encode(snapshot))
    actual = HairZone(restored.face, restored.pose, restored.hair_zone_offset)

    assert actual.ellipse_params["center"] == pytest.approx(
        expected.ellipse_params["center"]
    )
    assert actual.ellipse_params["semi_width"] == pytest.approx(
        expected.ellipse_params["semi_width"]
    )
    assert actual.ellipse_params["semi_height"] == pytest.approx(
        expected.ellipse_params["semi_height"]
    )
    assert actual.ellipse_params["yaw_ratio"] == pytest.approx(
        expected.ellipse_params["yaw_ratio"]
    )
    assert actual.forehead_line == pytest.approx(expected.forehead_line)
    # And the gate itself agrees, which is the only thing a user is reading
    # the window to find out.
    for x in (0.40, 0.45, 0.50, 0.55, 0.60):
        for y in (0.20, 0.25, 0.30, 0.35):
            assert actual.contains(x, y) == expected.contains(x, y)


def test_only_the_face_landmarks_the_zone_uses_are_sent():
    """478 face landmarks per tick would be most of the payload and none of
    the picture: the zone reads the forehead arc and nothing else."""
    snapshot = make_snapshot()

    restored = debugwire.decode(debugwire.encode(snapshot))

    assert set(restored.face.landmarks) == set(FOREHEAD_INDICES)
    assert restored.face.forehead_line() == pytest.approx(
        snapshot.face.forehead_line()
    )


def test_round_trip_preserves_pose_landmark_confidences():
    """The zone drops low-confidence landmarks, so confidence is load-bearing."""
    pose = make_pose()
    pose.landmarks["left_shoulder"] = Landmark(x=0.35, y=0.72, confidence=0.11)
    snapshot = make_snapshot(pose=pose)

    restored = debugwire.decode(debugwire.encode(snapshot))

    assert restored.pose.landmarks["left_shoulder"].confidence == pytest.approx(0.11)
    assert restored.pose.detected is True


def test_frame_survives_as_an_image_of_the_same_shape():
    snapshot = make_snapshot(frame=make_frame(320, 240))

    restored = debugwire.decode(debugwire.encode(snapshot))

    assert restored.frame.shape == (240, 320, 3)
    assert restored.frame.dtype == np.uint8


def test_a_large_frame_is_shrunk_for_the_wire_but_keeps_its_aspect():
    """A 720p frame is 2.7 MB raw; the window it lands in is 960 px wide."""
    snapshot = make_snapshot(frame=make_frame(1280, 720))

    payload = debugwire.encode(snapshot, max_width=960)
    restored = debugwire.decode(payload)

    assert restored.frame.shape[1] == 960
    assert restored.frame.shape[0] == 540
    assert len(payload) < 300_000


def test_a_small_frame_is_not_upscaled():
    snapshot = make_snapshot(frame=make_frame(320, 240))

    restored = debugwire.decode(debugwire.encode(snapshot, max_width=960))

    assert restored.frame.shape[:2] == (240, 320)


# ---------------------------------------------------------------------------
# Nothing detected — the case the panel exists for
# ---------------------------------------------------------------------------


def test_round_trip_with_nothing_detected():
    """An empty frame of results is the most interesting thing to look at:
    it is what the user sees when the detector is not seeing them."""
    snapshot = make_snapshot(
        pose=make_pose(detected=False),
        hands=[],
        face=make_face(detected=False),
        result=make_result(
            should_alert=False,
            hand_near_head=False,
            state="IDLE",
            pinch_ratio_raw=None,
            pinch_ratio_smooth=None,
            spread_index_pinky=None,
            hand_rotation=None,
            combined_score=None,
            model_score=None,
            fingertips_in_zone=False,
            consecutive_frames=0,
            detector="rule",
            in_zone_hand_index=None,
        ),
        model_threshold=None,
    )

    restored = debugwire.decode(debugwire.encode(snapshot))

    assert restored.pose.detected is False
    assert restored.pose.landmarks == {}
    assert restored.hands == []
    assert restored.face.detected is False
    assert restored.result.state == "IDLE"
    assert restored.result.model_score is None
    assert restored.result.combined_score is None
    assert restored.model_threshold is None


def test_raw_landmarks_are_not_shipped():
    """MediaPipe's own result object is not serialisable and nothing on the
    far side draws from it. It must come back None rather than half-formed."""
    restored = debugwire.decode(debugwire.encode(make_snapshot()))

    assert restored.pose.raw_landmarks is None


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def test_write_then_read_returns_the_payload():
    stream = io.BytesIO()

    debugwire.write_message(stream, b"payload")
    stream.seek(0)

    assert debugwire.read_message(stream) == b"payload"


def test_several_messages_read_back_in_order():
    stream = io.BytesIO()
    for payload in (b"one", b"two", b"three"):
        debugwire.write_message(stream, payload)
    stream.seek(0)

    assert debugwire.read_message(stream) == b"one"
    assert debugwire.read_message(stream) == b"two"
    assert debugwire.read_message(stream) == b"three"
    assert debugwire.read_message(stream) is None


def test_a_closed_stream_reads_as_end_of_feed():
    """The app exiting closes the pipe; the viewer must shut down, not hang."""
    assert debugwire.read_message(io.BytesIO(b"")) is None


def test_a_truncated_message_reads_as_end_of_feed():
    """Killing the app mid-write must not leave the viewer blocked forever
    on the rest of a frame that is never coming."""
    stream = io.BytesIO()
    debugwire.write_message(stream, b"x" * 500)
    truncated = io.BytesIO(stream.getvalue()[:100])

    assert debugwire.read_message(truncated) is None


def test_a_truncated_length_prefix_reads_as_end_of_feed():
    assert debugwire.read_message(io.BytesIO(b"\x00\x01")) is None


def test_an_oversized_length_prefix_is_refused():
    """A length field is a memory allocation, so it is bounded."""
    stream = io.BytesIO(struct.pack(">I", debugwire.MAX_MESSAGE_BYTES + 1))

    with pytest.raises(ValueError, match="слишком"):
        debugwire.read_message(stream)


# ---------------------------------------------------------------------------
# Refusing what it cannot read
# ---------------------------------------------------------------------------


def test_a_foreign_payload_is_refused():
    with pytest.raises(ValueError):
        debugwire.decode(b"not a snapshot at all")


def test_a_payload_from_another_protocol_version_is_refused():
    """Two versions of the app on one machine must not draw each other's
    numbers in the wrong boxes."""
    payload = debugwire.encode(make_snapshot())
    header_len = struct.unpack(">I", payload[:4])[0]
    header = json.loads(payload[4 : 4 + header_len].decode("utf-8"))
    header["protocol"] = debugwire.PROTOCOL + 1
    forged = json.dumps(header).encode("utf-8")
    tampered = struct.pack(">I", len(forged)) + forged + payload[4 + header_len :]

    with pytest.raises(ValueError, match="версия"):
        debugwire.decode(tampered)


def test_a_payload_with_a_missing_image_is_refused():
    payload = debugwire.encode(make_snapshot())
    header_len = struct.unpack(">I", payload[:4])[0]

    with pytest.raises(ValueError):
        debugwire.decode(payload[: 4 + header_len])
