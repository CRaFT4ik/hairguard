"""hairguard.debugwire — one tick of detector state, as it travels down a pipe.

The diagnostic window runs in a separate process and contains no camera code
at all. That is deliberate and it is the whole design: the running app holds
the webcam, and on Windows a second open of a held camera does not fail, it
returns black frames. This project has already lost a day to exactly that —
an auto-detection routine reporting a live camera as blank because another
instance held it. A debug view that opened its own camera would therefore
appear to work and show nothing, which is worse than refusing. So the viewer
is given no way to ask a camera for anything; everything it draws arrives
here, from the detection tick that already decided.

Three consequences shape the format.

*Fidelity over resemblance.* The far side recomputes :class:`HairZone` from
the same landmarks rather than being sent a picture of it, so the ellipse it
draws is the ellipse the analyzer gated on. Only the face landmarks the zone
actually reads are sent — the forehead arc, twelve of four hundred and
seventy-eight — because the rest are neither drawn nor consulted and would be
most of the payload.

*JSON and JPEG, not pickle.* Pickle over a pipe is code execution, and the
child is spawned from a user-visible menu entry in an app that watches a
webcam. A text header a person can read in a hex dump is also worth
something when this mechanism is itself the thing misbehaving.

*Every read can end.* The app exits, the user kills it, a write lands half
finished: each of those arrives as a short read, and a short read is the end
of the feed rather than a stall. The viewer's loop must be able to notice
the app is gone, because the app is under no obligation to say goodbye.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Optional

import numpy as np

from hairguard.analyzer import AnalysisResult
from hairguard.detector import Landmark, PoseResult
from hairguard.face_detector import FOREHEAD_INDICES, FaceLandmark, FaceResult
from hairguard.hand_detector import HandLandmark, HandResult

#: Bumped whenever the header's meaning changes. Two builds of the app can
#: sit on one machine; drawing one version's numbers in another version's
#: boxes would be a wrong answer presented confidently, which is the one
#: thing a diagnostic tool must never do.
PROTOCOL = 1

#: A length prefix is an allocation request from the other end of a pipe, so
#: it is bounded. Generous against a 960-wide JPEG, which runs well under
#: 200 KB.
MAX_MESSAGE_BYTES = 8 * 1024 * 1024

_LENGTH = struct.Struct(">I")

#: Width the frame is shrunk to before encoding. The window is opened at
#: 960 px and the landmarks are normalised, so the extra pixels of a 720p
#: capture would cost encode time and pipe bandwidth to draw the same
#: picture.
DEFAULT_MAX_WIDTH = 960

#: JPEG quality. High enough to see a strand of hair between two fingers,
#: low enough that a frame fits comfortably in a pipe buffer.
DEFAULT_QUALITY = 72


@dataclass
class Snapshot:
    """Everything the diagnostic window draws for one detection tick."""

    frame: np.ndarray
    pose: PoseResult
    hands: list[HandResult]
    face: FaceResult
    result: AnalysisResult
    fps: float
    combined_threshold: float
    hair_zone_offset: float
    model_threshold: Optional[float]
    detector_status: str
    tick: int


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def write_message(stream: BinaryIO, payload: bytes) -> None:
    """Write one length-prefixed message and flush it."""
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"сообщение слишком велико: {len(payload)} байт")
    stream.write(_LENGTH.pack(len(payload)))
    stream.write(payload)
    try:
        stream.flush()
    except AttributeError:
        pass


def _read_exactly(stream: BinaryIO, count: int) -> Optional[bytes]:
    """``count`` bytes, or None if the stream ended first.

    ``read`` on a pipe is allowed to return less than asked for, so this
    loops; a zero-length read is end of feed and anything short of ``count``
    at that point is a message that will never be completed.
    """
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(stream: BinaryIO) -> Optional[bytes]:
    """The next message, or None when the feed has ended.

    None covers every way the app can stop talking — clean exit, kill,
    a write cut in half — because the viewer's only correct response to all
    of them is to close its window.
    """
    header = _read_exactly(stream, _LENGTH.size)
    if header is None:
        return None
    (length,) = _LENGTH.unpack(header)
    if length > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"объявлена длина {length} байт — слишком много для кадра отладки"
        )
    return _read_exactly(stream, length)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _landmark_dict(landmark: Landmark) -> dict[str, float]:
    return {"x": landmark.x, "y": landmark.y, "c": landmark.confidence}


def _hand_dict(hand: HandResult) -> dict[str, Any]:
    return {
        "handedness": hand.handedness,
        "score": hand.score,
        # A flat list keyed by position: twenty-one landmarks per hand, twice
        # per tick, and the index is the MediaPipe landmark id.
        "landmarks": [
            [idx, lm.x, lm.y, lm.z, lm.presence]
            for idx, lm in sorted(hand.landmarks.items())
        ],
    }


def _result_dict(result: AnalysisResult) -> dict[str, Any]:
    return {
        "should_alert": result.should_alert,
        "hand_near_head": result.hand_near_head,
        "state": result.state,
        "episode_started": result.episode_started,
        "episode_ended": result.episode_ended,
        "episode_duration": result.episode_duration,
        "pinch_ratio_raw": result.pinch_ratio_raw,
        "pinch_ratio_smooth": result.pinch_ratio_smooth,
        "spread_index_pinky": result.spread_index_pinky,
        "hand_rotation": result.hand_rotation,
        "combined_score": result.combined_score,
        "fingertips_in_zone": result.fingertips_in_zone,
        "consecutive_frames": result.consecutive_frames,
        "model_score": result.model_score,
        "detector": result.detector,
        "in_zone_hand_index": result.in_zone_hand_index,
    }


def _shrink(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Narrow the frame to ``max_width``, keeping aspect. Never upscales."""
    width = int(frame.shape[1])
    if max_width <= 0 or width <= max_width:
        return frame
    import cv2

    scale = max_width / width
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def encode(
    snapshot: Snapshot,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    quality: int = DEFAULT_QUALITY,
) -> bytes:
    """Serialise one snapshot.

    Called on the feed's writer thread, never on the detection tick: the
    JPEG encode is milliseconds and the tick also drives the tkinter event
    loop, the overlay and the alert sound.
    """
    import cv2

    frame = _shrink(np.asarray(snapshot.frame), max_width)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("не удалось закодировать кадр в JPEG")
    image = buffer.tobytes()

    header = {
        "protocol": PROTOCOL,
        "image_bytes": len(image),
        "tick": snapshot.tick,
        "fps": snapshot.fps,
        "combined_threshold": snapshot.combined_threshold,
        "hair_zone_offset": snapshot.hair_zone_offset,
        "model_threshold": snapshot.model_threshold,
        "detector_status": snapshot.detector_status,
        "pose": {
            "detected": snapshot.pose.detected,
            "landmarks": {
                name: _landmark_dict(lm)
                for name, lm in snapshot.pose.landmarks.items()
            },
        },
        "hands": [_hand_dict(hand) for hand in snapshot.hands],
        "face": {
            "detected": snapshot.face.detected,
            # Only the forehead arc: it is what HairZone reads and what the
            # view draws. See the module docstring.
            "landmarks": [
                [idx, lm.x, lm.y]
                for idx in FOREHEAD_INDICES
                if (lm := snapshot.face.landmarks.get(idx)) is not None
            ],
        },
        "result": _result_dict(snapshot.result),
    }
    blob = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return _LENGTH.pack(len(blob)) + blob + image


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode(payload: bytes) -> Snapshot:
    """Rebuild a snapshot, or raise if the payload is not one."""
    if len(payload) < _LENGTH.size:
        raise ValueError("пустой кадр отладки")
    (header_len,) = _LENGTH.unpack(payload[: _LENGTH.size])
    start = _LENGTH.size
    end = start + header_len
    if header_len > MAX_MESSAGE_BYTES or end > len(payload):
        raise ValueError("заголовок кадра отладки не помещается в сообщение")
    try:
        header = json.loads(payload[start:end].decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"заголовок кадра отладки не читается: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("заголовок кадра отладки не похож на описание кадра")

    protocol = header.get("protocol")
    if protocol != PROTOCOL:
        raise ValueError(
            f"версия протокола {protocol!r} вместо {PROTOCOL} — "
            "приложение и окно диагностики из разных сборок"
        )

    image = payload[end:]
    expected = header.get("image_bytes")
    if not isinstance(expected, int) or len(image) != expected:
        raise ValueError(
            f"обещано {expected!r} байт изображения, получено {len(image)}"
        )

    import cv2

    frame = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("изображение кадра отладки не декодируется")

    pose_raw = header.get("pose") or {}
    pose = PoseResult(
        landmarks={
            name: Landmark(x=d["x"], y=d["y"], confidence=d["c"])
            for name, d in (pose_raw.get("landmarks") or {}).items()
        },
        # MediaPipe's own result object does not cross a process boundary and
        # nothing here draws from it.
        raw_landmarks=None,
        detected=bool(pose_raw.get("detected")),
    )

    hands = [
        HandResult(
            landmarks={
                int(idx): HandLandmark(x=x, y=y, z=z, presence=presence)
                for idx, x, y, z, presence in hand["landmarks"]
            },
            handedness=hand["handedness"],
            score=hand["score"],
        )
        for hand in (header.get("hands") or [])
    ]

    face_raw = header.get("face") or {}
    face = FaceResult(
        landmarks={
            int(idx): FaceLandmark(x=x, y=y)
            for idx, x, y in (face_raw.get("landmarks") or [])
        },
        detected=bool(face_raw.get("detected")),
    )

    result = AnalysisResult(**(header.get("result") or {}))

    return Snapshot(
        frame=frame,
        pose=pose,
        hands=hands,
        face=face,
        result=result,
        fps=float(header.get("fps") or 0.0),
        combined_threshold=float(header.get("combined_threshold") or 0.0),
        hair_zone_offset=float(header.get("hair_zone_offset") or 0.0),
        model_threshold=header.get("model_threshold"),
        detector_status=str(header.get("detector_status") or ""),
        tick=int(header.get("tick") or 0),
    )
