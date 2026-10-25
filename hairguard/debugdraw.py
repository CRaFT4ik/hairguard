"""hairguard.debugdraw — the diagnostic overlays, drawn onto a frame.

Split out of ``hairguard.debugview`` so that the window opened from the tray
can draw exactly what the standalone command draws without inheriting its
camera. ``debugview`` opens a ``CameraManager``; the tray viewer must not be
able to, because the running app already holds the device and a second open
returns black frames on Windows rather than failing. Keeping the drawing in
a module that imports no camera makes that a property of the import graph
rather than a thing to remember — see ``tests/test_debugfeed.py``, which
asserts it.

Pure rendering: every function here takes results and returns nothing but
marks on the image it was handed.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from hairguard.analyzer import AnalysisResult
from hairguard.detector import Landmark, PoseResult
from hairguard.face_detector import FaceResult
from hairguard.hand_detector import HandResult, INDEX_TIP, THUMB_TIP
from hairguard.head_zone import HairZone, HairZoneSmoother

# ---------------------------------------------------------------------------
# Color palette (BGR)
# ---------------------------------------------------------------------------

COLOR_GREEN  = (50, 205, 50)
COLOR_YELLOW = (0, 200, 220)
COLOR_RED    = (50, 50, 220)
COLOR_BLUE   = (200, 130, 50)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_GRAY   = (160, 160, 160)
COLOR_CYAN   = (200, 200, 0)
COLOR_ORANGE = (0, 140, 255)

# State -> background panel color
_STATE_BG: dict[str, tuple[int, int, int]] = {
    "IDLE":      (30, 100, 30),
    "TRIGGERED": (0, 130, 140),
    "ALERTING":  (30, 30, 140),
    "COOLDOWN":  (110, 60, 20),
}

# State -> text label color
_STATE_FG: dict[str, tuple[int, int, int]] = {
    "IDLE":      COLOR_GREEN,
    "TRIGGERED": COLOR_YELLOW,
    "ALERTING":  COLOR_RED,
    "COOLDOWN":  COLOR_ORANGE,
}

# Font settings
_FONT     = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM  = 0.45
_FONT_MD  = 0.55
_FONT_LG  = 0.70

PANEL_HEIGHT = 150  # pixels reserved at top for status panel

# Hand skeleton connections (MediaPipe hand topology)
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),        # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),         # index
    (0, 9),  (9, 10), (10, 11),(11, 12),        # middle
    (0, 13),(13, 14),(14, 15),(15, 16),          # ring
    (0, 17),(17, 18),(18, 19),(19, 20),          # pinky
    (5, 9), (9, 13),(13, 17),                   # palm
]


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    """Convert normalized (x, y) to pixel coordinates."""
    return (int(x * w), int(y * h))


def _lm_px(lm: Landmark, w: int, h: int) -> tuple[int, int]:
    return _px(lm.x, lm.y, w, h)


def _text_shadow(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), _FONT, scale, COLOR_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def text_width(text: str, scale: float = _FONT_MD, thickness: int = 1) -> int:
    """Pixel width of ``text`` in the panel font."""
    (width, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    return int(width)


#: Gap between a panel label and its value.
_LABEL_GAP = 8


def label_value(
    img: np.ndarray,
    label: str,
    value: str,
    pos: tuple[int, int],
    value_color: tuple[int, int, int],
    value_thickness: int = 1,
) -> int:
    """Draw ``label value`` and return where the value started.

    The value is placed after the measured width of the label rather than at
    a hand-tuned offset. The offsets this replaces were a pixel count per
    label — ``col1 + 125`` for "Hand near head:", ``col1 + 150`` for
    "Fingertips in zone:" — and the longer ones had drifted into their own
    values, so the panel read ``Hand near hea[YES]``. A diagnostic that is
    hard to read is most of the way to not being one.
    """
    x, y = pos
    cv2.putText(img, label, (x, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    value_x = x + text_width(label) + _LABEL_GAP
    cv2.putText(
        img, value, (value_x, y), _FONT, _FONT_MD, value_color, value_thickness,
        cv2.LINE_AA,
    )
    return value_x


# ---------------------------------------------------------------------------
# draw_hand_skeletons
# ---------------------------------------------------------------------------

def draw_hand_skeletons(
    img: np.ndarray,
    hands: list[HandResult],
    result: AnalysisResult,
    combined_threshold: float,
) -> None:
    """Draw all 21 landmarks + connections for every detected hand."""
    h, w = img.shape[:2]

    state = result.state
    if state == "ALERTING":
        base_color = COLOR_RED
    elif state in ("TRIGGERED", "COOLDOWN"):
        base_color = COLOR_YELLOW
    else:
        base_color = COLOR_GREEN

    for hand in hands:
        lms = hand.landmarks

        # --- Skeleton connections ---
        for a, b in HAND_CONNECTIONS:
            if a not in lms or b not in lms:
                continue
            pa = _px(lms[a].x, lms[a].y, w, h)
            pb = _px(lms[b].x, lms[b].y, w, h)
            cv2.line(img, pa, pb, base_color, 1, cv2.LINE_AA)

        # --- All 21 landmark dots ---
        for idx, lm in lms.items():
            pt = _px(lm.x, lm.y, w, h)
            if idx in (THUMB_TIP, INDEX_TIP):
                continue  # drawn separately below
            cv2.circle(img, pt, 3, base_color, -1, cv2.LINE_AA)

        # --- Fingertip highlights ---
        for tip_idx in (THUMB_TIP, INDEX_TIP):
            if tip_idx not in lms:
                continue
            lm = lms[tip_idx]
            pt = _px(lm.x, lm.y, w, h)

            if result.fingertips_in_zone and result.should_alert:
                tip_color = COLOR_RED
            elif result.fingertips_in_zone:
                tip_color = COLOR_YELLOW
            else:
                tip_color = COLOR_GREEN

            cv2.circle(img, pt, 8, COLOR_BLACK, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 8, tip_color, 2, cv2.LINE_AA)
            cv2.circle(img, pt, 4, tip_color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# draw_pinch_indicator
# ---------------------------------------------------------------------------

def draw_pinch_indicator(
    img: np.ndarray,
    hands: list[HandResult],
    combined_threshold: float,
) -> None:
    """Draw a line between thumb and index tip with the pinch_ratio label."""
    if not hands:
        return

    h, w = img.shape[:2]

    for hand in hands:
        lms = hand.landmarks
        if THUMB_TIP not in lms or INDEX_TIP not in lms:
            continue

        t = lms[THUMB_TIP]
        i = lms[INDEX_TIP]
        pt_thumb = _px(t.x, t.y, w, h)
        pt_index = _px(i.x, i.y, w, h)

        ratio = hand.pinch_ratio()
        line_color = COLOR_RED if ratio < 0.30 else COLOR_GREEN  # visual indicator only

        cv2.line(img, pt_thumb, pt_index, line_color, 1, cv2.LINE_AA)

        mid_x = (pt_thumb[0] + pt_index[0]) // 2
        mid_y = (pt_thumb[1] + pt_index[1]) // 2
        label = f"pr:{ratio:.2f}"
        _text_shadow(img, label, (mid_x + 4, mid_y - 4), _FONT_SM, line_color)


# ---------------------------------------------------------------------------
# draw_hair_zone
# ---------------------------------------------------------------------------

def draw_hair_zone(
    img: np.ndarray,
    face: FaceResult,
    pose: PoseResult,
    hair_zone_offset: float,
    smoother: Optional[HairZoneSmoother] = None,
) -> None:
    """Visualize the hair zone as a rotated ellipse + forehead polyline."""
    h, w = img.shape[:2]

    zone = HairZone(face, pose, hair_zone_offset)
    if smoother is not None:
        zone = smoother.smooth(zone)
    params = zone.ellipse_params

    cx, cy = params["center"]
    sw = params["semi_width"]
    sh = params["semi_height"]
    hh = params["head_horizontal"]
    uv = params["up_vec"]
    hc = params["head_center"]

    # --- Draw rotated ellipse ---
    # OpenCV ellipse needs: center in pixels, axes in pixels, angle in degrees
    center_px = (int(cx * w), int(cy * h))
    axes_px = (int(sw * w), int(sh * h))
    # Angle = angle of head_horizontal relative to screen horizontal
    angle_deg = math.degrees(math.atan2(hh[1], hh[0]))

    if axes_px[0] > 0 and axes_px[1] > 0:
        # Semi-transparent fill
        overlay = img.copy()
        cv2.ellipse(overlay, center_px, axes_px, angle_deg, 0, 360, (200, 200, 0), -1)
        cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)
        # Outline
        cv2.ellipse(img, center_px, axes_px, angle_deg, 0, 360, COLOR_ORANGE, 2, cv2.LINE_AA)

    # --- Hair center cross (same color as ellipse) ---
    cross_size = 8
    cv2.drawMarker(img, center_px, COLOR_ORANGE, cv2.MARKER_CROSS, cross_size, 2)

    # --- Forehead polyline (cyan) ---
    forehead_pts = zone.forehead_line
    if len(forehead_pts) >= 2:
        pixel_pts = [_px(x, y, w, h) for x, y in forehead_pts]
        pts_arr = np.array(pixel_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_arr], isClosed=False, color=COLOR_CYAN, thickness=2, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# draw_pose_landmarks
# ---------------------------------------------------------------------------

def draw_pose_landmarks(img: np.ndarray, pose: PoseResult) -> None:
    """Draw ears, nose, and shoulders as small labeled dots."""
    if not pose.detected:
        return

    h, w = img.shape[:2]
    lm = pose.landmarks

    _POSE_META = {
        "nose":           ("Nose",   COLOR_CYAN,   7),
        "left_ear":       ("L.Ear",  COLOR_ORANGE, 7),
        "right_ear":      ("R.Ear",  COLOR_ORANGE, 7),
        "left_shoulder":  ("L.Sh",   COLOR_GRAY,   6),
        "right_shoulder": ("R.Sh",   COLOR_GRAY,   6),
    }

    # Shoulder line baseline
    ls = lm.get("left_shoulder")
    rs = lm.get("right_shoulder")
    if ls is not None and rs is not None:
        lspt = _lm_px(ls, w, h)
        rspt = _lm_px(rs, w, h)
        cv2.line(img, lspt, rspt, COLOR_GRAY, 1, cv2.LINE_AA)
        mid = ((lspt[0] + rspt[0]) // 2, (lspt[1] + rspt[1]) // 2)
        _text_shadow(img, "shoulder baseline", (mid[0] - 55, mid[1] - 6), _FONT_SM, COLOR_GRAY)

    for name, (label, color, radius) in _POSE_META.items():
        landmark = lm.get(name)
        if landmark is None:
            continue
        pt = _lm_px(landmark, w, h)
        cv2.circle(img, pt, radius, COLOR_BLACK, -1)
        cv2.circle(img, pt, radius, color, 2)
        conf_txt = f"{label}({landmark.confidence:.2f})"
        _text_shadow(img, conf_txt, (pt[0] + radius + 3, pt[1] + 4), _FONT_SM, color)


# ---------------------------------------------------------------------------
# draw_status_panel
# ---------------------------------------------------------------------------

def draw_status_panel(
    img: np.ndarray,
    result: AnalysisResult,
    fps: float,
    combined_threshold: float,
    model_threshold: Optional[float] = None,
    hint: str = "Q/ESC: quit   C: cycle camera",
    title: str = "HairGuard Debug View  v2",
) -> None:
    """Render the status panel at the top of the frame."""
    h, w = img.shape[:2]
    state = result.state
    panel_bg = _STATE_BG.get(state, (40, 40, 40))
    state_fg = _STATE_FG.get(state, COLOR_WHITE)

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, PANEL_HEIGHT), panel_bg, -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
    cv2.line(img, (0, PANEL_HEIGHT), (w, PANEL_HEIGHT), COLOR_WHITE, 1)

    col1 = 10
    col2 = w // 2 + 10
    lh   = 22   # line height

    # ---- Title ----
    cv2.putText(img, title, (col1, 22), _FONT, _FONT_LG, COLOR_WHITE, 1, cv2.LINE_AA)

    # ---- Column 1 ----
    y = 22 + lh + 2

    # State
    label_value(img, "State:", state, (col1, y), state_fg, value_thickness=2)
    y += lh

    # Alert
    alert_txt   = "YES" if result.should_alert else "NO"
    alert_color = COLOR_RED if result.should_alert else COLOR_GREEN
    label_value(img, "Alert:", alert_txt, (col1, y), alert_color, value_thickness=2)
    y += lh

    # Hand near head
    hnh_txt   = "YES" if result.hand_near_head else "NO"
    hnh_color = COLOR_YELLOW if result.hand_near_head else COLOR_WHITE
    label_value(img, "Hand near head:", hnh_txt, (col1, y), hnh_color)
    y += lh

    # Fingertips in zone
    fiz_txt   = "YES" if result.fingertips_in_zone else "NO"
    fiz_color = COLOR_YELLOW if result.fingertips_in_zone else COLOR_WHITE
    label_value(img, "Fingertips in zone:", fiz_txt, (col1, y), fiz_color)
    y += lh

    # Consecutive frames
    cf_txt = str(result.consecutive_frames)
    cf_color = COLOR_RED if result.consecutive_frames >= 5 else COLOR_WHITE
    label_value(img, "Persist:", cf_txt, (col1, y), cf_color)

    # ---- Column 2 ----
    y = 22 + lh + 2

    # FPS
    label_value(img, "FPS:", f"{fps:.1f}", (col2, y), COLOR_CYAN)
    y += lh

    # Combined score (pinch*2 + spread)
    if result.combined_score is not None:
        cs_val = result.combined_score
        cs_color = COLOR_RED if cs_val < combined_threshold else COLOR_GREEN
        cs_txt = f"{cs_val:.3f}"
    else:
        cs_color = COLOR_GRAY
        cs_txt = "N/A"
    label_value(img, "Score(p*2+s):", cs_txt, (col2, y), cs_color)
    y += lh

    # Hand rotation
    if result.hand_rotation is not None:
        rot_val = result.hand_rotation
        rot_color = COLOR_RED if rot_val < 0.10 else COLOR_GREEN
        rot_txt = f"{rot_val:.3f}"
    else:
        rot_color = COLOR_GRAY
        rot_txt = "N/A"
    label_value(img, "Rotation:", rot_txt, (col2, y), rot_color)
    y += lh

    # Model score. Shown next to the rule's numbers rather than instead of
    # them: this panel is where the two detectors can be caught disagreeing.
    # "---" means the window is not full yet, which is itself worth seeing —
    # it is how long the model stays blind after a hand reaches the head.
    if result.detector == "model":
        if result.model_score is None:
            model_txt, model_color = "---", COLOR_GRAY
        else:
            fires = model_threshold is not None and result.model_score >= model_threshold
            model_txt = f"{result.model_score:.3f}"
            model_color = COLOR_RED if fires else COLOR_GREEN
        label = "Model:" if model_threshold is None else f"Model(>{model_threshold:.2f}):"
    else:
        model_txt, model_color, label = "rule", COLOR_GRAY, "Detector:"
    label_value(img, label, model_txt, (col2, y), model_color)
    y += lh

    # Key hints
    (tw, _), _ = cv2.getTextSize(hint, _FONT, _FONT_SM, 1)
    cv2.putText(img, hint, (w - tw - 8, PANEL_HEIGHT - 6), _FONT, _FONT_SM, COLOR_GRAY, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# draw_footer
# ---------------------------------------------------------------------------

def draw_footer(img: np.ndarray, lines: list[str]) -> None:
    """Write short status lines along the bottom edge, newest last.

    Used by the live viewer to say which detector is deciding and whether
    frames are still arriving — the two questions that distinguish "the
    detector is wrong" from "the detector stopped".
    """
    if not lines:
        return
    h, w = img.shape[:2]
    pad = 6
    line_h = 18
    top = h - pad - line_h * len(lines)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, top - pad), (w, h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    y = top + line_h - 5
    for line in lines:
        _text_shadow(img, line, (pad + 2, y), _FONT_SM, COLOR_WHITE)
        y += line_h
