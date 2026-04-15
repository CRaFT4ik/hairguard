"""
HairGuard Debug View
====================
Live camera feed with overlaid skeleton landmarks, detection zones, and state info.

Run with:
    python -m hairguard.debugview
    python -m hairguard.debugview --camera 1

Press 'q' or ESC to quit.
Press 'c' to cycle through cameras.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from hairguard.analyzer import GestureAnalyzer, AnalysisResult
from hairguard.camera import CameraManager
from hairguard.config import Config
from hairguard.detector import PoseDetector, PoseResult, Landmark

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

# Panel colors by state
_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "IDLE":      (30, 100, 30),
    "TRIGGERED": (0, 130, 140),
    "ALERTING":  (30, 30, 140),
    "COOLDOWN":  (110, 60, 20),
}

_STATE_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "IDLE":      COLOR_GREEN,
    "TRIGGERED": COLOR_YELLOW,
    "ALERTING":  COLOR_RED,
    "COOLDOWN":  COLOR_BLUE,
}

# Landmark display info: (label, role)
_LANDMARK_META: dict[str, tuple[str, str]] = {
    "left_wrist":    ("L.Wrist",    "wrist"),
    "right_wrist":   ("R.Wrist",    "wrist"),
    "left_index":    ("L.Index",    "finger"),
    "right_index":   ("R.Index",    "finger"),
    "left_pinky":    ("L.Pinky",    "finger"),
    "right_pinky":   ("R.Pinky",    "finger"),
    "left_ear":      ("L.Ear",      "ear"),
    "right_ear":     ("R.Ear",      "ear"),
    "left_shoulder": ("L.Shoulder", "shoulder"),
    "right_shoulder":("R.Shoulder", "shoulder"),
}

# Font settings
_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM   = 0.45
_FONT_MD   = 0.55
_FONT_LG   = 0.70
_THICKNESS = 1

PANEL_HEIGHT = 130  # pixels reserved at top for status panel


# ---------------------------------------------------------------------------
# Helper: pixel coords from normalized landmark
# ---------------------------------------------------------------------------

def _px(lm: Landmark, w: int, h: int) -> tuple[int, int]:
    return (int(lm.x * w), int(lm.y * h))


def _euclidean_px(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_text_with_shadow(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw text with a dark shadow for readability on any background."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), _FONT, font_scale, COLOR_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)


def _draw_landmark_dot(
    img: np.ndarray,
    pt: tuple[int, int],
    label: str,
    role: str,
    wrist_color: tuple[int, int, int],
    confidence: float,
) -> None:
    """Draw a labeled circle for one landmark."""
    if role == "finger":
        color = wrist_color
        radius = 6
    elif role == "wrist":
        color = wrist_color
        radius = 8
    elif role == "ear":
        color = COLOR_ORANGE
        radius = 7
    else:
        color = COLOR_GRAY
        radius = 6

    cv2.circle(img, pt, radius, COLOR_BLACK, -1)
    cv2.circle(img, pt, radius, color, 2)

    conf_text = f"{confidence:.2f}"
    lx = pt[0] + radius + 4
    ly = pt[1] + 4
    _draw_text_with_shadow(img, f"{label} ({conf_text})", (lx, ly), _FONT_SM, color)


def _draw_ear_zone(
    img: np.ndarray,
    ear_pt: tuple[int, int],
    radius_px: int,
) -> None:
    """Draw a dashed-style detection-threshold circle around an ear."""
    # OpenCV has no built-in dashed circle; approximate with small arc segments.
    n_segments = 20
    for i in range(n_segments):
        if i % 2 == 0:  # draw every other segment to simulate dashing
            angle_start = (360 / n_segments) * i
            angle_end   = angle_start + (360 / n_segments)
            cv2.ellipse(
                img,
                ear_pt,
                (radius_px, radius_px),
                0,
                angle_start,
                angle_end,
                COLOR_ORANGE,
                1,
                cv2.LINE_AA,
            )


def _draw_shoulder_line(
    img: np.ndarray,
    left_pt: tuple[int, int],
    right_pt: tuple[int, int],
) -> None:
    """Draw the shoulder baseline used for distance normalization."""
    cv2.line(img, left_pt, right_pt, COLOR_GRAY, 1, cv2.LINE_AA)
    mid = ((left_pt[0] + right_pt[0]) // 2, (left_pt[1] + right_pt[1]) // 2)
    _draw_text_with_shadow(img, "shoulder baseline", (mid[0] - 55, mid[1] - 6), _FONT_SM, COLOR_GRAY)


def _draw_wrist_ear_lines(
    img: np.ndarray,
    wrists: dict[str, tuple[int, int]],
    ears: dict[str, tuple[int, int]],
    lm: dict[str, Landmark],
    shoulder_dist_px: Optional[float],
    distance_threshold: float,
    min_confidence: float,
) -> dict[str, float]:
    """
    Draw lines from wrists to their nearest ear when wrist is near head.
    Returns a dict of "wrist_name -> min_ratio" for on-frame labeling.
    """
    ratios: dict[str, float] = {}

    for wname, wpt in wrists.items():
        wlm = lm.get(wname)
        if wlm is None or wlm.confidence < min_confidence:
            continue

        best_ratio: Optional[float] = None
        best_ept: Optional[tuple[int, int]] = None
        best_ename: Optional[str] = None

        for ename, ept in ears.items():
            elm = lm.get(ename)
            if elm is None or elm.confidence < min_confidence:
                continue

            if shoulder_dist_px and shoulder_dist_px > 1e-6:
                # Normalized ratio (same logic as analyzer)
                wlm_n = lm[wname]
                elm_n = lm[ename]
                norm_dist = math.sqrt(
                    (wlm_n.x - elm_n.x) ** 2 + (wlm_n.y - elm_n.y) ** 2
                )
                # Shoulder dist is in normalized space too
                left_sh = lm.get("left_shoulder")
                right_sh = lm.get("right_shoulder")
                if left_sh and right_sh:
                    sh_norm = math.sqrt(
                        (left_sh.x - right_sh.x) ** 2 + (left_sh.y - right_sh.y) ** 2
                    )
                    ratio = norm_dist / sh_norm if sh_norm > 1e-6 else 999.0
                else:
                    ratio = 999.0
            else:
                ratio = _euclidean_px(wpt, ept) / (shoulder_dist_px or 1.0)

            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_ept = ept
                best_ename = ename

        if best_ratio is not None and best_ept is not None:
            ratios[wname] = best_ratio
            if best_ratio < distance_threshold:
                # Hand near head — draw line and ratio label
                color = COLOR_RED if best_ratio < distance_threshold * 0.75 else COLOR_YELLOW
                cv2.line(img, wpt, best_ept, color, 1, cv2.LINE_AA)
                mid = ((wpt[0] + best_ept[0]) // 2, (wpt[1] + best_ept[1]) // 2)
                ratio_text = f"{best_ratio:.2f}"
                _draw_text_with_shadow(img, ratio_text, (mid[0] + 4, mid[1]), _FONT_MD, color)
            else:
                # Still show ratio but in neutral color
                mid = ((wpt[0] + best_ept[0]) // 2 if best_ept else wpt[0],
                       (wpt[1] + best_ept[1]) // 2 if best_ept else wpt[1])
                ratio_text = f"{best_ratio:.2f}"
                _draw_text_with_shadow(img, ratio_text, (wpt[0] + 10, wpt[1] - 10), _FONT_SM, COLOR_GRAY)

    return ratios


# ---------------------------------------------------------------------------
# Status panel
# ---------------------------------------------------------------------------

def _draw_status_panel(
    img: np.ndarray,
    state: str,
    should_alert: bool,
    hand_near_head: bool,
    min_ratio: Optional[float],
    fps: float,
    lm: dict[str, Landmark],
) -> None:
    """Render the status panel at the top of the frame."""
    h, w = img.shape[:2]
    panel_color = _STATE_COLORS.get(state, (40, 40, 40))

    # Semi-transparent panel background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, PANEL_HEIGHT), panel_color, -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

    # Divider line
    cv2.line(img, (0, PANEL_HEIGHT), (w, PANEL_HEIGHT), COLOR_WHITE, 1)

    state_color = _STATE_LABEL_COLORS.get(state, COLOR_WHITE)
    col1_x = 10
    col2_x = w // 2 + 10
    line_h = 22

    # --- Column 1 ---
    y = 22
    cv2.putText(img, "HairGuard Debug View", (col1_x, y), _FONT, _FONT_LG, COLOR_WHITE, 1, cv2.LINE_AA)

    y += line_h + 4
    cv2.putText(img, "State:", (col1_x, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, state, (col1_x + 55, y), _FONT, _FONT_MD, state_color, 2, cv2.LINE_AA)

    y += line_h
    alert_text  = "YES" if should_alert else "NO"
    alert_color = COLOR_RED if should_alert else COLOR_GREEN
    cv2.putText(img, "Alert:", (col1_x, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, alert_text, (col1_x + 55, y), _FONT, _FONT_MD, alert_color, 2, cv2.LINE_AA)

    y += line_h
    hand_text  = "YES" if hand_near_head else "NO"
    hand_color = COLOR_YELLOW if hand_near_head else COLOR_WHITE
    cv2.putText(img, "Hand near head:", (col1_x, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, hand_text, (col1_x + 125, y), _FONT, _FONT_MD, hand_color, 1, cv2.LINE_AA)

    y += line_h
    ratio_text  = f"{min_ratio:.3f}" if min_ratio is not None else "N/A"
    ratio_color = COLOR_RED if (min_ratio is not None and min_ratio < 0.6) else COLOR_WHITE
    cv2.putText(img, "Min ratio:", (col1_x, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, ratio_text, (col1_x + 80, y), _FONT, _FONT_MD, ratio_color, 1, cv2.LINE_AA)

    # --- Column 2 ---
    y = 22 + line_h + 4
    cv2.putText(img, f"FPS: {fps:.1f}", (col2_x, y), _FONT, _FONT_MD, COLOR_CYAN, 1, cv2.LINE_AA)

    y += line_h
    cv2.putText(img, "Landmark confidence:", (col2_x, y), _FONT, _FONT_SM, COLOR_GRAY, 1, cv2.LINE_AA)

    conf_names = [
        ("L.Idx",    "left_index"),
        ("R.Idx",    "right_index"),
        ("L.Pnk",    "left_pinky"),
        ("R.Pnk",    "right_pinky"),
        ("L.Ear",    "left_ear"),
        ("R.Ear",    "right_ear"),
    ]
    x_off = col2_x
    for i, (label, key) in enumerate(conf_names):
        conf = lm[key].confidence if key in lm else 0.0
        col = COLOR_GREEN if conf >= 0.5 else COLOR_RED
        col_x = col2_x + (i % 3) * 95
        row_y = y + line_h + (i // 3) * line_h
        cv2.putText(img, f"{label}:{conf:.2f}", (col_x, row_y), _FONT, _FONT_SM, col, 1, cv2.LINE_AA)

    # Key hints at bottom-right
    hint = "Q/ESC: quit   C: cycle camera"
    (tw, _), _ = cv2.getTextSize(hint, _FONT, _FONT_SM, 1)
    cv2.putText(img, hint, (w - tw - 8, PANEL_HEIGHT - 6), _FONT, _FONT_SM, COLOR_GRAY, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Frame overlay — main drawing pass
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    pose: PoseResult,
    result: AnalysisResult,
    fps: float,
    config: Config,
) -> np.ndarray:
    """Composite all debug overlays onto *a copy of* frame and return it."""
    img = frame.copy()
    h, w = img.shape[:2]

    lm = pose.landmarks
    distance_threshold: float = config.get("detection", "distance_threshold")
    min_confidence: float     = config.get("detection", "min_confidence")

    # Determine wrist color based on current state
    state = result.state
    if state == "ALERTING":
        wrist_color = COLOR_RED
    elif state in ("TRIGGERED", "COOLDOWN"):
        wrist_color = COLOR_YELLOW
    else:
        wrist_color = COLOR_GREEN

    # Compute pixel positions for available landmarks
    pts: dict[str, tuple[int, int]] = {}
    for name, lmk in lm.items():
        pts[name] = _px(lmk, w, h)

    # ---- Shoulder line ----
    if "left_shoulder" in pts and "right_shoulder" in pts:
        _draw_shoulder_line(img, pts["left_shoulder"], pts["right_shoulder"])

    # ---- Shoulder distance (normalized space) for zone radius ----
    shoulder_dist_norm: Optional[float] = None
    shoulder_dist_px: Optional[float]   = None
    left_sh  = lm.get("left_shoulder")
    right_sh = lm.get("right_shoulder")
    if (
        left_sh is not None and right_sh is not None
        and left_sh.confidence >= min_confidence
        and right_sh.confidence >= min_confidence
    ):
        shoulder_dist_norm = math.sqrt(
            (left_sh.x - right_sh.x) ** 2 + (left_sh.y - right_sh.y) ** 2
        )
        # Pixel-space shoulder distance (for zone visualization)
        shoulder_dist_px = _euclidean_px(pts["left_shoulder"], pts["right_shoulder"])

    # ---- Ear detection zones ----
    for ear_name in ("left_ear", "right_ear"):
        if ear_name not in pts:
            continue
        ear_lm = lm.get(ear_name)
        if ear_lm is None or ear_lm.confidence < min_confidence:
            continue
        if shoulder_dist_px is not None:
            zone_radius = int(distance_threshold * shoulder_dist_px)
            _draw_ear_zone(img, pts[ear_name], zone_radius)

    # ---- Hand-ear connection lines + ratio labels ----
    # Prefer finger points, fall back to wrists (same logic as analyzer)
    finger_pts = {k: v for k, v in pts.items() if "index" in k or "pinky" in k}
    wrist_only_pts = {k: v for k, v in pts.items() if "wrist" in k}
    wrist_pts = finger_pts if finger_pts else wrist_only_pts
    ear_pts   = {k: v for k, v in pts.items() if "ear" in k}

    ratios = _draw_wrist_ear_lines(
        img,
        wrist_pts,
        ear_pts,
        lm,
        shoulder_dist_px,
        distance_threshold,
        min_confidence,
    )

    # ---- Landmark dots with labels ----
    for name, (label, role) in _LANDMARK_META.items():
        if name not in pts:
            continue
        lmk = lm.get(name)
        if lmk is None:
            continue
        _draw_landmark_dot(img, pts[name], label, role, wrist_color, lmk.confidence)

    # ---- Min ratio across all wrist-ear pairs ----
    min_ratio: Optional[float] = min(ratios.values()) if ratios else None

    # ---- Status panel ----
    _draw_status_panel(img, state, result.should_alert, result.hand_near_head, min_ratio, fps, lm)

    return img


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(camera_index: int = 0) -> None:
    """Open the debug view window. Blocks until 'q'/ESC is pressed."""
    config  = Config()
    default_camera = camera_index if camera_index is not None else config.get("camera")

    print("[debugview] Initializing pose detector...")
    detector = PoseDetector(min_confidence=config.get("detection", "min_confidence"))
    analyzer = GestureAnalyzer(config)

    available_cameras = CameraManager.list_cameras()
    camera_indices = [c["index"] for c in available_cameras] or [0]
    try:
        current_cam_slot = camera_indices.index(default_camera)
    except ValueError:
        current_cam_slot = 0

    def open_camera(idx: int) -> CameraManager:
        print(f"[debugview] Opening camera {idx}...")
        return CameraManager(camera_index=idx)

    cam = open_camera(camera_indices[current_cam_slot])

    cv2.namedWindow("HairGuard Debug View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HairGuard Debug View", 960, 600)

    fps_counter = 0
    fps_value   = 0.0
    fps_ts      = time.monotonic()

    print("[debugview] Running — press 'q' or ESC to quit, 'c' to cycle cameras.")

    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                print("[debugview] No frame from camera.")
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            pose   = detector.process(frame)
            result = analyzer.analyze(pose, h, w)

            # FPS tracking
            fps_counter += 1
            now = time.monotonic()
            elapsed = now - fps_ts
            if elapsed >= 1.0:
                fps_value   = fps_counter / elapsed
                fps_counter = 0
                fps_ts      = now

            display = draw_overlay(frame, pose, result, fps_value, config)
            cv2.imshow("HairGuard Debug View", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord('c'):
                # Cycle to next camera
                cam.release()
                current_cam_slot = (current_cam_slot + 1) % len(camera_indices)
                next_idx = camera_indices[current_cam_slot]
                try:
                    cam = open_camera(next_idx)
                except RuntimeError as exc:
                    print(f"[debugview] Could not open camera {next_idx}: {exc}")
                    # Try to stay on current one — reopen previous
                    current_cam_slot = (current_cam_slot - 1) % len(camera_indices)
                    cam = open_camera(camera_indices[current_cam_slot])

    finally:
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        print("[debugview] Closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(camera: Optional[int] = None) -> None:
    """Public entry point used by CLI and direct module execution."""
    from hairguard.config import Config as _Config

    cfg = _Config()
    cam_idx = camera if camera is not None else cfg.get("camera")
    run(camera_index=cam_idx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HairGuard Debug View")
    parser.add_argument("--camera", type=int, default=None, metavar="N",
                        help="Camera index to use (default: from config)")
    args = parser.parse_args()
    main(camera=args.camera)
