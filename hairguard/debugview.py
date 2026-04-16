"""
HairGuard Debug View  (v2)
==========================
Live camera feed with v2 detection pipeline overlays:
  - Hand skeleton (21 landmarks, color-coded)
  - Fingertip highlights (thumb + index)
  - Pinch line with distance label
  - Hair zone (forehead polyline, ear vertical bounds)
  - Pose landmarks (ears, nose, shoulders)
  - Status panel (state, alert, pinch ratio, zone, grip, FPS)

Run with:
    python -m hairguard.debugview
    python -m hairguard.debugview --camera 1

Controls: q / ESC to quit, c to cycle cameras.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from hairguard.analyzer import GestureAnalyzer, AnalysisResult
from hairguard.camera import CameraManager
from hairguard.config import Config
from hairguard.detector import PoseDetector, PoseResult, Landmark
# FaceDetector no longer used for detection (hair mask replaces it)
from hairguard.hair_segmenter import HairSegmenter, HairMask, NoseAnchoredHairZone
from hairguard.hand_detector import (
    HandDetector,
    HandResult,
    THUMB_TIP,
    INDEX_TIP,
)

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


# ---------------------------------------------------------------------------
# draw_hand_skeletons
# ---------------------------------------------------------------------------

def draw_hand_skeletons(
    img: np.ndarray,
    hands: list[HandResult],
    result: AnalysisResult,
    pinch_threshold: float,
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
    pinch_threshold: float,
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
        line_color = COLOR_RED if ratio < pinch_threshold else COLOR_GREEN

        cv2.line(img, pt_thumb, pt_index, line_color, 1, cv2.LINE_AA)

        mid_x = (pt_thumb[0] + pt_index[0]) // 2
        mid_y = (pt_thumb[1] + pt_index[1]) // 2
        label = f"pr:{ratio:.2f}"
        _text_shadow(img, label, (mid_x + 4, mid_y - 4), _FONT_SM, line_color)


# ---------------------------------------------------------------------------
# draw_hair_mask
# ---------------------------------------------------------------------------

def draw_hair_mask(img: np.ndarray, hair_mask: HairMask) -> None:
    """Draw a semi-transparent green overlay on hair pixels."""
    if not hair_mask.detected:
        return

    mask = hair_mask.get_mask()

    # Ensure mask is 2D
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        return

    # Resize mask to display frame size if needed
    h, w = img.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Green-tinted overlay on hair pixels
    hair_bool = mask.astype(bool)
    if not hair_bool.any():
        return

    green_overlay = img.copy()
    green_overlay[hair_bool] = (
        green_overlay[hair_bool].astype(np.float32) * 0.3
        + np.array([0, 120, 0], dtype=np.float32)
    ).clip(0, 255).astype(np.uint8)
    cv2.addWeighted(green_overlay, 0.45, img, 0.55, 0, img)


# ---------------------------------------------------------------------------
# draw_nose_anchored_zone
# ---------------------------------------------------------------------------

def draw_nose_anchored_zone(
    img: np.ndarray,
    hair_zone: NoseAnchoredHairZone,
    pose: "PoseResult",
    hand_near_head: bool = False,
) -> None:
    """Draw the learned nose-anchored hair zone as a light-blue rectangle.

    Only drawn when:
    - Zone is learned (>= 10 clean frames)
    - Hand is near head (zone fallback is active)
    When hand is away, the live mask is used and this rectangle is hidden.
    """
    if not hand_near_head:
        return
    if not hair_zone.is_learned:
        return
    if not pose.detected:
        return

    nose_lm = pose.landmarks.get("nose")
    if nose_lm is None or (nose_lm.confidence or 0) < 0.5:
        return

    h, w = img.shape[:2]
    nx = int(nose_lm.x * w)
    ny = int(nose_lm.y * h)

    offsets, frame_count = hair_zone.zone_bounds
    if offsets is None:
        return
    top, bottom, left, right = offsets

    # Apply same 15% margin as the detection logic
    m = 1.15
    x0 = max(0, nx - int(left * m))
    y0 = max(0, ny - int(top * m))
    x1 = min(w - 1, nx + int(right * m))
    y1 = min(h - 1, ny + int(bottom * m))

    # Semi-transparent light-blue rectangle
    COLOR_LIGHT_BLUE = (200, 180, 80)   # BGR approximation of light blue
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), COLOR_LIGHT_BLUE, 1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Label in corner of the rectangle
    label = f"hair zone ({frame_count}f)"
    _text_shadow(img, label, (x0 + 4, y0 + 14), _FONT_SM, COLOR_LIGHT_BLUE)


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
    pinch_threshold: float,
    spread_threshold: float = 0.075,
    rotation_threshold: float = 0.075,
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
    val1 = 155  # column 1 value offset (after label)
    col2 = w // 2 + 10
    val2 = col2 + 120  # column 2 value offset (after label)
    lh   = 22   # line height

    # ---- Title ----
    cv2.putText(img, "HairGuard Debug View  v2", (col1, 22), _FONT, _FONT_LG, COLOR_WHITE, 1, cv2.LINE_AA)

    # ---- Column 1 ----
    y = 22 + lh + 2

    # State
    cv2.putText(img, "State:", (col1, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, state, (val1, y), _FONT, _FONT_MD, state_fg, 2, cv2.LINE_AA)
    y += lh

    # Alert
    alert_txt   = "YES" if result.should_alert else "NO"
    alert_color = COLOR_RED if result.should_alert else COLOR_GREEN
    cv2.putText(img, "Alert:", (col1, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, alert_txt, (val1, y), _FONT, _FONT_MD, alert_color, 2, cv2.LINE_AA)
    y += lh

    # Fingertips in zone
    fiz_txt   = "YES" if result.fingertips_in_zone else "NO"
    fiz_color = COLOR_YELLOW if result.fingertips_in_zone else COLOR_WHITE
    cv2.putText(img, "In zone:", (col1, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, fiz_txt, (val1, y), _FONT, _FONT_MD, fiz_color, 1, cv2.LINE_AA)
    y += lh

    # Consecutive frames
    cf_txt = str(result.consecutive_frames)
    cf_color = COLOR_RED if result.consecutive_frames >= 5 else COLOR_WHITE
    cv2.putText(img, "Persist:", (col1, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, cf_txt, (val1, y), _FONT, _FONT_MD, cf_color, 1, cv2.LINE_AA)

    # ---- Column 2 ----
    y = 22 + lh + 2

    # FPS
    cv2.putText(img, f"FPS: {fps:.1f}", (col2, y), _FONT, _FONT_MD, COLOR_CYAN, 1, cv2.LINE_AA)
    y += lh

    # Pinch smooth
    if result.pinch_ratio_smooth is not None:
        pr_val = result.pinch_ratio_smooth
        pr_color = COLOR_RED if pr_val < pinch_threshold else COLOR_GREEN
        pr_txt = f"{pr_val:.3f}"
    else:
        pr_color = COLOR_GRAY
        pr_txt = "N/A"
    cv2.putText(img, "Pinch(sm):", (col2, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, pr_txt, (val2, y), _FONT, _FONT_MD, pr_color, 1, cv2.LINE_AA)
    y += lh

    # Spread
    if result.spread_index_pinky is not None:
        sp_val = result.spread_index_pinky
        sp_color = COLOR_RED if sp_val < spread_threshold else COLOR_GREEN
        sp_txt = f"{sp_val:.3f}"
    else:
        sp_color = COLOR_GRAY
        sp_txt = "N/A"
    cv2.putText(img, "Spread:", (col2, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, sp_txt, (val2, y), _FONT, _FONT_MD, sp_color, 1, cv2.LINE_AA)
    y += lh

    # Hand rotation
    if result.hand_rotation is not None:
        rot_val = result.hand_rotation
        rot_color = COLOR_RED if rot_val < rotation_threshold else COLOR_GREEN
        rot_txt = f"{rot_val:.3f}"
    else:
        rot_color = COLOR_GRAY
        rot_txt = "N/A"
    cv2.putText(img, "Rotation:", (col2, y), _FONT, _FONT_MD, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, rot_txt, (val2, y), _FONT, _FONT_MD, rot_color, 1, cv2.LINE_AA)
    y += lh

    # Key hints
    hint = "Q/ESC: quit   C: cycle camera"
    (tw, _), _ = cv2.getTextSize(hint, _FONT, _FONT_SM, 1)
    cv2.putText(img, hint, (w - tw - 8, PANEL_HEIGHT - 6), _FONT, _FONT_SM, COLOR_GRAY, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(camera: Optional[int] = None) -> None:
    """Open the HairGuard debug view window. Blocks until 'q'/ESC is pressed."""
    config = Config()
    pinch_threshold: float = config.get("detection", "pinch_threshold")
    spread_threshold: float = config.get("detection", "spread_threshold")
    rotation_threshold: float = config.get("detection", "rotation_threshold")
    min_confidence: float  = config.get("detection", "min_confidence")
    hair_dilation_px: int  = int(config.get("detection", "hair_dilation_px"))

    default_camera = camera if camera is not None else (config.get("camera") or 0)

    print("[debugview] Initializing detectors...")
    pose_det = PoseDetector(min_confidence=min_confidence)
    hand_det = HandDetector(max_hands=2, min_confidence=min_confidence)
    hair_seg = HairSegmenter()
    analyzer = GestureAnalyzer(config)

    available_cameras = CameraManager.list_cameras()
    camera_indices = [c["index"] for c in available_cameras] or [0]
    try:
        current_cam_slot = camera_indices.index(int(default_camera))
    except (ValueError, TypeError):
        current_cam_slot = 0

    def open_camera(idx: int) -> CameraManager:
        print(f"[debugview] Opening camera {idx}...")
        return CameraManager(camera_index=idx)

    cam = open_camera(camera_indices[current_cam_slot])

    cv2.namedWindow("HairGuard Debug View", cv2.WINDOW_AUTOSIZE)

    fps_counter = 0
    fps_value   = 0.0
    fps_ts      = time.monotonic()

    # --- CSV recording state ---
    import csv
    csv_file = None
    csv_writer = None
    recording_label = ""

    print("[debugview] Running — press 'q' or ESC to quit, 'c' to cycle cameras.")
    print("[debugview] Press '1' to toggle TWISTING recording, '2' to toggle COMBING recording.")

    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                print("[debugview] No frame from camera.")
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            ts = int(time.monotonic() * 1000)

            pose      = pose_det.process(frame, timestamp_ms=ts)
            hands     = hand_det.detect(frame, timestamp_ms=ts + 1)
            hair_mask = hair_seg.segment(frame)

            result = analyzer.analyze(pose, hands, hair_mask, h, w)

            # FPS
            fps_counter += 1
            now = time.monotonic()
            elapsed = now - fps_ts
            if elapsed >= 1.0:
                fps_value   = fps_counter / elapsed
                fps_counter = 0
                fps_ts      = now

            # --- CSV recording ---
            if csv_writer is not None and hands:
                for hi, hand in enumerate(hands):
                    row = {
                        "timestamp": now,
                        "label": recording_label,
                        "hand_idx": hi,
                        "handedness": hand.handedness,
                        "pinch_ratio": hand.pinch_ratio(),
                        "hand_scale": hand.hand_scale(),
                        "is_gripping": hand.is_gripping(3),
                        "fingertips_in_zone": result.fingertips_in_zone,
                        "state": result.state,
                    }
                    # Add all 21 landmark positions
                    for lm_idx in range(21):
                        lm = hand.landmarks.get(lm_idx)
                        if lm:
                            row[f"lm{lm_idx}_x"] = f"{lm.x:.4f}"
                            row[f"lm{lm_idx}_y"] = f"{lm.y:.4f}"
                            row[f"lm{lm_idx}_z"] = f"{lm.z:.4f}"
                    csv_writer.writerow(row)

            # --- Build display frame ---
            # Draw everything on original resolution (coordinates match engine)
            display = frame.copy()

            draw_hair_mask(display, hair_mask.dilate(hair_dilation_px))
            draw_nose_anchored_zone(display, analyzer._hair_zone, pose, result.fingertips_in_zone)
            draw_pose_landmarks(display, pose)
            draw_hand_skeletons(display, hands, result, pinch_threshold)
            draw_pinch_indicator(display, hands, pinch_threshold)
            draw_status_panel(display, result, fps_value, pinch_threshold, spread_threshold, rotation_threshold)

            # Upscale for sharp display (after all drawing)
            dh, dw = display.shape[:2]
            target_w = 960
            if dw < target_w:
                scale = target_w / dw
                display = cv2.resize(display, (target_w, int(dh * scale)), interpolation=cv2.INTER_LINEAR)

            cv2.imshow("HairGuard Debug View", display)

            # --- Recording indicator on frame ---
            if csv_writer is not None:
                rec_text = f"REC [{recording_label}]"
                cv2.circle(display, (w - 30, 20), 8, (0, 0, 255), -1)
                cv2.putText(display, rec_text, (w - 200, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("HairGuard Debug View", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            # Detect window closed via X button
            if cv2.getWindowProperty("HairGuard Debug View", cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('1'):
                if csv_writer is not None and recording_label == "twisting":
                    # Toggle OFF
                    csv_file.close()
                    csv_file = None
                    csv_writer = None
                    print("[debugview] TWISTING recording stopped.")
                else:
                    # Toggle ON
                    if csv_file is not None:
                        csv_file.close()
                    csv_path = Path.home() / ".hairguard" / f"recording_twisting_{int(time.time())}.csv"
                    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
                    fieldnames = ["timestamp", "label", "hand_idx", "handedness", "pinch_ratio", "hand_scale", "is_gripping", "fingertips_in_zone", "state"]
                    fieldnames += [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
                    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
                    csv_writer.writeheader()
                    recording_label = "twisting"
                    print(f"[debugview] Recording TWISTING to {csv_path}")
            elif key == ord('2'):
                if csv_writer is not None and recording_label == "combing":
                    # Toggle OFF
                    csv_file.close()
                    csv_file = None
                    csv_writer = None
                    print("[debugview] COMBING recording stopped.")
                else:
                    # Toggle ON
                    if csv_file is not None:
                        csv_file.close()
                    csv_path = Path.home() / ".hairguard" / f"recording_combing_{int(time.time())}.csv"
                    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
                    fieldnames = ["timestamp", "label", "hand_idx", "handedness", "pinch_ratio", "hand_scale", "is_gripping", "fingertips_in_zone", "state"]
                    fieldnames += [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
                    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
                    csv_writer.writeheader()
                    recording_label = "combing"
                    print(f"[debugview] Recording COMBING to {csv_path}")
            elif key == ord('s'):
                # Stop recording
                if csv_file is not None:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None
                    print("[debugview] Recording stopped.")
            elif key == ord('c'):
                cam.release()
                current_cam_slot = (current_cam_slot + 1) % len(camera_indices)
                next_idx = camera_indices[current_cam_slot]
                try:
                    cam = open_camera(next_idx)
                except RuntimeError as exc:
                    print(f"[debugview] Could not open camera {next_idx}: {exc}")
                    current_cam_slot = (current_cam_slot - 1) % len(camera_indices)
                    cam = open_camera(camera_indices[current_cam_slot])

    finally:
        if csv_file is not None:
            csv_file.close()
            print("[debugview] Recording saved.")
        cam.release()
        pose_det.close()
        hand_det.close()
        # face_det removed — no longer used
        hair_seg.close()
        cv2.destroyAllWindows()
        print("[debugview] Closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HairGuard Debug View v2")
    parser.add_argument("--camera", type=int, default=None, metavar="N",
                        help="Camera index to use (default: from config)")
    args = parser.parse_args()
    main(camera=args.camera)
