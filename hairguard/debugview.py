"""
HairGuard Debug View  (v2) — the standalone command.
====================================================
Opens its own camera and runs its own copy of the pipeline, which makes it
the right tool for exactly one situation: the app is *not* running and you
want to see what the detectors make of you.

While the app is running, use the tray entry instead. A second open of a
held camera does not fail on Windows — it returns black frames — so this
command would show an empty picture and blame the detector. The guard in
``hairguard.cli`` refuses to start it in that case; the tray's window is fed
from the running detection tick and has no camera of its own (see
``hairguard.debugfeed``).

The drawing lives in ``hairguard.debugdraw`` and is shared with that window,
so both show the same thing.

Run with:
    python -m hairguard.debugview
    python -m hairguard.debugview --camera 1

Controls: q / ESC to quit, c to cycle cameras.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2

from hairguard.analyzer import GestureAnalyzer
from hairguard.camera import CameraManager, resolve_camera_index
from hairguard.config import Config
from hairguard.detector import PoseDetector
from hairguard.face_detector import FaceDetector
from hairguard.hand_detector import HandDetector
from hairguard.head_zone import HairZoneSmoother

# Re-exported so that `from hairguard.debugview import draw_status_panel`
# keeps working; the implementations moved to debugdraw when the tray's
# window needed them without a camera attached.
from hairguard.debugdraw import (  # noqa: F401
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_CYAN,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    HAND_CONNECTIONS,
    PANEL_HEIGHT,
    draw_hair_zone,
    draw_hand_skeletons,
    draw_pinch_indicator,
    draw_pose_landmarks,
    draw_status_panel,
)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(camera: Optional[int] = None) -> None:
    """Open the HairGuard debug view window. Blocks until 'q'/ESC is pressed."""
    config = Config()
    combined_threshold: float = config.get("detection", "combined_threshold") or 0.55
    min_confidence: float  = config.get("detection", "min_confidence")
    hair_zone_offset: float = config.get("detection", "hair_zone_offset")

    default_camera = camera if camera is not None else resolve_camera_index(config.get("camera"))

    print("[debugview] Initializing detectors...")
    pose_det = PoseDetector(min_confidence=min_confidence)
    hand_det = HandDetector(max_hands=2, min_confidence=min_confidence)
    face_det = FaceDetector(min_confidence=min_confidence)

    # Same decision the app makes, so what is watched here is what runs.
    from hairguard.twistnet import load_configured

    twist_model, detector_status = load_configured(config)
    print(f"[debugview] Детектор: {detector_status}")
    analyzer = GestureAnalyzer(config, twist_model=twist_model)
    model_threshold = twist_model.threshold if twist_model is not None else None

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

    cv2.namedWindow("HairGuard Debug View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HairGuard Debug View", 960, 600)

    fps_counter = 0
    fps_value   = 0.0
    fps_ts      = time.monotonic()
    zone_smoother = HairZoneSmoother(alpha=0.3)

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

            pose  = pose_det.process(frame, timestamp_ms=ts)
            hands = hand_det.detect(frame, timestamp_ms=ts + 1)
            face  = face_det.detect(frame, timestamp_ms=ts + 2)

            result = analyzer.analyze(pose, hands, face, h, w)

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
            display = frame.copy()

            draw_hair_zone(display, face, pose, hair_zone_offset, zone_smoother)
            draw_pose_landmarks(display, pose)
            draw_hand_skeletons(display, hands, result, combined_threshold)
            draw_pinch_indicator(display, hands, combined_threshold)
            draw_status_panel(display, result, fps_value, combined_threshold,
                              model_threshold=model_threshold)

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
        face_det.close()
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
