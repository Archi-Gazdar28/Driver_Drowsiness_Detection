#!/usr/bin/env python3
"""
DrowsyGuard — Real-time Drowsiness Detector
============================================
Entry point. Run with:
    python main.py
    python main.py --camera 1          # use second camera
    python main.py --no-sound          # disable beep alerts
    python main.py --ear 0.22          # custom EAR threshold
    python main.py --mar 0.65          # custom MAR threshold
    python main.py --frames 4          # custom consecutive-frame threshold
"""

import argparse
import time
import sys

import cv2
import numpy as np
import imutils
from imutils.video import WebcamVideoStream

from utils.detector import FaceDetector
from utils.alerter import Alerter
from utils.display import draw_frame
from utils.logger import DrowsinessLogger
from config import Config


def parse_args():
    ap = argparse.ArgumentParser(description="DrowsyGuard real-time drowsiness detector")
    ap.add_argument("--camera",  type=int,   default=0,     help="Camera index (default: 0)")
    ap.add_argument("--ear",     type=float, default=None,  help="EAR threshold override")
    ap.add_argument("--mar",     type=float, default=None,  help="MAR threshold override")
    ap.add_argument("--frames",  type=int,   default=None,  help="Consecutive closed frames threshold")
    ap.add_argument("--no-sound",action="store_true",       help="Disable audio alerts")
    ap.add_argument("--no-log",  action="store_true",       help="Disable CSV logging")
    ap.add_argument("--width",   type=int,   default=None,  help="Frame width override")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    # Apply CLI overrides
    if args.ear    is not None: cfg.EAR_THRESH        = args.ear
    if args.mar    is not None: cfg.MAR_THRESH        = args.mar
    if args.frames is not None: cfg.EAR_CONSEC_FRAMES = args.frames
    if args.width  is not None: cfg.FRAME_WIDTH       = args.width
    if args.no_sound:           cfg.SOUND_ENABLED     = False
    if args.no_log:             cfg.LOGGING_ENABLED   = False

    print("[DrowsyGuard] Loading facial landmark predictor...")
    try:
        face_detector = FaceDetector(cfg)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        print("\nDownload the model file:")
        print("  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("  Extract → place in ./dlib_shape_predictor/\n")
        sys.exit(1)

    alerter = Alerter(cfg)
    logger  = DrowsinessLogger(cfg) if cfg.LOGGING_ENABLED else None

    print(f"[DrowsyGuard] Starting camera (index {args.camera})...")
    vs = WebcamVideoStream(src=args.camera).start()
    time.sleep(2.0)

    print("[DrowsyGuard] Running — press 'q' to quit\n")
    print(f"  Thresholds → EAR: {cfg.EAR_THRESH}  "
          f"MAR_YAWN: {cfg.MAR_YAWN_THRESH}  "
          f"EAR_Frames: {cfg.EAR_CONSEC_FRAMES}  "
          f"YAW_Frames: {cfg.MAR_CONSEC_FRAMES}\n")

    closed_counter = 0
    yawning_counter = 0
    eyes_closed_start_time = None  # Track when eyes started closing
    continuous_yawn_counter = 0    # Track continuous yawning
    fps_timer      = time.time()
    frame_count    = 0
    fps_fps        = 0.0  # Store FPS for alert logic

    while True:
        frame = vs.read()
        if frame is None:
            continue

        # cv2.resize is safer than imutils.resize on Windows webcam drivers
        h, w  = frame.shape[:2]
        scale = cfg.FRAME_WIDTH / float(w)
        frame = cv2.resize(frame, (cfg.FRAME_WIDTH, int(h * scale)),
                           interpolation=cv2.INTER_AREA)

        # Ensure proper uint8 BGR array (guards against webcam quirks on Windows)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        frame_count += 1

        # Compute FPS
        elapsed = time.time() - fps_timer
        fps_fps = frame_count / elapsed if elapsed > 0 else 0

        result = face_detector.process(frame)

        if result is not None:
            ear, mar, pitch, yaw, roll, shape, rect = result

            # ── Eye Closure Detection ──────────────────────────────────────
            if ear < cfg.EAR_THRESH:
                closed_counter += 1
                # Track continuous eye closure time
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = time.time()
            else:
                closed_counter = 0
                eyes_closed_start_time = None

            # ── Improved Yawning Detection ─────────────────────────────────
            # Get proper yawn metrics (vertical and horizontal mouth opening)
            yawn_mar, mouth_width = face_detector._is_proper_yawn(shape)
            
            # Detect yawn: needs sufficient vertical AND horizontal mouth opening
            is_proper_yawn = (
                yawn_mar >= cfg.YAWN_MAR_MIN and 
                mouth_width >= cfg.YAWN_MOUTH_WIDTH_MIN
            )
            
            if is_proper_yawn:
                continuous_yawn_counter += 1
            else:
                continuous_yawn_counter = 0

            # Traditional detection for backward compatibility
            if mar > cfg.MAR_YAWN_THRESH:
                yawning_counter += 1
            else:
                yawning_counter = 0

            # ── Alert Conditions ───────────────────────────────────────────
            # Drowsy: eyes closed for consecutive frames
            drowsy = closed_counter >= cfg.EAR_CONSEC_FRAMES
            
            # Yawning: sustained continuous yawn (monitor only, no alert)
            yawning = continuous_yawn_counter >= cfg.YAWN_DURATION_FRAMES
            
            # Extended eye closure: track duration in seconds
            eyes_closed_prolonged = False
            if eyes_closed_start_time is not None:
                closed_duration = time.time() - eyes_closed_start_time
                if closed_duration >= cfg.EYES_CLOSED_DURATION_SECS:
                    eyes_closed_prolonged = True
                    drowsy = True  # Trigger drowsy alert

            # Alert ONLY on prolonged eye closure (5+ seconds), NOT on yawning
            alerter.check(eyes_closed_prolonged)


            # Logging
            if logger:
                logger.log(ear, mar, closed_counter, yawning_counter, drowsy, yawning, pitch, yaw, roll)

            # Console status line
            status = "DROWSY ⚠" if drowsy else ("YAWNING" if yawning else "ALERT ✓")
            print(
                f"  EAR:{ear:.3f}  MAR:{mar:.3f}  "
                f"Closed:{closed_counter:2d}  Yawn:{continuous_yawn_counter:2d}  "
                f"P:{pitch:5.1f} Y:{yaw:5.1f} R:{roll:5.1f}  "
                f"FPS:{fps_fps:4.1f}  [{status}]   ",
                end="\r"
            )

            # Draw annotations
            draw_frame(frame, cfg, shape, rect, ear, mar, closed_counter,
                       drowsy, yawning, pitch, yaw, roll, fps_fps)
        else:
            closed_counter = 0
            yawning_counter = 0
            continuous_yawn_counter = 0
            eyes_closed_start_time = None
            print(f"  No face detected — FPS:{fps_fps:4.1f}                              ", end="\r")
            draw_frame(frame, cfg, None, None, None, None, 0,
                       False, False, None, None, None, fps_fps)

        cv2.imshow("DrowsyGuard", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            # Reset counter
            closed_counter = 0
            yawning_counter = 0
            continuous_yawn_counter = 0
            eyes_closed_start_time = None
            frame_count    = 0
            fps_timer      = time.time()
            print("\n[DrowsyGuard] Counter reset.")

    print("\n[DrowsyGuard] Shutting down...")
    if logger:
        logger.close()
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    main()