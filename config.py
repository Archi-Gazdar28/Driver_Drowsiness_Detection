"""
config.py — All tunable parameters for DrowsyGuard
Edit this file to change detection thresholds, paths, and features.
"""

import os


class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
    PREDICTOR_PATH = os.path.join(BASE_DIR, "dlib_shape_predictor",
                                  "shape_predictor_68_face_landmarks.dat")
    LOG_DIR        = os.path.join(BASE_DIR, "logs")
    SOUND_FILE     = os.path.join(BASE_DIR, "sounds", "alert.wav")  # optional

    # ── Camera ─────────────────────────────────────────────────────────────
    FRAME_WIDTH  = 1024
    FRAME_HEIGHT = 576   # informational; imutils resize keeps aspect ratio

    # ── Detection thresholds ───────────────────────────────────────────────
    EAR_THRESH         = 0.22   # Eye Aspect Ratio — below = eyes closed
    MAR_THRESH         = 0.50   # Mouth Aspect Ratio — above = mouth open
    MAR_YAWN_THRESH    = 0.55   # Mouth Aspect Ratio — above = yawning
    EAR_CONSEC_FRAMES  = 4      # consecutive closed frames → drowsy alert
    MAR_CONSEC_FRAMES  = 3      # consecutive yawning frames → yawn alert
    
    # ── Improved Yawn Detection Thresholds ──────────────────────────────────
    YAWN_MAR_MIN       = 0.55   # Minimum MAR for proper yawn detection
    YAWN_MOUTH_WIDTH_MIN = 50   # Minimum mouth width (pixels) to filter false positives
    YAWN_DURATION_FRAMES = 5    # Frames needed for continuous yawn detection (monitor only)
    EYES_CLOSED_DURATION_SECS = 2.0  # Alert when eyes closed for 2+ seconds continuously

    # ── Features ───────────────────────────────────────────────────────────
    SOUND_ENABLED   = True    # beep on drowsy alert (requires pygame or playsound)
    LOGGING_ENABLED = True    # write CSV log to logs/

    # ── Display ────────────────────────────────────────────────────────────
    SHOW_LANDMARKS    = True
    SHOW_EYE_CONTOUR  = True
    SHOW_MOUTH_CONTOUR= True
    SHOW_HEAD_AXIS    = True
    SHOW_HUD          = True

    # ── Colors (BGR) ───────────────────────────────────────────────────────
    COLOR_OK      = (0, 255, 0)     # green
    COLOR_WARN    = (0, 165, 255)   # orange
    COLOR_ALERT   = (0, 0, 255)     # red
    COLOR_LANDMARK= (0, 0, 255)
    COLOR_AXIS    = (255, 0, 0)

    # ── Head pose 3D model points (standard face model) ───────────────────
    import numpy as np
    MODEL_3D_POINTS = np.array([
        (0.0,    0.0,    0.0),     # Nose tip       [33]
        (0.0,  -330.0,  -65.0),    # Chin           [8]
        (-225.0, 170.0, -135.0),   # Left eye left  [36]
        (225.0,  170.0, -135.0),   # Right eye right[45]
        (-150.0,-150.0, -125.0),   # Left mouth     [48]
        (150.0, -150.0, -125.0),   # Right mouth    [54]
    ], dtype="double")
