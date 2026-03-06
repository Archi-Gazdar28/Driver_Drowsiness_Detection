# DrowsyGuard 👁

Real-time drowsiness and yawn detection using dlib + OpenCV.
No frontend — pure Python, runs in a CV window.

---

## Folder structure

```
drowsy_guard/
├── main.py                          ← entry point
├── config.py                        ← all thresholds & settings
├── requirements.txt
├── utils/
│   ├── detector.py                  ← face detection, EAR, MAR, head pose
│   ├── display.py                   ← OpenCV drawing / HUD
│   ├── alerter.py                   ← audio / beep alerts
│   └── logger.py                    ← CSV session logger
├── dlib_shape_predictor/
│   └── shape_predictor_68_face_landmarks.dat   ← YOU MUST ADD THIS
├── logs/                            ← auto-created; CSV sessions saved here
└── sounds/
    └── alert.wav                    ← optional audio alert file
```

---

## Setup

### 1 — Python environment (recommended: venv)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **dlib on Windows** requires CMake + Visual Studio Build Tools.
> Easier option: install a prebuilt wheel:
> ```bash
> pip install cmake
> pip install dlib
> ```

### 3 — Download the landmark model

Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract the `.dat` file and place it at:
```
drowsy_guard/dlib_shape_predictor/shape_predictor_68_face_landmarks.dat
```

---

## Run

```bash
# Default (camera 0)
python main.py

# Custom camera index
python main.py --camera 1

# Adjust thresholds
python main.py --ear 0.22 --mar 0.65 --frames 4

# Disable audio alerts
python main.py --no-sound

# Disable CSV logging
python main.py --no-log
```

**Keyboard shortcuts while running:**
- `q` — quit
- `r` — reset the drowsiness counter and FPS timer

---

## Detection logic

| Metric | Full name | Trigger |
|--------|-----------|---------|
| **EAR** | Eye Aspect Ratio | < 0.25 → eyes closed |
| **MAR** | Mouth Aspect Ratio | > 0.60 → yawning |
| **Closed frames** | Consecutive EAR-below-threshold frames | ≥ 3 → DROWSY alert |

All thresholds are editable in `config.py` or via CLI flags.

---

## Logs

Each session creates a timestamped CSV in `logs/`:

```
timestamp, ear, mar, closed_frames, drowsy, yawning, pitch, yaw, roll
```

Use this to analyse patterns or tune thresholds.

---

## Optional audio alert

Place a `alert.wav` file in the `sounds/` folder.
Install pygame (`pip install pygame`) for playback.
Without it, the terminal bell (`\a`) is used as fallback.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `RuntimeError: Predictor not found` | Download the `.dat` file (Step 3 above) |
| Black / no camera window | Change `--camera 1` (or 2) |
| dlib install fails on Windows | Install CMake first: `pip install cmake` |
| Low FPS | Lower `FRAME_WIDTH` in `config.py` (e.g. 640) |
| False drowsy alerts | Increase `EAR_THRESH` in `config.py` (e.g. 0.28) |
