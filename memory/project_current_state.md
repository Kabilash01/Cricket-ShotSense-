---
name: Cricket-Angle pipeline current state (2026-06-03)
description: Where test3.py stands, what works, what's broken, and what's next
type: project
---

`src/test3.py` is the active pipeline. As of 2026-06-03:

**Environment**:
- OS: Ubuntu (migrated from Windows 2026-05-31)
- Conda env: `ball` — Python 3.11, torch 2.12+cu132, ultralytics 8.4, opencv 4.13, onnxruntime-gpu 1.26
- GPU: RTX 5060 Laptop (Blackwell sm_120, CUDA 13.2) — all models confirmed running on GPU
- Run command: `HEADLESS=1 VIDEO_PATH=<clip.mp4> conda run -n ball python -m src.test3`
- AV1-encoded `videoplayback.mp4` requires ffmpeg re-encode first: `conda run -n ball ffmpeg -ss HH:MM:SS -i videoplayback.mp4 -t HH:MM:SS -c:v libx264 -preset veryfast -crf 23 -an clip.mp4`

**Models in use**:
- Ball detector: `ball_test/weights/best.pt` (YOLOv8, conf=0.10, class 0=ball)
- Bat detector: `models/bat_detector_v8n/weights/best.pt` (YOLOv8, conf=0.05, bat_class_id=1)
- Shot classifier: `models/shot_classifier/shot_classifier.onnx` (EfficientNetB0+GRU, CUDAExecutionProvider)
- Pose detector: `models/pose/yolov8m-pose.pt` (YOLOv8-pose, conf=0.30) — **NEW**

**Contact triggers (3 parallel paths)**:
1. **BatBox** (Priority 1): ball centre within 35px of bat bbox — most precise
2. **Trajectory** (Priority 2): curvature >10° + accel spike >1.5 + speed >5 px/f
3. **WristVelocity** (Priority 3, NEW): peak wrist keypoint speed ≥80 px/f, gated on ball seen within 15 frames

**Shot classification logic**:
- `shot_angle` computed as `atan2(dy, dx)` (90° = straight toward bowler)
- Converted to README wagon-wheel coords: `geo_angle = (90 - shot_angle) % 360`
- Geometry `classify_shot(geo_angle)` is PRIMARY using 10-shot table
- EfficientNet overrides ONLY when confidence ≥97%
- `wagon_wheel_angle` in events.json stores `geo_angle` (not raw atan2 angle)

**Batsman tracking (pose)**:
- `POSE_BAT_CONF=0.55` + y-position filter: only high-conf bat detections in lower 40% of frame anchor the pose tracker
- IoU continuity (min 0.20) keeps tracker on same person across frames
- Prevents umpire/bowler from being tracked as batsman

**Key tuning constants in test3.py CONFIG block**:
```
WRIST_SPEED_THRESHOLD    = 80.0   px/frame
WRIST_REQUIRE_BALL_WITHIN = 15    frames
POSE_BAT_CONF            = 0.55
POSE_BAT_MIN_Y_FRAC      = 0.40
CONTACT_COOLDOWN_FRAMES  = 90
INTERP_MAX_GAP           = 3
```

**Validated results on test_clip2.mp4 (19:50–24:00, 4 min, 6 real balls)**:
- 7 trajectory-backed events saved (1 false positive vs 6 ground-truth balls)
- All events require real ball trajectory — EfficientNet-only saves were
  removed because they false-fire on batsman stance between deliveries
- Events: Flick, Edge, Leg Glance, Pull Shot, Lofted Drive, Late Cut, Edge
- No stale future_trajectory bug, no OOM, no stuck ball_id

**Event-count tuning decision (2026-06-03)**:
- Tested one-event-per-delivery → gave 5 (dropped a real Lofted Drive that
  shared delivery_id=2). Reverted: over-capture beats under-capture.
- Root cause of the extra event: delivery segmentation merges balls into one
  delivery_id, esp. at clip start (everything before first detected delivery
  is delivery_id=0). The real fix is better segmentation, needs cleaner
  full-delivery footage to tune — do NOT re-add the one-per-delivery gate.

**Delivery segmentation — gap-based (updated 2026-07-29)**:
- `test3.py` now increments `delivery_id` when the ball reappears after being
  absent ≥ `DELIVERY_GAP_FRAMES` (default 45, env-tunable). Replaced the old
  top-of-frame release heuristic, which needed the ball caught at release (rare
  at ~9% hit rate) and merged real deliveries into a `delivery 0` catch-all.
- Validated on test_clip2: same 7 events, but F136/F294 (two separate balls,
  ~5s apart) now split into d2/d3 instead of both collapsing to delivery 0.
  F939/F1045 stay one delivery (ball tracked continuously, gap never hit 45).
- **Replay ceiling (the real remaining limit)**: broadcast replays + camera cuts
  of the same ball each create a separate ball-activity burst, so delivery_id
  counts ~36 bursts on a 6-ball clip, not 6 umpire-deliveries. Phantom bursts
  stay event-free (event save still requires a real >4-pt trajectory), so it's
  cosmetic. Diagnostic: no clean bimodal gap exists between intra- vs inter-
  delivery — inter-detection gaps form a continuous spread. Lifting this ceiling
  to true per-ball counts needs **scene-cut detection** (frame-diff/histogram),
  not more gap tuning. Diagnostic script: scratchpad `ball_gaps.py`.
- Do NOT re-add the one-event-per-delivery gate (see below) — over-capture still
  beats under-capture.

**Known remaining issues**:
- Ball detection still sparse (~10% hit rate) — limits post-contact trajectory quality
- EfficientNet at 100% sometimes disagrees with geometry angle (misfires on follow-through frames)
- delivery_id over-segments on replays (see above) — needs scene-cut detection, not gap tuning
- Bowler may still get pose box occasionally if bat detector fires on the ball in his hand

**Analysis artifacts for the paper (added 2026-07-29)**:
- events_clip1/2/3.json REGENERATED with gap-based segmentation (clip1→test_clip.mp4=1
  event, clip2→test_clip2.mp4=7, clip3→test_clip3.mp4=9441f=7; n=15 total; clip1 dropped
  from 2→1 due to new tracker-reset timing).
- `docs/directional_results.md` — circular stats (mean dir 238°, Rayleigh p=0.039 +
  Watson U² p=0.050 marginally sig; Rao/Kuiper not; leg-side tendency, weak at n=15).
- `docs/system_performance.md` — runtime profile (ball detector is the 17-FPS bottleneck)
  + descriptive analytics. Figures + JSON in `docs/figures/`.
- `scripts/eval_events.py` + `docs/eval/` — precision/recall scaffold; NEEDS human
  ground-truth labels (not yet done — the one remaining gap for a publishable evaluation).
- test3.py has PROFILE env (default on) writing runtime_profile.json each run.

**Why**: User wants reliable per-delivery event extraction for analytics (`events.json`).

**How to apply**:
- Always re-read `src/test3.py` before quoting line numbers — file changes rapidly
- Don't re-implement things already in `shot_analyzer.py` — port from there if needed
- Next priority: scene-cut detection (to lift the delivery-segmentation replay
  ceiling), then EfficientNet reliability improvement
