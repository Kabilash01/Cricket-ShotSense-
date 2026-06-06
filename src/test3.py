import os
import cv2
import time
import json
import math
import numpy as np

from collections import deque

from src.detection.yolo_detector import YoloBallDetector
from src.detection.bat_detector import BatDetector
from src.detection.shot_classifier import ShotClassifier
from src.detection.pose_detector import PoseDetector, WristVelocityTrigger, iou as _bbox_iou
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball


# =========================================================
# CONFIG
# =========================================================

from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent

BALL_MODEL_PATH = str(_REPO_ROOT / "ball_test" / "weights" / "best.pt")
BAT_MODEL_PATH  = str(_REPO_ROOT / "models" / "bat_detector_v8n" / "weights" / "best.pt")
SHOT_ONNX_PATH  = str(_REPO_ROOT / "models" / "shot_classifier" / "shot_classifier.onnx")
POSE_MODEL_PATH = str(_REPO_ROOT / "models" / "pose" / "yolov8m-pose.pt")

VIDEO_PATH = os.environ.get(
    "VIDEO_PATH",
    str(_REPO_ROOT / "videoplayback.mp4"),
)
HEADLESS = os.environ.get("HEADLESS", "0") == "1"
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0")) or None

# Wrist-velocity contact trigger (third path, fires when bat-box + traj miss)
POSE_CONF                = 0.30
WRIST_SPEED_THRESHOLD    = 80.0   # px/frame — peak wrist speed during swing
WRIST_HISTORY            = 5      # frames of wrist history used for max-speed
WRIST_REQUIRE_BALL_WITHIN = 15    # frames; gate wrist trigger on recent ball detection

# Bat detection has two conf levels:
#   BAT_CONF (0.05) — low threshold kept for ball-proximity contact detection
#   POSE_BAT_CONF (0.25) — higher threshold used ONLY to anchor batsman
#   selection in pick_batsman. Prevents umpire/stumps/bowler-hand false
#   positives from locking the pose tracker onto the wrong person.
POSE_BAT_CONF = 0.55   # high threshold — only real bat detections pass
# Bat box must appear in the lower portion of the frame (batsman holds bat
# at crease level). Anything above this row is the bowler's hand / umpire.
POSE_BAT_MIN_Y_FRAC = 0.40   # bat box centre-y must be > 40% down the frame

# Frames to ignore new contacts after one fires (suppresses follow-through
# practice swings; was 40, raised to 90 ≈ 3 s at 30 fps)
CONTACT_COOLDOWN_FRAMES = 90

OUTPUT_VIDEO = "output_analysis.mp4"
EVENTS_JSON = "events.json"

METERS_PER_PIXEL = 18.5 / 520


# =========================================================
# INIT
# =========================================================

ball_detector = YoloBallDetector(
    model_path=BALL_MODEL_PATH,
    conf=0.07,
    ball_class_id=0
)

bat_detector = BatDetector(
    model_path=BAT_MODEL_PATH,
    conf=0.05,
    bat_class_id=1       # class 0='-', class 1='bat'
)

pose_detector = PoseDetector(
    model_path=POSE_MODEL_PATH,
    conf=POSE_CONF,
    device=0,
)

wrist_trigger = WristVelocityTrigger(
    speed_threshold=WRIST_SPEED_THRESHOLD,
    history=WRIST_HISTORY,
)

prev_batsman_bbox = None
frames_since_ball  = 999      # frames since last real ball detection

import os as _os
shot_classifier = (
    ShotClassifier(SHOT_ONNX_PATH)
    if _os.path.exists(SHOT_ONNX_PATH)
    else None
)
if shot_classifier is None:
    print("⚠  shot_classifier.onnx not found — run scripts/convert_to_onnx.py first")
    print("   Falling back to geometry-based shot classification")

tracker = BallTracker()

cap = cv2.VideoCapture(VIDEO_PATH)

assert cap.isOpened(), "❌ Failed to open video"

fps = cap.get(cv2.CAP_PROP_FPS) or 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =========================================================
# OUTPUT VIDEO
# =========================================================

# Try H.264 first (avc1) for cross-player compatibility; fall back to mp4v
# (avc1 requires OpenCV built with H.264 support — pip wheels often lack this)
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
if not out.isOpened():
    print("⚠  avc1 codec unavailable, falling back to mp4v (re-encode with ffmpeg for playback)")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# =========================================================
# GLOBALS
# =========================================================

events = []

ball_id    = 0
delivery_id = 0      # increments each time ball appears at bowler's end

frame_idx = 0

fps_frames = 0
fps_time = time.time()
display_fps = 0

# =========================================================
# TRAJECTORY
# =========================================================

trajectory_history = deque(maxlen=30)

post_contact_points = []

future_trajectory = []

# Speed history for acceleration detection
speed_history = deque(maxlen=15)

# Ball detection gap interpolation state
INTERP_MAX_GAP = 5          # fill gaps up to this many missed frames
last_ball_pos  = None       # last confirmed detection (x, y)
last_ball_frame = -999      # frame index of last_ball_pos

# Rolling 60-frame buffer stored at classifier resolution to save RAM.
# EfficientNet input is 224x224, so no info is lost for classification.
CLF_SIZE = (224, 224)
frame_buffer = deque(maxlen=60)

# Frames captured around contact for shot classification
contact_frames = []

# Latest bat bounding box (x1,y1,x2,y2) or None
bat_box = None

# =========================================================
# CONTACT STATE
# =========================================================

contact_detected = False
contact_frame = -999

# =========================================================
# SHOT INFO
# =========================================================

shot_angle = None   # raw atan2 value (internal)
geo_angle  = None   # wagon-wheel angle displayed/saved (0°=straight)
shot_name = None
predicted_distance = 0

# =========================================================
# HELPERS
# =========================================================

def calculate_speed(vx, vy):
    return math.sqrt(vx**2 + vy**2)


def ball_near_bat(ball_pos, bat_box, margin=30):
    """Return True when ball centre is inside (or within margin of) the bat box."""
    bx, by = ball_pos
    x1, y1, x2, y2 = bat_box
    return (x1 - margin <= bx <= x2 + margin and
            y1 - margin <= by <= y2 + margin)


def calculate_motion_angle(vx, vy):
    angle = math.degrees(
        math.atan2(-vy, vx)
    )
    if angle < 0:
        angle += 360
    return angle


def detect_trajectory_curvature(trajectory_points, window_size=5):
    """
    Detect trajectory curvature by comparing angles at different segments.
    High curvature → direction change → possible bat contact
    """
    if len(trajectory_points) < window_size + 2:
        return 0.0

    # Recent segment
    p1 = np.array(trajectory_points[-window_size - 1])
    p2 = np.array(trajectory_points[-1])
    recent_vector = p2 - p1

    # Older segment
    p3 = np.array(trajectory_points[-window_size * 2 - 1])
    p4 = np.array(trajectory_points[-window_size - 1])
    older_vector = p4 - p3

    # Normalize to magnitude for angle comparison
    mag_recent = np.linalg.norm(recent_vector)
    mag_older = np.linalg.norm(older_vector)

    if mag_recent < 1 or mag_older < 1:
        return 0.0

    # Angle between vectors
    cos_angle = np.dot(
        recent_vector / mag_recent,
        older_vector / mag_older
    )
    cos_angle = np.clip(cos_angle, -1, 1)
    curvature = math.degrees(math.acos(cos_angle))

    return curvature


def detect_acceleration_spike(speed_history, threshold=5.0, window=3):
    """
    Detect sudden speed changes (acceleration/deceleration).
    Contact with bat causes rapid speed change.
    """
    if len(speed_history) < window:
        return 0.0

    # Compare average speed before and after
    recent_avg = np.mean(speed_history[-window:])
    older_avg = np.mean(speed_history[-(window*2):-window])

    acceleration = abs(recent_avg - older_avg)
    return acceleration


# Centre angle for each shot name (wagon-wheel coords). Used to reject
# EfficientNet overrides that are physically inconsistent with ball direction.
SHOT_CENTRE_ANGLE = {
    # Geometry table shots
    "Straight Drive": 0,
    "Off Drive":      27,
    "Cover Drive":    57,
    "Square Cut":     90,
    "Late Cut":      127,
    "Edge":          175,
    "Leg Glance":    225,
    "Pull Shot":     267,
    "Flick":         302,
    "On Drive":      332,
    # EfficientNet-only labels (not in geometry table but need angle check)
    "Lofted Drive":   30,   # typically driven over off/mid-off
    "Hook Shot":     267,   # same region as pull, leg side
    "Sweep":         225,   # leg side, similar to leg glance
    "Defensive Shot": 0,    # ball goes straight back / minimal movement
}


def _angle_diff(a, b):
    """Smallest circular difference between two angles (0-180)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def classify_shot(angle):
    # Normalize to 0-360 (0° = straight back toward bowler, clockwise)
    angle = angle % 360

    # 10-shot wagon-wheel table (right-handed batsman)
    if 345 <= angle or angle < 15:
        return "Straight Drive"
    elif 15 <= angle < 40:
        return "Off Drive"
    elif 40 <= angle < 75:
        return "Cover Drive"
    elif 75 <= angle < 105:
        return "Square Cut"
    elif 105 <= angle < 150:
        return "Late Cut"
    elif 150 <= angle < 200:
        return "Edge"
    elif 200 <= angle < 250:
        return "Leg Glance"
    elif 250 <= angle < 285:
        return "Pull Shot"
    elif 285 <= angle < 320:
        return "Flick"
    else:  # 320-345
        return "On Drive"


def smooth_trajectory(points):

    if len(points) < 5:
        return points

    pts = np.array(points)

    x = pts[:, 0]
    y = pts[:, 1]

    kernel = np.ones(5) / 5

    smooth_x = np.convolve(
        x,
        kernel,
        mode='valid'
    )

    smooth_y = np.convolve(
        y,
        kernel,
        mode='valid'
    )

    return list(
        zip(
            smooth_x.astype(int),
            smooth_y.astype(int)
        )
    )


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1
    frame_buffer.append(cv2.resize(frame, CLF_SIZE))

    # =====================================================
    # FPS
    # =====================================================

    fps_frames += 1

    if time.time() - fps_time >= 1.0:

        display_fps = fps_frames

        fps_frames = 0

        fps_time = time.time()

    # =====================================================
    # DETECTION
    # =====================================================

    predicted = (
        tracker.predict()
        if tracker.initialized
        else None
    )

    detections = ball_detector.detect(
        frame,
        predicted
    )

    # =====================================================
    # BAT DETECTION
    # =====================================================

    bat_detections = bat_detector.detect(frame)

    pose_bat_box = None   # high-conf bat box for batsman selection
    if bat_detections:
        # Use the highest-confidence bat detection
        bat_detections.sort(key=lambda d: d[6], reverse=True)
        _, _, bx1, by1, bx2, by2, bat_conf = bat_detections[0]
        bat_box = (bx1, by1, bx2, by2)

        # High-conf bat box for pose anchoring: must be high-confidence AND
        # in the lower portion of the frame (batsman at crease level).
        min_y = int(height * POSE_BAT_MIN_Y_FRAC)
        high_conf = [
            d for d in bat_detections
            if d[6] >= POSE_BAT_CONF
            and d[1] > min_y                  # bat box centre-y below threshold
        ]
        if high_conf:
            _, _, hbx1, hby1, hbx2, hby2, _ = high_conf[0]
            pose_bat_box = (hbx1, hby1, hbx2, hby2)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 165, 0), 2)
        cv2.putText(
            frame, f"BAT {bat_conf:.2f}",
            (bx1, by1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2
        )

    # =====================================================
    # POSE DETECTION (batsman wrists → contact trigger)
    # =====================================================

    pose_dets = pose_detector.detect(frame)
    batsman = PoseDetector.pick_batsman(
        pose_dets,
        prev_bbox=prev_batsman_bbox,
        frame_shape=frame.shape,
        bat_box=pose_bat_box,      # high-conf only — prevents umpire/bowler locks
    )
    if batsman is not None:
        new_bbox = batsman["bbox"]
        # Identity-change guard: if the picked batsman doesn't overlap with
        # the prior frame's pick, the pose detector swapped people. Drop the
        # wrist history so we don't compute velocity across two persons.
        if prev_batsman_bbox is not None and _bbox_iou(new_bbox, prev_batsman_bbox) < 0.20:
            wrist_trigger.reset()
        prev_batsman_bbox = new_bbox
        bx1p, by1p, bx2p, by2p = new_bbox
        cv2.rectangle(frame, (bx1p, by1p), (bx2p, by2p), (200, 100, 255), 1)
        lw, rw = PoseDetector.wrist_positions(batsman, min_conf=0.25)
        if lw is not None:
            cv2.circle(frame, (int(lw[0]), int(lw[1])), 8, (255, 0, 255), -1)
        if rw is not None:
            cv2.circle(frame, (int(rw[0]), int(rw[1])), 8, (255, 0, 255), -1)

    wrist_speed = wrist_trigger.update(batsman)

    # =====================================================
    # DRAW BALL DETECTIONS
    # =====================================================

    for cx, cy, x1, y1, x2, y2, conf in detections:

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

    # =====================================================
    # DELIVERY SEGMENTATION
    # Ball appearing in the top 35% of frame after ≥30 missed frames
    # = bowler releasing a new delivery. Reset overlays + increment delivery_id.
    # =====================================================

    if detections and tracker.missed_frames >= 30:
        cx0, cy0, *_ = detections[0]
        if cy0 < height * 0.35:
            delivery_id       += 1
            shot_name          = None
            shot_angle         = None
            geo_angle          = None
            predicted_distance = 0
            future_trajectory  = []
            last_ball_pos      = None
            last_ball_frame    = -999
            wrist_trigger.reset()
            tracker.reset()             # force fresh ball_id on new delivery
            trajectory_history.clear()
            speed_history.clear()
            print(f"[F{frame_idx}] 🏏 NEW DELIVERY #{delivery_id}")

    # =====================================================
    # TRACKER UPDATE
    # =====================================================

    tracker_updated = False

    if detections:

        tracker.missed_frames = 0
        frames_since_ball = 0

        if not tracker.initialized:

            ball_id += 1

            cx, cy, *_ = detections[0]

            tracker.update((cx, cy))
            tracker_updated = True
            last_ball_pos   = (cx, cy)
            last_ball_frame = frame_idx

        else:

            match = associate_ball(
                detections,
                predicted
            )

            if match:

                cx, cy, *_ = match

                tracker.update((cx, cy))
                tracker_updated = True
                last_ball_pos   = (cx, cy)
                last_ball_frame = frame_idx

    else:

        tracker.missed_frames += 1
        frames_since_ball += 1

        # Gap interpolation: if ball was seen recently and gap is small,
        # synthesise a position by linear interpolation toward predicted pos.
        gap = frame_idx - last_ball_frame
        if (
            last_ball_pos is not None
            and gap <= INTERP_MAX_GAP
            and tracker.initialized
            and predicted is not None
        ):
            t = gap / (INTERP_MAX_GAP + 1)
            ix = int(last_ball_pos[0] * (1 - t) + predicted[0] * t)
            iy = int(last_ball_pos[1] * (1 - t) + predicted[1] * t)
            tracker.update((ix, iy))
            tracker_updated = True

        # Reset tracker if ball has been missing too long
        if tracker.missed_frames > 60:
            tracker.reset()
            trajectory_history.clear()
            speed_history.clear()

    # =====================================================
    # TRACKER LOGIC
    # =====================================================

    if tracker.initialized:

        x, y = tracker.get_position()

        vx, vy = tracker.get_velocity()

        speed = calculate_speed(vx, vy)

        # Phantom speed guard: tracker jumped to a false detection far away
        # Reset if speed is physically impossible (> 800 px/frame)
        if speed > 800:
            tracker.reset()
            trajectory_history.clear()
            speed_history.clear()
            print(f"[F{frame_idx}] ⚠ Phantom speed {speed:.0f} — tracker reset")
            continue

        trajectory_history.append((x, y))
        speed_history.append(speed)

        # =================================================
        # CONTACT DETECTION
        # Priority 1: bat box (precise)
        # Priority 2: trajectory signals (always active)
        # =================================================

        cooldown_ok = not contact_detected and (frame_idx - contact_frame) > CONTACT_COOLDOWN_FRAMES

        if cooldown_ok:

            triggered      = False
            trigger_reason = ""

            # --- Priority 1: ball inside bat bounding box ---
            if bat_box is not None and ball_near_bat((x, y), bat_box, margin=35):
                triggered      = True
                trigger_reason = f"BatBox ball=({x},{y})"

            # --- Priority 2: trajectory direction change ---
            elif len(trajectory_history) >= 12:
                curvature   = detect_trajectory_curvature(
                    list(trajectory_history), window_size=5
                )
                accel_spike = detect_acceleration_spike(
                    list(speed_history), window=3
                )

                if curvature > 10 and accel_spike > 1.5 and speed > 5.0:
                    triggered      = True
                    trigger_reason = (
                        f"Traj curv={curvature:.1f}° "
                        f"accel={accel_spike:.1f} spd={speed:.1f}"
                    )

            # --- Priority 3: wrist-velocity spike (fires when 1+2 miss) ---
            # Gated on recent ball detection — a wrist spike with no ball in
            # flight is almost always the umpire/batsman gesturing, not contact.
            if (
                not triggered
                and wrist_trigger.should_trigger()
                and frames_since_ball <= WRIST_REQUIRE_BALL_WITHIN
            ):
                triggered      = True
                trigger_reason = (
                    f"WristVel {wrist_speed:.1f}px/f "
                    f"(ball_seen {frames_since_ball}f ago)"
                )

            if triggered:
                contact_detected    = True
                contact_frame       = frame_idx
                post_contact_points = []
                future_trajectory   = []   # clear stale prediction from prior event
                contact_frames      = list(frame_buffer)[-30:]
                print(f"🏏 CONTACT [F{frame_idx}] {trigger_reason}")

        # Debug: every 20 frames
        if frame_idx % 20 == 0:
            bat_status = f"bat=✅" if bat_box else "bat=❌"
            ball_status = f"ball=✅({len(detections)})" if detections else "ball=❌"
            pose_status = f"pose=✅" if batsman is not None else "pose=❌"
            print(
                f"[F{frame_idx:4d}] Spd={speed:5.1f} | "
                f"{ball_status} | {bat_status} | {pose_status} | "
                f"wrist={wrist_speed:5.1f} | "
                f"tracker={'on' if tracker.initialized else 'off'} | "
                f"Events={len(events)}"
            )

        # =================================================
        # STORE POST CONTACT TRAJECTORY
        # =================================================

        if contact_detected:

            frames_since_contact = frame_idx - contact_frame

            if frames_since_contact < 45:

                # Collect post-contact frames for classifier (capped at 45)
                if len(contact_frames) < 45:
                    contact_frames.append(cv2.resize(frame, CLF_SIZE))

                # Only append if tracker was actually updated this frame
                if tracker_updated:
                    post_contact_points.append((x, y))

            else:
                # Window just closed — run analysis now with all collected points
                contact_detected = False

                # =================================================
                # ANALYZE SHOT (runs once, after full 45-frame window)
                # =================================================

                # Filter out duplicate/stale positions
                unique_points = []
                for pt in post_contact_points:
                    if not unique_points or (
                        abs(pt[0] - unique_points[-1][0]) > 2 or
                        abs(pt[1] - unique_points[-1][1]) > 2
                    ):
                        unique_points.append(pt)

                has_trajectory = len(unique_points) > 4

                # --- Trajectory-based classification (when ball tracked) ---
                geo_angle          = None
                geo_name           = None
                predicted_distance = 0

                if has_trajectory:
                    smooth_points = smooth_trajectory(unique_points)
                    if len(smooth_points) > 5:
                        p1 = smooth_points[0]
                        p2 = smooth_points[-1]
                        dx = p2[0] - p1[0]
                        dy = p1[1] - p2[1]
                        if abs(dx) >= 3 or abs(dy) >= 3:
                            raw_angle = math.degrees(math.atan2(dy, dx))
                            if raw_angle < 0:
                                raw_angle += 360
                            geo_angle = (90 - raw_angle) % 360
                            geo_name  = classify_shot(geo_angle)

                            predicted_distance = 0
                            for i in range(1, len(smooth_points)):
                                px1, py1 = smooth_points[i - 1]
                                px2, py2 = smooth_points[i]
                                predicted_distance += math.sqrt(
                                    (px2 - px1) ** 2 + (py2 - py1) ** 2
                                ) * METERS_PER_PIXEL

                            pts = np.array(smooth_points)
                            try:
                                x_vals = pts[:, 0]
                                y_vals = pts[:, 1]
                                if np.max(x_vals) - np.min(x_vals) > 20:
                                    coeffs = np.polyfit(x_vals, y_vals, 2)
                                    poly   = np.poly1d(coeffs)
                                    xs = np.linspace(x_vals[-1], x_vals[-1] + 250, 40)
                                    ys = poly(xs)
                                    future_trajectory = [
                                        (int(fpx), int(fpy))
                                        for fpx, fpy in zip(xs, ys)
                                        if 0 <= int(fpx) < width and 0 <= int(fpy) < height
                                    ]
                            except Exception:
                                pass
                    else:
                        has_trajectory = False  # smoothing ate the points

                # contact_point: first post-contact ball point, or bat_box centre,
                # or tracker position as last resort.
                if has_trajectory and smooth_points:
                    cp = smooth_points[0]
                elif bat_box is not None:
                    cp = ((bat_box[0] + bat_box[2]) // 2, (bat_box[1] + bat_box[3]) // 2)
                elif tracker.initialized:
                    tx, ty = int(tracker.get_position()[0]), int(tracker.get_position()[1])
                    # Reject tracker positions stuck at screen edges (false detections)
                    if 0 < tx < int(width * 0.95) and int(height * 0.03) < ty < height:
                        cp = (tx, ty)
                    else:
                        cp = None
                else:
                    cp = None

                # --- Shot classification ---
                clf_name = None
                clf_conf = 0.0

                if shot_classifier is not None and len(contact_frames) >= 10:
                    clf_name, clf_conf = shot_classifier.classify(contact_frames)

                if has_trajectory and geo_name is not None:
                    # Geometry available: use it as primary, EfficientNet secondary
                    clf_centre = SHOT_CENTRE_ANGLE.get(clf_name or "", 180)
                    angle_ok   = clf_name and _angle_diff(clf_centre, geo_angle) <= 100
                    if clf_conf >= 97.0 and angle_ok:
                        shot_name = clf_name
                        print(f"[F{frame_idx}] EfficientNet override -> {shot_name} "
                              f"({clf_conf:.1f}%)  geo={geo_name} geo_angle={geo_angle:.1f}")
                    else:
                        shot_name = geo_name
                        reason = f"{clf_conf:.1f}% < 97" if clf_conf < 97.0 else \
                                 f"angle diff {_angle_diff(clf_centre, geo_angle):.0f}° > 100"
                        print(f"[F{frame_idx}] geometry -> {shot_name} "
                              f"geo_angle={geo_angle:.1f}  "
                              f"EfficientNet={clf_name} ({reason} — ignored)")
                else:
                    # No trajectory = no reliable event — skip entirely.
                    # EfficientNet without ball trajectory generates too many
                    # false positives (model sees batsman stance between
                    # deliveries and calls it a shot).
                    print(f"[F{frame_idx}] Skip — no ball trajectory "
                          f"({len(unique_points)} pts)")
                    shot_name = None

                if shot_name is not None and cp is not None:
                    already_saved = any(
                        evt.get("frame") == contact_frame and
                        evt.get("ball_id") == ball_id
                        for evt in events
                    )
                    if not already_saved:
                        event = {
                            "ball_id":              ball_id,
                            "delivery_id":          delivery_id,
                            "event":                "ball_bat_contact",
                            "frame":                contact_frame,
                            "timestamp_sec":        round(contact_frame / fps, 2),
                            "wagon_wheel_angle":    round(geo_angle, 2) if geo_angle is not None and geo_angle >= 0 else None,
                            "shot_name":            shot_name,
                            "classifier_confidence": round(clf_conf, 2),
                            "predicted_distance_m": round(predicted_distance, 2),
                            "has_trajectory":       has_trajectory,
                            "contact_point":        {"x": int(cp[0]),  "y": int(cp[1])},
                            "future_trajectory":    [[fpx, fpy] for fpx, fpy in future_trajectory],
                        }
                        events.append(event)
                        print("SHOT EVENT:", event)

                post_contact_points = []
                contact_frames = []

        # =================================================
        # DRAW TRAJECTORY HISTORY
        # =================================================

        for i in range(
            1,
            len(trajectory_history)
        ):

            cv2.line(
                frame,
                trajectory_history[i - 1],
                trajectory_history[i],
                (255, 255, 0),
                2
            )

        # =================================================
        # DRAW FUTURE TRAJECTORY
        # =================================================

        for i in range(
            1,
            len(future_trajectory)
        ):

            cv2.line(
                frame,
                future_trajectory[i - 1],
                future_trajectory[i],
                (255, 0, 255),
                2
            )

        # =================================================
        # DRAW BALL
        # =================================================

        cv2.circle(
            frame,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )

        # =================================================
        # TEXT
        # =================================================

        cv2.putText(
            frame,
            f"Speed: {speed:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if shot_name:

            cv2.putText(
                frame,
                f"{shot_name}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Angle: {geo_angle:.1f}" if geo_angle is not None else "Angle: --",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Distance: "
                f"{predicted_distance:.1f}m",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2
            )

    # =====================================================
    # FPS
    # =====================================================

    cv2.putText(
        frame,
        f"FPS: {display_fps}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =====================================================
    # SHOW
    # =====================================================

    if not HEADLESS:
        cv2.imshow(
            "Cricket Shot Intelligence",
            frame
        )

    out.write(frame)

    if not HEADLESS and (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
        break


# =========================================================
# SAVE
# =========================================================

cap.release()

out.release()

cv2.destroyAllWindows()

with open(EVENTS_JSON, "w") as f:

    json.dump(
        events,
        f,
        indent=4
    )

print(
    f"✅ Saved "
    f"{len(events)} events → "
    f"{EVENTS_JSON}"
)

print(
    f"✅ Saved video → "
    f"{OUTPUT_VIDEO}"
)