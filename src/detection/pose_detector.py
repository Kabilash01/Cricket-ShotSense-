"""
PoseDetector — wraps YOLOv8-pose for batsman keypoint extraction.

Used by src/test3.py to add a wrist-velocity contact trigger that fires when
the bat-box trigger misses. Backend is swappable: this file currently uses
ultralytics YOLOv8-pose, but the public interface matches what a Detectron2
Keypoint R-CNN wrapper would return, so the backend can be swapped later
without touching test3.py.

Per-frame output is a list of dicts:
    {
        "bbox": (x1, y1, x2, y2),     # int pixel coords
        "conf": float,                 # person detection confidence
        "keypoints": np.ndarray,       # shape (17, 3): (x, y, conf) per joint
    }

COCO keypoint indices (constants exported below):
    9  = left wrist
    10 = right wrist
"""

from collections import deque

import numpy as np
from ultralytics import YOLO


# COCO keypoint indices
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PoseDetector:
    def __init__(self, model_path="yolov8m-pose.pt", conf=0.30, device=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, frame):
        """Run pose detection on one BGR frame. Returns list of person dicts."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        out = []
        for r in results:
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                continue
            kpts_xy = r.keypoints.xy.cpu().numpy()        # (N, 17, 2)
            if r.keypoints.conf is None:
                kpts_conf = np.ones(kpts_xy.shape[:2])
            else:
                kpts_conf = r.keypoints.conf.cpu().numpy()  # (N, 17)
            boxes = r.boxes.xyxy.cpu().numpy()             # (N, 4)
            confs = r.boxes.conf.cpu().numpy()             # (N,)
            for i in range(len(boxes)):
                kpts = np.concatenate(
                    [kpts_xy[i], kpts_conf[i][:, None]], axis=1
                )  # (17, 3)
                x1, y1, x2, y2 = boxes[i]
                out.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": float(confs[i]),
                    "keypoints": kpts,
                })
        return out

    @staticmethod
    def pick_batsman(detections, prev_bbox=None, frame_shape=None,
                     bat_box=None, min_iou=0.20):
        """Pick the most likely batsman from a frame's pose detections.

        Two reliable cues only — no spatial fallbacks that pick the bowler:

          1. Bat box — person whose bbox contains the bat box centre.
             This is the strongest signal and re-anchors tracking every frame
             the bat is visible.

          2. IoU continuity — if we already locked on a person, confirm they
             are still in frame by requiring IoU >= min_iou with their last
             known bbox.

        Returns None when neither cue fires. That means wrist tracking goes
        silent during the bowler's run-up (no bat visible, no prior lock) —
        which is intentional; we don't want wrist triggers during a run-up.
        """
        if not detections:
            return None

        # Cue 1: bat-box containment (strongest — re-anchors on every frame)
        if bat_box is not None:
            bx1, by1, bx2, by2 = bat_box
            bat_cx = (bx1 + bx2) / 2
            bat_cy = (by1 + by2) / 2
            containing = [
                d for d in detections
                if d["bbox"][0] <= bat_cx <= d["bbox"][2]
                and d["bbox"][1] <= bat_cy <= d["bbox"][3]
            ]
            if containing:
                return max(containing, key=lambda d: d["conf"])

        # Cue 2: IoU continuity — stay locked on the same person
        if prev_bbox is not None:
            best = None
            best_iou = 0.0
            for d in detections:
                this_iou = iou(d["bbox"], prev_bbox)
                if this_iou > best_iou:
                    best_iou = this_iou
                    best = d
            if best is not None and best_iou >= min_iou:
                return best

        # No reliable cue — return None rather than guess (avoids bowler picks)
        return None

    @staticmethod
    def wrist_positions(detection, min_conf=0.30):
        """Extract (left_wrist, right_wrist) as (x, y) or None per side."""
        if detection is None:
            return None, None
        kpts = detection["keypoints"]
        lw = kpts[L_WRIST]
        rw = kpts[R_WRIST]
        left = (float(lw[0]), float(lw[1])) if lw[2] >= min_conf else None
        right = (float(rw[0]), float(rw[1])) if rw[2] >= min_conf else None
        return left, right


class WristVelocityTrigger:
    """Detects bat-swing contact moments via wrist-keypoint speed spikes.

    Designed as a third parallel contact trigger alongside bat-box proximity
    and ball-trajectory curvature in src/test3.py — fires when those miss.
    """

    # Frame-to-frame wrist deltas above this are almost certainly identity
    # drift in the pose detector (batsman ↔ umpire/bowler swap), not real
    # wrist motion. A peak cricket-bat swing tops out near 250 px/frame at
    # 1080p / 30fps. Anything bigger gets discarded.
    MAX_SANE_DELTA_PX = 300.0

    def __init__(self, speed_threshold=40.0, history=5, min_kpt_conf=0.30):
        self.speed_threshold = speed_threshold
        self.min_kpt_conf = min_kpt_conf
        self.left_history = deque(maxlen=history)
        self.right_history = deque(maxlen=history)

    def update(self, batsman_detection):
        """Append latest wrist positions; return the max observed wrist speed."""
        lw, rw = PoseDetector.wrist_positions(
            batsman_detection, min_conf=self.min_kpt_conf
        )
        self.left_history.append(lw)
        self.right_history.append(rw)
        return self.max_speed()

    def max_speed(self):
        return max(
            self._side_max_speed(self.left_history),
            self._side_max_speed(self.right_history),
        )

    @classmethod
    def _side_max_speed(cls, history):
        max_delta = 0.0
        prev = None
        for pos in history:
            if pos is not None and prev is not None:
                dx = pos[0] - prev[0]
                dy = pos[1] - prev[1]
                d = (dx * dx + dy * dy) ** 0.5
                # Skip unphysical jumps caused by identity drift
                if d <= cls.MAX_SANE_DELTA_PX and d > max_delta:
                    max_delta = d
            if pos is not None:
                prev = pos
        return max_delta

    def should_trigger(self):
        return self.max_speed() >= self.speed_threshold

    def reset(self):
        self.left_history.clear()
        self.right_history.clear()
