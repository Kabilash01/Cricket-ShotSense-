---
name: Pose trigger — completed with YOLOv8-pose; Detectron2 deferred
description: Pose-based wrist-velocity contact trigger is done using YOLOv8-pose. Detectron2 remains a potential upgrade.
type: project
---

**Completed (2026-06-03)**: Wrist-velocity trigger implemented using YOLOv8-pose (`models/pose/yolov8m-pose.pt`), NOT Detectron2.

**Why YOLOv8-pose instead of Detectron2**:
- torch 2.12+cu132 on Blackwell sm_120 has no Detectron2 prebuilt wheel — source build risk was high
- YOLOv8-pose already installed in `ball` env, same 17 COCO keypoints, zero install cost
- Trigger logic is backend-agnostic (`src/detection/pose_detector.py` interface is swappable)

**Detectron2 is still a valid upgrade path if**:
- YOLOv8-pose keypoint quality proves insufficient on sweep/defensive shots (bent-over occlusion)
- To upgrade: swap the backend in `src/detection/pose_detector.py` only — `test3.py` is unchanged

**Current pose architecture**:
- `src/detection/pose_detector.py` — `PoseDetector` (YOLOv8-pose wrapper) + `WristVelocityTrigger`
- Batsman selection: high-conf bat box anchor (conf≥0.55, lower 40% of frame) + IoU continuity
- Wrist speed threshold: 80 px/frame, gated on ball seen within last 15 frames

**Do NOT**:
- Re-implement Detectron2 unless YOLOv8-pose proves insufficient on specific shot types
- Change the `PoseDetector` public interface — test3.py depends on `detect()`, `pick_batsman()`, `wrist_positions()`
