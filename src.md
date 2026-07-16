# `src/` overview

Multi-camera (out2, out4, out13) basketball player/ball tracking pipeline: dataset prep → training → 2D detection/tracking → camera calibration → 3D reconstruction → evaluation → visualization.

## Dataset preparation

### `prepare_dataset.py`
Downloads the annotated player/referee/ball dataset from Roboflow into `dataset/`. Remaps class IDs so the ball is class 1 and all persons (players/referees) collapse to class 0, then rewrites `dataset/data.yaml` accordingly (2 classes). Idempotent: skips the download/remap if `dataset/data.yaml` already shows the expected class names.

### `prepare_ball_dataset.py`
Builds a ball-only dataset (`dataset_ball/`) from `dataset/`, used to train a dedicated ball detector. Keeps only native 4K (3840x2160) frames, filters each label file down to ball-class annotations (remapped to class 0), and writes a matching `data.yaml` (1 class). Idempotent like `prepare_dataset.py`.

## Training

### `train.py`
Trains the person/ball YOLO model (YOLOv8m) on `dataset/data.yaml` at 3840x2160, 50 epochs, with a higher classification loss weight (`cls=4.0`) to counter class imbalance.

### `train_ball.py`
Trains a dedicated ball-only YOLO model (YOLOv8m) on `dataset_ball/data.yaml`, full-resolution rectangular images, small batch size (2), mosaic augmentation disabled and scale augmentation biased up — tuned for a small, fast-moving object.

## 2D tracking

### `tracking_2d.py`
Runs Ultralytics YOLO tracking (`model.track`, persistent IDs) over each camera video (out2/out4/out13) at reduced inference resolution, detecting classes `0` (person) and `32` (proxy ball class from the base COCO model, `yolo26n.pt`). Writes per-frame bounding boxes with track IDs to `tracking_results/tracking_2d/trajectories/2d_positions{0,1,2}.csv`. Optional live display with trailing motion paths.

### `sahi_tracking.py`
Alternative ball detector using SAHI (Slicing Aided Hyper Inference) with a custom-trained model (`runs/detect/train-6/weights/best.pt`) to catch small/far-away balls that plain YOLO inference misses. Slices each frame into overlapping tiles, runs detection per tile, and writes detections (no persistent tracking ID, `object_id=-1`) to `tracking_results/tracking_2d/ball_trajectories/2d_positions{0,1,2}.csv`.

### `track_ball.py`
Classical CV ball tracker, independent of any trained model: combines an HSV color mask (tuned for the ball's color) with frame-differencing (motion mask) and filters resulting contours by area. Meant as a lightweight/baseline alternative to the YOLO-based trackers above.

### `hsv_tuner.py`
Interactive OpenCV tool (trackbars for H/S/V thresholds) to visually tune the HSV color range used by `track_ball.py` against live video frames from `out2.mp4`.

## Camera calibration

### `calibration.py`
Calibrates the 3 static cameras independently using manually annotated correspondences between known 3D court coordinates (`CAL_POINTS`, ~12 points per camera) and pixel coordinates. Pipeline per camera: Hartley-normalize points → DLT (SVD) to get an initial projection matrix `P` → denormalize → RQ-decompose into intrinsics `K`, rotation `R`, translation `t` → nonlinear refinement (`scipy.optimize.least_squares`) of `K, R, t` plus radial/tangential distortion coefficients by minimizing reprojection error. Exports all matrices (`K`, `R`, `t`, camera center `C`, distortion, `P`, RMS reprojection error) to `camera_calibration/camera_calibration.csv`.

## 3D reconstruction

### `tracking_3d.py`
Fuses the independent per-camera 2D tracks into 3D trajectories. Steps: undistort 2D points using the calibration; for each camera pair, vote on which object IDs correspond to each other by checking if their triangulated point reprojects within a pixel threshold and wins a Hungarian assignment in enough frames (`MIN_VOTES`); resolve mutual-best votes into cross-camera groups (union-find) and assign each group a `global_id`; for every frame, triangulate (multi-view DLT) the 3D position of each global ID using all camera views that saw it, discarding triangulations that fall outside a plausible court bounding volume. Outputs `tracking_results/tracking_3d/3d_positions.csv` and the `global_id` mapping.

## Evaluation

### `evaluate_2d.py`
Compares 2D tracking output against COCO ground-truth annotations, per camera and per frame (downsampled to the 5 fps annotation rate). Computes an IoU cost matrix (players and ball separately) and solves optimal matching (Hungarian) to get TP/FP/FN, then reports precision, recall, F1, and MOTP (mean IoU of matched pairs) for players and ball.

### `evaluate_3d.py`
Evaluates the reconstructed 3D trajectories. Builds pseudo-ground-truth 3D points by triangulating the COCO ground-truth 2D boxes (undistorted) wherever a category is visible in >=2 cameras in the same frame, then matches them against the reconstructed 3D positions (Hungarian on 3D Euclidean distance, capped by `MAX_MATCH_DIST_M`). Reports coverage, MED (mean Euclidean distance) and RMSE, broken down by category, and writes match details to `tracking_results/tracking_3d/evaluation_3d.csv`.

## Visualization

### `visualize_2d.py`
Plays back each camera video with the 2D tracking results overlaid: bounding boxes (red for ball, green for players) with object IDs and a short trajectory trail per object.

### `visualize_3d.py`
Renders the reconstructed 3D trajectories over a basketball court model (drawn from the same court geometry as `calibration.py`). Supports a static 3D plot (saved as PNG) or an animated version (`--animate`, optionally saved as GIF with `--save`) that also shows the 3 synchronized camera feeds with matching-colored bounding boxes (via the `global_id` mapping) alongside the 3D view.
