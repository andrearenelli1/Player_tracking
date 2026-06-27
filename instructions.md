# Computer vision project

## Player Tracking in Sports - Multi-View tracking and 3D reconstruction

### 1) Annotate Bounding Boxes

Annotate the frames, 1 action from 3 camera views.
It is mandatory to use at least the action you have annotated.
The frames for annotation are at 5fps.

### 2) 2D Tracking

Perform 2D tracking of players, referees, and the ball across frames for each camera view.

#### Setup

- Annotation is performed at 5 fps.
- Tracking is performed at the native frame rate (25 fps).
- Tracking is done independently for each camera in the image plane.

#### Method

- Track the center of bounding boxes over time for each object.
- Maintain consistent identities across frames.

#### Evaluation

- Evaluate tracking performance using standard metrics from state of the art, such as:
  - Intersection over Union (IoU)
  - (optionally) MOTA / MOTP if identity tracking is considered

#### Output

- 2D trajectories for each object in each camera view.
- Numerical results computed on the annotated subset of frames.

### 3) 3D Tracking

Reconstruct 3D trajectories of players (center of bounding boxes) and the ball using triangulation across multiple camera views.

#### Steps

- Use 2D tracked positions (center of bounding boxes) from each camera view at 25 fps.
- Apply stereo rectification to align the image planes of the cameras.
  If rectification is applied, the same transformation must also be applied to the tracked 2D points and GT annotations.
- Use camera projection matrices \(P = K[R|t]\) to project points between image and world space.
- Perform triangulation using corresponding 2D points from at least two views to obtain 3D positions.

#### Output

- 3D positions of players and ball for each frame.
- Reconstructed 3D trajectories over time.

#### Evaluation

- Compare reconstructed 3D trajectories with ground truth (when available).
- Use standard metrics such as:
  - Mean Euclidean Distance (MED)
  - Root Mean Squared Error (RMSE)