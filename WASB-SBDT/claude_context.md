# Project context: Basketball Ball Tracking

## Goal
Track a basketball across **multiple independent views** (N cameras, each
view processed as an independent 2D detector — no 3D fusion for now).
Project **not connected** to the user's LIMO robot/ROS2 project (two
separate projects, don't conflate them).

**This repo's scope within the larger project** (as of session 4): the
larger project has 3 steps — (1) 2D tracking of ball + players on every
view (players via YOLO, elsewhere, works well), (2) performance
evaluation vs ground truth, (3) 3D tracking + performance eval. **This
repo covers only the ball side of step 1+2**: `track_ball.py` produces
the per-frame 2D detections (CSV) that step 3 needs, `eval_wasb.py`
produces the performance numbers for step 2. It was trimmed down
accordingly (see session 4) — no longer a general multi-sport/
multi-model framework, just this one pipeline.

## Debugging history (how we got here)

### 1. Starting point: fine-tuned YOLO was failing systematically
- Very low recall, ball almost never detected
- Identical failure whether the ball was still/sharp or moving/blurred
- **Root cause identified**: ball <15px in diameter on FHD/4K frames.
  YOLO's resize to 640x640 plus the backbone's aggressive downsampling
  (stride up to 32) makes the object disappear before the detection head
  can see it. Not a training problem but a structural/architectural one.
- SAHI (tiling) tried but didn't fix it (tile size likely not tuned for
  such an extreme ball/frame ratio).

### 2. SOTA search for ball tracking
Candidates examined:
- **TrackNet** (original, tennis/badminton) — heatmap-based, multi-frame stack
- **TOTNet** — designed for occlusion (3D conv, visibility-weighted loss),
  training code available: https://github.com/AugustRushG/TOTNet
- **WASB-SBDT** — general-purpose multi-sport baseline, **has a pretrained
  checkpoint specific to basketball (NBA dataset)**, training code NOT
  public (eval only). Repo: https://github.com/nttcom/WASB-SBDT

**Decision made**: start with WASB (basketball checkpoint already
available) for a quick zero-shot test, before investing in
training/fine-tuning TOTNet.

### 3. Environment setup
- Docker Ubuntu, image built from the WASB-SBDT repo's Dockerfile
- GPU: RTX 3060, 6GB VRAM
- Container launched with: `docker run -it --gpus all -v $(pwd):/workspace wasb-sbdt`
  (host repo root = `/workspace` inside the container)
- `nvidia-container-toolkit` configured and working
- Weights downloaded with `sh setup_scripts/setup_weights.sh` (inside
  `src/`), end up in `/workspace/pretrained_weights/`, including
  `wasb_basketball_best.pth.tar`
- **Note**: pip packages installed by hand inside the container (e.g.
  `attrdict`) are lost if the container is recreated from scratch with
  `docker run`. To keep working, use `docker start -ai <container_id>` on
  the same container instead of recreating a new one. Even better: add
  them to the Dockerfile.

### 4. Model architecture (for reference)
- Config: `src/configs/model/wasb.yaml` → `name: hrnet` (HRNet backbone)
- Model file: `src/models/hrnet.py`, factory in `src/models/__init__.py`
- **Input**: `(batch, 9, 288, 512)` — 3 consecutive RGB frames
  (`frames_in: 3`) concatenated on the channel axis, resized to 512(w)x288(h)
- **Output**: dict `{scale: tensor(batch, 3, 288, 512)}` — 3 heatmaps (one
  per frame in the window), **raw logits, needs an explicit sigmoid** (not
  included in the model's forward — must be applied separately, as their
  `detectors/postprocessor.py` does: `hms_ = preds_.sigmoid_()`)
- **Input normalization**: standard ImageNet —
  `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` after `ToTensor()`
- **Checkpoint loading**: the weights in `wasb_basketball_best.pth.tar` are
  wrapped under the key `'model_state_dict'` (not `'state_dict'` or
  `'model'` as more common conventions — watch out if writing a loader
  from scratch)
- The model config must be wrapped with `AttrDict` (the `attrdict`
  library) before passing it to `build_model()`, because `hrnet.py`'s
  internal code accesses fields both with dict-style
  (`cfg['MODEL']['EXTRA']`) and attribute-style (`cfg.MODEL.EXTRA`)
  notation in different places
- **Official heatmap decoding** (was in `detectors/postprocessor.py` of
  the original repo, **removed from this repo in session 4's cleanup**
  since we don't use the original hydra framework at all — replicated
  and kept in `src/heatmap_decode.py` instead): not a simple argmax, but
  connected components on the thresholded heatmap + weighted centroid of
  the blob's pixels (`_detect_blob_concomp`). Default threshold in their
  config: `score_threshold: 0.5` (tuned for NBA broadcast, turned out to
  be TOO HIGH for our domain, see below)

## Scripts written so far

### `track_ball.py` (originally `infer_video.py`, renamed in session 4)
Copied into `/workspace/src/` in the repo, to use the relative imports
(`from models import build_model`). **Updated in session 3** to use the
same validated blob decoding and tiling as `eval_wasb.py` (see below) —
originally used a simpler argmax decoding, now superseded. **Renamed and
extended with CSV export in session 4** (see below) once its role became
"the project's 2D ball tracking deliverable", not just a debug tool.

Does:
1. Loads the HRNet model + checkpoint weights (handles the
   `model_state_dict` key)
2. Reads a video with OpenCV, 3-frame sliding window (optionally per-tile,
   see tiling below)
3. Preprocessing (resize to 512x288 + ImageNet normalization) per frame/tile
4. Forward pass, sigmoid on the output, decoding with the blob-based
   method from `heatmap_decode.py` (connected components, see below),
   default confidence threshold **0.02** (calibrated against ground truth
   in session 2 — NOT the model's official 0.5, see below)
5. Prints a recall sweep at several candidate thresholds (0.001 → 0.5),
   unsupervised (no ground truth, just fraction of frames above each
   threshold — the real recall/precision sweep lives in `eval_wasb.py`)
6. With `--debug-dir`: saves the highest/lowest confidence frames to disk
   (resized to 960px width to avoid running out of RAM on 4K video) for
   manual visual inspection

### Early zero-shot results (test on a lab video, 4K, 525 frames)
- With the official 0.5 threshold: **recall ~0.2%** (essentially nil)
- Observed confidence range: min=0.0114, max=0.5437, mean=0.0274, median=0.0134
  → a very compressed "background noise" around 0.011-0.013, with a tail
  of genuine peaks up to 0.54
- **Visual inspection confirmed by the user**: the highest-confidence
  frames (top-10) correctly show the ball, well centered by the predicted
  circle. The lowest-confidence frames (bottom-5) are indeed frames with
  no visible ball.
  → **The signal is genuine, not random noise.** The problem is purely
  threshold calibration: the one tuned for NBA broadcast (0.5) doesn't
  apply to the lab domain.
- Threshold sweep (test video, 523 valid frames):
  - threshold 0.01 → 100% recall (but likely overcounting, at the noise floor)
  - threshold 0.02 → 30.8% recall
  - threshold 0.05 → 8.4% recall
  - threshold 0.5 (official) → 0.2% recall

## Session 2: real quantitative evaluation (recall + precision on ground truth)

### Evaluation dataset
- Annotations in `annotations/_annotations.coco.json` (Roboflow COCO
  export), with the images themselves copied into the same folder (309
  4K PNGs, ~1.2GB, **not tracked in git**, only the json is — see
  `.gitignore`)
- 309 annotated frames total across 3 videos (`out2`=99, `out4`=105,
  `out13`=105), 191 with a ball (`category_id` of the `"Ball"` class), 118
  without (negative frames, useful for precision)
- The annotations are real images from the subset assigned to the work
  group (not the full ~2000-image dataset available for training — here
  we only need the subset for zero-shot evaluation, no training)
- **Important**: the annotations are sampled at **5fps**, the source
  videos are at **25fps**. The annotated-frame → video-index mapping was
  verified empirically (pixel-by-pixel comparison, see
  `src/check_frame_offset.py`): `video_idx = (frame_num - 1) * 5 + 2`
  (constant +2 offset across all 3 videos, not the naive
  `(frame_num-1)*5` one would expect)

### Evaluation script: `src/eval_wasb.py`
For each annotated frame, reconstructs the triplet of 3 consecutive
frames from the original video (WASB needs temporal input), runs
inference, decodes the heatmap with the official blob method (connected
components, replicated from `detectors/postprocessor.py`, with one
modification — see below), and computes the **TrackNet/TrackNetV2-style**
metric: TP/FN/FP2(wrong position)/FP1(false alarm)/TN, with
Precision=TP/(TP+FP1+FP2), Recall=TP/(TP+FP2+FN).

Useful options added:
- `--exclude-videos out13` to isolate performance by excluding a video
- `--mask-rect video:x0,y0,x1,y1` to zero out heatmap regions (fixed
  graphical elements overlaid on a static camera)
- `--overlay-dir` saves images with the GT bbox (green) + prediction
  (red) + category (TP/FN/FP2/FP1/TN) in the file name, for visual
  inspection
- `--output-csv` saves the threshold→metrics table

### Bugs found and fixed during development
1. **Wrong frame-annotation offset**: I assumed `(N-1)*5`, the correct
   value is `(N-1)*5 + 2` (`--frame-offset` parameter). Without the fix,
   almost every prediction was compared against the ball at the wrong
   position → catastrophic initial results (very high FP2) that didn't
   reflect the model's real performance.
2. **Blob scoring criterion**: the official method picks the blob with
   the highest total sum; at the low thresholds required for our
   uncalibrated domain, this favors large-but-weak regions over
   small-but-confident ones. Changed to "highest peak" (area-invariant).
   Smaller impact than expected — it wasn't the main cause of the problems.

### Final results (after the fixes, threshold 0.02 = best for aggregate F1)
| video | precision | recall | F1 | note |
|---|---|---|---|---|
| out2  | 0.485 | 0.364 | 0.416 | close-up shot |
| out4  | 0.808 | 0.467 | **0.592** | close-up shot, the best one |
| out13 | 0.000 | 0.000 | 0.000 | full-court fisheye — **unusable** |

**Diagnosis of `out13`** (done with the overlay + quantitative analysis):
- Initially looked like a graphical confuser (a fixed logo/clock in the
  top-right corner, original coordinates ~x∈[3795,3836] y∈[582,601],
  found by searching for the red pixels drawn in the overlay). Masking it
  (`--mask-rect out13:3700,500,3840,680`) eliminated almost all the FP2s
  but **TP stayed at 0** — it wasn't the real cause, just an additional
  symptom.
- **Real, quantified cause**: the ball in `out13` has a mean bbox
  diagonal of **55px** (min 40, max 78) on 3840x2160 frames, versus 173px
  (out2) and 133px (out4). After the mandatory resize to 512x288 (÷7.5)
  it becomes **~7px** — below HRNet's minimum detectable size. It's the
  exact same structural phenomenon that made YOLO fail at the start of
  the project (see section 1), just less extreme. Not fixable with a
  threshold/mask: would need tiling on the heatmap or a crop/zoom before
  the network.

**Operational conclusion (session 2)**: WASB zero-shot is **usable** on
close-up shots like `out2`/`out4` (F1 up to 0.59, no training needed).
Full-court/small-ball shots like `out13` should be excluded or need a
different approach (tiling/crop), not simple recalibration. **Superseded
by session 3 below**, which partially recovers `out13` with tiling.

## Session 3: tiling to partially recover `out13`

### Why a static crop doesn't work
Checked where the ball actually moves in `out13`'s ground truth: x spans
almost the full frame width (386 to 3126 out of 3840), y is a narrow band
(828 to 1319 out of 2160). Since x-coverage is nearly full-width, a fixed
crop/zoom risks cutting the ball out at other moments of a real game —
not just in this test clip. Ruled out static cropping in favor of tiling,
which covers the whole frame without this risk.

### New shared modules
- **`src/tiling.py`**: `compute_tile_grid(frame_w, frame_h, n, overlap_frac)`
  builds an `n x n` grid of overlapping tiles. Using the same `n` for rows
  and columns is what keeps each tile's aspect ratio at 16:9 automatically
  (the frame itself is 3840x2160 = 16:9, matching the network's 512x288
  input ratio — no distortion). `overlap_frac` (default 0.1) inflates each
  tile so a ball sitting on a grid line still falls well inside at least
  one tile. Also has `clip_rect_to_tile()` to remap `--mask-rect` regions
  (given in full-frame coordinates) onto individual tiles.
- **`src/heatmap_decode.py`**: added `get_ranked_detections(record, thresh)`,
  a unified interface that works whether `record` has a single full-frame
  heatmap or a list of per-tile heatmaps (`record['tiles']`), always
  returning detections in original-image pixel coordinates, merged and
  ranked by peak confidence across tiles if applicable. `track_ball.py`
  now always goes through this (with `n=1` — a single tile spanning the
  whole frame — being exactly the old non-tiled behavior, so there's no
  special-casing/regression for videos that don't need tiling).

### How it works
For each requested frame, crop+resize+3-frame-window independently **per
tile** (so each tile gets its own forward pass — N-fold slower per frame,
N = number of tiles), then merge all tiles' detections into one ranked
list in original-pixel coordinates. `eval_wasb.py` got `--tile-videos`,
`--tile-n` (grid size, default 2), `--tile-overlap` (default 0.1).
`track_ball.py` got the same via `--tile-n`/`--tile-overlap` (no separate
video list since it only processes one video per run).

### Results on `out13` (mask-rect for the corner-graphic confuser still applied)
| config | tile size (px) | ball size in network space | precision | recall | F1 |
|---|---|---|---|---|---|
| no tiling (session 2) | 3840x2160 (full frame) | ~7px | 0.000 | 0.000 | 0.000 |
| tiling 2x2 (4 tiles) | ~2112x1188 | ~13px | 0.400 | 0.118 | 0.182 |
| **tiling 3x3 (9 tiles)** | ~1408x792 | ~20px | **0.905** | 0.186 | **0.309** |

Clear trend: smaller tiles → less downsampling → bigger apparent ball →
better metric, exactly as the resolution diagnosis from session 2
predicted. `recall_oracle` at low thresholds is meaningfully higher than
the actual recall with more tiles (e.g. 0.608 vs 0.245 at threshold 0.01
with 3x3) — a side effect of tiling: more tiles means more independent
candidate peaks competing for "highest confidence", so a false peak in
one tile more often outranks the true peak in another. Fixing this
properly would need temporal consistency (a tracker across frames), not
just more tiles.

**Decision**: stopped at 3x3 (didn't try 4x4) — diminishing returns
expected (16 tiles, even slower, worse ranking-ambiguity effect) without
a tracker to resolve the cross-tile competition. 3x3 is the recommended
setting for `out13`-like wide shots: doesn't match `out2`/`out4`'s F1,
but is a real recovery from "completely unusable" to "high precision,
partial recall" (useful if false positives are more costly downstream
than missed detections, e.g. if a tracker downstream can tolerate gaps
but not confirm wrong locations).

## Session 4: repo restructuring (2D-tracking module for the larger project)

User clarified the larger project's structure (see Goal section above)
and asked to turn this repo into just the 2D-ball-tracking module,
throwing away everything not needed for that. Also asked about
git-submoduling this repo into the main project repo.

### Submodule decision: no
Recommended against a git submodule (asked two clarifying questions
first: does the main repo exist yet — no, not decided; do we need to
pull upstream nttcom updates in the future — no, this is "ours" now).
Without the upstream-sync use case, a submodule only adds workflow risk
for a small student team (forgetting `git submodule update --init`,
double-commit dance) with no matching benefit. **Recommendation for
later**: when the main project repo exists, either `git subtree add`
(preserves this repo's commit history) or a plain copy (simpler, history
not needed) directly into a subfolder — not a submodule.

Also found and cleaned up before any of this: 5 stray empty files
(`--annotations`, `--checkpoint`, etc. — accidental artifacts from a
command pasted outside the container, each `--flag` interpreted as a
filename by bash) and confirmed the whole session's work was still
uncommitted with `origin` pointing at the read-only upstream
(`nttcom/WASB-SBDT`) — flagged for the user to fix before pushing
anywhere, not yet resolved as of this writing.

### New feature: per-frame CSV export (`track_ball.py`)
The project's step 3 (3D tracking) needs a continuous per-frame 2D
detection stream, not just the 309 annotated frames used for evaluation.
Added `--csv-output` to `track_ball.py`: one row per **video frame**
(every frame, not just annotated ones — confirmed with the user), columns
`frame,x,y,w,h,confidence,visible`. Two design decisions confirmed with
the user:
- **Box format, not a point**: even though the model only predicts a
  point, output a synthetic `(w,h)` box (matching the format a player
  detector like YOLO would produce, for uniform downstream handling).
  Implemented by having `decode_blob_concomp` (in `heatmap_decode.py`)
  also return each blob's own pixel-extent bounding box (not a fixed/
  arbitrary size) — centered on the same weighted centroid used as the
  point. `get_ranked_detections` now returns `(xy, peak, (w, h))` tuples
  instead of `(xy, peak)`; all callers in `eval_wasb.py` updated for the
  extra tuple element.
- **Every frame of the video**, not just the 309 annotated ones (the
  annotated subset remains what `eval_wasb.py` scores against).

The very first and last frame of each video have no CSV row (the 3-frame
sliding window has no center there yet/anymore) — negligible (2 frames),
noted in the script's log output rather than silently dropped.

### Rename: `infer_video.py` → `track_ball.py`
Reflects its new role as the actual pipeline deliverable (CSV + overlay
video), not just a debug/visualization tool. `eval_wasb.py`'s import
updated accordingly (`from track_ball import load_model, preprocess_frame`).

### Cleanup: what got removed and why
Checked first what the kept scripts actually import (`hrnet.py` has ZERO
dependencies on the rest of the original repo — not even the other model
files); everything below was dead weight for this fork's narrower scope:
- **Entire original hydra framework**: `src/dataloaders/`, `src/datasets/`,
  `src/detectors/`, `src/losses/`, `src/optimizers/`, `src/runners/`,
  `src/trackers/`, `src/utils/`, `src/main.py`, `src/configs/` (kept only
  `configs/model/wasb.yaml` — **almost deleted by mistake with a `rm -rf
  src/configs`, caught immediately since track_ball.py/eval_wasb.py both
  need it, restored with `git checkout HEAD -- src/configs/model/wasb.yaml`**)
- **Other model architectures**: `src/models/{ballseg,deepball,monotrack,
  resnetv1b,resunet2d,segbase,unet2d,unet2d_parts}.py` — only `hrnet.py`
  is used. `src/models/__init__.py` trimmed to just the HRNet factory entry.
- **34 of 35 checkpoints** in `pretrained_weights/` (all other models/
  sports) — kept only `wasb_basketball_best.pth.tar`. 590MB → 6MB.
- **3 of 4 files** in `src/setup_scripts/` (per-sport dataset download
  scripts, not needed since we use our own videos/annotations) — kept
  `setup_weights.sh`, trimmed to download only the WASB basketball checkpoint.
- **`GET_STARTED.md`, `MODEL_ZOO.md`** (docs for the full original
  framework) — `README.md` rewritten from scratch for this fork's scope.
- **Stale outputs**: all `eval_results*.csv` (superseded by fresh runs),
  `overlay_debug/` (worse: labeled with the FP1/FP2 convention from
  *before* it was corrected to match the official evaluator — actively
  wrong now, not just stale), `videos/debug_frames/` (old test),
  `videos/out13_result.mp4` (generated with the pre-fix argmax decoder
  and threshold 0.5, misleading to keep). `out2_result.mp4`/
  `out4_result.mp4` were kept (still valid, same decode logic as now) but
  ideally should be regenerated together with fresh CSVs for consistency
  — not yet done as of this writing.
- Many of the files above were owned by `root` (written from inside the
  container) and needed `docker exec heuristic_hoover rm ...` since the
  host user has no permission to delete them directly.

### Dockerfile slimmed too
Removed `hydra-core`, `tqdm`, `scikit-learn`, `scikit-image`, `pandas`,
`einops`, `timm`, `matplotlib` from the pip install (none used by the
surviving code). Added `attrdict` and `PyYAML` **permanently** (finally
fixes the recurring annoyance from session 1 of having to
`pip3 install attrdict` by hand every time the container is recreated).
Also fixed a pre-existing `MAINTAINER` deprecation lint warning
(→ `LABEL maintainer=...`).

### Verification
After cleanup: all scripts still parse (`ast.parse`), and inside the
container `build_model()` + importing `track_ball`, `eval_wasb`,
`heatmap_decode`, `tiling` all still work end-to-end.

## Session 5: imported into the main project repo, output paths + CSV schema aligned

User plain-copied `WASB-SBDT/` (own nested `.git`, not yet reconciled with
the main repo's git — still shows as a single untracked dir in `git
status` there) into the main project repo, as
`<main repo root>/WASB-SBDT/`. This session did the integration cleanup:

- **Removed the duplicate `videos/` folder**: `WASB-SBDT/videos/` had
  copies of `out2.mp4`/`out4.mp4`/`out13.mp4` identical (md5-verified) to
  the main repo's own `videos/`. Kept only the main repo's copy, deleted
  `WASB-SBDT/videos/` entirely (after moving out its result artifacts,
  see below). Source videos are now reached via `/workspace/videos` once
  the container mounts the main repo root (see Docker section below).
- **Result artifacts relocated** into the main project's own
  `tracking_results/tracking_2d/` layout, split by purpose:
  - overlay `.mp4`s (visual QC) → `tracking_results/tracking_2d/evaluation/`
  - per-frame detection CSVs (actual tracking data) →
    `tracking_results/tracking_2d/ball_trajectories/`
  This mirrors the main project's own convention: `tracking_2d/
  trajectories/` holds player positions (`tracking_players_2d.py`),
  `tracking_2d/ball_trajectories/` now holds ball positions, `tracking_2d/
  evaluation/` holds visual/metric evaluation outputs.
- **CSV schema changed to match the main project's convention**: the main
  project's `src/evaluate_2d.py` and `src/track_ball_classic.py` (a
  simple classic-CV ball detector living in the main repo, background-
  subtraction based) both read/write ball trajectory CSVs as
  `frame,cam_id,class_id,object_id,u,v,w,h` (same schema
  `tracking_players_2d.py` uses for players) — not the `frame,x,y,w,h,
  confidence,visible` schema `track_ball.py` used before. Changed
  `track_ball.py` directly (not a separate adapter script) to write the
  shared schema: added a required `--cam-id` argument (stamped into the
  `cam_id` column; convention `cam_0`=`out2`, `cam_1`=`out4`,
  `cam_2`=`out13`, matching `CAMERAS` in `tracking_players_2d.py`/
  `track_ball_classic.py`), `class_id` always `0`, `object_id` always
  `-1` (no per-instance tracking — matches `track_ball_classic.py`'s
  convention, since there's only ever one ball). **Rows for
  undetected frames are now omitted entirely** (previously written with
  empty `x,y,w,h` + `visible=False`), matching `track_ball_classic.py`'s
  behavior of only emitting a row when something was actually found.
  `run_tracking.sh`, `run_eval.sh`, `README.md` updated to match.
- **Docker mount convention changed**: since `WASB-SBDT` is now a
  subfolder of the main repo, the container must mount the **main repo
  root** at `/workspace` (not `WASB-SBDT`'s own root as before), so it
  can reach both the shared `videos/` and `tracking_results/`:
  `docker run -it --gpus all -v <main repo root>:/workspace -w
  /workspace/WASB-SBDT/src wasb-sbdt`. All checkpoint/config paths in
  `run_tracking.sh`/`run_eval.sh`/`README.md` got an extra `/WASB-SBDT`
  segment accordingly.
- **Stale image caught and rebuilt**: the existing `wasb-sbdt` docker
  image (and the old `heuristic_hoover` container, itself mounted on the
  pre-import standalone path `.../computer_vision/WASB-SBDT`, now
  obsolete) predated session 4's Dockerfile fix that bakes in `attrdict`/
  `PyYAML` — first attempt to rerun `run_tracking.sh` in a fresh
  container failed with `ModuleNotFoundError: No module named
  'attrdict'`. Rebuilt the image (`docker build -t wasb-sbdt .`) to
  actually pick up that Dockerfile fix, then reran successfully. Lesson:
  an image being older than a Dockerfile fix is an easy trap — the fix
  being in the Dockerfile doesn't mean the last-built image has it.
- **`WASB-SBDT/annotations/` deduped too** (same session, follow-up):
  spotted as *also* a duplicate of the main repo's own `annotations/`
  (md5-verified identical `_annotations.coco.json` + all 309 images) —
  deleted `WASB-SBDT/annotations/` entirely. Code updated to read from
  the outer one instead: `run_eval.sh` and `README.md`'s Case C example
  now pass `--annotations /workspace/annotations/_annotations.coco.json`
  (was `/workspace/WASB-SBDT/annotations/...`); `check_frame_offset.py`
  needed no path change (already used a bare relative `'annotations/...'`
  — its docstring was updated to say "run from the MAIN repo root", not
  WASB-SBDT's). `WASB-SBDT/.gitignore`'s now-pointless `annotations/*.png`/
  `*.jpg` entries removed (nothing left there to ignore).
  `WASB-SBDT`'s nested `.git` vs the main repo's git tracking is still
  unresolved (see session 4's original submodule-vs-subtree-vs-copy
  discussion) — the plain copy performed by the user before this session
  did not address it.
- **CSVs actually regenerated** (rebuilt image, `sh run_tracking.sh` on
  the new mount, all 3 videos, exit code 0): `ball_trajectories/
  out{2,4,13}_detections.csv` are now in the new schema, `evaluation/
  out{2,4,13}_result.mp4` refreshed to match. Ball-detected fraction at
  the baked-in threshold (0.02): `out2` 17.4%, `out4` 23.7%, `out13`
  (tiled 3x3) 100% — `out13`'s 100% is expected/known-optimistic (more
  tiles → more competing candidate peaks, see session 3's
  `recall_oracle` note; the real precision/recall against ground truth is
  what session 2/3's table reports, not this unsupervised fraction).
  Output files ended up owned by `root` (written from inside the
  container) — readable but not `chown`-able by the host user without
  `sudo`, consistent with session 4's note about container-written files.

## Next step (pending)

1. **Fix git remote + integrate `WASB-SBDT`'s nested `.git`** with the
   main repo (currently a plain copy with its own `.git`, showing as one
   untracked dir in the main repo's `git status`; `origin` inside it
   still points at the read-only upstream `nttcom/WASB-SBDT`) — decide
   subtree vs plain copy (no submodule, see session 4) and commit
2. ~~Dedupe `WASB-SBDT/annotations/` vs the main repo's own
   `annotations/`~~ — done in session 5
3. Assess whether `out13`'s tiled 3x3 result (precision 0.90, recall
   0.19) is good enough for downstream use as-is, or whether it's worth
   building a lightweight temporal tracker to resolve the cross-tile
   ranking ambiguity and push recall higher
4. With threshold 0.02 confirmed as reasonable on `out2`/`out4` (no
   tiling needed there), assess whether the achieved precision/recall
   (F1 ~0.42-0.59) is enough for downstream use, or whether fine-tuning
   is still needed (WASB's training code isn't public → consider TOTNet,
   which has one, or reverse-engineer WASB's training loop from the
   repo's issues)
5. Possibly extend the evaluation to the full dataset (~2000 images) if
   real training/fine-tuning is decided on
6. Run `src/evaluate_2d.py` (main repo) now that `ball_trajectories/`
   CSVs are in the schema it expects, to get real precision/recall/F1/
   MOTP numbers for the ball on the main project's own evaluator (as
   opposed to `eval_wasb.py`'s TrackNet-style numbers within this repo)

## Reference files and paths (inside the container, root = `/workspace` = the
MAIN PROJECT repo root as of session 5, not `WASB-SBDT`'s own root — see
Docker section below) — post session-5 reorg

- `/workspace/WASB-SBDT/src/track_ball.py` — runs the tracker on a full
  video → per-frame CSV (`--csv-output`, schema `frame,cam_id,class_id,
  object_id,u,v,w,h`, one row per DETECTED frame, `--cam-id` required) +
  overlay `.mp4` (`--output`); the project's 2D-tracking deliverable
- `/workspace/WASB-SBDT/src/eval_wasb.py` — quantitative evaluation vs
  ground truth (precision/recall/F1/accuracy/RMSE, matches the official
  evaluator convention — see session "SOTA evaluation method" notes below)
- `/workspace/WASB-SBDT/src/heatmap_decode.py` — shared blob decoding
  (`decode_blob_concomp`, modified from the official method to rank by
  peak instead of sum, and to also return each blob's pixel extent for
  the CSV's synthetic box) + `get_ranked_detections` (unifies single-
  frame and tiled detection, always original-image coordinates) +
  `apply_mask_rects`
- `/workspace/WASB-SBDT/src/tiling.py` — tile-grid computation +
  mask-rect remapping for tiled inference (session 3)
- `/workspace/WASB-SBDT/src/check_frame_offset.py` — verifies the
  frame↔annotation offset pixel-by-pixel, re-run if the annotations get
  re-exported
- `/workspace/WASB-SBDT/src/models/hrnet.py` + trimmed
  `src/models/__init__.py` — only surviving model architecture
- `/workspace/WASB-SBDT/src/configs/model/wasb.yaml` — model config (only
  survivor of `src/configs/`)
- `/workspace/WASB-SBDT/pretrained_weights/wasb_basketball_best.pth.tar`
  — only surviving checkpoint (was 35 files/590MB, now 1 file/6MB)
- `/workspace/videos/` — 25fps source videos (out2.mp4, out4.mp4,
  out13.mp4), shared with the rest of the main project (session 5: no
  longer duplicated inside `WASB-SBDT/`)
- `/workspace/tracking_results/tracking_2d/ball_trajectories/` — per-frame
  ball detection CSVs (session 5: moved here from `WASB-SBDT/videos/`,
  schema aligned with `tracking_players_2d.py`/`track_ball_classic.py`)
- `/workspace/tracking_results/tracking_2d/evaluation/` — overlay result
  `.mp4`s + `eval_wasb.py`'s threshold→metrics CSV (session 5: moved here
  from `WASB-SBDT/videos/` / `/workspace/eval_results.csv`)
- `/workspace/annotations/_annotations.coco.json` — ground truth (+
  images, not tracked in git). Session 5: no longer duplicated inside
  `WASB-SBDT/` (was `WASB-SBDT/annotations/`, deleted; `run_eval.sh`/
  `README.md` updated to point here)
- `WASB-SBDT/README.md` — rewritten for this fork's scope in session 4,
  paths updated in session 5 for the new mount convention
- Removed entirely (session 4): `src/dataloaders/`, `src/datasets/`,
  `src/detectors/` (including the original `postprocessor.py` the blob
  decoding was ported from), `src/losses/`, `src/optimizers/`,
  `src/runners/`, `src/trackers/`, `src/utils/`, `src/main.py`, most of
  `src/configs/`, most of `src/models/`, 34/35 checkpoints,
  `GET_STARTED.md`, `MODEL_ZOO.md`, stale debug outputs — see session 4
  above for the full list and why

## Docker setup — watch the mount
Since session 5, `WASB-SBDT` lives as a subfolder of the main project
repo, and the container must mount the **main repo root** (not
`WASB-SBDT`'s own root) so it can reach both the shared `videos/` and
`tracking_results/`:
```bash
cd <main project repo root>
docker run -it --gpus all -v $(pwd):/workspace -w /workspace/WASB-SBDT/src wasb-sbdt
```
Working dir is on the mounted path (not on `/root/src`, which is a static
copy taken at build time). `attrdict`/`PyYAML` are baked into the
Dockerfile (session 4) — but **check the image was actually rebuilt after
that fix landed**, not just that the Dockerfile has it: session 5 hit
`ModuleNotFoundError: No module named 'attrdict'` from a stale
pre-session-4 image and had to `docker build -t wasb-sbdt .` again before
it worked. **Always verify the mount with `docker inspect <container>
--format '{{json .Mounts}}'`** before assuming the files are in sync —
session 5 also found a leftover container (`heuristic_hoover`) mounted on
the old pre-import standalone path from before `WASB-SBDT` was copied
into the main repo.

## User's working style (Marco)
- Prefers step-by-step explanations with the "why", not just ready-made
  solutions
- Communicates in Italian in chat (this file and code comments are kept
  in English per the user's request)
- Graduate student in computer vision
- Builds understanding through targeted follow-up questions
- Note: **this project is not connected to his LIMO/ROS2 project**
  (separate projects, don't assume any connection between them)
