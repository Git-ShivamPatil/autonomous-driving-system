# Design

Three parts, each measured end to end:

1. **Steering.** A PilotNet-style CNN drives a car in the MetaDrive simulator from a single forward
   camera. It is trained on frames collected with DART noise injection and evaluated in closed loop
   on 1,000 procedurally generated test roads.
2. **Perception.** One YOLO11 backbone with three heads: object detection (12 classes), drivable-area
   segmentation and lane-line segmentation, trained on BDD100K. A classic OpenCV lane pipeline is
   the baseline for the lane head.
3. **Edge export.** The models are exported to TensorFlow SavedModel and TFLite INT8 and benchmarked.
   The steering model is also evaluated open loop on a real dashcam dataset (Sully Chen) to measure
   the sim-to-real gap.

Every advertised number is recorded in [CLAIMS.md](../CLAIMS.md) with its command, commit and
hardware.

## Datasets

| Dataset | Kind | Role |
|---|---|---|
| MetaDrive 0.4.3, collected here | Synthetic, procedurally generated roads | Steering training, validation and closed-loop test |
| BDD100K | Real dashcam, 100k images | Detection, drivable area and lane training and evaluation |
| Sully Chen driving dataset | Real dashcam with steering-wheel angle | Cross-domain open-loop evaluation of the steering model |

## Steering

### Road suite and seed splits

MetaDrive generates a road network from each seed. The splits are disjoint seed ranges:

| Split | Seeds | Use |
|---|---|---|
| train | 1000–8999 | DART data collection, consumed in order until 75,000 frames |
| val | 9000–9199 | Collection until 6,000 frames; early stopping |
| test | 0–999 | The 1,000 closed-loop test scenarios |

Each road has 4 blocks drawn from Curve 0.6, Straight 0.2, InRampOnStraight 0.1 and
OutRampOnStraight 0.1, with random lane width, random lane count and traffic density 0.1.
Intersections and roundabouts are excluded on purpose: the correct steering there depends on the
route, which a camera-only model cannot see.

### Expert and data collection

The expert is MetaDrive's IDM policy with lane changes disabled, constructed after every reset.
Collection uses DART (Laskey et al. 2017): the executed steering is the expert's steering plus AR(1)
noise (rho 0.9, per-episode stationary sigma drawn from {0, 0.05, 0.10, 0.15} by the seed), while the
recorded label is the expert's clean steering in the state actually visited. The car drifts and the
labels teach the recovery; no side cameras or steering offsets are involved.

Every second step is saved (0.2 s apart), frames below 2 km/h are skipped, and each seed's rows are
written atomically so an interrupted collection resumes. The camera is rendered only on saved steps;
`tools/frame_lag_check.py` checks that those frames are pixel-identical to a fully rendered run, and
that multi-threaded rendering does not lag the physics state.

### Model and training

* Camera: 320x160 at MetaDrive's default mount; crop rows 60–150, resize to 200x66, convert to YUV.
* Network: PilotNet (5 conv layers, FC 100/50/10) with the vehicle speed concatenated at the first
  FC layer; about 250k parameters.
* Loss: SmoothL1 weighted by the inverse frequency of the steering bin (21 bins over [-1, 1]), to
  counter the bias toward driving straight.
* Augmentation (OpenCV): brightness, random shadow, horizontal flip with negated steering, blur.
* AdamW, cosine schedule, early stopping on validation MAE. Mixed precision only on GPUs with fast
  FP16 (compute capability 7.0+).

### Closed-loop evaluation

The model steers and the expert controls speed, so the metrics isolate steering. A safety driver
takes over steering for 3 s whenever the lateral offset exceeds 1.2 m; each takeover is one
intervention. The expert also drives the same 1,000 roads as an upper-bound baseline.

MetaDrive by default ends an episode as soon as the chassis touches a solid lane line. With lanes
drawn 3.0–4.5 m wide and a 1.85 m wide car, that happens at a lateral offset of 0.54–1.29 m, before
the 1.2 m takeover on most roads, which would hide interventions. Both collection and evaluation
therefore turn that off: an episode leaves the road only when the vehicle centre leaves the lane
surface, and solid-line touches are reported as their own rate.

| Metric | Definition |
|---|---|
| Autonomy | 1 − interventions × 6 s / elapsed time (Bojarski et al. 2016), pooled |
| Success rate | Destination reached with no intervention, collision or road exit; Wilson 95% interval |
| Route completion | MetaDrive's route completion at episode end, averaged |
| Interventions / km, collisions / km | Pooled counts over pooled distance; a collision is a rising edge of the crash flag |
| Out-of-road rate | Episodes terminated because the vehicle centre left the lane surface |
| Solid-line touches / km | Rising edges of the chassis touching a solid lane line |
| Lateral offset | Mean and 95th percentile of \|offset from lane centre\| while the model steers |
| Steering jerk | Mean \|d²steering/dt²\| within uninterrupted model-controlled segments |
| Open-loop MAE | Model vs expert steering on expert-driven test states, split curve / straight |
| On-policy MAE and gap | Model vs expert steering on the model's own states; the ratio to open-loop MAE measures covariate shift |

## Perception

### Data

BDD100K is downloaded from Berkeley's server: the 100k images, the per-image 2018 labels and the
drivable-area id maps. The 2020 `det_20` detection labels are no longer available from an official
source, so detection uses the 2018 labels, mapped to `det_20`'s class names (person → pedestrian,
bike → bicycle, motor → motorcycle). Results are therefore not directly comparable with numbers
reported on `det_20`.

* **Detection classes (12):** pedestrian, rider, car, truck, bus, train, motorcycle, bicycle,
  traffic light red / yellow / green, traffic sign. Traffic lights are split by their
  `trafficLightColor` attribute. Lights labelled with no colour (seen from the side or behind, about a
  third of all lights) have no colour class and are left out of training and evaluation targets.
* **Drivable area:** binary, merging "direct" and "alternative" as YOLOP does.
* **Lane lines:** BDD100K publishes no lane masks for these labels, so they are drawn from the lane
  polylines (every category except crosswalk, direction "parallel", Bezier segments evaluated):
  8 px wide at 1280x720 for training and 2 px for evaluation, YOLOP's protocol.
* **Splits:** the official 70k train / 10k val split; all numbers are on val, as in YOLOP and A-YOLOM.

### Model and training

A shared YOLO11 backbone and neck (Ultralytics' model, unchanged) with three heads: Ultralytics'
detection head and two light segmentation decoders. Each decoder reads the neck's stride-8 P3
feature and the backbone's stride-4 feature, upsamples and fuses them, and predicts one logit per
pixel. Detection uses Ultralytics' loss; each segmentation head uses BCE + Dice. COCO-pretrained
weights initialise everything whose shape matches. Training follows Ultralytics' recipe (SGD,
nominal batch 64, warm-up, cosine decay, weight EMA, mixed precision) at a 640x384 letterboxed
input, with random scale, translation, flip and HSV jitter applied jointly to the image, masks and
boxes. It runs on a free Kaggle T4 notebook, checkpointing so a run can continue across sessions.

### Metrics

* Detection: mAP@0.5 and mAP@0.5:0.95 over the 12 classes (Ultralytics' matching and AP code), boxes
  mapped back to 1280x720.
* Drivable area: IoU per class and their mean, at 1280x720.
* Lane lines: lane IoU against the 2 px lines, plus both accuracy definitions found in the
  literature under one name: lane recall TP/(TP+FN) (YOLOP's code) and balanced accuracy (A-YOLOM).
* Speed: median latency and FPS of all three heads plus NMS at 640x384, with the device and
  precision stated.

### OpenCV baseline

Colour and gradient thresholds, a perspective warp, sliding-window second-order fits from up to four
histogram peaks, and the fitted curves drawn back in the image frame. Its region and thresholds are
tuned by grid search on 2,000 training images, then it is scored on val with the same lane metrics.

## Edge export

PyTorch → ONNX → TensorFlow SavedModel → TFLite (FP32 and full-integer INT8 with a representative
dataset). Latency is reported as median and p90 over repeated runs on named hardware, together with
the accuracy change against the PyTorch model.
