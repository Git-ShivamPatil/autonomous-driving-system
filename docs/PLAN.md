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
written atomically so an interrupted collection resumes.

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

| Metric | Definition |
|---|---|
| Autonomy | 1 − interventions × 6 s / elapsed time (Bojarski et al. 2016), pooled |
| Success rate | Destination reached with no intervention, collision or road exit; Wilson 95% interval |
| Route completion | MetaDrive's route completion at episode end, averaged |
| Interventions / km, collisions / km | Pooled counts over pooled distance; a collision is a rising edge of the crash flag |
| Out-of-road rate | Episodes terminated by leaving the road |
| Lateral offset | Mean and 95th percentile of \|offset from lane centre\| while the model steers |
| Steering jerk | Mean \|d²steering/dt²\| within uninterrupted model-controlled segments |
| Open-loop MAE | Model vs expert steering on expert-driven test states, split curve / straight |
| On-policy MAE and gap | Model vs expert steering on the model's own states; the ratio to open-loop MAE measures covariate shift |

## Perception

* Detection classes: BDD100K's 10 classes with `traffic light` split into red, yellow and green by
  BDD's colour attribute (12 classes).
* Architecture: a shared YOLO11 backbone and neck feeding a detection head and two segmentation
  heads (drivable area, lane lines), in the style of A-YOLOM and YOLOPv2.
* Metrics: detection mAP@0.5 and mAP@0.5:0.95, drivable-area mIoU, lane IoU and accuracy, and FPS on
  stated hardware.
* Baseline: perspective warp plus sliding-window lane fitting in OpenCV, scored on the same lane masks.
* Training hardware: a free cloud GPU notebook; evaluation and FPS are reported per device.

## Edge export

PyTorch → ONNX → TensorFlow SavedModel → TFLite (FP32 and full-integer INT8 with a representative
dataset). Latency is reported as median and p90 over repeated runs on named hardware, together with
the accuracy change against the PyTorch model.
