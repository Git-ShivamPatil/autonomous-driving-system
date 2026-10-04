# Claims ledger

Every number this project advertises is listed here with the command that produced it, the commit it
ran at and the hardware it ran on. A claim that has not been measured yet says **pending**; it is not
stated anywhere else as achieved.

## Steering (MetaDrive)

| Claim | Status | Value | Command | Commit | Hardware |
|---|---|---|---|---|---|
| Steering CNN trained on 70,000+ camera images | pending | | `python -m ads.sim.collect --split train --target 75000` | | |
| Validation set from disjoint roads | pending | | `python -m ads.sim.collect --split val --target 6000` | | |
| Open-loop validation MAE (all / curve / straight) | pending | | `python -m ads.train` | | |
| Closed-loop evaluation on 1,000 test scenarios (seeds 0–999) | pending | | `python -m ads.eval.closed_loop --driver model ...` | | |
| Expert baseline on the same 1,000 scenarios | pending | | `python -m ads.eval.closed_loop --driver expert ...` | | |
| Autonomy %, success rate (Wilson 95%), interventions/km, collisions/km | pending | | as above | | |
| Open-loop vs on-policy MAE gap | pending | | as above | | |

## Perception (BDD100K)

| Claim | Status | Value | Command | Commit | Hardware |
|---|---|---|---|---|---|
| Object detection over 12 classes (BDD100K's 10, traffic light split by colour) | pending | | | | |
| Detection mAP@0.5 / mAP@0.5:0.95 | pending | | | | |
| Drivable-area segmentation mIoU | pending | | | | |
| Lane-line segmentation IoU, learned head vs OpenCV baseline | pending | | | | |
| Real-time inference (FPS) | pending | | | | |

## Edge export and cross-dataset evaluation

| Claim | Status | Value | Command | Commit | Hardware |
|---|---|---|---|---|---|
| TFLite INT8 latency and accuracy change | pending | | | | |
| Steering open-loop MAE on Sully Chen's real dashcam dataset | pending | | | | |
