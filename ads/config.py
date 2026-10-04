"""Project-wide constants: seed splits, road suite, camera geometry, collection and evaluation settings."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

# Seed splits. Half-open ranges, pairwise disjoint. TEST_SEEDS is the 1,000-scenario closed-loop suite.
TRAIN_SEEDS = range(1000, 9000)
VAL_SEEDS = range(9000, 9200)
TEST_SEEDS = range(0, 1000)
SPLITS = {"train": TRAIN_SEEDS, "val": VAL_SEEDS, "test": TEST_SEEDS}

# Lane-keeping road suite. Intersections and roundabouts are excluded: the correct steering there
# depends on the route, which a camera-only model cannot see.
MAP_BLOCKS = 4
TRAFFIC_DENSITY = 0.1
BLOCK_DISTRIBUTION = {
    "Curve": 0.6,
    "Straight": 0.2,
    "InRampOnStraight": 0.1,
    "OutRampOnStraight": 0.1,
}

# Simulation timing: 0.02 s physics step x 5 = 0.1 s per environment step.
PHYSICS_DT = 0.02
DECISION_REPEAT = 5
STEP_DT = PHYSICS_DT * DECISION_REPEAT
EPISODE_HORIZON = 1500  # environment steps, i.e. 150 s

# Camera: MetaDrive's default sensor mount, rendered at 320x160, looking straight ahead.
CAM_W, CAM_H = 320, 160
# Channel order of obs["image"], checked on a saved frame (see README, "Camera").
CAMERA_CHANNEL_ORDER = "BGR"
# PilotNet geometry: crop rows [60, 150), resize to 200x66, convert to YUV.
CROP_TOP, CROP_BOTTOM = 60, 150
NET_W, NET_H = 200, 66

# DART data collection (Laskey et al. 2017). The executed steering is the expert's plus AR(1) noise
# whose stationary standard deviation is the per-episode sigma; the recorded label is the clean
# expert steering in the visited state.
DART_RHO = 0.9
DART_SIGMAS = (0.0, 0.05, 0.10, 0.15)
SAVE_EVERY = 2
MIN_SPEED_KMH = 2.0
TRAIN_TARGET_FRAMES = 75_000
VAL_TARGET_FRAMES = 6_000

CSV_COLUMNS = (
    "path",
    "seed",
    "step",
    "steering",
    "executed_steering",
    "throttle",
    "speed_kmh",
    "lateral_m",
    "curvature",
    "noise_sigma",
)

# Closed-loop evaluation.
TAKEOVER_LATERAL_M = 1.2
TAKEOVER_SECONDS = 3.0
NVIDIA_PENALTY_SECONDS = 6.0  # Bojarski et al. 2016: each intervention costs 6 s of autonomy
CURVE_CURVATURE = 1e-4  # 1/m; lanes with |curvature| above this count as curves (MetaDrive straights are exactly 0)

# Speed input normalisation for the steering network.
SPEED_SCALE_KMH = 50.0
