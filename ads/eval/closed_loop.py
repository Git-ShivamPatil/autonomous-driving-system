"""Closed-loop evaluation on the 1,000 test roads (seeds 0-999).

    python -m ads.eval.closed_loop --driver model --model runs/steering/pilotnet/best.pt --out runs/closed_loop/pilotnet
    python -m ads.eval.closed_loop --driver expert --model runs/steering/pilotnet/best.pt --out runs/closed_loop/expert

The driver steers and the IDM expert controls speed, so the metrics isolate steering. A safety driver
(the expert) takes over steering for TAKEOVER_SECONDS whenever |lateral offset| exceeds
TAKEOVER_LATERAL_M while the driver is in control; each takeover is one intervention.

With --driver expert the expert also steers; this is the upper-bound baseline. If --model is given in
that mode, the model predicts on the expert's states without acting, which yields the open-loop MAE on
the test roads. With --driver model, the model's prediction is compared with the expert's steering on
the model's own states (on-policy MAE). Their ratio is the open-loop / closed-loop gap.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from ads import config, provenance
from ads.eval import metrics

_STATE: dict = {}


def load_model(path: Path):
    import torch

    from ads.model import PilotNet

    torch.set_num_threads(1)
    model = PilotNet()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def predict(model, frame: np.ndarray, speed_kmh: float) -> float:
    import torch

    from ads.steering.preprocess import preprocess

    yuv = preprocess(frame)
    x = torch.from_numpy(np.ascontiguousarray(yuv.transpose(2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        return float(model(x, torch.tensor([speed_kmh])).clamp(-1, 1).item())


def _init_worker(model_path: str | None, camera: bool) -> None:
    from ads.sim.env import make_env

    _STATE["env"] = make_env("test", camera=camera, eval_mode=True)
    _STATE["model"] = load_model(Path(model_path)) if model_path else None


def _overlay(frame: np.ndarray, step: int, speed: float, driver_steer: float, expert_steer: float, takeover: bool) -> np.ndarray:
    import cv2

    bgr = frame if config.CAMERA_CHANNEL_ORDER == "BGR" else frame[..., ::-1]
    img = cv2.resize(np.ascontiguousarray(bgr), (config.CAM_W * 2, config.CAM_H * 2), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    cx, y0 = w // 2, h - 28
    cv2.rectangle(img, (cx - 150, y0 - 6), (cx + 150, y0 + 6), (60, 60, 60), -1)
    for value, colour, dy in ((expert_steer, (255, 255, 255), -10), (driver_steer, (0, 220, 0), 10)):
        x = int(cx + 150 * float(np.clip(value, -1, 1)))
        cv2.line(img, (x, y0 - 6 + dy), (x, y0 + 6 + dy), colour, 3)
    cv2.putText(img, f"step {step}  {speed:5.1f} km/h", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "green: driver  white: expert", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    if takeover:
        cv2.putText(img, "SAFETY TAKEOVER", (w - 210, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def run_episode(seed: int, driver: str, out: Path, video: bool) -> dict:
    import cv2

    from ads.sim.env import camera_frame, lane_state, speed_kmh
    from ads.sim.expert import expert_action, make_expert

    path = out / "episodes" / f"{seed:04d}.json"
    if path.exists():
        return json.loads(path.read_text())

    env, model = _STATE["env"], _STATE["model"]
    t0 = time.perf_counter()
    obs, info = env.reset(seed=seed)
    expert = make_expert(env, seed)
    takeover_steps = int(round(config.TAKEOVER_SECONDS / config.STEP_DT))
    takeover_left, interventions = 0, 0
    lateral_abs, segments, segment = [], [], []
    open_pairs = []  # (prediction, expert steering, curvature) on the visited states
    crash_flags = []
    distance, steps = 0.0, 0
    writer = None
    if video:
        writer = cv2.VideoWriter(
            str(out / "videos" / f"{seed:04d}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1 / config.STEP_DT,
            (config.CAM_W * 2, config.CAM_H * 2),
        )

    for step in range(config.EPISODE_HORIZON):
        expert_steer, accel = expert_action(expert)
        lateral, curvature = lane_state(env.agent)
        v = speed_kmh(env.agent)
        pred = predict(model, camera_frame(obs), v) if model is not None else None
        if pred is not None:
            open_pairs.append((pred, expert_steer, curvature))
        driver_steer = expert_steer if driver == "expert" else pred

        if takeover_left > 0:
            takeover_left -= 1
            executed, in_control = expert_steer, False
        elif abs(lateral) > config.TAKEOVER_LATERAL_M:
            interventions += 1
            takeover_left = takeover_steps - 1
            executed, in_control = expert_steer, False
        else:
            executed, in_control = driver_steer, True

        if in_control:
            segment.append(executed)
            lateral_abs.append(abs(lateral))
        elif segment:
            segments.append(segment)
            segment = []
        if writer is not None:
            writer.write(
                _overlay(
                    camera_frame(obs),
                    step,
                    v,
                    driver_steer if driver_steer is not None else expert_steer,
                    expert_steer,
                    not in_control,
                )
            )

        obs, _r, terminated, truncated, info = env.step([executed, accel])
        steps += 1
        distance += speed_kmh(env.agent) / 3.6 * config.STEP_DT
        crash_flags.append(bool(info.get("crash_vehicle") or info.get("crash_object")))
        if terminated or truncated:
            break
    if segment:
        segments.append(segment)
    if writer is not None:
        writer.release()

    result = metrics.EpisodeResult(
        seed=seed,
        elapsed_s=steps * config.STEP_DT,
        distance_m=distance,
        interventions=interventions,
        collisions=metrics.rising_edges(crash_flags),
        out_of_road=bool(info.get("out_of_road")),
        arrive_dest=bool(info.get("arrive_dest")),
        route_completion=float(info.get("route_completion", float("nan"))),
        lateral_abs_m=[round(x, 4) for x in lateral_abs],
        steering_segments=[[round(x, 5) for x in s] for s in segments],
    )
    record = dict(result.__dict__)
    record["success"] = result.success
    record["open_pairs"] = [[round(a, 5), round(b, 5), round(c, 6)] for a, b, c in open_pairs]
    record["wall_seconds"] = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
    return record


def _worker_episode(seed: int, driver: str, out: str, video: bool) -> dict:
    return run_episode(seed, driver, Path(out), video)


def summarize_records(records: list[dict], driver: str) -> dict:
    fields = metrics.EpisodeResult.__dataclass_fields__
    episodes = [metrics.EpisodeResult(**{k: r[k] for k in fields}) for r in records]
    summary = metrics.summarize(episodes, dt=config.STEP_DT, penalty_s=config.NVIDIA_PENALTY_SECONDS)
    pairs = [p for r in records for p in r.get("open_pairs", [])]
    if pairs:
        pred, target, curv = (np.array(x) for x in zip(*pairs, strict=True))
        key = "open_loop" if driver == "expert" else "on_policy"
        summary[key] = metrics.split_mae(pred, target, curv, config.CURVE_CURVATURE)
    return summary


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Closed-loop evaluation on the test roads.")
    p.add_argument("--driver", choices=["model", "expert"], required=True)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seeds", type=int, default=len(config.TEST_SEEDS), help="first N test seeds")
    p.add_argument("--videos", type=int, default=0, help="record MP4s for the first N seeds")
    args = p.parse_args(argv)
    if args.driver == "model" and args.model is None:
        p.error("--driver model needs --model")

    seeds = list(config.TEST_SEEDS)[: args.seeds]
    (args.out / "videos").mkdir(parents=True, exist_ok=True)
    camera = args.model is not None or args.videos > 0
    t_start = time.perf_counter()
    records: dict[int, dict] = {}
    initargs = (str(args.model) if args.model else None, camera)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=initargs) as pool:
        pending, idx = set(), 0
        while idx < len(seeds) or pending:
            while idx < len(seeds) and len(pending) < 2 * args.workers:
                s = seeds[idx]
                pending.add(pool.submit(_worker_episode, s, args.driver, str(args.out), idx < args.videos))
                idx += 1
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                r = fut.result()
                records[r["seed"]] = r
            if len(records) % 10 == 0 or not pending:
                elapsed = time.perf_counter() - t_start
                eta = (len(seeds) - len(records)) * elapsed / max(len(records), 1) / 3600
                ok = sum(r["success"] for r in records.values())
                print(f"[{args.driver}] {len(records)}/{len(seeds)} episodes | success {ok} | eta {eta:.2f} h", flush=True)

    ordered = [records[s] for s in seeds]
    summary = {
        "provenance": provenance.stamp("ads.eval.closed_loop"),
        "driver": args.driver,
        "model": str(args.model) if args.model else None,
        "seeds": [seeds[0], seeds[-1]],
        "wall_seconds": time.perf_counter() - t_start,
        **summarize_records(ordered, args.driver),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=1))
    return summary


if __name__ == "__main__":
    main()
