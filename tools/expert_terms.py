"""Decompose the IDM expert's steering into its PID terms while another driver steers.

MetaDrive's IDM steering is heading_PID + lateral_PID, and its PID integral accumulates without bound.
When the expert is queried but not obeyed (the model drives), the integral can wind up. This logs, per
step, the proportional, integral and derivative contributions so the effect can be measured.

    python tools/expert_terms.py --model runs/steering/pilotnet/best.pt --split val --seeds 3 --out runs/expert_terms.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ads import config
from ads.eval.closed_loop import load_model, predict
from ads.sim.env import camera_frame, lane_state, make_env, speed_kmh
from ads.sim.expert import expert_action, make_expert


def terms(pid) -> tuple[float, float, float]:
    """Contributions of the last get_result call: (-kp*p, -ki*i, -kd*d)."""
    return -pid.k_p * pid.p_error, -pid.k_i * pid.i_error, -pid.k_d * pid.d_error


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    model = load_model(args.model)
    env = make_env(args.split, camera=True, eval_mode=True)
    out = {}
    for seed in list(config.SPLITS[args.split])[: args.seeds]:
        obs, _ = env.reset(seed=seed)
        expert = make_expert(env, seed)
        rows = []
        for step in range(config.EPISODE_HORIZON):
            exp_steer, accel = expert_action(expert)
            h, lat_pid = terms(expert.heading_pid), terms(expert.lateral_pid)
            lateral, curvature = lane_state(env.agent)
            v = speed_kmh(env.agent)
            pred = predict(model, camera_frame(obs), v)
            rows.append([step, lateral, curvature, pred, exp_steer, *h, *lat_pid])
            obs, _r, term, trunc, _info = env.step([pred, accel])  # the model always steers; no safety driver
            if term or trunc or abs(lateral) > 2.5:
                break
        a = np.array(rows)
        integral = a[:, 6] + a[:, 9]
        out[seed] = {
            "steps": len(a),
            "mean_abs_expert": float(np.abs(a[:, 4]).mean()),
            "mean_abs_integral_term": float(np.abs(integral).mean()),
            "max_abs_integral_term": float(np.abs(integral).max()),
            "integral_share_of_expert": float(np.abs(integral).mean() / max(np.abs(a[:, 4]).mean(), 1e-9)),
            "integral_term_at_25_50_75_100pct": [float(integral[int(q * (len(a) - 1))]) for q in (0.25, 0.5, 0.75, 1.0)],
            "mean_abs_lateral": float(np.abs(a[:, 1]).mean()),
        }
        print(seed, json.dumps(out[seed]), flush=True)
    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
