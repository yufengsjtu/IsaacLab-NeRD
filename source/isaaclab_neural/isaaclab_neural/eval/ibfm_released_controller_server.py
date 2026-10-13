# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve the released I-BFM Push Box and Kick Box phase controllers over stdin and stdout.

Run with the Python of the I-BFM ``uv`` environment: ``python ibfm_released_controller_server.py ROOT TASK NUM_ENVS``,
where ``ROOT`` is ``I-BFM-Code/ibfm``. The controller flags and environment variables are those of the released
``tasks/push_box/run_interactive.sh`` and ``tasks/kick_box/run_interactive.sh``. Physics stays in Isaac Lab; see
:mod:`isaaclab_neural.eval.ibfm_box_demo`.

Protocol: one JSON line per request, one JSON line per reply.
  {"cmd": "latent", "metrics": [..], "state": [[..]], "privileged": [[..]], "last_action": [[..]],
   "history": [[..]], "base_rot": [[9]], "box_dir": [[2]], "goal_dir": [[2]]}
   -> {"latent": [[256]], "state": [[..]], "privileged": [[..]], "mode": [..], "phase": [..]}
  {"cmd": "update", "metrics": [..]} -> {"phase": [..]}
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
TASK = sys.argv[2]
NUM_ENVS = int(sys.argv[3])
HERE = ROOT / "tasks" / f"{TASK}_box"

if TASK == "push":
    env = {
        "UFO_PUSH_V2_START_WITH_REPOSITION": "1",
        "UFO_PUSH_V2_HOLD_DRIVE": "1",
        "UFO_PUSH_V2_REACQUIRE_BLEND_MAX": "0.0",
        "UFO_PUSH_V2_STOP_SPEED": "0.06",
        "UFO_PUSH_V2_STALL_PULSE_BLEND_MAX": "0.0",
        "UFO_PUSH_V2_STALL_PULSE_DELAY_SECONDS": "0.50",
        "UFO_PUSH_V2_STALL_PULSE_RAMP_SECONDS": "0.50",
        "UFO_PUSH_V2_ACQUIRE_DRIVE_BLEND_SECONDS": "0.40",
        "UFO_PUSH_TIMED_ACQUIRE_SECONDS": "0.65",
        "UFO_PUSH_TIMED_ACQUIRE_CROUCH_SECONDS": "0.65",
        "UFO_PUSH_TIMED_ACQUIRE_SETTLE_SECONDS": "0.25",
        "UFO_PUSH_TIMED_ACQUIRE_SETTLE_LATENT_DIR": str(HERE / "latents/selected/acquire_hand_layout_v2"),
        "UFO_PUSH_V2_HANDPUSH_HOLD_LATENT_DIR": str(HERE / "latents/selected/drive_handpush_hold_v1"),
        "UFO_PUSH_V2_HANDPUSH_HOLD_BLEND_BASE": "0.0",
        "UFO_PUSH_V2_HANDPUSH_HOLD_BLEND_MAX": "0.30",
        "UFO_PUSH_V2_HANDPUSH_HOLD_HEIGHT_ENTER": "0.65",
        "UFO_PUSH_V2_HANDPUSH_HOLD_HEIGHT_EXIT": "0.58",
        "UFO_PUSH_V2_HANDPUSH_HOLD_RISE_SECONDS": "0.50",
        "UFO_PUSH_V2_HANDPUSH_HOLD_FALL_SECONDS": "0.50",
        "UFO_PUSH_V2_HANDPUSH_HOLD_ON_CONTACT_LOSS": "0",
        "UFO_PUSH_V2_FORCE_DRIVE_CONTACT_OBS": "1",
    }
    script = HERE / "two_stage_timed_acquire_sim2sim.py"
    overlay = [HERE / "runtime_dependencies/code_overlay", ROOT / "tasks/common/code_overlay"]
    drive_dir, acquire_dir = HERE / "latents/selected/drive", HERE / "latents/acquire_openloop_grid_v1/crouch_sync_0p55"
    extra = [
        "--direct-push-max-turn-blend",
        "0.05",
        "--direct-push-stall-speed",
        "0.10",
        "--direct-push-stall-recovery-seconds",
        "5",
        "--direct-push-stop-distance",
        "0.40",
        "--direct-push-stop-exit-distance",
        "0.55",
        "--push-stop-enter-distance",
        "0.40",
        "--push-stop-exit-distance",
        "0.55",
        "--push-success-distance",
        "0.40",
        "--direct-push-hand-lateral-abs-max",
        "0.13",
        "--direct-push-hand-height-abs-max",
        "0.30",
    ]
else:
    env = {
        "UFO_KICK_START_WITH_REPOSITION": "1",
        "UFO_PUSH_V2_HOLD_DRIVE": "1",
        "UFO_PUSH_V2_REACQUIRE_BLEND_MAX": "0.0",
        "UFO_PUSH_V2_STALL_PULSE_BLEND_MAX": "0.0",
        "UFO_KICK_NOHAND_REKICK_GUARD": "0",
        "UFO_KICK_REKICK_LATENT_PATH": "",
    }
    script = HERE / "kick_box_sim2sim.py"
    overlay = [ROOT / "tasks/common/code_overlay"]
    drive_dir, acquire_dir = HERE / "latents/candidates/kick25_min75", HERE / "latents/selected/acquire"
    extra = [
        "--direct-push-max-turn-blend",
        "0.0",
        "--direct-push-stall-speed",
        "0.03",
        "--direct-push-stall-recovery-seconds",
        "10",
        "--direct-push-stop-distance",
        "0.20",
        "--direct-push-stop-exit-distance",
        "0.40",
        "--direct-push-hand-lateral-abs-max",
        "0.25",
        "--direct-push-hand-height-abs-max",
        "0.20",
    ]
os.environ.update(env)
sys.path[:0] = [str(p) for p in overlay] + [str(ROOT)]
os.chdir(ROOT)

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location(f"released_{TASK}", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rt = module.runtime if hasattr(module, "runtime") else module.module
latents = HERE / "latents"
sys.argv = [
    str(script),
    "--controller-mode",
    "push-direct",
    "--model-folder",
    str(ROOT / "tasks/common/checkpoint_ref"),
    "--latent-dir",
    str(drive_dir),
    "--direct-push-contact-acquire-latent-dir",
    str(acquire_dir),
    "--direct-push-contact-drive-enter-steps",
    "5",
    "--direct-push-contact-drive-exit-steps",
    "30",
    "--direct-push-turn-latent-dir",
    str(latents / "turn_stop"),
    "--direct-push-align-pivot-latent-dir",
    str(latents / "selected/align_pivot_no_contact_v1"),
    "--direct-push-stop-latent-dir",
    str(latents / "turn_stop"),
    "--get-up-latent",
    str(latents / "get_up/z_get_up_reward.pkl"),
    "--get-up-stand-latent",
    str(latents / "get_up/z_get_up_stand.pkl"),
    "--get-up-stand-seconds",
    "0.60",
    "--direct-push-heading-blend-full-deg",
    "45",
    "--direct-push-speed-target",
    "0.0",
    "--direct-push-speed-deadband",
    "0.05",
    "--direct-push-speed-filter-seconds",
    "0.15",
    "--direct-push-speed-min-drive",
    "0.25",
    "--direct-push-stop-enter-steps",
    "5",
    "--direct-push-stop-exit-steps",
    "10",
    "--direct-push-require-rear-hand-contact",
    "--direct-push-hand-rear-min",
    "0.04",
    "--direct-push-hand-rear-max",
    "0.38",
    "--direct-push-reposition-enter-steps",
    "5",
    "--direct-push-release-seconds",
    "0.60",
    "--direct-push-release-speed",
    "0.08",
    "--direct-push-release-upright-min",
    "0.90",
    "--direct-push-backoff-distance",
    "1.0",
    "--direct-push-backoff-max-seconds",
    "1.5",
    "--direct-push-backoff-stable-steps",
    "1",
    "--direct-push-backoff-blend-seconds",
    "0.4",
    "--direct-push-backoff-latent-dir",
    str(latents / "selected/backoff_replay_v1"),
    "--direct-push-backoff-latent-mode",
    "replay",
    "--direct-push-reposition-blend-seconds",
    "0.80",
    "--direct-push-reposition-max-turn-blend",
    "0.75",
    "--direct-push-reposition-turn-cap-min-recovery",
    "2",
    "--direct-push-reposition-turn-hysteresis-deg",
    "150",
    "--direct-push-reposition-side-clearance",
    "0.75",
    "--direct-push-reposition-behind-offset",
    "0.60",
    "--direct-push-reposition-min-path-clearance",
    "0.50",
    "--direct-push-reposition-replan-box-displacement",
    "0.10",
    "--direct-push-reposition-side-switch-clearance-gain",
    "0.10",
    "--push-standoff-distance",
    "0.80",
    "--push-staging-tolerance",
    "0.20",
    "--push-lateral-max",
    "0.30",
    "--direct-push-reposition-heading-tolerance-deg",
    "20",
    "--direct-push-align-heading-exit-tolerance-deg",
    "25",
    "--direct-push-align-ready-distance",
    "0.35",
    "--direct-push-reposition-align-exit-distance",
    "0.35",
    "--direct-push-reposition-align-exit-lateral",
    "0.40",
    "--direct-push-reposition-align-exit-behind-margin",
    "0.10",
    "--direct-push-overshoot-margin",
    "0.10",
    "--direct-push-overshoot-steps",
    "10",
    "--motion-cache",
    "unused.pkl",
    "--contact-source",
    "physical",
    "--device",
    "cpu",
    "--mask-approach-object-context",
    "--no-viewer",
] + extra
args = rt.parse_args()
device = torch.device("cpu")
z = {"push_direct": rt._load_latent(args.latent_dir, "z_push_direct", device)}
z["direct_turn"] = rt._load_latent_bank(args.direct_push_turn_latent_dir, "z_push_turn", device)
z["direct_approach"] = rt._load_latent(args.direct_push_turn_latent_dir, "z_approach", device)
z["direct_approach_turn"] = rt._load_latent_bank(args.direct_push_turn_latent_dir, "z_turn", device)
z["direct_align_pivot"] = rt._load_latent_bank(args.direct_push_align_pivot_latent_dir, "z_turn", device)
z["direct_stop"] = rt._load_latent(args.direct_push_stop_latent_dir, "z_push_stop", device)
z["direct_backoff"] = rt._load_latent(args.direct_push_backoff_latent_dir, "z_backoff", device)
z["push_contact_acquire"] = rt._load_latent(args.direct_push_contact_acquire_latent_dir, "z_push_direct", device)
z["get_up"] = rt._load_get_up_latent(args.get_up_latent, device)
z["get_up_stand"] = rt._load_get_up_latent(args.get_up_stand_latent, device)[-1:]


class Policy:
    @staticmethod
    def project_z(value):
        return value.shape[-1] ** 0.5 * torch.nn.functional.normalize(value, dim=-1)


policy = Policy()
controllers = [rt.DirectPushController(args, dt=rt.CONTROL_DT) for _ in range(NUM_ENVS)]
ground = 0.186028293
print(json.dumps({"ready": True, "phase": [c.phase for c in controllers]}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req["cmd"] == "update":
        for c, m in zip(controllers, req["metrics"]):
            c.update(m, goal_height=ground)
        print(json.dumps({"phase": [c.phase for c in controllers]}), flush=True)
        continue
    out = {"latent": [], "state": [], "privileged": [], "mode": [], "phase": []}
    for i, c in enumerate(controllers):
        obs = {
            k: torch.tensor([req[src][i]], dtype=torch.float32)
            for k, src in (
                ("state", "state"),
                ("privileged_state", "privileged"),
                ("last_action", "last_action"),
                ("history_actor", "history"),
            )
        }
        new_obs, latent, mode = c.latent(
            policy,
            obs,
            req["metrics"][i],
            base_rotation=np.asarray(req["base_rot"][i]).reshape(3, 3),
            box_direction=np.asarray(req["box_dir"][i]),
            goal_direction=np.asarray(req["goal_dir"][i]),
            z=z,
        )
        out["latent"].append(latent.reshape(-1).tolist())
        out["state"].append(new_obs["state"].reshape(-1).tolist())
        out["privileged"].append(new_obs["privileged_state"].reshape(-1).tolist())
        out["mode"].append(mode)
        out["phase"].append(c.phase)
    print(json.dumps(out), flush=True)
