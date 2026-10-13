# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run BFM-Zero latent demos on the G1 29-DoF robot in Newton MJWarp and optionally record them.

Demos (``--demo``):

* ``tracking``: follow the per-frame tracking latents of a LAFAN clip from the exported latent library.
* ``reward_chain``: switch between reward-inferred latents (walk, turn, side-step, crouch, arm motions).
* ``push_recovery``: get up from lying on the back, walk, and recover from horizontal pushes at the pelvis.

The environment is ``Isaac-G1-29DoF-BFM-Dataset-Gen-v0``; pushes are written as floating-base generalized forces. The
latent library is the ``.npz`` read by :class:`~isaaclab_neural.generate.samplers.g1_29dof_bfm.BfmLatentLibrary` with
additional ``tracking_names``, ``goal_names`` and ``reward_names`` arrays.

Example::

    PYTHONPATH="$PWD/source/isaaclab_neural:$PWD/source/isaaclab" BFM_ZERO_ACTOR_PATH=bfm_zero_actor_isaaclab.pt \\
      python -m isaaclab_neural.eval.bfm_zero_demo --demo reward_chain --library bfm_zero_latent_library.npz \\
      --video reward_chain.mp4
"""

from __future__ import annotations

import argparse
import json
import math

from isaaclab.app import add_launcher_args, launch_simulation

REWARD_CHAIN = (
    ("move-ego-0-0.7", 4.0, "walk forward 0.7 m/s"),
    ("rotate-z-5-0.5", 3.0, "turn in place"),
    ("move-ego-90-0.3", 3.0, "side-step left"),
    ("move-ego-180-0.3", 3.0, "walk backward"),
    ("crouch-0.25", 3.0, "crouch"),
    ("raisearms-m-m", 3.0, "raise arms"),
    ("spin-arms-5-l-l", 3.0, "spin arms"),
    ("sitonground", 4.0, "sit on the ground"),
    ("move-ego-0-0", 4.0, "stand up"),
)
"""Reward latent, duration [s] and caption of each ``reward_chain`` segment."""


def parse_args() -> argparse.Namespace:
    """Parse demo and launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", choices=("tracking", "reward_chain", "push_recovery"), required=True)
    parser.add_argument("--library", required=True, help="Exported BFM-Zero latent library (.npz).")
    parser.add_argument("--clip", default="dance1_subject2", help="Tracking clip name.")
    parser.add_argument("--clip-start-s", type=float, default=0.0, help="Tracking clip start time [s].")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--push-force", type=float, default=350.0, help="Push magnitude [N].")
    parser.add_argument("--push-duration", type=float, default=0.15, help="Push duration [s].")
    parser.add_argument(
        "--push-times",
        type=float,
        nargs="*",
        default=(8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0),
        help="Push onsets [s].",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="JSON summary with captions per time.")
    parser.add_argument("--video", default=None, help="Record a Newton GL mp4 following --video-env.")
    parser.add_argument("--video-env", type=int, default=0)
    parser.add_argument("--video-size", type=int, nargs=2, default=(960, 540), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--video-distance", type=float, default=3.2, help="Camera distance [m].")
    parser.add_argument("--video-focal-length", type=float, default=20.0, help="Camera focal length.")
    parser.add_argument("--video-azimuth", type=float, default=-60.0, help="Camera azimuth [deg].")
    add_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    """Run the demo."""
    args = parse_args()
    import isaaclab_neural.envs  # noqa: F401, PLC0415
    from isaaclab_neural.envs.g1_29dof_env_cfg import G129DofBfmEnvCfg  # noqa: PLC0415

    from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: PLC0415

    cfg = G129DofBfmEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.seconds + 5.0
    cfg.events.reset_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_base.params["pose_range"] = {"yaw": (0.0, 0.0)}
    cfg.sim.physics = resolve_presets(cfg.sim.physics)
    if args.video:
        from isaaclab_neural.eval.newton_video import newton_video_visualizer_cfg  # noqa: PLC0415

        cfg.sim.visualizer_cfgs = [newton_video_visualizer_cfg(*args.video_size, args.video_focal_length)]
    with launch_simulation(cfg, args):
        run(args, cfg)


def run(args: argparse.Namespace, cfg) -> None:
    """Roll out the latent schedule of the selected demo."""
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_neural.envs.mdp_g1_29dof import newton_free_joint_dof_starts, write_root_wrench

    import isaaclab.utils.math as math_utils

    library = np.load(args.library)
    env = gym.make("Isaac-G1-29DoF-BFM-Dataset-Gen-v0", cfg=cfg).unwrapped
    env.reset()
    n, device, dt = env.num_envs, env.device, env.step_dt
    robot = env.scene["robot"]
    steps = int(round(args.seconds / dt))
    reward_names, goal_names = list(library["reward_names"]), list(library["goal_names"])

    def reward(name: str) -> torch.Tensor:
        return torch.as_tensor(library["reward_z"][reward_names.index(name)], dtype=torch.float32, device=device)

    captions: list[tuple[float, str]] = []
    if args.demo == "tracking":
        names, offsets = list(library["tracking_names"]), library["tracking_offsets"]
        clip = names.index(args.clip)
        start = int(offsets[clip]) + int(round(args.clip_start_s / dt))
        frames = torch.as_tensor(
            library["tracking_z"][start : int(offsets[clip + 1])].astype(np.float32), device=device
        )
        schedule = [frames[min(step, len(frames) - 1)] for step in range(steps)]
        captions.append((0.0, f"motion tracking: LAFAN {args.clip}"))
    elif args.demo == "reward_chain":
        schedule = []
        for name, seconds, caption in REWARD_CHAIN:
            captions.append((len(schedule) * dt, f"reward prompt: {caption}"))
            schedule += [reward(name)] * int(round(seconds / dt))
        schedule = (schedule + [reward(REWARD_CHAIN[0][0])] * steps)[:steps]
    else:
        stand = torch.as_tensor(library["goal_z"][goal_names.index("walk2_subject1_2588")], device=device)
        walk = reward("move-ego-0-0.3")
        schedule = [stand if step * dt < 6.0 else walk for step in range(steps)]
        captions += [(0.0, "get up from lying"), (6.0, "walk 0.3 m/s"), (args.push_times[0], "walk under 350 N pushes")]
        pose = torch.zeros(n, 7, device=device)
        pose[:, :3] = env.scene.env_origins + torch.tensor([0.0, 0.0, 0.25], device=device)
        pose[:, 3:] = math_utils.quat_from_euler_xyz(
            torch.zeros(n, device=device), torch.full((n,), -math.pi / 2, device=device), torch.zeros(n, device=device)
        )
        robot.write_root_pose_to_sim_index(root_pose=pose)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device))
        import warp as wp  # noqa: PLC0415
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        dof_starts = newton_free_joint_dof_starts(n, device)
        joint_f = wp.to_torch(NewtonManager.get_control().joint_f)
    push_steps = {int(round(t / dt)) for t in args.push_times} if args.demo == "push_recovery" else set()
    push_length = max(1, int(round(args.push_duration / dt)))

    video = None
    if args.video:
        from isaaclab_neural.eval.newton_video import NewtonFollowVideo  # noqa: PLC0415

        video = NewtonFollowVideo(
            env.sim, args.video, fps=1.0 / dt, distance=args.video_distance, azimuth_deg=args.video_azimuth
        )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    push_left, wrench = 0, torch.zeros(n, 6, device=device)
    min_height = torch.full((n,), float("inf"), device=device)
    for step in range(steps):
        if args.demo == "push_recovery":
            if step in push_steps:
                angle = float(torch.rand(1, generator=generator)) * 2 * math.pi
                wrench[:, 0], wrench[:, 1] = args.push_force * math.cos(angle), args.push_force * math.sin(angle)
                push_left = push_length
            write_root_wrench(joint_f, dof_starts, wrench if push_left > 0 else torch.zeros_like(wrench))
            push_left -= 1
        env.step(torch.stack([schedule[step]] * n))
        height = robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        if step * dt > 6.0:
            min_height = torch.minimum(min_height, height)
        if video is not None:
            video.capture(robot.data.root_link_pos_w.torch[args.video_env].cpu().numpy())
    if video is not None:
        video.close()
    summary = {
        "demo": args.demo,
        "captions": captions,
        "min_root_height_after_6s": min_height.cpu().tolist(),
        "fell_after_6s": (min_height < 0.45).float().mean().item(),
    }
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(summary, handle, indent=1)
    env.close()


if __name__ == "__main__":
    main()
