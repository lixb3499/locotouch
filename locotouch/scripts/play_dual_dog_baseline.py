"""Play-only dual-dog baseline driven by a single-dog locomotion policy.

This script is meant as a lightweight benchmark before building a proper
dual-robot transport task:

- a long cylinder is spawned across two side-by-side robots
- both robots receive the same shared velocity command
- each robot builds the same proprioceptive observation used in single-dog locomotion
- the same pretrained single-dog PPO policy is applied to both robots independently
"""

from __future__ import annotations

import argparse
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args


parser = argparse.ArgumentParser(description="Play a dual-dog transport baseline with a shared single-dog PPO policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-DualDogBenchmark-LocoTouch-Play-v1",
    help="Registered dual-dog benchmark task that provides the scene and shared-command defaults.",
)
parser.add_argument(
    "--policy_task",
    type=str,
    default="Isaac-Locomotion-LocoTouch-Play-v1",
    help="Single-dog locomotion task whose PPO config/checkpoint will be loaded for both robots.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Unused compatibility flag. This script always uses one scene.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--video", action="store_true", default=False, help="Unused compatibility flag.")
parser.add_argument("--command_mode", type=str, default="random", choices=("fixed", "random"))
parser.add_argument("--command_interval_s", type=float, default=None, help="Shared command resampling period. Defaults to the source task config.")
parser.add_argument("--command_vx", type=float, default=0.3, help="Fixed forward velocity command.")
parser.add_argument("--command_vy", type=float, default=0.0, help="Fixed lateral velocity command.")
parser.add_argument("--command_wz", type=float, default=0.0, help="Fixed yaw-rate command.")
parser.add_argument("--max_vx", type=float, default=None, help="Override |vx| range for random commands.")
parser.add_argument("--max_vy", type=float, default=None, help="Override |vy| range for random commands.")
parser.add_argument("--max_wz", type=float, default=None, help="Override |wz| range for random commands.")
parser.add_argument("--stand_prob", type=float, default=None, help="Override the probability of sampling a zero command.")
parser.add_argument("--episode_length_s", type=float, default=None, help="Episode length before resetting the whole benchmark scene.")
parser.add_argument("--settle_steps", type=int, default=40, help="Physics steps to settle after reset.")
parser.add_argument(
    "--initial_zero_command_steps",
    type=int,
    default=None,
    help="Initial control steps with zero command after reset. Defaults to the source task command config.",
)
parser.add_argument("--robot_separation_y", type=float, default=0.9, help="Lateral distance between the two robots.")
parser.add_argument("--payload_length", type=float, default=3.5)
parser.add_argument("--payload_radius", type=float, default=0.06)
parser.add_argument("--payload_mass", type=float, default=6.0)
parser.add_argument("--payload_center_x", type=float, default=0.0)
parser.add_argument("--payload_center_z", type=float, default=0.38)
parser.add_argument("--payload_yaw_deg", type=float, default=0.0, help="Yaw of the payload. 0 means aligned with +Y.")
parser.add_argument("--payload_drop_height", type=float, default=0.20, help="Reset when the payload center falls below this z.")
parser.add_argument("--robot_fall_height", type=float, default=0.15, help="Reset when either robot base falls below this z.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab_tasks.utils import parse_env_cfg

from locotouch import *  # noqa: F401
from locotouch.utils.dual_dog_baseline_runtime import (
    build_policy_obs,
    build_scene_cfg,
    commanded_velocity,
    create_obs_state,
    extract_command_cfg,
    extract_policy_obs_cfg,
    make_policy,
    reset_scene,
    sample_shared_command,
    settle_scene,
)


def main():
    if args_cli.num_envs != 1:
        print("[INFO] This benchmark only uses one scene. Overriding num_envs to 1.")

    sim_device = args_cli.device if args_cli.device is not None else "cuda:0"
    task_env_cfg = parse_env_cfg(
        args_cli.task,
        device=sim_device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    policy_env_cfg = parse_env_cfg(
        args_cli.policy_task,
        device=sim_device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    policy_obs_cfg = extract_policy_obs_cfg(policy_env_cfg)
    command_cfg = extract_command_cfg(task_env_cfg, args_cli)
    control_decimation = int(task_env_cfg.decimation)

    scene_cfg = build_scene_cfg(
        task_env_cfg,
        robot_separation_y=args_cli.robot_separation_y,
        payload_length=args_cli.payload_length,
        payload_radius=args_cli.payload_radius,
        payload_mass=args_cli.payload_mass,
        payload_center_x=args_cli.payload_center_x,
        payload_center_z=args_cli.payload_center_z,
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=task_env_cfg.sim.dt,
        device=sim_device,
        use_fabric=not args_cli.disable_fabric,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[7.5, 8.0, 4.5], target=[0.0, 0.0, 0.35])
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    sim.play()
    print("[INFO] Dual-dog benchmark scene created.")

    robots = {
        "robot_left": scene["robot_left"],
        "robot_right": scene["robot_right"],
    }
    payload = scene["payload"]
    num_actions = robots["robot_left"].num_joints
    policy = make_policy(
        policy_obs_cfg,
        num_actions=num_actions,
        device=scene.device,
        policy_task=args_cli.policy_task,
        args_cli=args_cli,
    )

    default_joint_pos = {name: robot.data.default_joint_pos.clone() for name, robot in robots.items()}
    obs_states = {
        name: create_obs_state(num_actions=num_actions, history_length=policy_obs_cfg.history_length, device=scene.device)
        for name in robots
    }

    reset_scene(
        scene,
        sim,
        robot_separation_y=args_cli.robot_separation_y,
        payload_center_x=args_cli.payload_center_x,
        payload_center_z=args_cli.payload_center_z,
        payload_yaw_deg=args_cli.payload_yaw_deg,
    )
    settle_scene(scene, sim, robots, default_joint_pos, args_cli.settle_steps)

    shared_command = sample_shared_command(
        scene.device,
        command_cfg,
        args_cli.command_mode,
        (args_cli.command_vx, args_cli.command_vy, args_cli.command_wz),
    )
    warmup_steps_remaining = command_cfg.initial_zero_command_steps
    episode_steps = max(1, int(command_cfg.episode_length_s / sim.get_physics_dt() / control_decimation))
    command_interval_steps = max(1, int(command_cfg.interval_s / sim.get_physics_dt() / control_decimation))

    print(
        "[INFO] Shared command settings:",
        {
            "lin_vel_x_range": command_cfg.lin_vel_x_range,
            "lin_vel_y_range": command_cfg.lin_vel_y_range,
            "ang_vel_z_range": command_cfg.ang_vel_z_range,
            "interval_s": command_cfg.interval_s,
            "stand_prob": command_cfg.stand_prob,
            "initial_zero_command_steps": command_cfg.initial_zero_command_steps,
        },
    )
    print(f"[INFO] Left robot root pos {scene['robot_left'].data.root_pos_w[0].tolist()}")
    print(f"[INFO] Right robot root pos {scene['robot_right'].data.root_pos_w[0].tolist()}")
    print(f"[INFO] Payload root pos {payload.data.root_pos_w[0].tolist()}")

    control_step = 0
    episode_step = 0
    just_reset = True

    while simulation_app.is_running():
        if sim.is_stopped():
            sim.play()
            sim.step()
            continue
        if not sim.is_playing():
            sim.step()
            continue

        left_height = scene["robot_left"].data.root_pos_w[0, 2].item()
        right_height = scene["robot_right"].data.root_pos_w[0, 2].item()
        payload_height = payload.data.root_pos_w[0, 2].item()
        needs_reset = (
            payload_height < args_cli.payload_drop_height
            or left_height < args_cli.robot_fall_height
            or right_height < args_cli.robot_fall_height
            or episode_step >= episode_steps
        )

        if needs_reset:
            print(
                f"[INFO] Resetting benchmark: payload_z={payload_height:.3f}, "
                f"left_z={left_height:.3f}, right_z={right_height:.3f}, episode_step={episode_step}"
            )
            reset_scene(
                scene,
                sim,
                robot_separation_y=args_cli.robot_separation_y,
                payload_center_x=args_cli.payload_center_x,
                payload_center_z=args_cli.payload_center_z,
                payload_yaw_deg=args_cli.payload_yaw_deg,
            )
            obs_states = {
                name: create_obs_state(num_actions=num_actions, history_length=policy_obs_cfg.history_length, device=scene.device)
                for name in robots
            }
            settle_scene(scene, sim, robots, default_joint_pos, args_cli.settle_steps)
            shared_command = sample_shared_command(
                scene.device,
                command_cfg,
                args_cli.command_mode,
                (args_cli.command_vx, args_cli.command_vy, args_cli.command_wz),
            )
            warmup_steps_remaining = command_cfg.initial_zero_command_steps
            episode_step = 0
            just_reset = True

        if args_cli.command_mode == "random" and not just_reset and control_step % command_interval_steps == 0:
            shared_command = sample_shared_command(
                scene.device,
                command_cfg,
                args_cli.command_mode,
                (args_cli.command_vx, args_cli.command_vy, args_cli.command_wz),
            )

        command = commanded_velocity(shared_command, warmup_steps_remaining)
        batch_obs = torch.cat(
            [build_policy_obs(robot, obs_states[name], command, policy_obs_cfg) for name, robot in robots.items()],
            dim=0,
        )

        with torch.inference_mode():
            raw_actions = policy(batch_obs)

        if policy_obs_cfg.clip_raw_actions:
            scaled_actions = torch.clamp(raw_actions, -policy_obs_cfg.raw_action_clip_value, policy_obs_cfg.raw_action_clip_value)
        else:
            scaled_actions = raw_actions
        scaled_actions = scaled_actions * policy_obs_cfg.raw_action_scale

        targets = {}
        for action_idx, (name, robot) in enumerate(robots.items()):
            action = scaled_actions[action_idx : action_idx + 1]
            obs_states[name].last_action = action.clone()
            targets[name] = default_joint_pos[name] + action

        for _ in range(control_decimation):
            for name, robot in robots.items():
                robot.set_joint_position_target(targets[name])
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        if control_step % 100 == 0:
            cmd_str = ", ".join(f"{value:.3f}" for value in command[0].tolist())
            left_xyz = scene["robot_left"].data.root_pos_w[0].tolist()
            right_xyz = scene["robot_right"].data.root_pos_w[0].tolist()
            payload_xyz = payload.data.root_pos_w[0].tolist()
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"cmd=[{cmd_str}] "
                f"left=({left_xyz[0]:.3f}, {left_xyz[1]:.3f}, {left_xyz[2]:.3f}) "
                f"right=({right_xyz[0]:.3f}, {right_xyz[1]:.3f}, {right_xyz[2]:.3f}) "
                f"payload=({payload_xyz[0]:.3f}, {payload_xyz[1]:.3f}, {payload_xyz[2]:.3f})"
            )

        control_step += 1
        episode_step += 1
        if warmup_steps_remaining > 0:
            warmup_steps_remaining -= 1
        just_reset = False

    simulation_app.close()


if __name__ == "__main__":
    main()
