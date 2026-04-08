"""Play-only dual-dog baseline driven by a single-dog student policy.

This script mirrors the dual-dog teacher baseline but swaps the policy input to:

- proprioception matching the single-dog student backbone input
- tactile observations from each robot's local tactile taxels

The same pretrained single-dog student policy weights are loaded twice so the
left/right robots keep independent recurrent state while sharing parameters.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Play a dual-dog baseline with a shared single-dog student policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-DualDogBenchmark-LocoTouch-Play-v1",
    help="Registered dual-dog benchmark task that provides the scene and shared-command defaults.",
)
parser.add_argument(
    "--student_task",
    type=str,
    default="Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-Play-v1",
    help="Single-dog student source task whose distillation config/checkpoint will be loaded for both robots.",
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
parser.add_argument(
    "--disable_payload",
    action="store_true",
    default=False,
    help="Do not spawn the long cylinder payload. Useful for bring-up without object contacts.",
)
parser.add_argument("--payload_center_x", type=float, default=0.0)
parser.add_argument("--payload_center_z", type=float, default=0.38)
parser.add_argument("--payload_yaw_deg", type=float, default=0.0, help="Yaw of the payload. 0 means aligned with +Y.")
parser.add_argument("--payload_drop_height", type=float, default=0.20, help="Reset when the payload center falls below this z.")
parser.add_argument("--robot_fall_height", type=float, default=0.15, help="Reset when either robot base falls below this z.")
cli_args.add_distillation_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Distillation config parsing expects a few generic CLI fields that are not part
# of this lightweight play script.
if not hasattr(args_cli, "logger"):
    args_cli.logger = None
if not hasattr(args_cli, "training"):
    args_cli.training = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab_tasks.utils import parse_env_cfg

from locotouch import *  # noqa: F401
from locotouch.assets.dual_dog_long_cylinder import make_dual_dog_robot_cfgs
from locotouch.assets.locotouch import LocoTouch_CFG
from locotouch.utils.dual_dog_baseline_runtime import (
    build_scene_cfg,
    commanded_velocity,
    extract_command_cfg,
    reset_scene,
    sample_shared_command,
    settle_scene,
)
from locotouch.utils.dual_dog_student_baseline_runtime import (
    build_student_binary_tactile_obs,
    build_student_proprio_obs,
    create_student_obs_state,
    extract_student_proprio_obs_cfg,
    extract_student_tactile_obs_cfg,
    make_student_policy,
    reset_student_policies,
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
    student_env_cfg = parse_env_cfg(
        args_cli.student_task,
        device=sim_device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    proprio_obs_cfg = extract_student_proprio_obs_cfg(student_env_cfg)
    tactile_obs_cfg = extract_student_tactile_obs_cfg(student_env_cfg)
    command_cfg = extract_command_cfg(student_env_cfg, args_cli)
    control_decimation = int(task_env_cfg.decimation)

    scene_cfg = build_scene_cfg(
        task_env_cfg,
        robot_separation_y=args_cli.robot_separation_y,
        payload_length=args_cli.payload_length,
        payload_radius=args_cli.payload_radius,
        payload_mass=args_cli.payload_mass,
        payload_center_x=args_cli.payload_center_x,
        payload_center_z=args_cli.payload_center_z,
        disable_payload=args_cli.disable_payload,
    )
    student_left_cfg, student_right_cfg = make_dual_dog_robot_cfgs(
        base_robot_cfg=LocoTouch_CFG,
        left_prim_path="{ENV_REGEX_NS}/RobotLeft",
        right_prim_path="{ENV_REGEX_NS}/RobotRight",
        lateral_separation_y=args_cli.robot_separation_y,
    )
    scene_cfg.robot_left = student_left_cfg
    scene_cfg.robot_right = student_right_cfg
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
    print("[INFO] Dual-dog student benchmark scene created.")

    robots = {
        "robot_left": scene["robot_left"],
        "robot_right": scene["robot_right"],
    }
    tactile_sensors = {
        "robot_left": scene.sensors["robot_left_tactile_contact_sensor"],
        "robot_right": scene.sensors["robot_right_tactile_contact_sensor"],
    }
    payload = scene["payload"] if not args_cli.disable_payload else None
    num_actions = robots["robot_left"].num_joints
    tactile_dim = 2 * tactile_obs_cfg.tactile_signal_shape[0] * tactile_obs_cfg.tactile_signal_shape[1]
    proprio_dim = proprio_obs_cfg.history_length * (3 + 3 + 3 + num_actions + num_actions + num_actions)
    student_policies = {
        name: make_student_policy(
            num_actions=num_actions,
            proprioception_dim=proprio_dim,
            tactile_signal_dim=tactile_dim,
            device=scene.device,
            student_task=args_cli.student_task,
            args_cli=args_cli,
        )
        for name in robots
    }

    default_joint_pos = {name: robot.data.default_joint_pos.clone() for name, robot in robots.items()}
    obs_states = {
        name: create_student_obs_state(
            num_actions=num_actions,
            history_length=proprio_obs_cfg.history_length,
            device=scene.device,
        )
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
    reset_student_policies(student_policies)

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
    if payload is None:
        print("[INFO] Payload disabled for this student benchmark run.")
    else:
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
        payload_height = payload.data.root_pos_w[0, 2].item() if payload is not None else float("inf")
        needs_reset = (
            (payload is not None and payload_height < args_cli.payload_drop_height)
            or left_height < args_cli.robot_fall_height
            or right_height < args_cli.robot_fall_height
            or episode_step >= episode_steps
        )

        if needs_reset:
            print(
                f"[INFO] Resetting student benchmark: payload_z={payload_height:.3f}, "
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
                name: create_student_obs_state(
                    num_actions=num_actions,
                    history_length=proprio_obs_cfg.history_length,
                    device=scene.device,
                )
                for name in robots
            }
            settle_scene(scene, sim, robots, default_joint_pos, args_cli.settle_steps)
            reset_student_policies(student_policies)
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

        raw_actions = {}
        with torch.inference_mode():
            for name, robot in robots.items():
                proprio_obs = build_student_proprio_obs(robot, obs_states[name], command, proprio_obs_cfg)
                tactile_obs = build_student_binary_tactile_obs(robot, tactile_sensors[name], tactile_obs_cfg)
                raw_actions[name] = student_policies[name](proprio_obs, tactile_obs)

        targets = {}
        for name in robots:
            action = raw_actions[name]
            if proprio_obs_cfg.clip_raw_actions:
                action = torch.clamp(action, -proprio_obs_cfg.raw_action_clip_value, proprio_obs_cfg.raw_action_clip_value)
            scaled_action = action * proprio_obs_cfg.raw_action_scale
            obs_states[name].last_action = scaled_action.clone()
            targets[name] = default_joint_pos[name] + scaled_action

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
            if payload is None:
                payload_text = "payload=disabled"
            else:
                payload_xyz = payload.data.root_pos_w[0].tolist()
                payload_text = f"payload=({payload_xyz[0]:.3f}, {payload_xyz[1]:.3f}, {payload_xyz[2]:.3f})"
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"cmd=[{cmd_str}] "
                f"left=({left_xyz[0]:.3f}, {left_xyz[1]:.3f}, {left_xyz[2]:.3f}) "
                f"right=({right_xyz[0]:.3f}, {right_xyz[1]:.3f}, {right_xyz[2]:.3f}) "
                f"{payload_text}"
            )

        control_step += 1
        episode_step += 1
        if warmup_steps_remaining > 0:
            warmup_steps_remaining -= 1
        just_reset = False

    simulation_app.close()


if __name__ == "__main__":
    main()
