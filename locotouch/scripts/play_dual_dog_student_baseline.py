"""Play a dual-dog long-rod benchmark by deploying the same single-dog tactile student on both dogs."""

import argparse
import os
import signal
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args


parser = argparse.ArgumentParser(description="Play a dual-dog benchmark with single-dog tactile student policies.")
parser.add_argument("--video", action="store_true", default=False, help="Record a benchmark video.")
parser.add_argument("--video_length", type=int, default=600, help="Recorded video length in policy steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-DualDogBenchmark-LocoTouch-Play-v1", help="Benchmark task.")
parser.add_argument(
    "--student_task",
    type=str,
    default="Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-Play-v1",
    help="Single-dog tactile student task used to parse the distillation config.",
)
cli_args.add_distillation_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg

from locotouch import *  # noqa: F401,F403
from locotouch.distill.tactile_recorder import TactileRecorder
from locotouch.utils.dual_dog_student_baseline_runtime import (
    get_side_observations,
    make_student_policy,
)


def _tactile_dim_from_shape(tactile_shape: torch.Size):
    return tactile_shape[0] if len(tactile_shape) == 1 else tuple(tactile_shape)


_STOP_REQUESTED = False


def _handle_sigint(signum, frame):
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("[INFO] SIGINT received. Finishing current step and closing cleanly...")


def main():
    env = None
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        env_obs, _ = env.reset()
        left_proprio_dim = env_obs["left_policy"].shape[-1]
        left_tactile_dim = _tactile_dim_from_shape(env_obs["left_tactile"].shape[1:])
        action_dim = env.unwrapped.action_manager.get_term("joint_pos_left").raw_actions.shape[-1]

        left_student, resume_path, distillation_cfg = make_student_policy(
            args_cli.student_task,
            args_cli,
            proprioception_dim=left_proprio_dim,
            tactile_signal_dim=left_tactile_dim,
            action_dim=action_dim,
        )
        right_student, _, _ = make_student_policy(
            args_cli.student_task,
            args_cli,
            proprioception_dim=left_proprio_dim,
            tactile_signal_dim=left_tactile_dim,
            action_dim=action_dim,
        )
        print(f"[INFO] Loading student checkpoint from: {resume_path}")

        if args_cli.video:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_kwargs = {
                "video_folder": os.path.join(os.path.dirname(resume_path), "videos", "dual_dog_benchmark"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
                "name_prefix": f"dual-dog-benchmark-{timestamp}",
            }
            print("[INFO] Recording benchmark video.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
            env_obs, _ = env.reset()

        left_tactile_recorder = TactileRecorder(
            env.unwrapped.device,
            env.unwrapped.num_envs,
            left_tactile_dim,
            distillation_cfg.min_delay,
            distillation_cfg.max_delay,
        )
        right_tactile_recorder = TactileRecorder(
            env.unwrapped.device,
            env.unwrapped.num_envs,
            left_tactile_dim,
            distillation_cfg.min_delay,
            distillation_cfg.max_delay,
        )

        timestep = 0
        with torch.inference_mode():
            while simulation_app.is_running():
                if _STOP_REQUESTED:
                    break
                left_obs = get_side_observations(env_obs, "left")
                right_obs = get_side_observations(env_obs, "right")
                shared_command = env.unwrapped.command_manager.get_command("base_velocity")[0]
                print(f"[CMD] base_velocity = {shared_command.tolist()}")

                left_tactile_recorder.record_new_tactile_signals(left_obs["tactile"])
                left_obs["tactile"] = left_tactile_recorder.get_tactile_signals().clone()
                right_tactile_recorder.record_new_tactile_signals(right_obs["tactile"])
                right_obs["tactile"] = right_tactile_recorder.get_tactile_signals().clone()

                left_actions = left_student.extract_input_and_forward(left_obs)
                right_actions = right_student.extract_input_and_forward(right_obs)
                actions = torch.cat((left_actions, right_actions), dim=-1)

                env_obs, _, terminated, time_outs, _ = env.step(actions)
                dones = terminated | time_outs
                if dones.any():
                    done_idx = dones.nonzero(as_tuple=False).flatten()
                    left_tactile_recorder.reset(done_idx)
                    right_tactile_recorder.reset(done_idx)
                    left_student.reset(dones)
                    right_student.reset(dones)

                if args_cli.video:
                    timestep += 1
                    if timestep >= args_cli.video_length:
                        break
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
