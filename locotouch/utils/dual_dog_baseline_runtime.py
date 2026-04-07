from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass

import torch

import cli_args
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils import get_checkpoint_path


@dataclass
class PolicyObsCfg:
    history_length: int
    base_ang_vel_scale: float
    joint_vel_scale: float
    raw_action_scale: float
    raw_action_clip_value: float
    clip_raw_actions: bool


@dataclass
class CommandCfg:
    lin_vel_x_range: tuple[float, float]
    lin_vel_y_range: tuple[float, float]
    ang_vel_z_range: tuple[float, float]
    interval_s: float
    stand_prob: float
    episode_length_s: float
    initial_zero_command_steps: int


@dataclass
class RobotObsState:
    last_action: torch.Tensor
    cmd_history: deque[torch.Tensor]
    ang_vel_history: deque[torch.Tensor]
    gravity_history: deque[torch.Tensor]
    joint_pos_history: deque[torch.Tensor]
    joint_vel_history: deque[torch.Tensor]
    action_history: deque[torch.Tensor]


class DummyPolicyEnv:
    """Small VecEnv-like stub used only to reconstruct the actor from checkpoint."""

    def __init__(self, obs_dim: int, num_actions: int, device: torch.device):
        self.num_envs = 1
        self.num_actions = num_actions
        self.max_episode_length = 1
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}
        self._obs = {"policy": torch.zeros((self.num_envs, obs_dim), device=device)}

    def get_observations(self):
        return self._obs

    def reset(self):
        return self._obs

    def step(self, actions: torch.Tensor):
        raise RuntimeError("DummyPolicyEnv is inference-only.")


def extract_policy_obs_cfg(env_cfg) -> PolicyObsCfg:
    policy_obs_cfg = env_cfg.observations.policy
    history_lengths = [
        policy_obs_cfg.velocity_commands.history_length,
        policy_obs_cfg.base_ang_vel.history_length,
        policy_obs_cfg.projected_gravity.history_length,
        policy_obs_cfg.joint_pos.history_length,
        policy_obs_cfg.joint_vel.history_length,
        policy_obs_cfg.last_action.history_length,
    ]
    if len(set(history_lengths)) != 1:
        raise ValueError(f"Expected identical observation history lengths, got {history_lengths}.")

    action_cfg = env_cfg.actions.joint_pos
    return PolicyObsCfg(
        history_length=history_lengths[0],
        base_ang_vel_scale=policy_obs_cfg.base_ang_vel.scale,
        joint_vel_scale=policy_obs_cfg.joint_vel.scale,
        raw_action_scale=action_cfg.raw_action_scale,
        raw_action_clip_value=action_cfg.raw_action_clip_value,
        clip_raw_actions=action_cfg.clip_raw_actions,
    )


def extract_command_cfg(env_cfg, args_cli) -> CommandCfg:
    base_velocity_cfg = env_cfg.commands.base_velocity
    lin_vel_x_range = (
        (-args_cli.max_vx, args_cli.max_vx)
        if args_cli.max_vx is not None
        else tuple(base_velocity_cfg.ranges.lin_vel_x)
    )
    lin_vel_y_range = (
        (-args_cli.max_vy, args_cli.max_vy)
        if args_cli.max_vy is not None
        else tuple(base_velocity_cfg.ranges.lin_vel_y)
    )
    ang_vel_z_range = (
        (-args_cli.max_wz, args_cli.max_wz)
        if args_cli.max_wz is not None
        else tuple(base_velocity_cfg.ranges.ang_vel_z)
    )

    interval_s = args_cli.command_interval_s
    if interval_s is None:
        interval_s = float(base_velocity_cfg.resampling_time_range[0])

    stand_prob = (
        args_cli.stand_prob
        if args_cli.stand_prob is not None
        else float(getattr(base_velocity_cfg, "rel_standing_envs", 0.0))
    )
    episode_length_s = (
        args_cli.episode_length_s if args_cli.episode_length_s is not None else float(env_cfg.episode_length_s)
    )
    initial_zero_command_steps = args_cli.initial_zero_command_steps
    if initial_zero_command_steps is None:
        initial_zero_command_steps = int(getattr(base_velocity_cfg, "initial_zero_command_steps", 0))

    return CommandCfg(
        lin_vel_x_range=lin_vel_x_range,
        lin_vel_y_range=lin_vel_y_range,
        ang_vel_z_range=ang_vel_z_range,
        interval_s=interval_s,
        stand_prob=stand_prob,
        episode_length_s=episode_length_s,
        initial_zero_command_steps=initial_zero_command_steps,
    )


def create_obs_state(num_actions: int, history_length: int, device: torch.device) -> RobotObsState:
    zeros_cmd = torch.zeros(1, 3, device=device)
    zeros_base = torch.zeros(1, 3, device=device)
    zeros_gravity = torch.tensor([[0.0, 0.0, -1.0]], device=device)
    zeros_joint = torch.zeros(1, num_actions, device=device)
    return RobotObsState(
        last_action=zeros_joint.clone(),
        cmd_history=deque([zeros_cmd.clone() for _ in range(history_length)], maxlen=history_length),
        ang_vel_history=deque([zeros_base.clone() for _ in range(history_length)], maxlen=history_length),
        gravity_history=deque([zeros_gravity.clone() for _ in range(history_length)], maxlen=history_length),
        joint_pos_history=deque([zeros_joint.clone() for _ in range(history_length)], maxlen=history_length),
        joint_vel_history=deque([zeros_joint.clone() for _ in range(history_length)], maxlen=history_length),
        action_history=deque([zeros_joint.clone() for _ in range(history_length)], maxlen=history_length),
    )


def _roll_history(history: deque[torch.Tensor], value: torch.Tensor) -> torch.Tensor:
    history.append(value)
    return torch.cat(list(history), dim=-1)


def build_policy_obs(robot, obs_state: RobotObsState, command: torch.Tensor, obs_cfg: PolicyObsCfg) -> torch.Tensor:
    joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
    joint_vel_rel = robot.data.joint_vel
    base_ang_vel = robot.data.root_ang_vel_b * obs_cfg.base_ang_vel_scale
    projected_gravity = robot.data.projected_gravity_b

    cmd_hist = _roll_history(obs_state.cmd_history, command)
    ang_hist = _roll_history(obs_state.ang_vel_history, base_ang_vel)
    grav_hist = _roll_history(obs_state.gravity_history, projected_gravity)
    jpos_hist = _roll_history(obs_state.joint_pos_history, joint_pos_rel)
    jvel_hist = _roll_history(obs_state.joint_vel_history, joint_vel_rel * obs_cfg.joint_vel_scale)
    act_hist = _roll_history(obs_state.action_history, obs_state.last_action)

    return torch.cat([cmd_hist, ang_hist, grav_hist, jpos_hist, jvel_hist, act_hist], dim=-1)


def sample_shared_command(
    device: torch.device,
    command_cfg: CommandCfg,
    command_mode: str,
    fixed_command: tuple[float, float, float],
) -> torch.Tensor:
    if command_mode == "fixed":
        return torch.tensor([list(fixed_command)], device=device)

    if torch.rand(1, device=device).item() < command_cfg.stand_prob:
        return torch.zeros(1, 3, device=device)

    command = torch.empty(1, 3, device=device)
    command[:, 0].uniform_(*command_cfg.lin_vel_x_range)
    command[:, 1].uniform_(*command_cfg.lin_vel_y_range)
    command[:, 2].uniform_(*command_cfg.ang_vel_z_range)
    return command


def commanded_velocity(shared_command: torch.Tensor, warmup_steps_remaining: int) -> torch.Tensor:
    if warmup_steps_remaining > 0:
        return torch.zeros_like(shared_command)
    return shared_command


def reset_robot(robot, root_x: float, root_y: float, env_ids: torch.Tensor) -> None:
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0] = root_x
    root_state[:, 1] = root_y
    root_state[:, 7:] = 0.0
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def reset_payload(payload, pos_x: float, pos_y: float, pos_z: float, yaw_deg: float, env_ids: torch.Tensor) -> None:
    payload_state = payload.data.default_root_state.clone()
    payload_state[:, 0] = pos_x
    payload_state[:, 1] = pos_y
    payload_state[:, 2] = pos_z
    yaw = torch.full((1,), math.radians(yaw_deg), device=payload_state.device)
    quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    payload_state[:, 3:7] = quat
    payload_state[:, 7:] = 0.0
    payload.write_root_pose_to_sim(payload_state[:, :7], env_ids=env_ids)
    payload.write_root_velocity_to_sim(payload_state[:, 7:], env_ids=env_ids)


def reset_scene(
    scene: InteractiveScene,
    sim: SimulationContext,
    *,
    robot_separation_y: float,
    payload_center_x: float,
    payload_center_z: float,
    payload_yaw_deg: float,
) -> None:
    env_ids = torch.tensor([0], dtype=torch.long, device=scene.device)
    reset_robot(scene["robot_left"], 0.0, -robot_separation_y / 2.0, env_ids)
    reset_robot(scene["robot_right"], 0.0, robot_separation_y / 2.0, env_ids)
    reset_payload(
        scene["payload"],
        payload_center_x,
        0.0,
        payload_center_z,
        payload_yaw_deg,
        env_ids,
    )
    scene.reset(env_ids)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


def settle_scene(
    scene: InteractiveScene,
    sim: SimulationContext,
    robots: dict[str, object],
    default_joint_pos: dict[str, torch.Tensor],
    settle_steps: int,
) -> None:
    for _ in range(settle_steps):
        for name, robot in robots.items():
            robot.set_joint_position_target(default_joint_pos[name])
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


def build_scene_cfg(
    task_env_cfg,
    *,
    robot_separation_y: float,
    payload_length: float,
    payload_radius: float,
    payload_mass: float,
    payload_center_x: float,
    payload_center_z: float,
):
    scene_cfg = task_env_cfg.scene
    left_x, _, left_z = scene_cfg.robot_left.init_state.pos
    right_x, _, right_z = scene_cfg.robot_right.init_state.pos
    scene_cfg.robot_left.init_state.pos = (left_x, -robot_separation_y / 2.0, left_z)
    scene_cfg.robot_right.init_state.pos = (right_x, robot_separation_y / 2.0, right_z)
    scene_cfg.payload.spawn.radius = payload_radius
    scene_cfg.payload.spawn.height = payload_length
    scene_cfg.payload.spawn.mass_props.mass = payload_mass
    scene_cfg.payload.init_state.pos = (payload_center_x, 0.0, payload_center_z)
    return scene_cfg


def make_policy(
    policy_obs_cfg: PolicyObsCfg,
    num_actions: int,
    device: torch.device,
    *,
    policy_task: str,
    args_cli,
):
    from loco_rl.runners import OnPolicyRunner

    obs_dim = policy_obs_cfg.history_length * (3 + 3 + 3 + num_actions + num_actions + num_actions)
    dummy_env = DummyPolicyEnv(obs_dim=obs_dim, num_actions=num_actions, device=device)

    agent_cfg = cli_args.parse_rsl_rl_cfg(policy_task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    print(f"[INFO] Loading single-dog PPO checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(dummy_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    ppo_runner.load(resume_path)
    return ppo_runner.get_inference_policy(device=device)
