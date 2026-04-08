# python locotouch/scripts/play_dual_dog_baseline.py   --task Isaac-DualDogBenchmark-LocoTouch-Play-v1   --policy_task Isaac-RandCylinderTransportTeacher-LocoTouch-Play-v1   --resume_experiment locotouch_rand_cylinder_transport_teacher   --load_run 2025-09-01_21-03-58   --checkpoint model_15000.pt --payload_mass 0.5


from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

import cli_args
import locotouch.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_inv, quat_mul
from isaaclab_tasks.utils import get_checkpoint_path


@dataclass
class PolicyObsCfg:
    history_length: int
    base_ang_vel_scale: float
    joint_vel_scale: float
    raw_action_scale: float
    raw_action_clip_value: float
    clip_raw_actions: bool
    include_object_state: bool
    object_state_scale: tuple[float, ...] | None
    object_state_non_contact_obs: tuple[float, ...] | None
    object_state_last_contact_time_threshold: float
    object_state_current_contact_time_threshold: float


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
    object_state_history: deque[torch.Tensor] | None


class _CommandManagerAdapter:
    def __init__(self, command: torch.Tensor):
        self._command = command

    def get_command(self, _command_name: str | None = None) -> torch.Tensor:
        return self._command


class _ActionTermAdapter:
    def __init__(self, raw_actions: torch.Tensor):
        self.raw_actions = raw_actions


class _ActionManagerAdapter:
    def __init__(self, last_action: torch.Tensor):
        self._term = _ActionTermAdapter(last_action)

    def get_term(self, _action_name: str | None = None) -> _ActionTermAdapter:
        return self._term


class _SceneAdapter:
    def __init__(self, robot, payload=None, payload_contact_sensor=None):
        self._entities = {
            "robot": robot,
            "object": payload,
        }
        self.sensors = {"object_contact_sensor": payload_contact_sensor} if payload_contact_sensor is not None else {}

    def __getitem__(self, name: str):
        return self._entities[name]


class _SingleDogObsEnvAdapter:
    def __init__(self, robot, command: torch.Tensor, last_action: torch.Tensor, payload=None, payload_contact_sensor=None):
        self.scene = _SceneAdapter(robot, payload=payload, payload_contact_sensor=payload_contact_sensor)
        self.command_manager = _CommandManagerAdapter(command)
        self.action_manager = _ActionManagerAdapter(last_action)


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
    include_object_state = hasattr(policy_obs_cfg, "object_state")
    object_state_scale = None
    object_state_non_contact_obs = None
    object_state_last_contact_time_threshold = 0.0
    object_state_current_contact_time_threshold = 0.0
    if include_object_state:
        history_lengths.append(policy_obs_cfg.object_state.history_length)
        raw_object_state_scale = policy_obs_cfg.object_state.params.get("scale", [1.0] * 13)
        if isinstance(raw_object_state_scale, (float, int)):
            object_state_scale = tuple([float(raw_object_state_scale)] * 13)
        else:
            object_state_scale = tuple(raw_object_state_scale)
        object_state_non_contact_obs = tuple(policy_obs_cfg.object_state.params.get("non_contact_obs", [0.0] * 13))
        object_state_last_contact_time_threshold = float(
            policy_obs_cfg.object_state.params.get("last_contact_time_threshold", 0.0)
        )
        object_state_current_contact_time_threshold = float(
            policy_obs_cfg.object_state.params.get("current_contact_time_threshold", 0.0)
        )
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
        include_object_state=include_object_state,
        object_state_scale=object_state_scale,
        object_state_non_contact_obs=object_state_non_contact_obs,
        object_state_last_contact_time_threshold=object_state_last_contact_time_threshold,
        object_state_current_contact_time_threshold=object_state_current_contact_time_threshold,
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
        object_state_history=None,
    )


def _roll_history(history: deque[torch.Tensor], value: torch.Tensor) -> torch.Tensor:
    history.append(value)
    return torch.cat(list(history), dim=-1)


def _object_state_in_robot_frame(
    robot,
    payload,
    payload_contact_sensor,
    obs_cfg: PolicyObsCfg,
) -> torch.Tensor:
    if payload is None:
        if obs_cfg.object_state_non_contact_obs is not None:
            object_state = torch.tensor(
                obs_cfg.object_state_non_contact_obs,
                device=robot.data.root_pos_w.device,
                dtype=robot.data.root_pos_w.dtype,
            ).unsqueeze(0)
            if obs_cfg.object_state_scale is not None:
                scale = torch.tensor(
                    obs_cfg.object_state_scale,
                    device=object_state.device,
                    dtype=object_state.dtype,
                ).unsqueeze(0)
                object_state = object_state * scale
            return object_state
        return torch.zeros(1, 13, device=robot.data.root_pos_w.device, dtype=robot.data.root_pos_w.dtype)

    robot_quat_w = robot.data.root_quat_w
    pos_in_robot_frame = quat_apply_inverse(robot_quat_w, payload.data.root_pos_w - robot.data.root_pos_w)
    lin_vel_in_robot_frame = quat_apply_inverse(robot_quat_w, payload.data.root_lin_vel_w - robot.data.root_lin_vel_w)
    quat_in_robot_frame = quat_mul(quat_inv(robot_quat_w), payload.data.root_quat_w)
    ang_vel_in_robot_frame = quat_apply_inverse(robot_quat_w, payload.data.root_ang_vel_w - robot.data.root_ang_vel_w)
    object_state = torch.cat(
        [pos_in_robot_frame, lin_vel_in_robot_frame, quat_in_robot_frame, ang_vel_in_robot_frame],
        dim=-1,
    )

    if obs_cfg.object_state_scale is not None:
        scale = torch.tensor(obs_cfg.object_state_scale, device=object_state.device, dtype=object_state.dtype)
        object_state = object_state * scale

    if payload_contact_sensor is None or obs_cfg.object_state_non_contact_obs is None:
        return object_state

    last_contact_time = payload_contact_sensor.data.last_contact_time.reshape(object_state.shape[0], -1)
    current_contact_time = payload_contact_sensor.data.current_contact_time.reshape(object_state.shape[0], -1)
    no_contact = torch.logical_and(
        torch.max(last_contact_time, dim=1).values < obs_cfg.object_state_last_contact_time_threshold,
        torch.max(current_contact_time, dim=1).values < obs_cfg.object_state_current_contact_time_threshold,
    ).unsqueeze(-1)
    if torch.any(no_contact):
        non_contact_obs = torch.tensor(
            obs_cfg.object_state_non_contact_obs,
            device=object_state.device,
            dtype=object_state.dtype,
        ).unsqueeze(0)
        if obs_cfg.object_state_scale is not None:
            scale = torch.tensor(obs_cfg.object_state_scale, device=object_state.device, dtype=object_state.dtype).unsqueeze(0)
            non_contact_obs = non_contact_obs * scale
        object_state = torch.where(no_contact, non_contact_obs, object_state)
    return object_state


def build_policy_obs(
    robot,
    obs_state: RobotObsState,
    command: torch.Tensor,
    obs_cfg: PolicyObsCfg,
    *,
    payload=None,
    payload_contact_sensor=None,
) -> torch.Tensor:
    obs_env = _SingleDogObsEnvAdapter(
        robot,
        command,
        obs_state.last_action,
        payload=payload,
        payload_contact_sensor=payload_contact_sensor,
    )

    command_term = mdp.generated_commands(obs_env, "base_velocity")
    base_ang_vel = mdp.base_ang_vel(obs_env) * obs_cfg.base_ang_vel_scale
    projected_gravity = mdp.projected_gravity(obs_env)
    joint_pos_rel = mdp.joint_pos_rel(obs_env)
    joint_vel_rel = mdp.joint_vel_rel(obs_env)
    last_action = mdp.last_action(obs_env, "joint_pos")

    cmd_hist = _roll_history(obs_state.cmd_history, command_term)
    ang_hist = _roll_history(obs_state.ang_vel_history, base_ang_vel)
    grav_hist = _roll_history(obs_state.gravity_history, projected_gravity)
    jpos_hist = _roll_history(obs_state.joint_pos_history, joint_pos_rel)
    jvel_hist = _roll_history(obs_state.joint_vel_history, joint_vel_rel * obs_cfg.joint_vel_scale)
    act_hist = _roll_history(obs_state.action_history, last_action)

    obs_terms = [cmd_hist, ang_hist, grav_hist, jpos_hist, jvel_hist, act_hist]

    if obs_cfg.include_object_state:
        if obs_state.object_state_history is None:
            zeros_object = torch.zeros(1, 13, device=joint_pos_rel.device, dtype=joint_pos_rel.dtype)
            obs_state.object_state_history = deque(
                [zeros_object.clone() for _ in range(obs_cfg.history_length)],
                maxlen=obs_cfg.history_length,
            )
        if payload is None:
            object_state = _object_state_in_robot_frame(robot, payload, payload_contact_sensor, obs_cfg)
        else:
            object_state = mdp.object_state_in_robot_frame(
                obs_env,
                robot_cfg=SceneEntityCfg("robot"),
                object_cfg=SceneEntityCfg("object"),
                sensor_cfg=SceneEntityCfg("object_contact_sensor", body_names="Object"),
                last_contact_time_threshold=obs_cfg.object_state_last_contact_time_threshold,
                current_contact_time_threshold=obs_cfg.object_state_current_contact_time_threshold,
                non_contact_obs=list(obs_cfg.object_state_non_contact_obs) if obs_cfg.object_state_non_contact_obs is not None else [0.0] * 13,
                add_uniform_noise=False,
                scale=list(obs_cfg.object_state_scale) if obs_cfg.object_state_scale is not None else 1.0,
            )
        object_hist = _roll_history(obs_state.object_state_history, object_state)
        obs_terms.append(object_hist)

    return torch.cat(obs_terms, dim=-1)


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
    if "payload" in scene.keys():
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
    disable_payload: bool = False,
):
    scene_cfg = task_env_cfg.scene
    left_x, _, left_z = scene_cfg.robot_left.init_state.pos
    right_x, _, right_z = scene_cfg.robot_right.init_state.pos
    scene_cfg.robot_left.init_state.pos = (left_x, -robot_separation_y / 2.0, left_z)
    scene_cfg.robot_right.init_state.pos = (right_x, robot_separation_y / 2.0, right_z)
    if disable_payload:
        scene_cfg.payload = None
        scene_cfg.payload_contact_sensor = None
        return scene_cfg

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

    per_step_obs_dim = 3 + 3 + 3 + num_actions + num_actions + num_actions
    if policy_obs_cfg.include_object_state:
        per_step_obs_dim += 13
    obs_dim = policy_obs_cfg.history_length * per_step_obs_dim
    dummy_env = DummyPolicyEnv(obs_dim=obs_dim, num_actions=num_actions, device=device)

    agent_cfg = cli_args.parse_rsl_rl_cfg(policy_task, args_cli)
    repo_root = Path(__file__).resolve().parents[2]

    if args_cli.checkpoint is not None:
        checkpoint_path = Path(args_cli.checkpoint)
        if checkpoint_path.is_absolute() and checkpoint_path.exists():
            resume_path = str(checkpoint_path)
            log_dir = str(checkpoint_path.parent)
            print(f"[INFO] Loading single-dog PPO checkpoint from absolute path: {resume_path}")
            ppo_runner = OnPolicyRunner(dummy_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
            ppo_runner.load(resume_path)
            return ppo_runner.get_inference_policy(device=device)

    experiment_name = args_cli.resume_experiment if args_cli.resume_experiment is not None else agent_cfg.experiment_name
    log_root_path = repo_root / "logs" / "rsl_rl" / experiment_name
    if not log_root_path.exists():
        available_experiments = []
        logs_root = repo_root / "logs" / "rsl_rl"
        if logs_root.exists():
            available_experiments = sorted(path.name for path in logs_root.iterdir() if path.is_dir())
        raise FileNotFoundError(
            "Could not find the single-dog policy experiment directory.\n"
            f"Expected: {log_root_path}\n"
            f"policy_task: {policy_task}\n"
            f"experiment_name: {agent_cfg.experiment_name}\n"
            f"resume_experiment: {args_cli.resume_experiment}\n"
            "You can fix this by either:\n"
            "1. passing --resume_experiment <actual_experiment_dir>, or\n"
            "2. passing --checkpoint <absolute_path_to_model.pt>.\n"
            f"Available experiment directories under {logs_root}: {available_experiments}"
        )

    resume_path = get_checkpoint_path(str(log_root_path), agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    print(f"[INFO] Loading single-dog PPO checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(dummy_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    ppo_runner.load(resume_path)
    return ppo_runner.get_inference_policy(device=device)
