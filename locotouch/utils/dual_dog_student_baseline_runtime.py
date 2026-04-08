#   python locotouch/scripts/play_dual_dog_student_baseline.py \
#     --task Isaac-DualDogBenchmark-LocoTouch-Play-v1 \
#     --student_task Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-Play-v1 \
#     --log_dir_distill 2025-09-02_23-27-14 \
#     --checkpoint_distill model_7.pt \
#     --disable_payload

#  python locotouch/scripts/play_dual_dog_student_baseline.py \
#     --task Isaac-DualDogBenchmark-LocoTouch-Play-v1 \
#     --student_task Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-Play-v1 \
#     --log_dir_distill 2025-09-02_23-27-14 \
#     --checkpoint_distill model_7.pt \
#     --command_mode fixed \
#     --command_vx 0.3 \
#     --command_vy 0.0 \
#     --command_wz 0.0

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

import cli_args
import locotouch.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils import get_checkpoint_path

from locotouch.distill.student import Student


@dataclass
class StudentProprioObsCfg:
    history_length: int
    base_ang_vel_scale: float
    joint_vel_scale: float
    raw_action_scale: float
    raw_action_clip_value: float
    clip_raw_actions: bool


@dataclass
class StudentTactileObsCfg:
    tactile_signal_shape: tuple[int, int]
    contact_threshold: float
    maximal_force: float
    add_threshold_noise: bool
    threshold_n_min: float
    threshold_n_max: float
    contact_dropout_prob: float
    contact_addition_prob: float
    add_force_noise: bool
    force_n_prop_min: float
    force_n_prop_max: float


@dataclass
class StudentRobotObsState:
    last_action: torch.Tensor
    cmd_history: deque[torch.Tensor]
    ang_vel_history: deque[torch.Tensor]
    gravity_history: deque[torch.Tensor]
    joint_pos_history: deque[torch.Tensor]
    joint_vel_history: deque[torch.Tensor]
    action_history: deque[torch.Tensor]


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
    def __init__(self, robot):
        self._entities = {"robot": robot}

    def __getitem__(self, name: str):
        return self._entities[name]


class _SingleDogProprioEnvAdapter:
    def __init__(self, robot, command: torch.Tensor, last_action: torch.Tensor):
        self.scene = _SceneAdapter(robot)
        self.command_manager = _CommandManagerAdapter(command)
        self.action_manager = _ActionManagerAdapter(last_action)


def extract_student_proprio_obs_cfg(env_cfg) -> StudentProprioObsCfg:
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
        raise ValueError(f"Expected identical student proprio history lengths, got {history_lengths}.")

    action_cfg = env_cfg.actions.joint_pos
    return StudentProprioObsCfg(
        history_length=history_lengths[0],
        base_ang_vel_scale=policy_obs_cfg.base_ang_vel.scale,
        joint_vel_scale=policy_obs_cfg.joint_vel.scale,
        raw_action_scale=action_cfg.raw_action_scale,
        raw_action_clip_value=action_cfg.raw_action_clip_value,
        clip_raw_actions=action_cfg.clip_raw_actions,
    )


def extract_student_tactile_obs_cfg(env_cfg) -> StudentTactileObsCfg:
    tactile_cfg = env_cfg.observations.tactile.tactile_signals
    params = tactile_cfg.params
    return StudentTactileObsCfg(
        tactile_signal_shape=tuple(params.get("tactile_signal_shape", (17, 13))),
        contact_threshold=float(params.get("contact_threshold", 0.05)),
        maximal_force=float(params.get("maximal_force", 3.0)),
        add_threshold_noise=bool(params.get("add_threshold_noise", False)),
        threshold_n_min=float(params.get("threshold_n_min", 0.0)),
        threshold_n_max=float(params.get("threshold_n_max", 0.0)),
        contact_dropout_prob=float(params.get("contact_dropout_prob", 0.0)),
        contact_addition_prob=float(params.get("contact_addition_prob", 0.0)),
        add_force_noise=bool(params.get("add_force_noise", False)),
        force_n_prop_min=float(params.get("force_n_prop_min", 0.0)),
        force_n_prop_max=float(params.get("force_n_prop_max", 0.0)),
    )


def create_student_obs_state(num_actions: int, history_length: int, device: torch.device) -> StudentRobotObsState:
    zeros_cmd = torch.zeros(1, 3, device=device)
    zeros_base = torch.zeros(1, 3, device=device)
    zeros_gravity = torch.tensor([[0.0, 0.0, -1.0]], device=device)
    zeros_joint = torch.zeros(1, num_actions, device=device)
    return StudentRobotObsState(
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


def build_student_proprio_obs(
    robot,
    obs_state: StudentRobotObsState,
    command: torch.Tensor,
    obs_cfg: StudentProprioObsCfg,
) -> torch.Tensor:
    obs_env = _SingleDogProprioEnvAdapter(robot, command, obs_state.last_action)

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
    return torch.cat([cmd_hist, ang_hist, grav_hist, jpos_hist, jvel_hist, act_hist], dim=-1)


def build_student_binary_tactile_obs(robot, tactile_contact_sensor, tactile_cfg: StudentTactileObsCfg) -> torch.Tensor:
    robot_sensor_body_ids, _ = robot.find_bodies("sensor_.*", preserve_order=True)
    tactile_body_ids, _ = tactile_contact_sensor.find_bodies("sensor_.*", preserve_order=True)
    body_quat_w = robot.data.body_quat_w[:, robot_sensor_body_ids]
    net_forces_w = tactile_contact_sensor.data.net_forces_w[:, tactile_body_ids]
    normal_forces = -quat_apply_inverse(body_quat_w, net_forces_w)[..., 2]

    threshold = tactile_cfg.contact_threshold
    if tactile_cfg.add_threshold_noise:
        threshold = threshold + torch.rand_like(normal_forces) * (
            tactile_cfg.threshold_n_max - tactile_cfg.threshold_n_min
        ) + tactile_cfg.threshold_n_min

    contact_taxels = normal_forces > threshold

    if tactile_cfg.add_force_noise:
        noisy_forces = normal_forces.clone()
        contact_mask = contact_taxels
        noisy_forces[contact_mask] *= 1.0 + (
            torch.rand_like(noisy_forces[contact_mask])
            * (tactile_cfg.force_n_prop_max - tactile_cfg.force_n_prop_min)
            + tactile_cfg.force_n_prop_min
        )
        normal_forces = torch.clamp(noisy_forces, min=0.0)
        contact_taxels = normal_forces > threshold

    # Match BinaryTactileSignals: two identical contact channels flattened.
    return torch.stack([contact_taxels.float(), contact_taxels.float()], dim=1).flatten(start_dim=1)


def make_student_policy(
    num_actions: int,
    proprioception_dim: int,
    tactile_signal_dim: int | tuple[int, ...],
    device: torch.device,
    *,
    student_task: str,
    args_cli,
):
    distillation_cfg = cli_args.parse_distillation_cfg(student_task, args_cli)
    student = Student(
        distillation_cfg,
        proprioception_dim,
        tactile_signal_dim,
        num_actions,
    ).to(device)

    repo_root = Path(__file__).resolve().parents[2]
    distill_log_root = repo_root / distillation_cfg.log_root_path / distillation_cfg.experiment_name
    resume_path = get_checkpoint_path(str(distill_log_root), distillation_cfg.log_dir_distill, distillation_cfg.checkpoint_distill)
    print(f"[INFO] Loading single-dog student checkpoint from: {resume_path}")
    student.load_checkpoint(resume_path)
    student.eval()
    return student


def reset_student_policies(student_policies: dict[str, Student], done_mask: torch.Tensor | None = None) -> None:
    for student in student_policies.values():
        student.reset(done_mask)
