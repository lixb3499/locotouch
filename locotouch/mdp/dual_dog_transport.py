from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def robot_pair_midpoint_pos_w(
    env: ManagerBasedEnv,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    left_robot: RigidObject = env.scene[left_cfg.name]
    right_robot: RigidObject = env.scene[right_cfg.name]
    return 0.5 * (left_robot.data.root_pos_w + right_robot.data.root_pos_w)


def robot_pair_midpoint_lin_vel_w(
    env: ManagerBasedEnv,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    left_robot: RigidObject = env.scene[left_cfg.name]
    right_robot: RigidObject = env.scene[right_cfg.name]
    return 0.5 * (left_robot.data.root_lin_vel_w + right_robot.data.root_lin_vel_w)


def robot_pair_yaw_quat_w(
    env: ManagerBasedEnv,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    left_robot: RigidObject = env.scene[left_cfg.name]
    right_robot: RigidObject = env.scene[right_cfg.name]
    forward_axis = torch.tensor([1.0, 0.0, 0.0], device=left_robot.device, dtype=left_robot.data.root_pos_w.dtype)
    left_forward_w = quat_apply(yaw_quat(left_robot.data.root_quat_w), forward_axis.unsqueeze(0).expand(left_robot.data.root_quat_w.shape[0], -1))
    right_forward_w = quat_apply(yaw_quat(right_robot.data.root_quat_w), forward_axis.unsqueeze(0).expand(right_robot.data.root_quat_w.shape[0], -1))
    pair_forward_w = left_forward_w + right_forward_w
    pair_forward_norm = torch.linalg.norm(pair_forward_w[:, :2], dim=1, keepdim=True).clamp_min(1.0e-6)
    pair_forward_xy = pair_forward_w[:, :2] / pair_forward_norm
    pair_yaw = torch.atan2(pair_forward_xy[:, 1], pair_forward_xy[:, 0])
    zeros = torch.zeros_like(pair_yaw)
    return quat_from_euler_xyz(zeros, zeros, pair_yaw)


def robot_pair_midpoint_lin_vel_in_left_frame(
    env: ManagerBasedEnv,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    pair_quat_w = robot_pair_yaw_quat_w(env, left_cfg, right_cfg)
    pair_lin_vel_w = robot_pair_midpoint_lin_vel_w(env, left_cfg, right_cfg)
    return quat_apply_inverse(pair_quat_w, pair_lin_vel_w)


def robot_pair_mean_ang_vel_in_left_frame(
    env: ManagerBasedEnv,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    right_robot: RigidObject = env.scene[right_cfg.name]
    left_robot: RigidObject = env.scene[left_cfg.name]
    pair_quat_w = robot_pair_yaw_quat_w(env, left_cfg, right_cfg)
    pair_ang_vel_w = 0.5 * (left_robot.data.root_ang_vel_w + right_robot.data.root_ang_vel_w)
    return quat_apply_inverse(pair_quat_w, pair_ang_vel_w)


def payload_center_in_pair_frame(
    env: ManagerBasedEnv,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    payload: RigidObject = env.scene[payload_cfg.name]
    left_robot: RigidObject = env.scene[left_cfg.name]
    right_robot: RigidObject = env.scene[right_cfg.name]
    pair_quat_w = robot_pair_yaw_quat_w(env, left_cfg, right_cfg)
    pair_midpoint = 0.5 * (left_robot.data.root_pos_w + right_robot.data.root_pos_w)
    return quat_apply_inverse(pair_quat_w, payload.data.root_pos_w - pair_midpoint)


def payload_lin_vel_in_pair_frame(
    env: ManagerBasedEnv,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    payload: RigidObject = env.scene[payload_cfg.name]
    left_robot: RigidObject = env.scene[left_cfg.name]
    right_robot: RigidObject = env.scene[right_cfg.name]
    pair_quat_w = robot_pair_yaw_quat_w(env, left_cfg, right_cfg)
    pair_midpoint_lin_vel = 0.5 * (left_robot.data.root_lin_vel_w + right_robot.data.root_lin_vel_w)
    return quat_apply_inverse(pair_quat_w, payload.data.root_lin_vel_w - pair_midpoint_lin_vel)


def payload_ang_vel_in_pair_frame(
    env: ManagerBasedEnv,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    payload: RigidObject = env.scene[payload_cfg.name]
    pair_quat_w = robot_pair_yaw_quat_w(env, left_cfg, right_cfg)
    return quat_apply_inverse(pair_quat_w, payload.data.root_ang_vel_w)


def payload_axis_in_pair_frame(
    env: ManagerBasedEnv,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
    rod_axis_payload: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    payload: RigidObject = env.scene[payload_cfg.name]
    pair_quat_w = robot_pair_yaw_quat_w(env, left_cfg, right_cfg)
    axis_payload = torch.tensor(rod_axis_payload, device=payload.device, dtype=payload.data.root_pos_w.dtype).unsqueeze(0)
    axis_world = quat_apply(payload.data.root_quat_w, axis_payload.expand(payload.data.root_quat_w.shape[0], -1))
    return quat_apply_inverse(pair_quat_w, axis_world)


def payload_support_points_in_robot_frames(
    env: ManagerBasedEnv,
    support_half_span: float,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> tuple[torch.Tensor, torch.Tensor]:
    payload: RigidObject = env.scene[payload_cfg.name]
    left_robot: RigidObject = env.scene[left_cfg.name]
    right_robot: RigidObject = env.scene[right_cfg.name]

    support_offset_payload = torch.tensor(
        [[0.0, -support_half_span, 0.0], [0.0, support_half_span, 0.0]],
        device=payload.device,
        dtype=payload.data.root_pos_w.dtype,
    )
    left_support_offset_world = quat_apply(
        payload.data.root_quat_w,
        support_offset_payload[0].unsqueeze(0).expand(payload.data.root_quat_w.shape[0], -1),
    )
    right_support_offset_world = quat_apply(
        payload.data.root_quat_w,
        support_offset_payload[1].unsqueeze(0).expand(payload.data.root_quat_w.shape[0], -1),
    )
    left_support_w = payload.data.root_pos_w + left_support_offset_world
    right_support_w = payload.data.root_pos_w + right_support_offset_world
    left_support_b = quat_apply_inverse(left_robot.data.root_quat_w, left_support_w - left_robot.data.root_pos_w)
    right_support_b = quat_apply_inverse(right_robot.data.root_quat_w, right_support_w - right_robot.data.root_pos_w)
    return left_support_b, right_support_b


def payload_support_points_lin_vel_in_robot_frames(
    env: ManagerBasedEnv,
    support_half_span: float,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> tuple[torch.Tensor, torch.Tensor]:
    payload: RigidObject = env.scene[payload_cfg.name]
    left_robot: RigidObject = env.scene[left_cfg.name]
    right_robot: RigidObject = env.scene[right_cfg.name]
    support_offset_payload = torch.tensor(
        [[0.0, -support_half_span, 0.0], [0.0, support_half_span, 0.0]],
        device=payload.device,
        dtype=payload.data.root_pos_w.dtype,
    )
    support_offset_world = quat_apply(
        payload.data.root_quat_w.repeat_interleave(2, dim=0),
        support_offset_payload.repeat(payload.data.root_quat_w.shape[0], 1),
    ).view(payload.data.root_quat_w.shape[0], 2, 3)
    support_ang_component = torch.cross(
        payload.data.root_ang_vel_w.unsqueeze(1).expand(-1, 2, -1),
        support_offset_world,
        dim=-1,
    )
    support_lin_vel_w = payload.data.root_lin_vel_w.unsqueeze(1) + support_ang_component
    left_support_lin_vel_b = quat_apply_inverse(
        left_robot.data.root_quat_w,
        support_lin_vel_w[:, 0] - left_robot.data.root_lin_vel_w,
    )
    right_support_lin_vel_b = quat_apply_inverse(
        right_robot.data.root_quat_w,
        support_lin_vel_w[:, 1] - right_robot.data.root_lin_vel_w,
    )
    return left_support_lin_vel_b, right_support_lin_vel_b


def left_support_point_in_left_robot_frame(
    env: ManagerBasedEnv,
    support_half_span: float,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    left_support_b, _ = payload_support_points_in_robot_frames(env, support_half_span, payload_cfg, left_cfg, right_cfg)
    return left_support_b


def right_support_point_in_right_robot_frame(
    env: ManagerBasedEnv,
    support_half_span: float,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    _, right_support_b = payload_support_points_in_robot_frames(env, support_half_span, payload_cfg, left_cfg, right_cfg)
    return right_support_b


def left_support_point_lin_vel_in_left_robot_frame(
    env: ManagerBasedEnv,
    support_half_span: float,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    left_support_lin_vel_b, _ = payload_support_points_lin_vel_in_robot_frames(
        env, support_half_span, payload_cfg, left_cfg, right_cfg
    )
    return left_support_lin_vel_b


def right_support_point_lin_vel_in_right_robot_frame(
    env: ManagerBasedEnv,
    support_half_span: float,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    _, right_support_lin_vel_b = payload_support_points_lin_vel_in_robot_frames(
        env, support_half_span, payload_cfg, left_cfg, right_cfg
    )
    return right_support_lin_vel_b


def robots_relative_pos_in_left_frame(
    env: ManagerBasedEnv,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    left_robot: RigidObject = env.scene[left_cfg.name]
    right_robot: RigidObject = env.scene[right_cfg.name]
    pair_quat_w = robot_pair_yaw_quat_w(env, left_cfg, right_cfg)
    return quat_apply_inverse(pair_quat_w, right_robot.data.root_pos_w - left_robot.data.root_pos_w)


def robots_relative_lin_vel_in_left_frame(
    env: ManagerBasedEnv,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    left_robot: RigidObject = env.scene[left_cfg.name]
    right_robot: RigidObject = env.scene[right_cfg.name]
    pair_quat_w = robot_pair_yaw_quat_w(env, left_cfg, right_cfg)
    return quat_apply_inverse(
        pair_quat_w,
        right_robot.data.root_lin_vel_w - left_robot.data.root_lin_vel_w,
    )


def shared_command_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
    sigma: float = 0.25,
) -> torch.Tensor:
    lin_reward = shared_command_lin_tracking_reward(env, command_name, left_cfg, right_cfg, sigma)
    ang_reward = shared_command_ang_tracking_reward(env, command_name, left_cfg, right_cfg, sigma)
    return 0.5 * (lin_reward + ang_reward)


def shared_command_lin_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
    sigma: float = 0.25,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    pair_lin_vel_b = robot_pair_midpoint_lin_vel_in_left_frame(env, left_cfg, right_cfg)
    lin_vel_error = torch.sum(torch.square(command[:, :2] - pair_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / sigma)


def shared_command_ang_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
    sigma: float = 0.25,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    pair_ang_vel_b = robot_pair_mean_ang_vel_in_left_frame(env, left_cfg, right_cfg)
    ang_vel_error = torch.square(command[:, 2] - pair_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / sigma)


def payload_upright_reward(
    env: ManagerBasedRLEnv,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
) -> torch.Tensor:
    payload: RigidObject = env.scene[payload_cfg.name]
    # projected gravity z close to -1 means the payload stays upright in roll/pitch.
    return torch.square(payload.data.projected_gravity_b[:, 2])


def payload_height_tracking_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    sigma: float = 0.05,
) -> torch.Tensor:
    payload: RigidObject = env.scene[payload_cfg.name]
    height_error = torch.square(payload.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error / sigma)


def payload_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
) -> torch.Tensor:
    payload: RigidObject = env.scene[payload_cfg.name]
    return -torch.sum(torch.square(payload.data.root_ang_vel_w), dim=1)


def payload_support_point_height_difference_penalty(
    env: ManagerBasedRLEnv,
    support_half_span: float,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
) -> torch.Tensor:
    payload: RigidObject = env.scene[payload_cfg.name]
    support_offset_payload = torch.tensor(
        [[0.0, -support_half_span, 0.0], [0.0, support_half_span, 0.0]],
        device=payload.device,
        dtype=payload.data.root_pos_w.dtype,
    )
    left_offset_world = quat_apply(
        payload.data.root_quat_w,
        support_offset_payload[0].unsqueeze(0).expand(payload.data.root_quat_w.shape[0], -1),
    )
    right_offset_world = quat_apply(
        payload.data.root_quat_w,
        support_offset_payload[1].unsqueeze(0).expand(payload.data.root_quat_w.shape[0], -1),
    )
    left_support_w = payload.data.root_pos_w + left_offset_world
    right_support_w = payload.data.root_pos_w + right_offset_world
    return -torch.abs(left_support_w[:, 2] - right_support_w[:, 2])


def payload_support_point_velocity_difference_penalty(
    env: ManagerBasedRLEnv,
    support_half_span: float,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
) -> torch.Tensor:
    payload: RigidObject = env.scene[payload_cfg.name]
    support_offset_payload = torch.tensor(
        [[0.0, -support_half_span, 0.0], [0.0, support_half_span, 0.0]],
        device=payload.device,
        dtype=payload.data.root_pos_w.dtype,
    )
    support_offset_world = quat_apply(
        payload.data.root_quat_w.repeat_interleave(2, dim=0),
        support_offset_payload.repeat(payload.data.root_quat_w.shape[0], 1),
    ).view(payload.data.root_quat_w.shape[0], 2, 3)
    support_ang_component = torch.cross(
        payload.data.root_ang_vel_w.unsqueeze(1).expand(-1, 2, -1),
        support_offset_world,
        dim=-1,
    )
    support_lin_vel_w = payload.data.root_lin_vel_w.unsqueeze(1) + support_ang_component
    return -torch.linalg.norm(support_lin_vel_w[:, 0] - support_lin_vel_w[:, 1], dim=1)


def payload_center_xy_deviation_penalty(
    env: ManagerBasedRLEnv,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot_right"),
) -> torch.Tensor:
    center_b = payload_center_in_pair_frame(env, payload_cfg, left_cfg, right_cfg)
    return -torch.linalg.norm(center_b[:, :2], dim=1)


def payload_below_minimum_height(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
) -> torch.Tensor:
    payload: RigidObject = env.scene[payload_cfg.name]
    return payload.data.root_pos_w[:, 2] < minimum_height
