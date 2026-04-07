from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .actions import JointPositionActionPrevPrev

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def shared_generated_commands(
    env: ManagerBasedEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def base_ang_vel_asset(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity_asset(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def joint_pos_rel_asset(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_vel_rel_asset(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def last_action_term(
    env: ManagerBasedEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    action_term: JointPositionActionPrevPrev = env.action_manager.get_term(action_name)  # type: ignore
    return action_term.raw_actions


def action_rate_term_ngt(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    action_term: JointPositionActionPrevPrev = env.action_manager.get_term(action_name)  # type: ignore
    return torch.sum(torch.square(action_term.raw_actions - action_term.prev_raw_actions), dim=1)


def bad_orientation_asset(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    cos_tilt = torch.clamp(-projected_gravity[:, 2], min=-1.0, max=1.0)
    tilt_angle = torch.acos(cos_tilt)
    return tilt_angle > limit_angle


def root_height_below_minimum_asset(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def illegal_body_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot_contact_sensor"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_mag = torch.max(
        torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
    )[0]
    return torch.any(contact_mag > threshold, dim=1)
