from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import locotouch.mdp as mdp


@configclass
class DualDogCommandsCfg:
    """Shared command stream used by both robots in the dual-dog benchmark."""

    base_velocity = mdp.UniformVelocityCommandGaitLoggingCfg(
        asset_name="robot_left",
        sensor_cfg=SceneEntityCfg("robot_left_contact_sensor", body_names=".*foot"),
        resampling_time_range=(8.0, 8.0),
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandGaitLoggingCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-math.pi / 2, math.pi / 2),
        ),
        rel_standing_envs=0.1,
    )


@configclass
class DualDogObservationsCfg:
    @configclass
    class NoisyDualDogProprioceptionCfg(ObservationGroupCfg):
        velocity_commands = ObservationTermCfg(
            func=mdp.shared_generated_commands,
            scale=1.0,
            params={"command_name": "base_velocity"},
            history_length=6,
        )

        left_base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel_asset,
            scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            params={"asset_cfg": SceneEntityCfg("robot_left")},
            history_length=6,
        )
        left_projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity_asset,
            scale=1.0,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot_left")},
            history_length=6,
        )
        left_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel_asset,
            scale=1.0,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot_left")},
            history_length=6,
        )
        left_joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel_asset,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot_left")},
            history_length=6,
        )
        left_last_action = ObservationTermCfg(
            func=mdp.last_action_term,
            scale=1.0,
            params={"action_name": "joint_pos_left"},
            history_length=6,
        )

        right_base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel_asset,
            scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            params={"asset_cfg": SceneEntityCfg("robot_right")},
            history_length=6,
        )
        right_projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity_asset,
            scale=1.0,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot_right")},
            history_length=6,
        )
        right_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel_asset,
            scale=1.0,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot_right")},
            history_length=6,
        )
        right_joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel_asset,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot_right")},
            history_length=6,
        )
        right_last_action = ObservationTermCfg(
            func=mdp.last_action_term,
            scale=1.0,
            params={"action_name": "joint_pos_right"},
            history_length=6,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DenoisedDualDogProprioceptionCfg(NoisyDualDogProprioceptionCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy: NoisyDualDogProprioceptionCfg = NoisyDualDogProprioceptionCfg()
    critic: DenoisedDualDogProprioceptionCfg = DenoisedDualDogProprioceptionCfg()


@configclass
class DualDogActionsCfg:
    joint_pos_left = mdp.JointPositionActionPrevPrevCfg(
        asset_name="robot_left",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
        clip_raw_actions=True,
        raw_action_clip_value=100.0,
        raw_action_scale=0.25,
    )
    joint_pos_right = mdp.JointPositionActionPrevPrevCfg(
        asset_name="robot_right",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
        clip_raw_actions=True,
        raw_action_clip_value=100.0,
        raw_action_scale=0.25,
    )


@configclass
class DualDogRewardsCfg:
    """Placeholder reward block for the play-only dual-dog benchmark."""

    placeholder = RewardTermCfg(func=mdp.is_alive, weight=0.0)


@configclass
class DualDogEventCfg:
    """Benchmark play/train task keeps events empty for a minimal baseline."""

    pass


@configclass
class DualDogTerminationsCfg:
    """Keep only time-out termination for the play-only benchmark task."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class DualDogCurriculumCfg:
    velocity_commands = None


@configclass
class DualDogBenchmarkBaseEnvCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = MISSING
    viewer = ViewerCfg(
        eye=(8.0, 8.0, 4.5),
        resolution=(1920, 1080),
        lookat=(0.0, 0.0, 0.3),
        origin_type="world",
        env_index=0,
        asset_name="payload",
    )

    commands: DualDogCommandsCfg = DualDogCommandsCfg()
    observations: DualDogObservationsCfg = DualDogObservationsCfg()
    actions: DualDogActionsCfg = DualDogActionsCfg()
    rewards: DualDogRewardsCfg = DualDogRewardsCfg()
    terminations: DualDogTerminationsCfg = DualDogTerminationsCfg()
    events: DualDogEventCfg = DualDogEventCfg()
    curriculum: DualDogCurriculumCfg = DualDogCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**16

        if getattr(self.scene, "robot_left_contact_sensor", None) is not None:
            self.scene.robot_left_contact_sensor.update_period = self.sim.dt
        if getattr(self.scene, "robot_left_tactile_contact_sensor", None) is not None:
            self.scene.robot_left_tactile_contact_sensor.update_period = self.sim.dt
        if getattr(self.scene, "robot_right_contact_sensor", None) is not None:
            self.scene.robot_right_contact_sensor.update_period = self.sim.dt
        if getattr(self.scene, "robot_right_tactile_contact_sensor", None) is not None:
            self.scene.robot_right_tactile_contact_sensor.update_period = self.sim.dt
        if getattr(self.scene, "payload_contact_sensor", None) is not None:
            self.scene.payload_contact_sensor.update_period = self.sim.dt


def smaller_dual_dog_scene_for_playing(env_cfg: DualDogBenchmarkBaseEnvCfg) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 8.0


@configclass
class DualDogBenchmarkBaseEnvCfg_PLAY(DualDogBenchmarkBaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        smaller_dual_dog_scene_for_playing(self)
