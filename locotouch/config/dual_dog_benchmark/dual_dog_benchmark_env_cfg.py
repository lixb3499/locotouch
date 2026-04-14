from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import locotouch.mdp as mdp
from locotouch.assets.dual_dog_long_cylinder import (
    DEFAULT_DOG_SEPARATION_Y,
    DualDogLongCylinderSceneCfg,
    DualDogLongCylinderSceneCfg_PLAY,
    make_dual_dog_robot_cfgs,
    make_long_cylinder_payload_cfg,
)
from locotouch.assets.locotouch import LocoTouch_CFG, LocoTouch_Instanceable_CFG
from locotouch.config.dual_dog_transport.dual_dog_transport_base_env_cfg import (
    DualDogTransportActionsCfg,
    DualDogTransportCommandsCfg,
)


LONG_ROD_MASS_KG = 3.0

BENCHMARK_LEFT_CFG, BENCHMARK_RIGHT_CFG = make_dual_dog_robot_cfgs(
    base_robot_cfg=LocoTouch_Instanceable_CFG
)
BENCHMARK_LEFT_PLAY_CFG, BENCHMARK_RIGHT_PLAY_CFG = make_dual_dog_robot_cfgs(
    base_robot_cfg=LocoTouch_CFG
)
BENCHMARK_PAYLOAD_CFG = make_long_cylinder_payload_cfg(mass=LONG_ROD_MASS_KG)


@configclass
class DualDogStudentBenchmarkSceneCfg(DualDogLongCylinderSceneCfg):
    robot_left = BENCHMARK_LEFT_CFG
    robot_right = BENCHMARK_RIGHT_CFG
    payload = BENCHMARK_PAYLOAD_CFG


@configclass
class DualDogStudentBenchmarkSceneCfg_PLAY(DualDogLongCylinderSceneCfg_PLAY):
    robot_left = BENCHMARK_LEFT_PLAY_CFG
    robot_right = BENCHMARK_RIGHT_PLAY_CFG
    payload = BENCHMARK_PAYLOAD_CFG


TACTILE_BINARY_PARAMS = {
    "tactile_signal_shape": (17, 13),
    "contact_threshold": 0.05,
    "add_threshold_noise": True,
    "threshold_n_min": -0.05 * 0.2,
    "threshold_n_max": 0.05 * 0.2,
    "contact_dropout_prob": 0.005,
    "contact_addition_prob": 0.005,
    "add_continuous_artifact": 0.0,
    "artifact_taxel_num_min": 0,
    "artifact_taxel_num_max": 3,
    "add_force_noise": True,
    "force_n_prop_min": -0.1,
    "force_n_prop_max": 0.1,
    "maximal_force": 3.0,
    "total_levels": 5,
    "add_level_noise": True,
    "level_n_min": -1,
    "level_n_max": 1,
}


@configclass
class LeftStudentPolicyCfg(ObservationGroupCfg):
    velocity_commands = ObservationTermCfg(
        func=mdp.shared_generated_commands,
        params={"command_name": "base_velocity"},
        history_length=6,
    )
    base_ang_vel = ObservationTermCfg(
        func=mdp.base_ang_vel_asset,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
        scale=0.25,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        history_length=6,
    )
    projected_gravity = ObservationTermCfg(
        func=mdp.projected_gravity_asset,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
        noise=Unoise(n_min=-0.05, n_max=0.05),
        history_length=6,
    )
    joint_pos = ObservationTermCfg(
        func=mdp.joint_pos_rel_asset,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=6,
    )
    joint_vel = ObservationTermCfg(
        func=mdp.joint_vel_rel_asset,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
        scale=0.05,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        history_length=6,
    )
    last_action = ObservationTermCfg(
        func=mdp.last_action_term,
        params={"action_name": "joint_pos_left"},
        history_length=6,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class RightStudentPolicyCfg(ObservationGroupCfg):
    velocity_commands = ObservationTermCfg(
        func=mdp.shared_generated_commands,
        params={"command_name": "base_velocity"},
        history_length=6,
    )
    base_ang_vel = ObservationTermCfg(
        func=mdp.base_ang_vel_asset,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
        scale=0.25,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        history_length=6,
    )
    projected_gravity = ObservationTermCfg(
        func=mdp.projected_gravity_asset,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
        noise=Unoise(n_min=-0.05, n_max=0.05),
        history_length=6,
    )
    joint_pos = ObservationTermCfg(
        func=mdp.joint_pos_rel_asset,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=6,
    )
    joint_vel = ObservationTermCfg(
        func=mdp.joint_vel_rel_asset,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
        scale=0.05,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        history_length=6,
    )
    last_action = ObservationTermCfg(
        func=mdp.last_action_term,
        params={"action_name": "joint_pos_right"},
        history_length=6,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class LeftBinaryTactileCfg(ObservationGroupCfg):
    tactile = ObservationTermCfg(
        func=mdp.BinaryTactileSignals,
        params={
            "asset_cfg": SceneEntityCfg("robot_left", body_names="sensor_.*"),
            "sensor_cfg": SceneEntityCfg("robot_left_tactile_contact_sensor", body_names="sensor_.*"),
            **TACTILE_BINARY_PARAMS,
        },
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class RightBinaryTactileCfg(ObservationGroupCfg):
    tactile = ObservationTermCfg(
        func=mdp.BinaryTactileSignals,
        params={
            "asset_cfg": SceneEntityCfg("robot_right", body_names="sensor_.*"),
            "sensor_cfg": SceneEntityCfg("robot_right_tactile_contact_sensor", body_names="sensor_.*"),
            **TACTILE_BINARY_PARAMS,
        },
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class DualDogStudentBenchmarkObservationsCfg:
    # Keep a conventional policy group so generic env tooling still works.
    policy: LeftStudentPolicyCfg = LeftStudentPolicyCfg()
    left_policy: LeftStudentPolicyCfg = LeftStudentPolicyCfg()
    left_tactile: LeftBinaryTactileCfg = LeftBinaryTactileCfg()
    right_policy: RightStudentPolicyCfg = RightStudentPolicyCfg()
    right_tactile: RightBinaryTactileCfg = RightBinaryTactileCfg()


@configclass
class DualDogStudentBenchmarkRewardsCfg:
    placeholder = RewardTermCfg(func=mdp.is_alive, weight=0.0)


@configclass
class DualDogStudentBenchmarkTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class DualDogStudentBenchmarkEventCfg:
    reset_left_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_left"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_right_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_right"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_left_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_left"),
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_right_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_right"),
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_payload = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("payload"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.04, 0.04),
                "z": (0.02, 0.02),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi / 12, math.pi / 12),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class DualDogStudentBenchmarkBaseEnvCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = MISSING
    viewer = ViewerCfg(
        eye=(8.0, 8.0, 4.5),
        resolution=(1920, 1080),
        lookat=(0.0, 0.0, 0.3),
        origin_type="world",
        env_index=0,
        asset_name="payload",
    )

    commands: DualDogTransportCommandsCfg = DualDogTransportCommandsCfg()
    observations: DualDogStudentBenchmarkObservationsCfg = DualDogStudentBenchmarkObservationsCfg()
    actions: DualDogTransportActionsCfg = DualDogTransportActionsCfg()
    rewards: DualDogStudentBenchmarkRewardsCfg = DualDogStudentBenchmarkRewardsCfg()
    terminations: DualDogStudentBenchmarkTerminationsCfg = DualDogStudentBenchmarkTerminationsCfg()
    events: DualDogStudentBenchmarkEventCfg = DualDogStudentBenchmarkEventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 8, math.pi / 8)

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        if getattr(self.scene, "robot_left_contact_sensor", None) is not None:
            self.scene.robot_left_contact_sensor.update_period = self.sim.dt
        if getattr(self.scene, "robot_right_contact_sensor", None) is not None:
            self.scene.robot_right_contact_sensor.update_period = self.sim.dt
        if getattr(self.scene, "robot_left_tactile_contact_sensor", None) is not None:
            self.scene.robot_left_tactile_contact_sensor.update_period = 0.025
        if getattr(self.scene, "robot_right_tactile_contact_sensor", None) is not None:
            self.scene.robot_right_tactile_contact_sensor.update_period = 0.025
        if getattr(self.scene, "payload_contact_sensor", None) is not None:
            self.scene.payload_contact_sensor.update_period = self.sim.dt


@configclass
class DualDogStudentBenchmarkEnvCfg(DualDogStudentBenchmarkBaseEnvCfg):
    scene = DualDogStudentBenchmarkSceneCfg(num_envs=64, env_spacing=8.0)

    def __post_init__(self):
        super().__post_init__()


@configclass
class DualDogStudentBenchmarkEnvCfg_PLAY(DualDogStudentBenchmarkEnvCfg):
    scene = DualDogStudentBenchmarkSceneCfg_PLAY(
        num_envs=1,
        env_spacing=8.0,
        replicate_physics=False,
        lazy_sensor_update=False,
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 8, math.pi / 8)
        self.commands.base_velocity.rel_standing_envs = self.commands.base_velocity.final_rel_standing_envs
        self.commands.base_velocity.initial_zero_command_steps = (
            self.commands.base_velocity.final_initial_zero_command_steps
        )
