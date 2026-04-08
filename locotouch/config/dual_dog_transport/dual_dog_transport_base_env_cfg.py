from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
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
from locotouch.assets.dual_dog_long_cylinder import DEFAULT_DOG_SEPARATION_Y


ROD_HALF_LENGTH = 1.75
SUPPORT_HALF_SPAN = 0.5 * DEFAULT_DOG_SEPARATION_Y


@configclass
class DualDogTransportCommandsCfg:
    base_velocity = mdp.WarmupUniformVelocityCommandMultiSamplingCfg(
        asset_name="robot_left",
        resampling_time_range=(8.0, 8.0),
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.15, 0.15),
            lin_vel_y=(-0.08, 0.08),
            ang_vel_z=(-math.pi / 10, math.pi / 10),
        ),
        rel_standing_envs=0.2,
        initial_zero_command_steps=50,
        final_rel_standing_envs=0.1,
        final_initial_zero_command_steps=50,
        new_command_probs=0.15,
    )


@configclass
class DualDogTransportObservationsCfg:
    @configclass
    class TeacherPolicyCfg(ObservationGroupCfg):
        velocity_commands = ObservationTermCfg(
            func=mdp.shared_generated_commands,
            params={"command_name": "base_velocity"},
            history_length=6,
        )

        left_base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel_asset,
            params={"asset_cfg": SceneEntityCfg("robot_left")},
            scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=6,
        )
        left_projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity_asset,
            params={"asset_cfg": SceneEntityCfg("robot_left")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=6,
        )
        left_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel_asset,
            params={"asset_cfg": SceneEntityCfg("robot_left")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=6,
        )
        left_joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel_asset,
            params={"asset_cfg": SceneEntityCfg("robot_left")},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=6,
        )
        left_last_action = ObservationTermCfg(
            func=mdp.last_action_term,
            params={"action_name": "joint_pos_left"},
            history_length=6,
        )

        right_base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel_asset,
            params={"asset_cfg": SceneEntityCfg("robot_right")},
            scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=6,
        )
        right_projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity_asset,
            params={"asset_cfg": SceneEntityCfg("robot_right")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=6,
        )
        right_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel_asset,
            params={"asset_cfg": SceneEntityCfg("robot_right")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=6,
        )
        right_joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel_asset,
            params={"asset_cfg": SceneEntityCfg("robot_right")},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=6,
        )
        right_last_action = ObservationTermCfg(
            func=mdp.last_action_term,
            params={"action_name": "joint_pos_right"},
            history_length=6,
        )

        payload_center = ObservationTermCfg(
            func=mdp.payload_center_in_pair_frame,
            history_length=6,
        )
        payload_lin_vel = ObservationTermCfg(
            func=mdp.payload_lin_vel_in_pair_frame,
            history_length=6,
        )
        payload_ang_vel = ObservationTermCfg(
            func=mdp.payload_ang_vel_in_pair_frame,
            scale=0.25,
            history_length=6,
        )
        payload_axis = ObservationTermCfg(
            func=mdp.payload_axis_in_pair_frame,
            history_length=6,
        )
        left_support_point = ObservationTermCfg(
            func=mdp.left_support_point_in_left_robot_frame,
            params={"support_half_span": SUPPORT_HALF_SPAN},
            history_length=6,
        )
        left_support_point_lin_vel = ObservationTermCfg(
            func=mdp.left_support_point_lin_vel_in_left_robot_frame,
            params={"support_half_span": SUPPORT_HALF_SPAN},
            scale=0.5,
            history_length=6,
        )
        right_support_point = ObservationTermCfg(
            func=mdp.right_support_point_in_right_robot_frame,
            params={"support_half_span": SUPPORT_HALF_SPAN},
            history_length=6,
        )
        right_support_point_lin_vel = ObservationTermCfg(
            func=mdp.right_support_point_lin_vel_in_right_robot_frame,
            params={"support_half_span": SUPPORT_HALF_SPAN},
            scale=0.5,
            history_length=6,
        )
        robots_relative_pos = ObservationTermCfg(
            func=mdp.robots_relative_pos_in_left_frame,
            history_length=6,
        )
        robots_relative_lin_vel = ObservationTermCfg(
            func=mdp.robots_relative_lin_vel_in_left_frame,
            scale=0.5,
            history_length=6,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(TeacherPolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy: TeacherPolicyCfg = TeacherPolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class DualDogTransportActionsCfg:
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
class DualDogTransportRewardsCfg:
    alive = RewardTermCfg(func=mdp.is_alive, weight=10.0)
    track_shared_lin_vel = RewardTermCfg(
        func=mdp.shared_command_lin_tracking_reward,
        weight=3.0,
        params={"command_name": "base_velocity", "sigma": 0.25},
    )
    track_shared_ang_vel = RewardTermCfg(
        func=mdp.shared_command_ang_tracking_reward,
        weight=1.5,
        params={"command_name": "base_velocity", "sigma": 0.25},
    )
    payload_upright = RewardTermCfg(func=mdp.payload_upright_reward, weight=2.0)
    payload_height = RewardTermCfg(
        func=mdp.payload_height_tracking_reward,
        weight=2.0,
        params={"target_height": 0.38, "sigma": 0.02},
    )
    payload_center_xy = RewardTermCfg(
        func=mdp.payload_center_xy_deviation_penalty,
        weight=1.0,
    )
    support_point_height_balance = RewardTermCfg(
        func=mdp.payload_support_point_height_difference_penalty,
        weight=2.0,
        params={"support_half_span": SUPPORT_HALF_SPAN},
    )
    support_point_velocity_balance = RewardTermCfg(
        func=mdp.payload_support_point_velocity_difference_penalty,
        weight=1.0,
        params={"support_half_span": SUPPORT_HALF_SPAN},
    )
    payload_ang_vel = RewardTermCfg(func=mdp.payload_ang_vel_penalty, weight=0.5)

    left_gait = RewardTermCfg(
        func=mdp.AdaptiveSymmetricGaitReward,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot_left"),
            "sensor_cfg": SceneEntityCfg("robot_left_contact_sensor"),
            "synced_feet_pair_names": (("a_FR_foot", "d_RL_foot"), ("b_FL_foot", "c_RR_foot")),
            "judge_time_threshold": 1.0e-6,
            "air_time_gait_bound": 0.5,
            "contact_time_gait_bound": 0.5,
            "async_time_tolerance": 0.05,
            "stance_rwd_scale": 1.0,
            "encourage_symmetricity_and_low_frequency": 1.0,
            "soft_minimum_frequency": 2.0,
            "tolerance_proportion": 0.2,
            "rwd_upper_bound": 1.0,
            "rwd_lower_bound": -5.0,
            "vel_tracking_exp_sigma": 0.25,
            "task_performance_ratio": 1.0,
        },
    )
    right_gait = RewardTermCfg(
        func=mdp.AdaptiveSymmetricGaitReward,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot_right"),
            "sensor_cfg": SceneEntityCfg("robot_right_contact_sensor"),
            "synced_feet_pair_names": (("a_FR_foot", "d_RL_foot"), ("b_FL_foot", "c_RR_foot")),
            "judge_time_threshold": 1.0e-6,
            "air_time_gait_bound": 0.5,
            "contact_time_gait_bound": 0.5,
            "async_time_tolerance": 0.05,
            "stance_rwd_scale": 1.0,
            "encourage_symmetricity_and_low_frequency": 1.0,
            "soft_minimum_frequency": 2.0,
            "tolerance_proportion": 0.2,
            "rwd_upper_bound": 1.0,
            "rwd_lower_bound": -5.0,
            "vel_tracking_exp_sigma": 0.25,
            "task_performance_ratio": 1.0,
        },
    )

    left_track_base_height = RewardTermCfg(
        func=mdp.track_base_height_ngt,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot_left"), "target_height": 0.27},
    )
    right_track_base_height = RewardTermCfg(
        func=mdp.track_base_height_ngt,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot_right"), "target_height": 0.27},
    )
    left_base_z_velocity = RewardTermCfg(
        func=mdp.base_z_velocity_ngt,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
    )
    right_base_z_velocity = RewardTermCfg(
        func=mdp.base_z_velocity_ngt,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
    )
    left_base_roll_pitch_angle = RewardTermCfg(
        func=mdp.base_roll_pitch_angle_ngt,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
    )
    right_base_roll_pitch_angle = RewardTermCfg(
        func=mdp.base_roll_pitch_angle_ngt,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
    )
    left_base_roll_pitch_velocity = RewardTermCfg(
        func=mdp.base_roll_pitch_velocity_ngt,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
    )
    right_base_roll_pitch_velocity = RewardTermCfg(
        func=mdp.base_roll_pitch_velocity_ngt,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
    )

    left_joint_position_limit = RewardTermCfg(
        func=mdp.joint_position_limit_ngt,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
    )
    right_joint_position_limit = RewardTermCfg(
        func=mdp.joint_position_limit_ngt,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
    )
    left_joint_position = RewardTermCfg(
        func=mdp.joint_position_ngt,
        weight=-2.5e-1,
        params={
            "asset_cfg": SceneEntityCfg("robot_left"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    right_joint_position = RewardTermCfg(
        func=mdp.joint_position_ngt,
        weight=-2.5e-1,
        params={
            "asset_cfg": SceneEntityCfg("robot_right"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    left_joint_acceleration = RewardTermCfg(
        func=mdp.joint_acceleration_ngt,
        weight=-2.5e-6,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
    )
    right_joint_acceleration = RewardTermCfg(
        func=mdp.joint_acceleration_ngt,
        weight=-2.5e-6,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
    )
    left_joint_velocity = RewardTermCfg(
        func=mdp.joint_velocity_ngt,
        weight=-2.5e-3,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
    )
    right_joint_velocity = RewardTermCfg(
        func=mdp.joint_velocity_ngt,
        weight=-2.5e-3,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
    )
    left_joint_torque = RewardTermCfg(
        func=mdp.joint_torque_ngt,
        weight=-1.25e-4,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
    )
    right_joint_torque = RewardTermCfg(
        func=mdp.joint_torque_ngt,
        weight=-1.25e-4,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
    )

    left_action_rate = RewardTermCfg(
        func=mdp.action_rate_term_ngt,
        weight=-0.125,
        params={"action_name": "joint_pos_left"},
    )
    right_action_rate = RewardTermCfg(
        func=mdp.action_rate_term_ngt,
        weight=-0.125,
        params={"action_name": "joint_pos_right"},
    )
    left_foot_slip = RewardTermCfg(
        func=mdp.foot_slipping_ngt,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot_left", body_names=".*foot"),
            "sensor_cfg": SceneEntityCfg("robot_left_contact_sensor", body_names=".*foot"),
            "threshold": 0.5,
        },
    )
    right_foot_slip = RewardTermCfg(
        func=mdp.foot_slipping_ngt,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot_right", body_names=".*foot"),
            "sensor_cfg": SceneEntityCfg("robot_right_contact_sensor", body_names=".*foot"),
            "threshold": 0.5,
        },
    )
    left_foot_dragging = RewardTermCfg(
        func=mdp.foot_dragging_ngt,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot_left", body_names=".*foot"),
            "height_threshold": 0.03,
            "foot_vel_xy_threshold": 0.1,
        },
    )
    right_foot_dragging = RewardTermCfg(
        func=mdp.foot_dragging_ngt,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot_right", body_names=".*foot"),
            "height_threshold": 0.03,
            "foot_vel_xy_threshold": 0.1,
        },
    )
    left_thigh_calf_collision = RewardTermCfg(
        func=mdp.thigh_calf_collision_ngt,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg("robot_left_contact_sensor", body_names=[".*thigh", ".*calf"]),
            "threshold": 0.1,
        },
    )
    right_thigh_calf_collision = RewardTermCfg(
        func=mdp.thigh_calf_collision_ngt,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg("robot_right_contact_sensor", body_names=[".*thigh", ".*calf"]),
            "threshold": 0.1,
        },
    )


@configclass
class DualDogTransportEventCfg:
    # startup randomization
    randomize_left_trunk_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_left", body_names="trunk"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    randomize_right_trunk_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_right", body_names="trunk"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    randomize_left_foot_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_left", body_names=".*foot"),
            "static_friction_range": (0.6, 1.5),
            "dynamic_friction_range": (0.6, 1.5),
            "make_consistent": True,
            "restitution_range": (0.0, 0.3),
            "num_buckets": 4000,
        },
    )
    randomize_right_foot_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_right", body_names=".*foot"),
            "static_friction_range": (0.6, 1.5),
            "dynamic_friction_range": (0.6, 1.5),
            "make_consistent": True,
            "restitution_range": (0.0, 0.3),
            "num_buckets": 4000,
        },
    )

    # reset randomization
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
                "z": (-0.01, 0.01),
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
    randomize_payload_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("payload", body_names=".*"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 1.0),
            "make_consistent": True,
            "restitution_range": (0.0, 0.2),
            "num_buckets": 2000,
        },
    )
    randomize_payload_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("payload", body_names=".*"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # interval perturbations
    push_left_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot_left"),
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.15, 0.15),
                "z": (-0.05, 0.05),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.15, 0.15),
            },
        },
    )
    push_right_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot_right"),
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.15, 0.15),
                "z": (-0.05, 0.05),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.15, 0.15),
            },
        },
    )


@configclass
class DualDogTransportTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    left_bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation_asset,
        params={"limit_angle": math.pi / 3, "asset_cfg": SceneEntityCfg("robot_left")},
    )
    right_bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation_asset,
        params={"limit_angle": math.pi / 3, "asset_cfg": SceneEntityCfg("robot_right")},
    )
    left_fall = TerminationTermCfg(
        func=mdp.root_height_below_minimum_asset,
        params={"minimum_height": 0.15, "asset_cfg": SceneEntityCfg("robot_left")},
    )
    right_fall = TerminationTermCfg(
        func=mdp.root_height_below_minimum_asset,
        params={"minimum_height": 0.15, "asset_cfg": SceneEntityCfg("robot_right")},
    )
    payload_drop = TerminationTermCfg(
        func=mdp.payload_below_minimum_height,
        params={"minimum_height": 0.20, "payload_cfg": SceneEntityCfg("payload")},
    )


@configclass
class DualDogTransportCurriculumCfg:
    velocity_commands = CurriculumTermCfg(
        func=mdp.ModifyVelCommandsRangeBasedonReward,
        params={
            "command_name": "base_velocity",
            "command_maximum_ranges": [0.5, 0.25, math.pi / 4],
            "curriculum_bins": [12, 12, 12],
            "reset_envs_episode_length": 0.9,
            "reward_name_lin": "track_shared_lin_vel",
            "reward_name_ang": "track_shared_ang_vel",
            "error_threshold_lin": 0.08,
            "error_threshold_ang": 0.12,
            "repeat_times_lin": 2,
            "repeat_times_ang": 2,
            "max_distance_bins": 3,
        },
    )


@configclass
class DualDogTransportBaseEnvCfg(ManagerBasedRLEnvCfg):
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
    observations: DualDogTransportObservationsCfg = DualDogTransportObservationsCfg()
    actions: DualDogTransportActionsCfg = DualDogTransportActionsCfg()
    rewards: DualDogTransportRewardsCfg = DualDogTransportRewardsCfg()
    terminations: DualDogTransportTerminationsCfg = DualDogTransportTerminationsCfg()
    events: DualDogTransportEventCfg = DualDogTransportEventCfg()
    curriculum: DualDogTransportCurriculumCfg = DualDogTransportCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        # This teacher-stage task uses privileged rod state without tactile input.
        self.scene.robot_left_tactile_contact_sensor = None
        self.scene.robot_right_tactile_contact_sensor = None

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
        if getattr(self.scene, "robot_right_contact_sensor", None) is not None:
            self.scene.robot_right_contact_sensor.update_period = self.sim.dt
        if getattr(self.scene, "payload_contact_sensor", None) is not None:
            self.scene.payload_contact_sensor.update_period = self.sim.dt


def smaller_dual_dog_transport_scene_for_playing(env_cfg: DualDogTransportBaseEnvCfg) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 8.0


@configclass
class DualDogTransportBaseEnvCfg_PLAY(DualDogTransportBaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        smaller_dual_dog_transport_scene_for_playing(self)
