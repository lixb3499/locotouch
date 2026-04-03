"""Play-only baseline for two dogs carrying a long cylinder with a shared locomotion policy.

This script is intentionally simple:

- one locomotion checkpoint is loaded once
- two robots run the same policy independently
- both robots receive the same shared velocity command
- the payload is a long rigid cylinder spawned between the two robots

The script does not reuse the manager-based training environment. It builds a standalone
scene so it can serve as a baseline before designing a proper dual-robot RL environment.
"""

import argparse
import inspect
import os
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from isaaclab.app import AppLauncher
import cli_args


parser = argparse.ArgumentParser(description="Play a dual-dog transport baseline with a shared single-robot policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Locomotion-LocoTouch-Play-v1",
    help="Single-robot task whose policy config/checkpoint should be loaded.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Unused. Present only for CLI compatibility.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Unused compatibility flag.")
parser.add_argument("--video", action="store_true", default=False, help="Unused compatibility flag.")
parser.add_argument("--command_mode", type=str, default="fixed", choices=("fixed", "random"))
parser.add_argument("--command_interval_s", type=float, default=8.0, help="How often to resample shared commands.")
parser.add_argument("--command_vx", type=float, default=0.3, help="Fixed forward velocity command.")
parser.add_argument("--command_vy", type=float, default=0.0, help="Fixed lateral velocity command.")
parser.add_argument("--command_wz", type=float, default=0.0, help="Fixed yaw-rate command.")
parser.add_argument("--max_vx", type=float, default=0.4, help="Max |vx| when using random commands.")
parser.add_argument("--max_vy", type=float, default=0.2, help="Max |vy| when using random commands.")
parser.add_argument("--max_wz", type=float, default=0.4, help="Max |wz| when using random commands.")
parser.add_argument("--stand_prob", type=float, default=0.1, help="Probability of sampling a zero command in random mode.")
parser.add_argument("--episode_length_s", type=float, default=20.0)
parser.add_argument("--settle_steps", type=int, default=80, help="Physics steps to settle after every reset.")
parser.add_argument(
    "--initial_zero_command_steps",
    type=int,
    default=50,
    help="Control steps with zero command after each group reset, matching the transport play env stabilization period.",
)
parser.add_argument("--robot_separation_y", type=float, default=0.9, help="Lateral distance between the two robots.")
parser.add_argument("--payload_length", type=float, default=3.5)
parser.add_argument("--payload_radius", type=float, default=0.06)
parser.add_argument("--payload_mass", type=float, default=6.0)
parser.add_argument("--payload_center_x", type=float, default=0.0)
parser.add_argument("--payload_center_z", type=float, default=0.38)
parser.add_argument("--payload_yaw_deg", type=float, default=0.0, help="Yaw of the payload. 0 means aligned with +Y.")
parser.add_argument("--payload_drop_height", type=float, default=0.20, help="Reset if payload center falls below this z.")
parser.add_argument("--compare_payload_length", type=float, default=0.30)
parser.add_argument("--compare_payload_mass", type=float, default=1.0)
parser.add_argument("--compare_payload_center_z", type=float, default=0.39)
cli_args.add_rsl_rl_args(parser)
cli_args.add_distillation_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from locotouch import *  # noqa: F401
from locotouch.assets.locotouch import LocoTouch_CFG
from locotouch.config.locotouch.agents.distillation_cfg import DistillationCfg
from locotouch.distill.student import Student
from locotouch.distill.tactile_recorder import TactileRecorder


OBS_HISTORY_LENGTH = 6
COMMAND_DIM = 3
BASE_ANG_VEL_DIM = 3
PROJECTED_GRAVITY_DIM = 3
TACTILE_IMG_SHAPE = (2, 17, 13)
TACTILE_SIGNAL_DIM = TACTILE_IMG_SHAPE[0] * TACTILE_IMG_SHAPE[1] * TACTILE_IMG_SHAPE[2]


@configclass
class DualDogBaselineSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot_left = LocoTouch_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotLeft")
    robot_right = LocoTouch_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotRight")
    robot_right.init_state.pos = (0.0, 0.9, 0.28)
    robot_cmp_1 = LocoTouch_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotCmp1")
    robot_cmp_1.init_state.pos = (4.0, 0.0, 0.28)
    robot_cmp_2 = LocoTouch_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotCmp2")
    robot_cmp_2.init_state.pos = (8.0, 0.0, 0.28)
    robot_left_tactile_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/sensor_.*",
        update_period=0.025,
        history_length=3,
        track_air_time=True,
    )
    robot_right_tactile_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotRight/sensor_.*",
        update_period=0.025,
        history_length=3,
        track_air_time=True,
    )
    robot_cmp_1_tactile_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotCmp1/sensor_.*",
        update_period=0.025,
        history_length=3,
        track_air_time=True,
    )
    robot_cmp_2_tactile_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotCmp2/sensor_.*",
        update_period=0.025,
        history_length=3,
        track_air_time=True,
    )

    payload = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Payload",
        spawn=sim_utils.CylinderCfg(
            radius=0.06,
            height=5.0,
            axis="Y",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=1.0e-9,
                rest_offset=-0.002,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=6.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2), opacity=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    payload_cmp_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PayloadCmp1",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=0.30,
            axis="Y",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=1.0e-9,
                rest_offset=-0.002,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.45, 0.1), opacity=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 0.39), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    payload_cmp_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PayloadCmp2",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=0.30,
            axis="Y",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=1.0e-9,
                rest_offset=-0.002,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 0.85), opacity=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 0.0, 0.39), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.7, 0.7, 0.7), intensity=1500.0),
    )


def _filtered_kwargs(target, kwargs: dict) -> dict:
    signature = inspect.signature(target.__init__)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    valid_keys = {
        name
        for name, param in signature.parameters.items()
        if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in valid_keys}


def _term_history(term: torch.Tensor) -> torch.Tensor:
    return torch.cat([term] * OBS_HISTORY_LENGTH, dim=-1)


def _roll_history(history: deque[torch.Tensor], value: torch.Tensor) -> torch.Tensor:
    history.append(value)
    return torch.cat(list(history), dim=-1)


@dataclass
class RobotObsState:
    last_action: torch.Tensor
    cmd_history: deque[torch.Tensor]
    ang_vel_history: deque[torch.Tensor]
    gravity_history: deque[torch.Tensor]
    joint_pos_history: deque[torch.Tensor]
    joint_vel_history: deque[torch.Tensor]
    action_history: deque[torch.Tensor]


def create_obs_state(num_actions: int, device: torch.device) -> RobotObsState:
    zeros_cmd = torch.zeros(1, COMMAND_DIM, device=device)
    zeros_base = torch.zeros(1, BASE_ANG_VEL_DIM, device=device)
    zeros_gravity = torch.tensor([[0.0, 0.0, -1.0]], device=device)
    zeros_joint = torch.zeros(1, num_actions, device=device)
    return RobotObsState(
        last_action=zeros_joint.clone(),
        cmd_history=deque([zeros_cmd.clone() for _ in range(OBS_HISTORY_LENGTH)], maxlen=OBS_HISTORY_LENGTH),
        ang_vel_history=deque([zeros_base.clone() for _ in range(OBS_HISTORY_LENGTH)], maxlen=OBS_HISTORY_LENGTH),
        gravity_history=deque([zeros_gravity.clone() for _ in range(OBS_HISTORY_LENGTH)], maxlen=OBS_HISTORY_LENGTH),
        joint_pos_history=deque([zeros_joint.clone() for _ in range(OBS_HISTORY_LENGTH)], maxlen=OBS_HISTORY_LENGTH),
        joint_vel_history=deque([zeros_joint.clone() for _ in range(OBS_HISTORY_LENGTH)], maxlen=OBS_HISTORY_LENGTH),
        action_history=deque([zeros_joint.clone() for _ in range(OBS_HISTORY_LENGTH)], maxlen=OBS_HISTORY_LENGTH),
    )


class ManualBinaryTactile:
    def __init__(self, robot, contact_sensor, tactile_signal_shape=(17, 13), contact_threshold=0.05):
        self.robot = robot
        self.contact_sensor = contact_sensor
        self.tactile_signal_shape = tactile_signal_shape
        self.contact_threshold = contact_threshold
        self.body_ids = [
            idx for idx, body_name in enumerate(robot.body_names) if re.fullmatch(r"sensor_.*", body_name) is not None
        ]
        self.body_ids = torch.tensor(self.body_ids, dtype=torch.long, device=robot.device)
        self.contact_threshold_envs_sensors = torch.ones(
            (1, tactile_signal_shape[0], tactile_signal_shape[1]),
            device=robot.device,
            dtype=torch.float32,
        ) * self.contact_threshold

    def __call__(self) -> torch.Tensor:
        net_forces_w = self.contact_sensor.data.net_forces_w
        body_quat_w = self.robot.data.body_quat_w[:, self.body_ids]
        normal_forces = -quat_apply_inverse(body_quat_w, net_forces_w)[..., 2].reshape(
            self.robot.num_instances, *self.tactile_signal_shape
        )
        contact_taxels = normal_forces > self.contact_threshold_envs_sensors
        return torch.stack([contact_taxels.float(), contact_taxels.float()], dim=1).flatten(start_dim=1)


def build_policy_obs(robot, obs_state: RobotObsState, command: torch.Tensor) -> torch.Tensor:
    joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
    joint_vel_rel = robot.data.joint_vel
    base_ang_vel = robot.data.root_ang_vel_b * 0.25
    projected_gravity = robot.data.projected_gravity_b
    last_action = obs_state.last_action

    cmd_hist = _roll_history(obs_state.cmd_history, command)
    ang_hist = _roll_history(obs_state.ang_vel_history, base_ang_vel)
    grav_hist = _roll_history(obs_state.gravity_history, projected_gravity)
    jpos_hist = _roll_history(obs_state.joint_pos_history, joint_pos_rel)
    jvel_hist = _roll_history(obs_state.joint_vel_history, joint_vel_rel * 0.05)
    act_hist = _roll_history(obs_state.action_history, last_action)

    return torch.cat([cmd_hist, ang_hist, grav_hist, jpos_hist, jvel_hist, act_hist], dim=-1)


def update_last_action(obs_state: RobotObsState, raw_action: torch.Tensor) -> None:
    obs_state.last_action = raw_action


def sample_shared_command(device: torch.device) -> torch.Tensor:
    if args_cli.command_mode == "fixed":
        return torch.tensor([[args_cli.command_vx, args_cli.command_vy, args_cli.command_wz]], device=device)

    if torch.rand(1, device=device).item() < args_cli.stand_prob:
        return torch.zeros(1, COMMAND_DIM, device=device)

    command = torch.empty(1, COMMAND_DIM, device=device)
    command[:, 0].uniform_(-args_cli.max_vx, args_cli.max_vx)
    command[:, 1].uniform_(-args_cli.max_vy, args_cli.max_vy)
    command[:, 2].uniform_(-args_cli.max_wz, args_cli.max_wz)
    return command


def current_group_command(group_name: str, shared_command: torch.Tensor, warmup_steps_remaining: dict[str, int]) -> torch.Tensor:
    if warmup_steps_remaining[group_name] > 0:
        return torch.zeros_like(shared_command)
    return shared_command


def build_student(
    distillation_cfg: DistillationCfg,
    proprioception_dim: int,
    num_actions: int,
):
    student = Student(
        distillation_cfg,
        proprioception_dim,
        TACTILE_SIGNAL_DIM,
        num_actions,
    )
    return student


def reset_robot(robot, root_x: float, root_y: float, env_ids: torch.Tensor) -> None:
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0] = root_x
    root_state[:, 1] = root_y
    root_state[:, 2] = 0.28
    root_state[:, 7:] = 0.0
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def reset_payload(
    payload,
    pos_x: float,
    pos_y: float,
    pos_z: float,
    yaw_deg: float,
    env_ids: torch.Tensor,
) -> None:
    payload_state = payload.data.default_root_state.clone()
    payload_state[:, 0] = pos_x
    payload_state[:, 1] = pos_y
    payload_state[:, 2] = pos_z
    payload_quat = quat_from_euler_xyz(
        torch.zeros(1, device=payload.device),
        torch.zeros(1, device=payload.device),
        torch.full((1,), torch.deg2rad(torch.tensor(yaw_deg)).item(), device=payload.device),
    )
    payload_state[:, 3:7] = payload_quat
    payload_state[:, 7:] = 0.0
    payload.write_root_pose_to_sim(payload_state[:, :7], env_ids=env_ids)
    payload.write_root_velocity_to_sim(payload_state[:, 7:], env_ids=env_ids)


def reset_group(scene: InteractiveScene, sim: SimulationContext, group_name: str) -> None:
    env_ids = torch.tensor([0], dtype=torch.long, device=scene.device)
    if group_name == "dual":
        reset_robot(scene["robot_left"], 0.0, -args_cli.robot_separation_y / 2.0, env_ids)
        reset_robot(scene["robot_right"], 0.0, args_cli.robot_separation_y / 2.0, env_ids)
        reset_payload(
            scene["payload"],
            args_cli.payload_center_x,
            0.0,
            args_cli.payload_center_z,
            args_cli.payload_yaw_deg,
            env_ids,
        )
    elif group_name == "cmp1":
        reset_robot(scene["robot_cmp_1"], 4.0, 0.0, env_ids)
        reset_payload(
            scene["payload_cmp_1"],
            4.0,
            0.0,
            args_cli.compare_payload_center_z,
            0.0,
            env_ids,
        )
    elif group_name == "cmp2":
        reset_robot(scene["robot_cmp_2"], 8.0, 0.0, env_ids)
        reset_payload(
            scene["payload_cmp_2"],
            8.0,
            0.0,
            args_cli.compare_payload_center_z,
            0.0,
            env_ids,
        )
    else:
        raise ValueError(f"Unknown group name: {group_name}")

    scene.reset(env_ids)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


def reset_scene(scene: InteractiveScene, sim: SimulationContext) -> None:
    for group_name in ("dual", "cmp1", "cmp2"):
        reset_group(scene, sim, group_name)


def main():
    if args_cli.num_envs != 1:
        print("[INFO] This baseline only uses one scene. Overriding num_envs to 1.")

    scene_cfg = DualDogBaselineSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False, lazy_sensor_update=False)
    scene_cfg.robot_right.init_state.pos = (0.0, args_cli.robot_separation_y / 2.0, 0.28)
    scene_cfg.payload.spawn.radius = args_cli.payload_radius
    scene_cfg.payload.spawn.height = args_cli.payload_length
    scene_cfg.payload.spawn.mass_props.mass = args_cli.payload_mass
    scene_cfg.payload.init_state.pos = (args_cli.payload_center_x, 0.0, args_cli.payload_center_z)
    scene_cfg.payload_cmp_1.spawn.height = args_cli.compare_payload_length
    scene_cfg.payload_cmp_1.spawn.mass_props.mass = args_cli.compare_payload_mass
    scene_cfg.payload_cmp_1.init_state.pos = (4.0, 0.0, args_cli.compare_payload_center_z)
    scene_cfg.payload_cmp_2.spawn.height = args_cli.compare_payload_length
    scene_cfg.payload_cmp_2.spawn.mass_props.mass = args_cli.compare_payload_mass
    scene_cfg.payload_cmp_2.init_state.pos = (8.0, 0.0, args_cli.compare_payload_center_z)

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
        use_fabric=not args_cli.disable_fabric,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[8.5, 10.0, 5.5], target=[4.0, 0.0, 0.45])
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    sim.play()
    print("[INFO]: Dual-dog baseline scene created.")

    distillation_cfg: DistillationCfg = cli_args.parse_distillation_cfg(args_cli.task, args_cli)
    distillation_log_root = os.path.abspath(os.path.join(distillation_cfg.log_root_path, distillation_cfg.experiment_name))
    student_resume_path = get_checkpoint_path(
        distillation_log_root,
        distillation_cfg.log_dir_distill,
        distillation_cfg.checkpoint_distill,
    )
    print(f"[INFO]: Loading shared student checkpoint from: {student_resume_path}")

    num_actions = scene["robot_left"].num_joints
    proprioception_dim = OBS_HISTORY_LENGTH * (
        COMMAND_DIM + BASE_ANG_VEL_DIM + PROJECTED_GRAVITY_DIM + num_actions + num_actions + num_actions
    )
    distillation_cfg.device = scene.device
    student = build_student(distillation_cfg, proprioception_dim=proprioception_dim, num_actions=num_actions)
    student.load_checkpoint(student_resume_path)
    student.eval()

    robot_names = ["robot_left", "robot_right", "robot_cmp_1", "robot_cmp_2"]
    robots = {name: scene[name] for name in robot_names}
    robot_index = {name: idx for idx, name in enumerate(robot_names)}
    groups = {
        "dual": {"robots": ["robot_left", "robot_right"], "payload": "payload"},
        "cmp1": {"robots": ["robot_cmp_1"], "payload": "payload_cmp_1"},
        "cmp2": {"robots": ["robot_cmp_2"], "payload": "payload_cmp_2"},
    }
    tactile_sensor_names = {
        "robot_left": "robot_left_tactile_sensor",
        "robot_right": "robot_right_tactile_sensor",
        "robot_cmp_1": "robot_cmp_1_tactile_sensor",
        "robot_cmp_2": "robot_cmp_2_tactile_sensor",
    }
    tactile_helpers = {
        name: ManualBinaryTactile(robots[name], scene[tactile_sensor_names[name]])
        for name in robot_names
    }
    tactile_recorders = {
        name: TactileRecorder(
            scene.device,
            1,
            TACTILE_SIGNAL_DIM,
            distillation_cfg.min_delay,
            distillation_cfg.max_delay,
        )
        for name in robot_names
    }
    payload_names = ["payload", "payload_cmp_1", "payload_cmp_2"]
    payloads = {name: scene[name] for name in payload_names}
    default_joint_pos = {name: robot.data.default_joint_pos.clone() for name, robot in robots.items()}
    obs_states = {name: create_obs_state(num_actions=num_actions, device=scene.device) for name in robot_names}
    group_episode_steps = {group_name: 0 for group_name in groups}
    group_warmup_steps_remaining = {
        group_name: args_cli.initial_zero_command_steps for group_name in groups
    }

    reset_scene(scene, sim)
    shared_command = sample_shared_command(scene.device)
    print(f"[INFO]: Shared command initialized to {shared_command[0].tolist()}")
    print(f"[INFO]: Left robot root pos {scene['robot_left'].data.root_pos_w[0].tolist()}")
    print(f"[INFO]: Right robot root pos {scene['robot_right'].data.root_pos_w[0].tolist()}")
    print(f"[INFO]: Payload root pos {scene['payload'].data.root_pos_w[0].tolist()}")
    print(f"[INFO]: Compare payload 1 root pos {scene['payload_cmp_1'].data.root_pos_w[0].tolist()}")
    print(f"[INFO]: Compare payload 2 root pos {scene['payload_cmp_2'].data.root_pos_w[0].tolist()}")
    episode_steps = max(1, int(args_cli.episode_length_s / sim.get_physics_dt() / 4))
    command_interval_steps = max(1, int(args_cli.command_interval_s / sim.get_physics_dt() / 4))

    control_step = 0
    episode_step = 0

    while simulation_app.is_running():
        if sim.is_stopped():
            sim.play()
            sim.step()
            continue
        if not sim.is_playing():
            sim.step()
            continue

        if episode_step == 0:
            shared_command = sample_shared_command(scene.device)

        groups_to_reset = []
        for group_name, group_cfg in groups.items():
            payload_height = payloads[group_cfg["payload"]].data.root_pos_w[0, 2].item()
            if payload_height < args_cli.payload_drop_height or group_episode_steps[group_name] >= episode_steps:
                groups_to_reset.append(group_name)

        if groups_to_reset:
            done_mask = torch.zeros(len(robot_names), dtype=torch.bool, device=scene.device)
            for group_name in groups_to_reset:
                payload_name = groups[group_name]["payload"]
                payload_height = payloads[payload_name].data.root_pos_w[0, 2].item()
                print(f"[INFO]: Resetting group '{group_name}' with payload '{payload_name}' at z={payload_height:.3f}")
                reset_group(scene, sim, group_name)
                group_episode_steps[group_name] = 0
                group_warmup_steps_remaining[group_name] = args_cli.initial_zero_command_steps
                for robot_name in groups[group_name]["robots"]:
                    obs_states[robot_name] = create_obs_state(num_actions=num_actions, device=scene.device)
                    tactile_recorders[robot_name].reset()
                    done_mask[robot_index[robot_name]] = True
            student.reset(done_mask)

            for _ in range(args_cli.settle_steps):
                for group_name in groups_to_reset:
                    for robot_name in groups[group_name]["robots"]:
                        robots[robot_name].set_joint_position_target(default_joint_pos[robot_name])
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())

        if args_cli.command_mode == "random" and control_step % command_interval_steps == 0:
            shared_command = sample_shared_command(scene.device)

        batch_obs = []
        batch_tactile = []
        for name in robot_names:
            group_name = next(group_name for group_name, group_cfg in groups.items() if name in group_cfg["robots"])
            proprio_obs = build_policy_obs(
                robots[name],
                obs_states[name],
                current_group_command(group_name, shared_command, group_warmup_steps_remaining),
            )
            tactile_signal = tactile_helpers[name]()
            tactile_recorders[name].record_new_tactile_signals(tactile_signal)
            delayed_tactile = tactile_recorders[name].get_tactile_signals().clone()
            batch_obs.append(proprio_obs)
            batch_tactile.append(delayed_tactile)
        batch_obs = torch.cat(batch_obs, dim=0)
        batch_tactile = torch.cat(batch_tactile, dim=0)

        with torch.inference_mode():
            raw_actions = student(batch_obs, batch_tactile)

        scaled_raw_actions = torch.clamp(raw_actions, -100.0, 100.0) * 0.25
        targets = {}
        for action_idx, name in enumerate(robot_names):
            action = scaled_raw_actions[action_idx : action_idx + 1]
            update_last_action(obs_states[name], action)
            targets[name] = default_joint_pos[name] + action

        for _ in range(4):
            for name, robot in robots.items():
                robot.set_joint_position_target(targets[name])
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        if control_step % 100 == 0:
            cmd_str = ", ".join(f"{v:.3f}" for v in shared_command[0].tolist())
            payload_xyz = scene["payload"].data.root_pos_w[0].tolist()
            cmp1_xyz = scene["payload_cmp_1"].data.root_pos_w[0].tolist()
            cmp2_xyz = scene["payload_cmp_2"].data.root_pos_w[0].tolist()
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"cmd=[{cmd_str}] "
                f"dual_payload=({payload_xyz[0]:.3f}, {payload_xyz[1]:.3f}, {payload_xyz[2]:.3f}) "
                f"cmp1=({cmp1_xyz[0]:.3f}, {cmp1_xyz[1]:.3f}, {cmp1_xyz[2]:.3f}) "
                f"cmp2=({cmp2_xyz[0]:.3f}, {cmp2_xyz[1]:.3f}, {cmp2_xyz[2]:.3f})"
            )

        control_step += 1
        episode_step += 1
        for group_name in groups:
            group_episode_steps[group_name] += 1
            if group_warmup_steps_remaining[group_name] > 0:
                group_warmup_steps_remaining[group_name] -= 1

    simulation_app.close()


if __name__ == "__main__":
    main()
