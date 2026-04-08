import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .locotouch import (
    LocoTouch_Without_Tactile_CFG,
    LocoTouch_Without_Tactile_Instanceable_CFG,
)


DEFAULT_DOG_SEPARATION_Y = 0.9
DEFAULT_PAYLOAD_LENGTH = 3.5
DEFAULT_PAYLOAD_RADIUS = 0.06
DEFAULT_PAYLOAD_MASS = 6.0
DEFAULT_PAYLOAD_CENTER = (0.0, 0.0, 0.38)
DEFAULT_PAYLOAD_COLOR = (0.1, 0.7, 0.2)


def make_dual_dog_robot_cfgs(
    base_robot_cfg=LocoTouch_Without_Tactile_Instanceable_CFG,
    left_prim_path: str = "{ENV_REGEX_NS}/RobotLeft",
    right_prim_path: str = "{ENV_REGEX_NS}/RobotRight",
    lateral_separation_y: float = DEFAULT_DOG_SEPARATION_Y,
):
    """Create a left/right robot pair centered around y=0 for cooperative transport."""
    left_cfg = base_robot_cfg.replace(prim_path=left_prim_path)
    right_cfg = base_robot_cfg.replace(prim_path=right_prim_path)

    base_x, _, base_z = base_robot_cfg.init_state.pos
    half_sep = 0.5 * lateral_separation_y
    left_cfg.init_state.pos = (base_x, -half_sep, base_z)
    right_cfg.init_state.pos = (base_x, half_sep, base_z)
    return left_cfg, right_cfg


def make_long_cylinder_payload_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Payload",
    length: float = DEFAULT_PAYLOAD_LENGTH,
    radius: float = DEFAULT_PAYLOAD_RADIUS,
    mass: float = DEFAULT_PAYLOAD_MASS,
    center: tuple[float, float, float] = DEFAULT_PAYLOAD_CENTER,
    color: tuple[float, float, float] = DEFAULT_PAYLOAD_COLOR,
):
    """Create a long rigid cylinder aligned with +Y, suitable as a shared payload."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=length,
            axis="Y",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=1.0e-9,
                rest_offset=-0.002,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=center, rot=(1.0, 0.0, 0.0, 0.0)),
    )


DUAL_DOG_LEFT_CFG, DUAL_DOG_RIGHT_CFG = make_dual_dog_robot_cfgs(
    base_robot_cfg=LocoTouch_Without_Tactile_Instanceable_CFG
)
DUAL_DOG_LEFT_PLAY_CFG, DUAL_DOG_RIGHT_PLAY_CFG = make_dual_dog_robot_cfgs(
    base_robot_cfg=LocoTouch_Without_Tactile_CFG
)
LONG_CYLINDER_PAYLOAD_CFG = make_long_cylinder_payload_cfg()


@configclass
class DualDogLongCylinderSceneCfg(InteractiveSceneCfg):
    """Reusable scene template with two dogs and one long cylindrical payload."""

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

    robot_left = DUAL_DOG_LEFT_CFG
    robot_right = DUAL_DOG_RIGHT_CFG

    robot_left_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/(?!sensor.*).*",
        history_length=3,
        track_air_time=True,
    )
    robot_left_tactile_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/sensor_.*",
        update_period=0.025,
        history_length=3,
        track_air_time=True,
    )
    robot_right_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotRight/(?!sensor.*).*",
        history_length=3,
        track_air_time=True,
    )
    robot_right_tactile_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotRight/sensor_.*",
        update_period=0.025,
        history_length=3,
        track_air_time=True,
    )

    payload = LONG_CYLINDER_PAYLOAD_CFG
    payload_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Payload",
        history_length=3,
        track_air_time=True,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.6, 0.6, 0.6), intensity=1000.0),
    )


@configclass
class DualDogLongCylinderSceneCfg_PLAY(DualDogLongCylinderSceneCfg):
    robot_left = DUAL_DOG_LEFT_PLAY_CFG
    robot_right = DUAL_DOG_RIGHT_PLAY_CFG
