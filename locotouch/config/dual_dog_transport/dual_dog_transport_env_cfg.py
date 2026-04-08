from isaaclab.utils import configclass

from locotouch.assets.dual_dog_long_cylinder import (
    DualDogLongCylinderSceneCfg,
    DualDogLongCylinderSceneCfg_PLAY,
)

from .dual_dog_transport_base_env_cfg import (
    DualDogTransportBaseEnvCfg,
    smaller_dual_dog_transport_scene_for_playing,
)


@configclass
class DualDogTransportEnvCfg(DualDogTransportBaseEnvCfg):
    scene = DualDogLongCylinderSceneCfg(num_envs=64, env_spacing=8.0)

    def __post_init__(self):
        super().__post_init__()


@configclass
class DualDogTransportEnvCfg_PLAY(DualDogTransportEnvCfg):
    scene = DualDogLongCylinderSceneCfg_PLAY(
        num_envs=1,
        env_spacing=8.0,
        replicate_physics=False,
        lazy_sensor_update=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        smaller_dual_dog_transport_scene_for_playing(self)
