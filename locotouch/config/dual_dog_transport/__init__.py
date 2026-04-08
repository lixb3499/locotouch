import gymnasium as gym

from . import dual_dog_transport_env_cfg
from .agents import rsl_rl_ppo_cfg


gym.register(
    id="Isaac-DualDogTransport-LocoTouch-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dual_dog_transport_env_cfg.DualDogTransportEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.DualDogTransportPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-DualDogTransport-LocoTouch-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dual_dog_transport_env_cfg.DualDogTransportEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.DualDogTransportPPORunnerCfg,
    },
)

"""
python locotouch/scripts/train.py --task Isaac-DualDogTransport-LocoTouch-v1 --num_envs=64 --headless
python locotouch/scripts/play.py --task Isaac-DualDogTransport-LocoTouch-Play-v1 --num_envs=1
"""
