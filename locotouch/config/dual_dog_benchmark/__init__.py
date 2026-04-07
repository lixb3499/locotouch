import gymnasium as gym

from .agents import rsl_rl_ppo_cfg
from . import dual_dog_benchmark_env_cfg


gym.register(
    id="Isaac-DualDogBenchmark-LocoTouch-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dual_dog_benchmark_env_cfg.DualDogBenchmarkEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.DualDogBenchmarkPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-DualDogBenchmark-LocoTouch-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dual_dog_benchmark_env_cfg.DualDogBenchmarkEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.DualDogBenchmarkPPORunnerCfg,
    },
)
"""
python locotouch/scripts/play_dual_dog_baseline.py --task Isaac-DualDogBenchmark-LocoTouch-Play-v1 --policy_task Isaac-Locomotion-LocoTouch-Play-v1 --load_run=<single_dog_run>
"""
