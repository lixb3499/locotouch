import gymnasium as gym

from . import dual_dog_benchmark_env_cfg


gym.register(
    id="Isaac-DualDogBenchmark-LocoTouch-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dual_dog_benchmark_env_cfg.DualDogStudentBenchmarkEnvCfg,
    },
)

gym.register(
    id="Isaac-DualDogBenchmark-LocoTouch-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dual_dog_benchmark_env_cfg.DualDogStudentBenchmarkEnvCfg_PLAY,
    },
)

"""
python locotouch/scripts/play_dual_dog_student_baseline.py \
    --task Isaac-DualDogBenchmark-LocoTouch-Play-v1 \
    --student_task Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-Play-v1 \
    --log_dir_distill 2025-09-02_23-27-14 \
    --checkpoint_distill model_7.pt \
    --num_envs 1
"""
