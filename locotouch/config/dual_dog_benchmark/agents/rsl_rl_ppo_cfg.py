from isaaclab.utils import configclass

from locotouch.config.locotouch.agents.rsl_rl_ppo_cfg import LocomotionPPORunnerCfg


@configclass
class DualDogBenchmarkPPORunnerCfg(LocomotionPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "locotouch_dual_dog_benchmark"
        self.wandb_project = "Dual_Dog_Benchmark"
