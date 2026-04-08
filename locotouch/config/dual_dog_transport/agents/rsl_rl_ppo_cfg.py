from isaaclab.utils import configclass

from locotouch.config.locotouch.agents.rsl_rl_ppo_cfg import LocomotionPPORunnerCfg


@configclass
class DualDogTransportPPORunnerCfg(LocomotionPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "locotouch_dual_dog_transport_teacher"
        self.wandb_project = "Dual_Dog_Transport"
