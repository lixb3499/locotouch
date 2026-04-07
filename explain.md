python locotouch/scripts/train.py --task Isaac-Locomotion-LocoTouch-v1
  --num_envs=4096 --headless

  时，主流程可以概括成这条链：

  train.py 解析参数 -> 启动 Isaac Sim -> 导入 locotouch 触发任务注册 -> 用 task
  名字拿到环境配置和 PPO 配置 -> gym.make() 实例化环境 -> 包一层
  RslRlVecEnvWrapper 适配 loco_rl -> OnPolicyRunner 建策略和 PPO -> 循环做
  rollout 和 update。

  1. 入口脚本先做什么

  在 locotouch/scripts/train.py:16 到 locotouch/scripts/train.py:40，脚本先用
  argparse 解析 CLI。

  你这条命令里关键参数的作用是：

  - --task Isaac-Locomotion-LocoTouch-v1：指定训练哪个已注册任务。
  - --num_envs=4096：覆盖环境默认并行环境数。
  - --headless：这是 AppLauncher.add_app_launcher_args() 加进来的参数，传给
    Isaac Sim，表示无界面启动。
  - 没传 --video，所以不会启用 camera 和 RecordVideo。
  - 没传 --resume，所以不会加载历史 checkpoint。

  接着 locotouch/scripts/train.py:35 用 AppLauncher(args_cli) 启动 Omniverse/
  Isaac Sim 应用。这一步相当于把仿真器进程和渲染/物理后端先拉起来。

  2. task 名字是怎么变成具体环境类和配置类的

  关键在这句：

  locotouch/scripts/train.py:64

  它会触发 locotouch/__init__.py:1 的 import_packages(__name__,
  _BLACKLIST_PKGS)，把 locotouch 包下面的模块都导进来。这样 locotouch/
  config/.../__init__.py 里的 gym.register(...) 就执行了。

  对你这个 task，注册位置在：

  locotouch/config/locotouch/__init__.py:7

  这里做了两件事：

  - 把 task id Isaac-Locomotion-LocoTouch-v1 绑定到环境类型
    isaaclab.envs:ManagerBasedRLEnv
  - 把这个 task 对应的两个入口类挂到 registry：
      - env_cfg_entry_point = LocomotionEnvCfg
      - rsl_rl_cfg_entry_point = LocomotionPPORunnerCfg

  也就是说，task 只是一个字符串，但它背后已经通过 Gym registry 绑定了“环境类 +
  环境配置类 + 算法配置类”。

  3. hydra_task_config(...) 怎么把配置注入进 main()

  main 上面这层装饰器最关键：

  locotouch/scripts/train.py:82

  @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
  def main(env_cfg, agent_cfg):

  它的职责可以从调用方式反推出来：

  - 根据 args_cli.task 去 registry 查这个 task
  - 取出该 task 的 env_cfg_entry_point
  - 再按第二个参数 "rsl_rl_cfg_entry_point" 取出 agent 配置入口
  - 实例化后，把 env_cfg 和 agent_cfg 作为参数传进 main(...)

  所以这里的接口其实是：

  - Gym registry：负责从 task name 找到 entry points
  - hydra_task_config(task_name, cfg_key)：负责把 entry point 类实例化成配置对象
    并注入函数参数

  之后 locotouch/scripts/train.py:85 会再做一层 CLI override：

  - env_cfg.scene.num_envs = 4096
  - env_cfg.seed = agent_cfg.seed
  - env_cfg.sim.device = args_cli.device，如果你没显式传，一般沿用配置默认
  - agent_cfg.max_iterations 可被 CLI 覆盖

  4. 环境配置本身是怎么组起来的

  你的 task 对应环境配置类是：

  locotouch/config/locotouch/locomotion_env_cfg.py:16

  LocomotionEnvCfg 继承 LocomotionBaseEnvCfg，自己只做了一件事：把机器人资产替换
  成 LocoTouch 的 locomotion 训练模型：

  locotouch/config/locotouch/locomotion_env_cfg.py:7

  它最终使用的是这个 USD：

  locotouch/assets/locotouch.py:18

  也就是 locotouch_without_tactile_instanceable.usd。说明这个 locomotion 任务训
  练时用的是“不带触觉传感器”的 instanceable 资产。

  真正把环境拼起来的是基类：

  locotouch/config/base/locomotion_base_env_cfg.py:289

  这是一个 ManagerBasedRLEnvCfg，里面定义了 IsaacLab manager-based 环境需要的几
  块：

  - scene：地面、机器人、接触传感器、灯光
    见 locotouch/config/base/locomotion_base_env_cfg.py:16
  - commands：速度指令采样
    见 locotouch/config/base/locomotion_base_env_cfg.py:52
  - observations：policy 和 critic 两组观测，都是通过
    ObservationTermCfg(func=mdp.xxx) 声明式绑定到 locotouch.mdp 里的函数
    见 locotouch/config/base/locomotion_base_env_cfg.py:67
  - actions：动作接口是 JointPositionActionPrevPrevCfg
    见 locotouch/config/base/locomotion_base_env_cfg.py:93
  - rewards：奖励项通过 RewardTermCfg(func=mdp.xxx) 绑定
    见 locotouch/config/base/locomotion_base_env_cfg.py:106
  - events：startup/reset/interval 随机化和扰动
    见 locotouch/config/base/locomotion_base_env_cfg.py:189
  - terminations：超时、姿态异常、机身碰撞等
    见 locotouch/config/base/locomotion_base_env_cfg.py:263

  这里最关键的接口是 IsaacLab 的 manager 配置接口：

  - ObservationTermCfg(func=...)
  - RewardTermCfg(func=...)
  - EventTermCfg(func=...)
  - TerminationTermCfg(func=...)

  也就是：配置文件不直接“跑训练”，它只是把 locotouch.mdp 里的函数注册成环境里的
  observation/reward/event/termination term。真正 gym.make() 环境时，
  ManagerBasedRLEnv 会读这些配置并组装出可执行环境。

  5. gym.make() 时发生了什么

  在 locotouch/scripts/train.py:107：

  env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

  因为这个 task 注册时的 entry_point 是 isaaclab.envs:ManagerBasedRLEnv，所以这
  里实例化的是一个 IsaacLab 的 manager-based 环境对象。

  也就是说，这里通过 Gym 的统一接口完成了：

  - 用 task id 找环境类
  - 把 cfg=env_cfg 传给该环境类构造函数
  - 环境内部再根据 scene / observations / actions / rewards / ... 配置完成
    manager 装配

  这是第一层重要的“接口边界”：train.py 并不直接 import 某个具体环境类，而是通过
  gym.make(task, cfg=...) 解耦。

  6. 为什么还要再包一层 RslRlVecEnvWrapper

  在 locotouch/scripts/train.py:125：

  env = RslRlVecEnvWrapper(env)

  因为后面的 loco_rl runner 期望拿到的是一个满足 VecEnv 协议的环境对象。

  loco_rl 这边定义的接口在：

  loco_rl/loco_rl/env/vec_env.py:12

  它要求环境至少提供：

  - get_observations()
  - reset()
  - step(actions)
  - 属性 num_envs
  - 属性 num_actions
  - 属性 episode_length_buf
  - 属性 max_episode_length
  - 属性 device
  - 属性 cfg

  所以 RslRlVecEnvWrapper 的职责就是把 IsaacLab 环境适配成这个接口形状。这是第二
  层关键接口边界：IsaacLab env -> VecEnv adapter。

  7. PPO 配置是怎么传到 loco_rl 的

  你的 task 对应 agent 配置类是：

  locotouch/config/locotouch/agents/rsl_rl_ppo_cfg.py:5

  默认值包括：

  - num_steps_per_env = 24
  - max_iterations = 30000
  - actor/critic MLP hidden dims: [512, 256, 128]
  - PPO:
      - num_learning_epochs = 5
      - num_mini_batches = 4
      - learning_rate = 1e-3
      - gamma = 0.99
      - lam = 0.95
      - entropy_coef = 0.01
  - experiment_name = "locotouch_locomotion"
  - logger = "wandb"

  随后在 locotouch/scripts/train.py:141：

  runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir,
  device=agent_cfg.device)

  这里又是一个明显的接口转换：

  - 上游还是 IsaacLab 风格的 config object
  - 到 loco_rl runner 这一层时，变成普通 dict

  8. OnPolicyRunner 初始化时做了哪些事

  在 loco_rl/loco_rl/runners/on_policy_runner.py:25：

  1. 从 env.get_observations() 取出初始观测。
  2. 约定 env_obs["policy"] 给 actor 用，env_obs["critic"] 给 critic 用。
     这和你在环境配置里定义的 observations.policy / observations.critic 正好对
     应。
  3. 根据 policy_cfg["class_name"] 和维度，构造 actor-critic 网络。
  4. 根据 algorithm 配置构造 PPO 实例。
  5. 初始化 rollout storage，大小是：
      - num_envs = 4096
      - num_steps_per_env = 24

  所以每个 iteration 一次会先采 4096 * 24 = 98304 条环境步。

  这里有个非常关键的跨模块约定：

  - 环境通过 get_observations() 返回一个 dict，至少有 policy
  - runner 假设这个 dict 里可能还有 critic、rnd_state
  - 这个约定不是 Gym 原生接口，而是 RslRlVecEnvWrapper / loco_rl 之间的私有协议

  9. 正式训练循环里，每一轮发生什么

  train.py 最后调用：

  locotouch/scripts/train.py:165

  runner.learn(num_learning_iterations=agent_cfg.max_iterations,
  init_at_random_ep_len=True)

  主循环在：

  loco_rl/loco_rl/runners/on_policy_runner.py:118

  每个 iteration 的过程是：

  1. 先拿当前观测 obs 和 critic_obs。
  2. 连续 rollout num_steps_per_env=24 步。
  3. 每步里：
      - actions = self.alg.act(obs, critic_obs)
        见 loco_rl/loco_rl/algorithms/ppo.py:129
      - next_obs, rewards, dones, infos = self.env.step(actions)
        见 loco_rl/loco_rl/runners/on_policy_runner.py:176
      - 把 next_obs 塞回 infos["observations"]
      - self.alg.process_env_step(rewards, dones, infos) 把 transition 存进
        rollout storage
        见 loco_rl/loco_rl/algorithms/ppo.py:143

  rollout 收完后：

  4. self.alg.compute_returns(critic_obs) 计算 GAE / returns
     见 loco_rl/loco_rl/algorithms/ppo.py:172
  5. self.alg.update() 做 PPO 更新
     见 loco_rl/loco_rl/algorithms/ppo.py:179

  update() 里面的核心是：

  - 从 storage 取 mini-batch
  - 用当前策略重新算 log_prob、value
  - 算 PPO clipped surrogate loss
  - 算 value loss
  - 算 entropy bonus
  - 如果配置了 symmetry / RND，再附加对应 loss
  - optimizer.step()

  具体损失构造在：

  - surrogate loss: loco_rl/loco_rl/algorithms/ppo.py:283
  - value loss: loco_rl/loco_rl/algorithms/ppo.py:291
  - 总 loss: loco_rl/loco_rl/algorithms/ppo.py:302

  10. 日志、checkpoint、配置落盘

  训练开始前，train.py 会创建：

  - logs/rsl_rl/locotouch_locomotion/<timestamp>/

  并把环境和 agent 配置写到：

  - params/env.yaml
  - params/agent.yaml
  - params/env.pkl
  - params/agent.pkl

  见 locotouch/scripts/train.py:97 和 locotouch/scripts/train.py:159

  训练过程中 OnPolicyRunner.log() 会写 wandb/tensorboard，并且每
  save_interval=50 轮存一个模型：

  见 loco_rl/loco_rl/runners/on_policy_runner.py:245

  按“接口”总结一遍

  这条训练命令里，模块之间主要靠这些接口接起来：

  - CLI 到入口脚本：argparse
  - 入口脚本到仿真器：AppLauncher(args_cli)
  - task 字符串到任务定义：gym.register(...) / gym.make(task, ...)
  - task 到配置实例：hydra_task_config(task_name, "rsl_rl_cfg_entry_point")
  - 环境配置到具体 MDP 逻辑：ObservationTermCfg/RewardTermCfg/EventTermCfg/
    TerminationTermCfg(func=mdp.xxx)
  - IsaacLab 环境到 RL 框架：RslRlVecEnvWrapper
  - RL runner 需要的环境协议：VecEnv.get_observations/reset/step
  - 配置对象到 runner：agent_cfg.to_dict()
  - runner 到 PPO：PPO.act / process_env_step / compute_returns / update
