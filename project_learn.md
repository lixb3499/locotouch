可以把这个项目按“先跑通主线，再往下拆层”的方式看。最省力的读法不是从文件夹一个个硬啃，而是先抓住一条完整链路：

`命令行脚本 -> task 名字 -> 环境配置 -> MDP 组成 -> 机器人资产 -> RL/蒸馏后端`

你如果按这条线看，整个仓库会顺很多。

**先建立全局地图**
这个仓库本质上由两大块组成：

- `locotouch/`：任务与环境本体
- `loco_rl/`：训练算法后端

根目录的 [README.md](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/README.md) 先看一遍，它告诉你这个项目主要做三件事：

- 四足机器人 locomotion
- 物体搬运 teacher policy
- 带触觉输入的 student policy distillation

所以你可以把项目理解成：

- `teacher`：用完整状态信息训练一个强策略
- `student`：只看更受限的输入，比如触觉 + 本体感觉，再通过蒸馏学 teacher

**第一步：先看入口脚本**
最值得先看的，是 `locotouch/scripts/` 目录：

- [train.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/scripts/train.py)
- [play.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/scripts/play.py)
- [distill.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/scripts/distill.py)
- [cli_args.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/scripts/cli_args.py)

建议你按这个顺序读：

1. `train.py`
2. `play.py`
3. `distill.py`
4. `cli_args.py`

因为这三个脚本就是整个项目最顶层的“用户接口”。

你在 `train.py` 里重点看这几件事：

- 它怎么接收 `--task`
- 它怎么 `from locotouch import *`
- 它怎么 `gym.make(args_cli.task, cfg=env_cfg, ...)`
- 它怎么构造 `OnPolicyRunner`

这会让你立刻明白：  
这个项目不是自己手写训练循环，而是把 Isaac Lab 环境构好，再交给 `loco_rl` 的 runner 去训练。

`play.py` 和 `train.py` 结构很像，只是改成加载 checkpoint 然后不断推理。

`distill.py` 则是另一条主线，它进入 [locotouch/distill/distillation.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/distill/distillation.py)，这是 student-teacher 蒸馏的总控。

**第二步：看 task 是怎么注册出来的**
你现在最该看的文件是：

- [locotouch/__init__.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/__init__.py)
- [locotouch/config/locotouch/__init__.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/locotouch/__init__.py)
- [locotouch/config/go1/__init__.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/go1/__init__.py)

这里是整个项目的“任务目录”。

当 `train.py` 里执行 `from locotouch import *` 时，`locotouch` 包会导入各个子包，最终触发 `gym.register(...)`。  
所以命令里的：

```bash
python locotouch/scripts/train.py --task Isaac-Locomotion-Go1-v1
```

本质是在 `config/go1/__init__.py` 里找到这个 task 对应的：

- `env_cfg_entry_point`
- `rsl_rl_cfg_entry_point`
- 有时还有 `distillation_cfg_entry_point`

你读这些 `__init__.py` 时，不要逐字细看，先只回答三个问题：

- 项目一共注册了哪些任务？
- 每个任务对应哪个 env cfg？
- 每个任务对应哪个 agent cfg？

这个阶段的目标是“认识地图”，不是抠细节。

**第三步：看环境配置是怎么拼起来的**
接下来进入最重要的一层：`config/`

建议顺序：

1. [locotouch/config/base/locomotion_base_env_cfg.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/base/locomotion_base_env_cfg.py)
2. [locotouch/config/go1/locomotion_env_cfg.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/go1/locomotion_env_cfg.py)
3. [locotouch/config/go1/locomotion_vel_cur_env_cfg.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/go1/locomotion_vel_cur_env_cfg.py)
4. `locotouch/config/locotouch/` 下面对应版本
5. 搬运任务配置，如 [object_transport_teacher_env_cfg.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/locotouch/object_transport_teacher_env_cfg.py)

`locomotion_base_env_cfg.py` 是环境骨架，你可以把它当成“所有任务的模板类”。  
它里面大致分为这些块：

- `MySceneCfg`：场景里有什么
- `CommandsCfg`：速度指令怎么采样
- `ObservationsCfg`：观测由哪些项组成
- `ActionsCfg`：动作是什么
- `RewardsCfg`：奖励有哪些项
- `EventCfg`：初始化、重置、扰动等随机事件
- `TerminationCfg`：什么时候结束
- 最后组合成整个 env cfg

你读这个文件时，重点不是记参数，而是建立一个概念：

Isaac Lab 的 manager-based env，本质就是在配置里声明：
- 看什么
- 做什么
- 奖什么
- 罚什么
- 什么时候重置

后面的具体任务配置文件，比如 `go1/locomotion_env_cfg.py`，通常只是做两件事：

- 继承 base cfg
- 替换机器人资产或局部参数

所以这类文件可以理解成“任务实例化层”。

**第四步：理解 `mdp/` 才是逻辑本体**
当你在配置里看到：

- `func=mdp.base_ang_vel`
- `func=mdp.track_lin_vel_xy_pst`
- `func=mdp.object_state_in_robot_frame`

这些函数都来自 `locotouch/mdp/`。

入口在 [locotouch/mdp/__init__.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/mdp/__init__.py)，它把几类逻辑聚合起来：

- `observations.py`：观测项
- `commands.py`：命令生成
- `actions.py`：动作处理
- `rewards.py`：奖励函数
- `curriculums.py`：课程学习
- `terminations.py`：终止条件
- `events.py`：初始化/重置/扰动

这层非常关键。你可以这样读：

1. 先在配置文件里挑一个你关心的项  
   例如 `track_lin_vel_xy_pst`
2. 再去 `mdp/rewards.py` 里找实现
3. 看它输入什么，输出什么，为什么这样设计

这样读会比直接硬翻整个 `rewards.py` 更轻松。

如果你问“这个项目真正的机器人任务逻辑在哪里”，答案基本就是：`mdp/`。

**第五步：搞清楚机器人资产层**
看这些文件：

- [locotouch/assets/go1.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/assets/go1.py)
- [locotouch/assets/locotouch.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/assets/locotouch.py)

它们不是训练逻辑，而是在定义“机器人长什么样、初始姿态是什么、电机参数是什么、用哪个 USD”。

你可以这样理解：

- `go1.py`：基础四足机器人定义
- `locotouch.py`：在 Go1 基础上换成带/不带触觉的不同机器人 USD 版本

而 `locotouch/assets/go1/`、`locotouch/assets/locotouch/` 下面的 `.usd`、`Props/`、`config.yaml`，就是仿真世界真正加载的资源文件。

如果你以后想改机器人模型、碰撞、传感器、mesh、instanceable 配置，这一层最重要。

**第六步：看训练后端 `loco_rl/`**
这个目录最好单独看，不要和 `locotouch/` 混在一起。

推荐顺序：

1. [loco_rl/README.md](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/loco_rl/README.md)
2. [loco_rl/loco_rl/runners/on_policy_runner.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/loco_rl/loco_rl/runners/on_policy_runner.py)
3. `loco_rl/loco_rl/algorithms/ppo.py`
4. `loco_rl/loco_rl/modules/`
5. `loco_rl/loco_rl/storage/rollout_storage.py`

它的职责很清晰：

- `runners/`：组织训练流程
- `algorithms/`：PPO 更新逻辑
- `modules/`：策略网络结构
- `storage/`：rollout buffer
- `models/`：底层网络模块
- `utils/`：日志和工具

你可以把它看成一个独立 PPO 框架。  
`locotouch` 负责定义任务，`loco_rl` 负责训练这个任务。

如果你是第一次读，先死盯 [on_policy_runner.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/loco_rl/loco_rl/runners/on_policy_runner.py) 就行。因为这个文件把：

- 环境观测怎么拿
- actor_critic 怎么构造
- rollout 怎么收集
- PPO 怎么 update

全串起来了。

**第七步：看 agent 配置**
训练超参数和网络结构主要在这些地方：

- [locotouch/config/locotouch/agents/rsl_rl_ppo_cfg.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/locotouch/agents/rsl_rl_ppo_cfg.py)
- `locotouch/config/go1/agents/`
- `locotouch/config/locotouch/agents/distillation_cfg.py`

这里定义了：

- PPO 迭代数
- actor/critic hidden dims
- 学习率
- logger
- experiment name
- distillation 参数

也就是说：

- `env cfg` 决定“任务长什么样”
- `agent cfg` 决定“怎么学”

**第八步：蒸馏模块要单独理解**
如果你关心这篇工作的核心贡献，那 `distill/` 要认真看。

推荐顺序：

1. [locotouch/scripts/distill.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/scripts/distill.py)
2. [locotouch/distill/distillation.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/distill/distillation.py)
3. `student.py`
4. `replay_buffer.py`
5. `tactile_recorder.py`
6. `distillation_cfg.py`

你读 `distillation.py` 时重点抓这几个问题：

- teacher policy 从哪里加载？
- student 输入由什么组成？
- tactile 信号怎么处理、有没有延迟？
- 数据怎么收集？
- 训练是 BC 还是 DAgger？
- play 模式和 training 模式差在哪里？

这部分实际上是项目最有“论文味”的地方。

**第九步：给你一条最推荐的实际阅读路线**
如果你真想系统看懂，我建议按这个 2 轮方式来。

第一轮，只看结构，不抠数学：

1. [README.md](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/README.md)
2. [locotouch/scripts/train.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/scripts/train.py)
3. [locotouch/config/locotouch/__init__.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/locotouch/__init__.py)
4. [locotouch/config/base/locomotion_base_env_cfg.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/base/locomotion_base_env_cfg.py)
5. [locotouch/mdp/__init__.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/mdp/__init__.py)
6. [loco_rl/loco_rl/runners/on_policy_runner.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/loco_rl/loco_rl/runners/on_policy_runner.py)

第二轮，选一个具体任务打穿。最建议选 `Go1 locomotion`，因为最简单：

1. [locotouch/config/go1/locomotion_env_cfg.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/go1/locomotion_env_cfg.py)
2. [locotouch/config/go1/locomotion_vel_cur_env_cfg.py](/Users/xuebin/BaiduSyncdisk/my_research/2026/locotouch/locotouch/locotouch/config/go1/locomotion_vel_cur_env_cfg.py)
3. 在 base cfg 里找到它继承了哪些默认观测/奖励
4. 去 `mdp/` 找这些函数实现
5. 回到 `rsl_rl_ppo_cfg.py` 看训练超参数
6. 回到 `train.py` 看它如何被跑起来

这样你能从“会跑”升级到“知道为什么这样设计”。

**每个目录一句话总结**
你可以先把整个仓库记成下面这样：

- `locotouch/scripts/`：入口
- `locotouch/config/`：任务注册与环境配置
- `locotouch/mdp/`：任务逻辑实现
- `locotouch/assets/`：机器人和仿真资源
- `locotouch/distill/`：teacher-student 蒸馏
- `locotouch/utils/`：一些辅助脚本，尤其是 URDF 处理
- `loco_rl/`：PPO 与策略网络训练框架
- `logs/`：训练输出
- `rm_figs/`：论文/文档图片
- `2505.23175v2.pdf`：论文正文

**如果你的目标不同，看法也不同**
如果你主要想改任务：
先看 `config/` + `mdp/`

如果你主要想改机器人：
先看 `assets/`

如果你主要想改 PPO/网络：
先看 `loco_rl/`

如果你主要想复现论文核心：
先看 `distill/` + 相关 student task cfg

如果你主要想先跑起来：
先看 `README.md` + `scripts/`

**最后给你一个非常实用的阅读方法**
每读一个文件，只回答这 3 个问题：

1. 这个文件定义了什么对象？
2. 它被谁调用？
3. 它依赖谁？

比如读 `locomotion_env_cfg.py`：

1. 它定义了一个具体 locomotion env cfg
2. 它被 `config/.../__init__.py` 注册给 task 名称使用
3. 它依赖 `locomotion_base_env_cfg.py` 和机器人资产定义

这样你不会陷进参数细节里出不来。

如果你愿意，我下一条可以直接给你整理一版“从 `Isaac-Locomotion-Go1-v1` 这一个任务出发的逐文件导读”，带着你把这个任务完整走一遍。