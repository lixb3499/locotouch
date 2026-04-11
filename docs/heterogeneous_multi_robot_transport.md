# 异构多机器人无通信编队搬运研究设想

## 研究动机

当前项目中的双狗协同搬运任务是多机器人协同搬运的第一步。更有学术价值的方向是进一步扩展到：

- 多个机器人协同搬运；
- 机器人之间存在异构性；
- 搬运过程中需要保持固定或半固定编队；
- 机器人之间不使用显式通信；
- 协同主要通过共享载荷的接触耦合和局部感知实现。

一个具体场景可以是：

- 四个异构小型四足机器人围绕一个大型轮式载体；
- 大型轮式载体类似“被搬运的大狗”，本身带有轮子、质量、惯量和滚动约束；
- 四个小狗分别处在载体的前左、前右、后左、后右位置；
- 每个小狗只根据自身局部观测和与载体的接触反馈行动；
- 系统整体需要按照速度命令或路径目标移动。

这个问题已经不是简单的物体搬运，而是**接触耦合约束下的异构多机器人编队搬运控制**。

## 核心科学问题

可以将核心科学问题凝练为：

**在无显式通信条件下，异构多机器人如何仅依靠局部本体感知、接触反馈和共享载荷的机械耦合，实现稳定的编队协同搬运？**

进一步可以拆成几个子问题：

- 异构机器人能力不同，如何自动形成合理分工？
- 多机器人通过同一个载荷形成强耦合动力学系统时，如何同时保持编队和载荷稳定？
- 没有显式通信时，机器人能否通过载荷运动和接触力变化实现隐式协同？
- 策略能否泛化到不同机器人能力、不同载荷质量、不同地面摩擦和不同编队规模？

## 问题特点

该问题和传统编队控制、普通物体搬运都有明显不同：

- 机器人之间不是独立运动，而是通过载荷发生动力学耦合；
- 载荷不是普通被动刚体，可以是具有轮式约束的动态平台；
- 机器人能力不同，不能简单要求所有机器人执行相同行为；
- 不使用显式通信，协同需要从物理交互中涌现；
- 任务同时包含全局载荷运动控制、局部接触稳定和编队保持。

## 推荐强化学习建模方式

建议采用：

**集中式训练，去中心化执行，无显式通信。**

也就是 CTDE 框架：

- 训练时 critic 可以看到全局 privileged state；
- 执行时每个机器人 actor 只能看到自己的局部观测；
- agent 之间不传递 message；
- 协同通过 payload 的接触动力学和局部反馈实现。

每个小狗可以看作一个 agent：

```text
a_i = pi(o_i, role_i, capability_i)
```

其中：

- `o_i`：第 `i` 个机器人的局部观测；
- `role_i`：机器人在编队中的角色，例如前左、前右、后左、后右；
- `capability_i`：机器人能力参数，表示异构性；
- `a_i`：该机器人的动作。

critic 可以使用全局信息：

```text
V = V(s_global)
```

其中 `s_global` 可以包含所有机器人状态、载荷状态、接触状态、编队误差和任务命令。

## 同一个 Actor 如何处理异构机器人

同一个 actor 不等于只能控制同构机器人。关键是 actor 输入中包含机器人自身的能力信息和角色信息。

策略形式可以写成：

```text
a_i = pi(o_i, role_i, capability_i)
```

虽然所有机器人共享同一个策略网络 `pi`，但是不同机器人输入不同，所以输出动作也可以不同。

例如：

- 强机器人可以承担更多推力或牵引任务；
- 弱机器人可以承担辅助稳定或轻负载支撑任务；
- 前侧机器人可以更多负责方向控制；
- 后侧机器人可以更多负责稳定和推进；
- 有触觉传感器的机器人可以更准确地调节接触；
- 无触觉机器人可以更多依靠本体状态和相对位姿。

### Capability Embedding

能力嵌入可以包含：

- 机器人质量；
- 机身尺寸；
- 腿长；
- 最大关节力矩；
- 最大速度；
- 动作尺度；
- 接触高度；
- 是否有触觉传感器；
- 控制延迟；
- 额定承载能力。

例如：

```text
capability_i = [
  mass,
  body_length,
  body_width,
  max_torque_scale,
  action_scale,
  max_speed,
  has_tactile,
  contact_height
]
```

### Role Embedding

角色嵌入可以包含：

- 期望接触点相对载荷的位置；
- 编队槽位，例如前左、前右、后左、后右；
- 期望局部朝向；
- 期望负载分配比例。

例如四机器人编队中：

- front-left；
- front-right；
- rear-left；
- rear-right。

共享 actor 加 capability embedding 和 role embedding 是处理异构多机器人问题的推荐第一版方案。

## 无通信协同机制

无通信不是没有协同。多个机器人共同接触一个载荷时，载荷本身就是物理信息通道。

机器人可以通过以下信号感知其他机器人的影响：

- 载荷位姿变化；
- 载荷速度变化；
- 接触力变化；
- 本体姿态变化；
- 局部相对载荷位姿变化；
- 局部触觉反馈。

这可以称为：

- contact-mediated coordination；
- embodied implicit communication；
- mechanical communication。

在论文中可以强调：机器人没有显式传递消息，但通过共享载荷的物理耦合实现隐式协同。

## 共享命令是否算通信

如果所有机器人都接收同一个任务命令，例如整体速度命令 `vx, vy, wz`，通常不认为这是机器人间通信。

它更像是任务目标，例如“把载荷向前移动”。

但如果想让问题更难，可以设计三种设置：

- 所有机器人都知道全局命令；
- 只有 leader 知道全局命令；
- 没有机器人直接知道全局命令，只通过载荷状态或局部目标推断任务。

这些可以作为论文中的消融实验。

## 观测设计

### Privileged Teacher 观测

teacher 可以使用全局状态：

- 载荷位置、姿态、线速度、角速度；
- 轮式载体的轮子状态；
- 所有机器人 base 状态；
- 所有机器人关节状态；
- 每个机器人相对载荷的位姿；
- 编队误差；
- 接触力或接触状态；
- capability embedding；
- role embedding；
- 全局速度或路径命令。

### Decentralized Student 观测

每个 robot actor 只使用局部观测：

- 自身 base angular velocity；
- 自身 projected gravity；
- 自身 joint position / velocity；
- 自身 last action；
- 自身 tactile/contact 信号；
- 自身相对载荷的局部位姿；
- 自身相对载荷的局部速度；
- 自身 role embedding；
- 自身 capability embedding；
- 可选的任务命令。

不应该使用：

- 其他机器人的关节状态；
- 其他机器人的动作；
- 其他机器人的 hidden state；
- 显式通信消息。

## Reward 设计

reward 应该同时约束载荷任务、编队、接触、负载分配和单体稳定。

### 载荷任务奖励

- 载荷速度跟踪；
- 载荷路径跟踪；
- 载荷 yaw 跟踪；
- 载荷 roll/pitch 稳定；
- 载荷高度稳定；
- 载荷不触地；
- 载荷角速度惩罚。

### 编队奖励

- 每个机器人保持在相对载荷的目标位置；
- 编队形状保持；
- 机器人之间相对位置稳定；
- 机器人之间相对速度一致；
- 机器人与载荷局部朝向对齐。

### 负载分配奖励

- 接触力保持在合理范围；
- 负载按机器人能力分配；
- 避免某个机器人过载；
- 避免弱机器人承担过大负载；
- 强机器人在需要时承担更多负载。

### 接触奖励

- 保持有效接触；
- 避免冲击；
- 避免打滑；
- 避免间歇性失去接触；
- 惩罚载荷和地面接触。

### 单体稳定奖励

- 存活奖励；
- base 高度保持；
- roll/pitch 角度惩罚；
- roll/pitch 角速度惩罚；
- foot slip 惩罚；
- action rate 惩罚；
- joint velocity / torque 正则。

## Termination 设计

termination 应该表示不可恢复或无效的搬运状态：

- 任一机器人摔倒；
- 载荷触地；
- 载荷姿态超过安全范围；
- 载荷远离编队中心；
- 某个机器人持续失去接触；
- episode timeout。

需要注意：不能直接照搬纯 locomotion 中的 illegal body contact termination，因为在搬运任务里，机身和载荷接触可能是正常行为。

## 算法路线

### Baseline

- 单个 centralized policy 控制所有机器人；
- 使用全局观测；
- 作为调试和性能上界。

### 主方法

- 共享 decentralized actor；
- centralized privileged critic；
- capability-conditioned policy；
- role-conditioned policy；
- 无显式通信；
- 通过载荷接触实现隐式协同。

### 进阶版本

- graph neural network critic；
- attention over robot-payload relations；
- role-conditioned actor；
- capability-conditioned actor；
- privileged teacher 到 tactile/contact student 的蒸馏；
- recurrent policy 处理接触历史。

## 实验设计

可以设计以下实验：

- 同构队伍 vs 异构队伍；
- 无通信 vs 有通信；
- 有 capability embedding vs 无 capability embedding；
- 有 role embedding vs 无 role embedding；
- 有 tactile/contact 观测 vs 无 tactile/contact 观测；
- 被动轮式载体 vs 普通刚体载荷；
- 不同载荷质量；
- 不同载荷重心；
- 不同地面摩擦；
- 不同机器人数量；
- 某个机器人能力下降；
- 某个机器人失效或掉队。

## 评价指标

建议不要只看 reward，可以定义论文级指标：

- task success rate；
- payload velocity tracking error；
- payload path tracking error；
- payload roll/pitch/yaw error；
- formation error；
- payload-ground contact rate；
- robot fall rate；
- payload drop rate；
- load sharing imbalance；
- contact consistency；
- energy consumption；
- generalization performance。

## 从当前项目出发的开发路线

建议按以下步骤扩展：

1. 稳定当前同构双狗 teacher；
2. 加入更完整的 payload-ground、formation failure 约束；
3. 把双狗 centralized policy 改成 decentralized shared actor；
4. 加入 left/right 或 front/rear role embedding；
5. 加入 capability embedding，先做轻度异构；
6. 从双狗扩展到四个机器人围绕载荷；
7. 将被动圆杆替换成轮式载体；
8. 训练 privileged teacher；
9. 蒸馏到局部 proprioception + tactile/contact student；
10. 做无通信和有通信版本的对比实验。

## 可能的论文定位

可以考虑以下题目方向：

- 无通信异构多智能体强化学习用于接触耦合编队搬运；
- 基于能力条件化策略的异构多机器人协同搬运；
- 通过共享载荷实现隐式通信的多四足机器人编队搬运；
- 面向动态轮式载体的异构机器人无通信协同搬运。

一句话贡献可以写成：

**本文研究无显式通信条件下的异构多机器人编队搬运问题，提出一种能力和角色条件化的多智能体强化学习框架，使去中心化机器人仅依赖局部感知和共享载荷的接触耦合实现稳定协同。**

