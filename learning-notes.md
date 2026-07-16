# LeRobot 学习笔记

> 基于 `lerobot 0.6.0`，个人学习记录
> 分支：`learning-notes`
> GPU：RTX 4000 Ada / CUDA 13.0 / PyTorch 2.11

---

## 学习计划

- [ ] 第一阶段：环境验证 + 核心概念
- [ ] 第二阶段：数据集（LeRobotDataset）
- [ ] 第三阶段：策略基类（PreTrainedPolicy）
- [ ] 第四阶段：训练流程（train.py）
- [ ] 第五阶段：硬件抽象（Robot 接口）
- [ ] 第六阶段：第一个策略——ACT
- [ ] 第七阶段：数据采集（teleoperate + record）
- [ ] 第八阶段：评估（eval + rollout）
- [ ] 第九阶段：进阶——扩散策略 / VLA 大模型

---

## 第一阶段：环境验证 + 核心概念

### 目标
- [ ] 跑通 `lerobot-info`（已完成）
- [ ] 了解项目整体架构
- [ ] 跑通一个最简单的 demo

### 笔记

#### 项目架构速览
```
src/lerobot/
├── scripts/           # CLI 入口（lerobot-train, lerobot-eval, lerobot-record 等）
├── configs/           # draccus 配置系统，ChoiceRegistry 多态
├── policies/          # 每个策略一个子目录，继承 PreTrainedPolicy
├── datasets/          # LeRobotDataset 核心
├── envs/              # 仿真环境（Gymnasium 集成）
├── robots/            # 硬件抽象层
├── motors/            # 电机控制
├── cameras/           # 相机驱动
└── teleoperators/     # 遥操作设备（数据采集）
```

#### 技术栈
- Python 3.12+ / PyTorch / Hugging Face / draccus（配置）/ Gymnasium（环境）
- 包管理：`uv`
- 数据格式：Parquet（状态/动作）+ MP4（视频）

#### 关键命令
```bash
# 安装
pip install lerobot

# 开发环境
uv sync --locked

# 训练
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_mobile_cabinet

# 评估
lerobot-eval --policy.path=lerobot/pi0_libero_finetuned --env.type=libero --env.task=libero_object

# 数据采集
lerobot-teleoperate
lerobot-record
```

---

## 第二阶段：数据集（LeRobotDataset）

### 目标
- [ ] 理解 LeRobotDataset 格式
- [ ] 学会加载 Hugging Face Hub 上的数据集
- [ ] 理解 episode / frame 的概念
- [ ] 学会查看数据集结构

### 笔记

#### 数据格式
- **视频**：MP4 或图片序列（observation）
- **状态/动作**：Parquet 文件
- **存储**：Parquet + MP4，托管在 Hugging Face Hub

#### 核心代码
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# 访问数据
episode_index = 0
print(f"{dataset[episode_index]['action'].shape=}")
```

#### 关键类
- `LeRobotDataset`：episode 级采样 + 视频解码
- `LeRobotDatasetMetadata`：数据集元信息

#### 常用数据集
- `lerobot/aloha_mobile_cabinet`
- `lerobot/so100_pick_single`

---

## 第三阶段：策略基类（PreTrainedPolicy）

### 目标
- [ ] 理解 PreTrainedPolicy 基类
- [ ] 理解策略工厂（factory.py）
- [ ] 理解 draccus ChoiceRegistry 多态机制

### 笔记

#### PreTrainedPolicy
- 位置：`src/lerobot/policies/pretrained.py`
- 继承：`nn.Module` + `HubMixin`
- 所有策略的父类

#### 多态机制
- `@register_subclass("name")` 装饰器
- 通过 `factory.py` 延迟导入
- 配置类使用 draccus ChoiceRegistry

#### 已有策略列表

**模仿学习（Imitation Learning）**
- [ ] ACT
- [ ] Diffusion
- [ ] VQ-BeT
- [ ] Multitask DiT

**强化学习（Reinforcement Learning）**
- [ ] HIL-SERL
- [ ] TDMPC

**VLA 大模型**
- [ ] Pi0
- [ ] Pi0.5（Pi0Fast）
- [ ] GR00T N1.5
- [ ] SmolVLA
- [ ] XVLA
- [ ] EO-1
- [ ] MolmoAct2
- [ ] WALL-OSS

**世界模型**
- [ ] VLA-JEPA

**Reward Model**
- [ ] SARM
- [ ] TOPReward
- [ ] Robometer

---

## 第四阶段：训练流程

### 目标
- [ ] 理解 train.py 入口
- [ ] 理解 TrainPipelineConfig 配置
- [ ] 理解训练循环的关键步骤
- [ ] 掌握如何配置训练参数

### 笔记

#### 入口
```bash
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

#### 配置文件
- 位置：`src/lerobot/configs/`
- `train.py` → `TrainPipelineConfig`（顶层）
- `policies.py` → `PreTrainedConfig`（基类）

#### 训练流程
1. 加载数据集
2. 创建策略（factory 多态）
3. 构建 DataProcessorPipeline / PolicyProcessorPipeline
4. 训练循环（forward / backward / optimizer.step）
5. 定期 eval / 保存 checkpoint

---

## 第五阶段：硬件抽象（Robot 接口）

### 目标
- [ ] 理解 Robot 基类
- [ ] 理解 motors / cameras / teleoperators 的抽象
- [ ] 学会接入新硬件

### 笔记

#### Robot 接口
```python
from lerobot.robots.myrobot import MyRobot

robot = MyRobot(config=...)
robot.connect()
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

#### 已支持硬件
SO100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones, OpenARM, Unitree G1, reBot B601

#### 自定义机器人步骤
1. 实现 `Robot` 接口
2. 复用数据采集、训练、可视化工具
3. 分享到 HF Hub

---

## 第六阶段：第一个策略——ACT

### 目标
- [ ] 理解 ACT（Action Chunking with Transformers）原理
- [ ] 运行 ACT 训练
- [ ] 理解代码实现

### 笔记

#### 原理
- Action Chunking：预测一段动作序列而非单步
- Transformer 编码观测，解码动作
- 用于模仿学习

#### 训练
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

#### 资源
- 文档：`docs/source/policy_act_README.md`
- 代码：`src/lerobot/policies/act/`

---

## 第七阶段：数据采集

### 目标
- [ ] 理解遥操作（teleoperation）流程
- [ ] 学会使用 lerobot-teleoperate
- [ ] 学会使用 lerobot-record 采集数据

### 笔记

#### 遥操作
```bash
lerobot-teleoperate
```

#### 数据采集
```bash
lerobot-record
```

#### 关键组件
- `teleoperators/`：遥操作设备驱动
- `cameras/`：相机采集
- `record` 脚本：录制 episode 到 LeRobotDataset 格式

---

## 第八阶段：评估

### 目标
- [ ] 理解 lerobot-eval 用法
- [ ] 理解 lerobot-rollout 用法
- [ ] 学会在仿真环境中评估

### 笔记

#### 评估
```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

#### 支持 benchmark
- LIBERO
- MetaWorld

#### Rollout
```bash
lerobot-rollout
```

---

## 第九阶段：进阶

### VLA 大模型（需要多卡 GPU）

#### Pi0 / Pi0.5
- 最强的 VLA 策略之一
- 需要大量 GPU 显存

#### GR00T / SmolVLA
- 开源 VLA 方案
- 相对轻量

### 扩散策略（Diffusion Policy）
- 基于扩散模型的机器人策略
- 文档：`docs/source/policy_diffusion_README.md`

---

## 学习问题记录

### Q&A

| 日期 | 问题 | 答案 |
|------|------|------|
|      |      |      |

### 代码笔记

| 日期 | 文件 | 关键点 |
|------|------|--------|
|      |      |        |

---

## 参考资料

- [官方文档](https://huggingface.co/docs/lerobot/index)
- [Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial)
- [Discord 社区](https://discord.gg/q8Dzzpym3f)
- [中文教程（子豪兄）](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)
