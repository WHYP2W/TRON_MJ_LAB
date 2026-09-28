# TRON2 SFYG：Windows 原生训练

面向 **SFYG_TRON2A** 的独立 mjlab 项目，强化学习策略负责行走，上肢关节由独立控制器控制。
无需 WSL，也不依赖参考训练仓库。本文中的命令均在仓库根目录执行。

本项目提供可运行的平地运动控制训练基线，不包含已训练成熟的行走或操作策略，也不是实机部署工具包。

## 已验证环境

- Windows 原生环境，NVIDIA RTX 4060 Laptop GPU，8 GB 显存，驱动版本 591.59。
- Python 3.11.15、mjlab 1.6.0、MuJoCo/MuJoCo Warp 3.11.0。
- PyTorch 2.10.0+cu128、Warp 1.17.0；完整依赖版本见 `uv.lock`。
- 默认使用 64 个并行训练环境，物理仿真频率 200 Hz，控制频率 50 Hz。
- Windows 原生运行已在本机验证，但不代表上游框架官方支持所有 Windows 配置。目前仅验证了单 GPU 运行。

## 安装与初始化

前置依赖：NVIDIA 驱动、Git 和 [uv](https://docs.astral.sh/uv/)。
在仓库根目录打开 PowerShell，执行：

```powershell
uv sync --locked
.\scripts\setup_assets.ps1
```

`uv` 会在项目内创建 `.venv`。在 VS Code 中打开项目后，请选择
`.venv\Scripts\python.exe` 作为 Python 解释器。CUDA 版 PyTorch 的下载量约为 2.7 GiB。
首次仿真需要编译 Warp 内核，可能耗时数分钟；后续运行会复用缓存。
Windows 使用原生 GLFW 渲染后端，不要设置 `MUJOCO_GL=egl`。

资产脚本仅从 [LimX 官方模型仓库](https://github.com/limxdynamics/tron2-robot-description)
下载 SFYG 的 XML 和网格目录，并固定到提交 `f547f5bc949f2a4c98e076e61cf6d3ca73d179a0`。
资产仓库中的许可证和声明会保留；已有本地修改的资产检出目录不会被覆盖。
如需使用其他兼容的模型仓库副本，可将 `TRON2_ASSET_ROOT` 指向包含 `tron2a/` 的目录。

## 训练任务

| 任务 | 策略动作维度 | 策略网络观测维度 |
| --- | ---: | ---: |
| `Mjlab-Velocity-Flat-TRON2-SFYG-External` | 10 | 74 |

mjlab 通过 Python 包入口点自动发现该任务。本项目仅注册 SFYG 独立控臂任务，不修改上游 mjlab 的内置任务。

- **独立控臂：** 强化学习策略只控制行走，由独立控制器设置 6 个机械臂关节和 2 个夹爪关节的目标。
  默认使用平滑变化的正弦目标，使行走策略在训练和普通回放时都能接触到运动中的上肢负载。
- 策略观测上肢关节位置、速度、期望目标，以及经过变化率限制后实际施加的目标。10 个腿部关节均采用位置控制。
- 策略动作空间只包含腿部关节，不包含机械臂和夹爪，策略没有上肢控制权。当前未加入抓取或末端跟踪奖励。

本项目不使用固定姿态的锁臂控制器。机械臂位置执行器的刚度/阻尼增益为 40/4，仿真力矩上限为 30 Nm；
夹爪增益为 100/5，力上限为 10 N。目标受关节软限位和变化率限制约束，机械臂为 1.5 rad/s，夹爪为 0.04 m/s。
这些参数仅作为仿真起点，不是经过标定的硬件限值。SFYG 保留必要的髋偏航初始偏置 -pi/+pi，没有焊接固定任何关节。

## 训练

```powershell
uv run train Mjlab-Velocity-Flat-TRON2-SFYG-External --env.scene.num-envs 64 --agent.max-iterations 1500
```

每次只运行一个训练命令。在这张 8 GB 显卡上建议从 64 个环境开始；如果其他程序占用显存，可降至 16 或 32 个。
从仓库根目录启动时，日志和检查点保存在 `logs/rsl_rl/tron2_sfyg_external/<timestamp>/`。
默认使用 TensorBoard，无需登录 W&B，也不会上传模型。

```powershell
uv run tensorboard --logdir logs/rsl_rl
uv run train Mjlab-Velocity-Flat-TRON2-SFYG-External --agent.resume True --agent.load-run "<run-directory-name>" --agent.load-checkpoint "model_100.pt"
```

续训时必须使用同一任务的检查点，并保持观测和动作结构一致。Isaac Lab 参考项目的检查点与本项目任务不兼容。

## 策略回放

选择本地训练生成的检查点：

```powershell
$Checkpoint = '.\logs\rsl_rl\tron2_sfyg_external\<run>\model_100.pt'
uv run play Mjlab-Velocity-Flat-TRON2-SFYG-External --checkpoint-file $Checkpoint --viewer native --num-envs 1
```

查看未训练的模型和独立控制器驱动的机械臂运动：

```powershell
uv run play Mjlab-Velocity-Flat-TRON2-SFYG-External --agent zero --viewer native --num-envs 1
```

未训练策略或零动作策略可能使机器人摔倒，随后环境会自动重置。

## Xbox 手柄回放（pygame）

通过 USB 或蓝牙将 Xbox 手柄连接到 Windows，使用独立回放入口：

```powershell
uv run tron2-play-gamepad --list-controllers
uv run tron2-play-gamepad --checkpoint-file $Checkpoint
```

`$Checkpoint` 使用上一节所选的本项目 External 任务检查点。该入口固定使用一个环境和原生 MuJoCo 窗口，
默认选择第一个 SDL 支持的手柄；多个手柄时可添加 `--controller-index 0`，编号以列举结果为准。
pygame 使用 SDL 标准 Xbox 映射，不需要手动配置摇杆轴编号，MuJoCo 窗口获得焦点后仍可读取手柄。

启动后先松开 LB，将摇杆回中并松开两个扳机，再按住 LB 使能。窗口左上角显示连接状态、使能状态、
当前机械臂关节组和速度指令。

| 操作 | 功能 |
| --- | --- |
| 按住 LB | 使能行走、控臂和夹爪；松开即清零速度指令并保持已施加的上肢目标 |
| LB + 左摇杆上下 | 前进 / 后退，最大 1.0 / 0.5 m/s |
| LB + 左摇杆左右 | 左转 / 右转，最大 0.6 rad/s；不发送未训练的侧移指令 |
| 十字键左 / 右 | 循环选择 arm1/arm2、arm3/arm4、arm5/arm6，默认第一组 |
| LB + 右摇杆左右 / 上下 | 分别增减当前组的第一个 / 第二个关节角度；向右 / 向上为增加 |
| LB + RT / LT | 张开 / 闭合夹爪，两个夹指同步反向运动 |
| LB + A | 上肢和夹爪平滑返回默认姿态，保持 LB 直到完成 |
| B | 停止并锁定输入；松开 LB、摇杆和扳机回中后才能重新使能 |
| View / Back | 重置仿真中的机器人，并解除使能 |
| Menu / Start | 暂停 / 继续仿真；继续后需要重新使能 |

手柄断连后会清零速度指令并保持已施加的上肢目标，重新连接后需要重新使能。摔倒后的自动重置也会解除使能，
防止旧指令立即作用于重置后的机器人。手柄模式取消随机速度和自动正弦上肢运动，回合上限为 1 小时，
仍保留摔倒检测。普通 `play` 和训练行为不变，策略动作及观测维度不变。

默认摇杆 / 扳机死区为 `0.15`，机械臂目标变化速度为 `0.6 rad/s`，单个夹指为 `0.03 m/s`。
可使用 `--deadzone 0.2 --arm-speed 0.4 --gripper-speed 0.02` 调整；目标继续受关节软限位及现有控制器的
速度限制约束。这里是关节空间控制，不包含末端逆运动学、自动抓取或避碰。

只查看模型及手柄控制、不加载策略时：

```powershell
uv run tron2-play-gamepad --agent zero
```

零动作策略无法完成平衡或行走。清零速度指令也不保证机器人立刻静止；实际运动取决于策略能力和仿真动力学。
B 键仅停止软件控制指令，不是实机急停。关闭 MuJoCo 窗口或在终端按 Ctrl+C 退出回放。

## 独立控制器接口

使用 `make_env_cfg()` 配置创建 `ManagerBasedRlEnv` 环境后调用：

```python
import torch

upper_body = env.action_manager.get_term("upper_body")
upper_body.set_targets(
    torch.tensor(
        [0.3, 0.5, -0.9, 0.1, 0.2, 0.0, 0.03, -0.03],
        device=env.device,
    )
)
observations, rewards, terminated, truncated, info = env.step(leg_actions)
upper_body.release_targets()
```

目标顺序为 `arm1_Joint` 至 `arm6_Joint`（单位：弧度），然后是 `gripper1_Joint` 和
`gripper2_Joint`（单位：米，两者的取值范围符号相反）。支持形状为 `(8,)` 的广播目标，
或形状为 `(selected_envs, 8)` 的批量目标；可通过可选的 `env_ids` 张量指定环境。

目标持续有效，直到被替换、主动释放，或对应环境重置。实时外部控制器应在每次步进前更新命令，环境重置后也不例外。
接口会拒绝包含非有限数值或形状错误的目标。腿部策略步进不会覆盖已经设置的外部上肢目标。

## 模型说明

本项目不提供训练成熟的行走策略。奖励调优、长期训练收敛、复杂地形、操作任务和仿真到实机验证不在本基线范围内。

模型适配器仅修改内存中的模型描述：移除模型自带的独立地面和电机，添加 mjlab 执行器与定位点，
并按 MuJoCo Warp MULTICCD 的要求将碰撞裕量设为零。场景时间步和接触容量由 mjlab 配置管理，不使用独立 XML 中的对应选项。
官方 XML 和网格文件保持不变。模型资产、虚拟环境、缓存、日志和生成产物仅保存在本地，不纳入 Git 或 Python 分发包。
