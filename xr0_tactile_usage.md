# XR0 触觉集成使用文档

> 基于 `gczx_Evo-RL` 项目的 `feature/xr0-vla-integration` 分支

---

## 目录

1. [触觉传感器可视化与参数调优](#1-触觉传感器可视化与参数调优)
2. [采集带触觉的数据集](#2-采集带触觉的数据集)
3. [数据处理与训练准备](#3-数据处理与训练准备)
4. [训练 XR0 模型](#4-训练-xr0-模型)
5. [推理部署](#5-推理部署)
6. [附录：完整命令行参考](#6-附录)

---

## 1. 触觉传感器可视化与参数调优

### 1.1 安装依赖

```bash
cd F:/gczx_Evo-RL
pip install -e ".[transformers-dep]"
```

触觉模块所需的 `pyserial` 和 `scipy` 已在项目核心依赖中。

### 1.2 命令行监控 (`lerobot-tactile-monitor`)

用于快速验证触觉传感器硬件连接和读数：

```bash
# 基础用法：查看左手指尖传感器读数
lerobot-tactile-monitor --port /dev/ttyACM0 --calibrate

# 输出示例：
# [00001] index=( 0.015, -0.023,  1.245) middle=(-0.008,  0.031,  0.892)
# [00002] index=( 0.018, -0.020,  1.267) middle=(-0.005,  0.028,  0.905)
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--port` | `/dev/ttyACM0` | 触觉控制器串口路径 |
| `--baudrate` | `921600` | 波特率（通常不需要改） |
| `--calibrate` | (flag) | 启动时自动做零位校准 |
| `--calibration-samples` | `50` | 校准采样数量 |
| `--calibration-warmup-frames` | `20` | 校准前丢弃的预热帧 |
| `--calibration-reducer` | `median` | 校准聚合方式：`median` 抗噪声更好，`mean` 更快 |
| `--output-dir` | 无 | 设置后定时保存 PNG 截图和 NPZ 原始数据 |
| `--save-every` | `0` | 每 N 帧保存一次，0=不保存 |
| `--max-frames` | `0` | 最大帧数后自动停止，0=无限 |

### 1.3 浏览器实时预览 (`lerobot-tactile-preview`)

启动本地 HTTP 服务器，在网页端实时查看触觉热力图：

```bash
# 启动左手传感器预览
lerobot-tactile-preview --port /dev/ttyACM0 --http-port 8765 --calibrate

# 另开终端，启动右手传感器预览
lerobot-tactile-preview --port /dev/ttyACM1 --http-port 8766 --calibrate
```

浏览器打开 `http://localhost:8765` 查看 2×2 面板热力图。

**输出图像说明**：每帧是一个 2×2 网格：
```
┌─────────────────────┬─────────────────────┐
│ Index 食指 Fz 热力图  │ Middle 中指 Fz 热力图  │  ← 第1行：压力分布 heatmap
├─────────────────────┼─────────────────────┤
│ Index RGB 力矢量图    │ Middle RGB 力矢量图    │  ← 第2行：RGB 力矢量 (B=Fz, G=Fy, R=Fx)
└─────────────────────┴─────────────────────┘
```

### 1.4 可视化参数调优（推荐参数）

以下参数决定了热力图的视觉效果和训练质量，**这是经过实践验证的最佳默认值**：

| 参数 | 推荐值 | 作用 | 调优建议 |
|---|---|---|---|
| `--output-size` | `256` | 每个子面板分辨率，最终图像是 `512×512×3` | 不要改。256 是 Qwen3-VL 最优平衡点（细节够用 + token 消耗适中） |
| `--heatmap-colormap` | `turbo` | Fz 热力图配色 | **turbo 最好用**。备选：`inferno`(对比度高)、`viridis`(色盲友好)、`jet`(传统但不推荐) |
| `--heatmap-vmin` | `0.0` | 热力图下限(N) | 一般保持 0。如果你做精细操作需要放大微小压力变化，可以设为负值（如 -5.0） |
| `--heatmap-vmax` | `25.5` | 热力图上限(N) | **关键参数！** 如果接触压力普遍很低（轻触 < 5N），降低到 10.0 能获得更多色彩层次；如果抓取重物（>30N），提高到 40.0 避免饱和 |
| `--heatmap-gamma` | `0.75` | Gamma 校正 | **0.75 是最佳实践**。<1 让低压力变化更可见（推荐轻触场景），=1 线性，>1 突出高压区域 |
| `--rgb-vmax-fz` | `25.5` | RGB 图法向力上限 | 与 `heatmap-vmax` 保持一致 |
| `--rgb-vmax-shear` | `12.8` | RGB 图切向力上限 | 通常约为法向力上限的一半 |

**快速调优工作流**：

```bash
# 1. 先不校准，看原始读数范围
lerobot-tactile-monitor --port /dev/ttyACM0 --max-frames 100

# 2. 根据看到的实际力范围调整 vmax
#    例如最大 Fz ≈ 8N，则设置 heatmap-vmax=10.0
lerobot-tactile-monitor --port /dev/ttyACM0 --calibrate \
    --heatmap-vmax 10.0 --rgb-vmax-fz 10.0 --rgb-vmax-shear 5.0

# 3. 保存一些样张到磁盘检查
lerobot-tactile-monitor --port /dev/ttyACM0 --calibrate \
    --heatmap-vmax 10.0 --rgb-vmax-fz 10.0 \
    --output-dir ./tactile_samples --save-every 10 --max-frames 50

# 4. 微调 gamma 使热力图层次丰富
lerobot-tactile-preview --port /dev/ttyACM0 --http-port 8765 --calibrate \
    --heatmap-vmax 10.0 --heatmap-gamma 0.6
```

---

## 2. 采集带触觉的数据集

### 2.1 方式一：专用采集脚本（推荐）

`lerobot-record-piper-tactile` 提供键盘驱动的灵活 episode 控制：

```bash
lerobot-record-piper-tactile \
    --task "Pick up the red cube and place it in the box" \
    --robot-type bi_piperx_follower \
    --left-follower-can can0 \
    --right-follower-can can1 \
    --left-leader-can can2 \
    --right-leader-can can3 \
    --top-camera 241322074213 \
    --left-wrist-camera 241322074214 \
    --right-wrist-camera 241322074215 \
    --ego-camera-side left \
    --left-tactile-port /dev/ttyACM0 \
    --right-tactile-port /dev/ttyACM1 \
    --tactile-output-size 256 \
    --tactile-heatmap-vmax 25.5 \
    --tactile-colormap turbo \
    --fps 30 \
    --follower-speed-ratio 100 \
    --dataset.repo_id my_org/bimanual_pick_place_tactile \
    --dataset.vcodec h264
```

**键盘操作**：

| 按键 | 功能 |
|---|---|
| `ENTER` | 开始新 episode（脚本会提示） |
| 录制中按 `ENTER` | 结束当前 episode，弹出标签选择 |
| 录制中按 `s` | 标记 **success** 并立即结束 |
| 录制中按 `f` | 标记 **failure** 并立即结束 |
| 录制中按 `o` | 标记 **ongoing** 并保存 |
| 录制中按 `d` | **丢弃** 当前 episode |
| 录制中按 `q` | **退出** 采集会话 |
| `Ctrl-C` | 紧急中断 |

**标签选择提示**（ENTER 结束后）：
```
Save as [s]uccess/[f]ailure/[o]ngoing/[d]iscard/[q]uit (ENTER=success):
```

### 2.2 方式二：使用标准 `lerobot-record`（时间驱动）

如果你的采集流程是固定时长的，也可以用标准脚本：

```bash
lerobot-record \
    --robot.type=bi_piperx_follower \
    --robot.id=bi_piper_collect \
    --robot.left_arm_config.port=can0 \
    --robot.right_arm_config.port=can1 \
    --robot.left_arm_config.cameras='{
        wrist: {type: intelrealsense, serial_number_or_name: "241322074214", width: 640, height: 480, fps: 30, warmup_s: 2},
        ego: {type: intelrealsense, serial_number_or_name: "241322074213", width: 640, height: 480, fps: 30, warmup_s: 2},
        tactile: {type: tactile, port: "/dev/ttyACM0", output_size: 256, heatmap_vmax: 25.5, calibrate_on_connect: true}
    }' \
    --robot.right_arm_config.cameras='{
        wrist: {type: intelrealsense, serial_number_or_name: "241322074215", width: 640, height: 480, fps: 30, warmup_s: 2},
        tactile: {type: tactile, port: "/dev/ttyACM1", output_size: 256, heatmap_vmax: 25.5, calibrate_on_connect: true}
    }' \
    --robot.right_arm_config.require_calibration=false \
    --robot.left_arm_config.require_calibration=false \
    --teleop.type=bi_piperx_leader \
    --teleop.id=bi_piper_leader \
    --teleop.left_arm_config.port=can2 \
    --teleop.right_arm_config.port=can3 \
    --teleop.left_arm_config.require_calibration=false \
    --teleop.right_arm_config.require_calibration=false \
    --dataset.repo_id=my_org/tactile_dataset \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=30 \
    --dataset.single_task="Pick and place the cube" \
    --dataset.vcodec=h264 \
    --enable_episode_outcome_labeling=true \
    --episode_success_key=s \
    --episode_failure_key=f \
    --display_data=true
```

### 2.3 不接触觉传感器时也可以采集

如果暂时没有触觉硬件，直接去掉 `--left-tactile-port` 和 `--right-tactile-port` 参数即可。采集脚本会自动跳过触觉相机，只录制摄像头数据。这样采集的数据集和正常 LeRobot 数据集完全一致。

### 2.4 采集后的数据集结构

```
~/.cache/huggingface/lerobot/my_org/tactile_dataset/
├── data/
│   └── chunk-0000/
│       ├── episode_000000.parquet                    # 状态 + 动作 (表格数据)
│       ├── episode_000000_observation.images.left_ego.mp4
│       ├── episode_000000_observation.images.left_wrist.mp4
│       ├── episode_000000_observation.images.left_tactile.mp4  ← 触觉视频
│       ├── episode_000000_observation.images.right_wrist.mp4
│       ├── episode_000000_observation.images.right_tactile.mp4  ← 触觉视频
│       └── ...
├── meta/
│   ├── info.json     # 特征 schema（自动包含 tactile 图像特征）
│   └── stats.json    # 归一化统计量（训练时自动计算）
```

每条 episode 中 `observation.images.left_tactile` 和 `observation.images.right_tactile` 是 **512×512×3 的 RGB 视频**（由 TactileVisualizer 实时渲染的热力图）。

---

## 3. 数据处理与训练准备

### 3.1 数据检查

```bash
# 查看数据集元信息和统计
lerobot-dataset-report --dataset my_org/tactile_dataset

# 可视化检查（包含触觉通道）
lerobot-dataset-viz --dataset my_org/tactile_dataset --episode 0
```

预期输出中应该看到 image keys 包含 `observation.images.left_tactile` 和 `observation.images.right_tactile`。

### 3.2 无需额外预处理

**采集输出的 LeRobot 格式数据集可以直接用于训练**，不需要：

- ~~`convert_lerobot_to_xr0.py`~~（XR0 自定义格式转换）
- ~~`prepare_xr0_dataset.py`~~（计算 mean/std、生成训练配置）

这些步骤由 `lerobot-train` 内部的 normalization pipeline 自动完成。

### 3.3 （可选）事后向已有数据集注入触觉视图

如果你有一个已经采集好的数据集但没有触觉通道，可以用触觉视频文件替换：

```bash
# 将 wrist 视频复制为 tactile 视频（用于测试，不是真实触觉数据）
python -c "
from lerobot.cameras.tactile import TactileCameraConfig
# 或者手工复制 mp4 文件并修改 info.json 添加 tactile image features
"
```

> 注意：这只是用于冒烟测试。训练应该使用真实触觉传感器采集的数据。

---

## 4. 训练 XR0 模型

### 4.1 单 GPU 训练

```bash
lerobot-train \
    --dataset.repo_id=my_org/tactile_dataset \
    --policy.type=xr0 \
    --policy.qwen_variant=Qwen/Qwen3-VL-4B-Instruct \
    --policy.freeze_vlm=true \
    --policy.dit_num_layers=16 \
    --policy.dit_hidden_size=1024 \
    --policy.num_inference_steps=5 \
    --policy.flow_sampling=beta \
    --policy.max_state_dim=32 \
    --policy.max_action_dim=32 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.optimizer_lr=2.5e-5 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=30000 \
    --batch_size=16 \
    --steps=30000 \
    --output_dir=outputs/train/xr0_tactile \
    --job_name=xr0_tactile_v1 \
    --wandb.enable=true
```

### 4.2 多 GPU 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --mixed_precision=bf16 \
    $(which lerobot-train) \
    --dataset.repo_id=my_org/tactile_dataset \
    --policy.type=xr0 \
    --policy.qwen_variant=Qwen/Qwen3-VL-4B-Instruct \
    --policy.freeze_vlm=true \
    --batch_size=8 \
    --steps=30000 \
    --output_dir=outputs/train/xr0_tactile \
    --job_name=xr0_tactile_v1_mgpu \
    --wandb.enable=true
```

### 4.3 XR0 收敛行为

训练过程中 XR0 自动处理 **5-7 张图像输入**（取决于是否有触觉 + 是否缺左腕摄像头）：

```
输入:
  observation.images.left_ego       → Qwen3-VL
  observation.images.left_wrist     → Qwen3-VL
  observation.images.left_tactile   → Qwen3-VL  ← 512×512 热力图
  observation.images.right_wrist    → Qwen3-VL
  observation.images.right_tactile  → Qwen3-VL  ← 512×512 热力图
  observation.state                 → State Projector → DiT
  task text (含 ACP tags)           → Qwen3-VL text encoder

VLM 输出:
  KV-cache (包含所有图像的视觉理解)

DiT:
  交叉注意力到 KV-cache → Rectified Flow → action (32-dim)
```

**训练超参建议**：

| 参数 | 有触觉 | 无触觉 | 说明 |
|---|---|---|---|
| `batch_size` | `16` | `32` | 触觉增加了 2 张 512×512 图像，显存消耗更大 |
| `freeze_vlm` | `true` | `true` | 冻结 VLM 只训练 DiT，收敛更稳定 |
| `dit_num_layers` | `16` | `16` | 触觉增加了信息量，16 层足够 |
| `optimizer_lr` | `2.5e-5` | `2.5e-5` | 不变 |
| `steps` | `30000` | `30000` | 有触觉时数据更丰富，相同步数可能收敛更好 |

### 4.4 监控训练

```bash
# 查看 wandb 面板
# https://wandb.ai/<your_entity>/xr0_tactile_v1

# 关键指标：
# - train/loss: 流匹配 velocity MSE loss
# - train/action_mse: action 预测误差
# - val/loss: 验证集 loss
```

---

## 5. 推理部署

### 5.1 单步推理（测试）

```python
from lerobot.policies.xr0 import XR0Policy, XR0Config
import torch

# 加载训练好的模型
config = XR0Config.from_pretrained("outputs/train/xr0_tactile/checkpoints/latest")
policy = XR0Policy(config)
policy.eval()

# 准备输入（与数据集格式一致）
batch = {
    "observation.state": torch.randn(1, 32),
    "observation.images.left_ego": torch.randn(1, 3, 480, 640),
    "observation.images.left_wrist": torch.randn(1, 3, 480, 640),
    "observation.images.left_tactile": torch.randn(1, 3, 512, 512),   # ← 触觉
    "observation.images.right_wrist": torch.randn(1, 3, 480, 640),
    "observation.images.right_tactile": torch.randn(1, 3, 512, 512),  # ← 触觉
    "task": "Pick up the red cube",
}

with torch.no_grad():
    action = policy.predict_action(batch)  # (1, 50, 32)
    print(action.shape)
```

### 5.2 真实机器人推理

用 Evo-RL 的 `lerobot-human-inloop-record` 配合训练好的 policy：

```bash
lerobot-human-inloop-record \
    --robot.type=bi_piperx_follower \
    --robot.left_arm_config.port=can0 \
    --robot.right_arm_config.port=can1 \
    --robot.left_arm_config.cameras='{
        wrist: {type: intelrealsense, serial_number_or_name: "...", width: 640, height: 480, fps: 30, warmup_s: 2},
        ego: {type: intelrealsense, serial_number_or_name: "...", width: 640, height: 480, fps: 30, warmup_s: 2},
        tactile: {type: tactile, port: "/dev/ttyACM0", output_size: 256}
    }' \
    --robot.right_arm_config.cameras='{
        wrist: {type: intelrealsense, serial_number_or_name: "...", width: 640, height: 480, fps: 30, warmup_s: 2},
        tactile: {type: tactile, port: "/dev/ttyACM1", output_size: 256}
    }' \
    --teleop.type=bi_piperx_leader \
    --teleop.left_arm_config.port=can2 \
    --teleop.right_arm_config.port=can3 \
    --policy.path=outputs/train/xr0_tactile/checkpoints/latest \
    --dataset.repo_id=my_org/rollout_round2 \
    --dataset.single_task="Pick up the red cube" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=60 \
    --display_data=true
```

### 5.3 推理时触觉热力图不需要可视化

注意：推理时 TactileCamera 只在进程内渲染热力图 → 喂给 VLM，**不需要**启动浏览器或保存图像。如果你想实时查看触觉状态，可以另开终端运行 `lerobot-tactile-preview`。

---

## 6. 附录

### 6.1 完整命令行参考

```bash
# === 触觉工具 ===
lerobot-tactile-monitor --help
lerobot-tactile-preview --help

# === 采集 ===
lerobot-record-piper-tactile --help

# === 训练 ===
lerobot-train --help

# === 数据集 ===
lerobot-dataset-report --help
lerobot-dataset-viz --help
```

### 6.2 已知限制

| 问题 | 影响 | 解决方案 |
|---|---|---|
| 触觉帧率 ~10-20Hz vs 摄像头 30fps | 触觉视频有重复帧 | 可接受，VLM 主要学习空间压力模式而非时序 |
| Qwen3-VL token 消耗随图像数量线性增长 | 7 张图像时推理较慢 | `freeze_vlm=true` 时 VLM 只跑一次，DiT 推理很快 |
| 触觉串口可能断连 | 采集中断 | TactileCamera 有 10 次重试机制 + 自动重连 |
| 512×512 触觉图像增加显存 | batch_size 需要减小 | 参考上表，建议 16 而非 32 |

### 6.3 文件改动汇总

| 文件 | 操作 |
|---|---|
| `src/lerobot/cameras/tactile/__init__.py` | 新建 |
| `src/lerobot/cameras/tactile/configuration_tactile.py` | 新建 |
| `src/lerobot/cameras/tactile/camera_tactile.py` | 新建 |
| `src/lerobot/cameras/tactile/driver.py` | 新建 |
| `src/lerobot/cameras/tactile/protocol.py` | 新建 |
| `src/lerobot/cameras/tactile/runtime.py` | 新建 |
| `src/lerobot/cameras/tactile/types.py` | 新建 |
| `src/lerobot/cameras/tactile/visualizer.py` | 新建 |
| `src/lerobot/scripts/lerobot_tactile_monitor.py` | 新建 |
| `src/lerobot/scripts/lerobot_tactile_preview.py` | 新建 |
| `src/lerobot/scripts/lerobot_record_piper_tactile.py` | 新建 |
| `src/lerobot/cameras/__init__.py` | 修改 (+2行) |
| `src/lerobot/cameras/utils.py` | 修改 (+3行) |
| `src/lerobot/scripts/lerobot_record.py` | 修改 (+1行) |
| `src/lerobot/scripts/lerobot_teleoperate.py` | 修改 (+1行) |
| `pyproject.toml` | 修改 (+3行 CLI 入口) |
| **XR0 Policy (`policies/xr0/`)** | **零改动** ✅ |
