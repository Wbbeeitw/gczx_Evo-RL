# XR0 触觉集成 — 部署与调试记录

> 日期: 2026-06-07 | 服务器: enine@192.168.110.85 | 分支: `feature/xr0-vla-integration`

---

## 一、环境配置

### 1.1 克隆与安装

```bash
git clone https://github.com/Wbbeeitw/gczx_Evo-RL.git ~/gczx_Evo-RL_tactile
cd ~/gczx_Evo-RL_tactile
git checkout feature/xr0-vla-integration
conda create -y -n evo_rl_xr0 python=3.10
conda activate evo_rl_xr0
pip install -e ".[transformers-dep]"
pip install "scipy>=1.10.1"
pip install "pyrealsense2>=2.55.1"
```

### 1.2 系统依赖

```bash
# Piper Leader 重力补偿需要
sudo apt-get install -y liburdfdom-dev
cd /usr/lib/x86_64-linux-gnu
for lib in liburdfdom_model liburdfdom_sensor liburdfdom_world liburdfdom_model_state; do
    sudo ln -sf ${lib}.so.3.0 ${lib}.so.4.0
done
sudo ldconfig
```

---

## 二、硬件映射

### 2.1 CAN → Piper 机械臂

| CAN 口 | 机械臂 |
|---|---|
| can0 | **右从臂** (right follower) |
| can1 | **左从臂** (left follower) |
| can2 | **左主臂** (left leader) |
| can3 | **右主臂** (right leader) |

### 2.2 RealSense D435 摄像头

| Serial Number | 位置 |
|---|---|
| 420222071960 | 顶视 (ego) |
| 419522073184 | 左腕 (left wrist) |
| 153122074100 | 右腕 (right wrist) |

### 2.3 Paxini 触觉传感器

| 设备路径 | 位置 | 传感器 |
|---|---|---|
| /dev/ttyACM0 | 左手指尖 | index_middle (68点) + middle_middle (68点) |
| /dev/ttyACM1 | 右手指尖 | index_middle (68点) + middle_middle (68点) |

---

## 三、成功采集命令

### 3.1 前置步骤

```bash
# 每次采集前执行
pkill -9 python3 2>/dev/null; sleep 1

# 启动 CAN 接口（重启后需要）
sudo ip link set can0 type can bitrate 1000000 && sudo ip link set up can0
sudo ip link set can1 type can bitrate 1000000 && sudo ip link set up can1
sudo ip link set can2 type can bitrate 1000000 && sudo ip link set up can2
sudo ip link set can3 type can bitrate 1000000 && sudo ip link set up can3

# 清理旧数据集
rm -rf ~/.cache/huggingface/lerobot/enine/202606*
```

### 3.2 采集命令

```bash
cd ~/gczx_Evo-RL_tactile

PYTHONUNBUFFERED=1 lerobot-record-piper-tactile \
    --task "Pick up the red cube and place it in the box" \
    --robot-type bi_piperx_follower \
    --left-follower-can can1 --right-follower-can can0 \
    --left-leader-can can2 --right-leader-can can3 \
    --top-camera 420222071960 \
    --left-wrist-camera 419522073184 --right-wrist-camera 153122074100 \
    --ego-camera-side left \
    --left-tactile-port /dev/ttyACM0 --right-tactile-port /dev/ttyACM1 \
    --tactile-heatmap-vmax 5.0 --tactile-gamma 0.55 \
    --dataset.repo_id enine/$(date +%Y%m%d_%H%M)_pick_place_tactile \
    --fps 30 --follower-speed-ratio 100 --follower-high-follow \
    --leader-command-speed-ratio 100 --leader-command-high-follow
```

### 3.3 键盘操作

| 按键 | 功能 |
|---|---|
| `ENTER` | 开始新 episode / 录制中结束 + 弹出标签选择 |
| `s` | 标记 **success** 并立即结束 |
| `f` | 标记 **failure** 并立即结束 |
| `d` | **丢弃** 当前 episode |
| `q` | **退出** 采集会话 |
| `Ctrl-C` | 紧急中断 |

---

## 四、触觉传感器工具

### 4.1 监控（CLI）

```bash
lerobot-tactile-monitor --port /dev/ttyACM0 --calibrate \
    --calibration-warmup-frames 30 --calibration-samples 80 \
    --calibration-interval 0.03 --calibration-reducer median \
    --max-frames 20
```

### 4.2 浏览器实时预览

```bash
lerobot-tactile-preview --port /dev/ttyACM0 --http-port 8765 --calibrate \
    --heatmap-vmin 0.0 --heatmap-vmax 5.0 --heatmap-gamma 0.55 \
    --host 0.0.0.0

# 浏览器打开 http://192.168.110.85:8765
```

### 4.3 触觉可视化最佳参数

| 参数 | 值 | 说明 |
|---|---|---|
| `heatmap-vmin` | `0.0` | 零压力基准 |
| `heatmap-vmax` | `5.0` | 适配 Paxini 传感器力度范围 |
| `heatmap-gamma` | `0.55` | <1 让低压力变化更可见 |
| `heatmap-colormap` | `turbo` | 蓝→绿→黄→红渐变 |
| `output-size` | `256` | 每面板 256×256，总输出 512×512 |
| `calibration-reducer` | `median` | 抗噪声 |

---

## 五、训练命令

```bash
lerobot-train \
    --dataset.repo_id=enine/20260607_XXXX_pick_place_tactile \
    --policy.type=xr0 \
    --policy.qwen_variant=Qwen/Qwen3-VL-4B-Instruct \
    --policy.freeze_vlm=true \
    --policy.dit_num_layers=16 \
    --policy.num_inference_steps=5 \
    --batch_size=16 \
    --steps=30000 \
    --output_dir=outputs/train/xr0_tactile \
    --job_name=xr0_tactile_v1
```

---

## 六、调试问题记录

### 6.1 Paxini 串口协议常量错误

**现象**：`No supported tactile sensors detected`

**原因**：最初移植 protocol.py 时使用了错误的协议头/指令码/校验算法

| 常量 | 错误值 | 正确值 |
|---|---|---|
| 请求头 | `\x5A\xA5` | `\x55\xAA` |
| 响应头 | `\x5A\xA5` | `\xAA\x55` |
| 自动推送头 | `\xA5\x5A` | `\xAA\x56` |
| FUNC_READ | `0x01` | `0x03` |
| FUNC_WRITE | `0x02` | `0x10` |
| 校验 | CRC-16 | **LRC** |

**修复**：直接用原版 `Xiaomi-Robotics-0/xr0/mibot/tactile/protocol.py`

### 6.2 串口读取无限循环

**现象**：脚本卡在 `read_response` 不退出

**原因**：每收到数据就重置超时计时器，触觉控制器持续发 auto-push 帧导致永不超时

**修复**：只在找到期望的 header 时才重置计时器

### 6.3 遥操卡顿

**现象**：从臂运动不连续

**原因**：
1. `TactileCamera.async_read()` 阻塞等待新帧（触觉 10Hz → 阻塞 50-100ms）
2. `dataset.add_frame()` 在主线程同步写入

**修复**：
1. `async_read` 改为非阻塞：立刻返回缓存帧（相同帧重复不使用）
2. `add_frame` 移到后台线程

### 6.4 RealSense USB Hub 带宽问题

**现象**：三台 D435 在同一个 USB Hub 上时 `Couldn't resolve requests`

**根因**：LeRobot 的 `RealSenseCamera` 在相机间做 warmup 读取帧，占用了 USB 带宽。纯 `pyrealsense2` API 同时打开三台没问题

**缓解**：回退到 `8c7a162`（最后一次正常采集版本），确保相机分布在不同 USB 控制器上

### 6.5 采集后 crash 污染硬件

**现象**：can2 失败 → crash → ACM0 坏 → 重插 → can2 又失败 → 循环

**解决**：采集前统一 `pkill -9 python3` + 重插 USB + CAN 口重激活

### 6.6 采集脚本缩进 bug

**现象**：录制中按键无响应

**原因**：print/键盘检测/帧率控制代码在 `while True` 循环外面

**修复**：增加一级缩进

### 6.7 scipy 依赖缺失

```bash
pip install "scipy>=1.10.1"
```

已加入 `pyproject.toml` 核心依赖。

### 6.8 urdfdom 库版本不匹配

pinocchio 需要 `liburdfdom_*.so.4.0`，但系统装的是 `3.0`。用符号链接解决。

---

## 七、数据集输出结构

```
~/.cache/huggingface/lerobot/enine/20260607_XXXX_pick_place_tactile/
├── data/chunk-0000/
│   ├── episode_000000.parquet                         # 状态+动作
│   ├── episode_000000_observation.images.left_ego.mp4        # 640×480
│   ├── episode_000000_observation.images.left_wrist.mp4      # 640×480
│   ├── episode_000000_observation.images.left_tactile.mp4    # 512×512 触觉
│   ├── episode_000000_observation.images.right_wrist.mp4     # 640×480
│   └── episode_000000_observation.images.right_tactile.mp4   # 512×512 触觉
├── meta/
│   ├── info.json     # 特征 schema（含 tactile image features）
│   └── stats.json    # 归一化统计
```

> 注意：之前使用 `--ego-camera-side right` 的采集，ego 视频在 `right_ego` 而非 `left_ego`

---

## 八、新增/修改的文件

### 新建（12 个）

| 文件 | 说明 |
|---|---|
| `src/lerobot/cameras/tactile/__init__.py` | 模块入口 |
| `src/lerobot/cameras/tactile/configuration_tactile.py` | TactileCameraConfig |
| `src/lerobot/cameras/tactile/camera_tactile.py` | TactileCamera 实现 |
| `src/lerobot/cameras/tactile/driver.py` | 串口传感器驱动 |
| `src/lerobot/cameras/tactile/protocol.py` | Paxini 串口协议 |
| `src/lerobot/cameras/tactile/runtime.py` | 采集运行时 |
| `src/lerobot/cameras/tactile/types.py` | 数据类型 |
| `src/lerobot/cameras/tactile/visualizer.py` | 热力图渲染 |
| `src/lerobot/scripts/lerobot_tactile_monitor.py` | CLI 触觉监控 |
| `src/lerobot/scripts/lerobot_tactile_preview.py` | 浏览器预览 |
| `src/lerobot/scripts/lerobot_record_piper_tactile.py` | 双臂 Piper + 触觉采集 |
| `xr0_tactile_setup_log.md` | 本文档 |

### 修改（6 个）

| 文件 | 改动 |
|---|---|
| `pyproject.toml` | +scipy 依赖, +3 个 CLI 入口 |
| `src/lerobot/cameras/__init__.py` | 导出 TactileCamera |
| `src/lerobot/cameras/utils.py` | make_cameras_from_configs 支持 tactile |
| `src/lerobot/scripts/lerobot_record.py` | import TactileCameraConfig |
| `src/lerobot/scripts/lerobot_teleoperate.py` | import TactileCameraConfig |

### 无需修改

- **XR0 Policy (`policies/xr0/`)**：零改动。自动发现所有 `image_features`，触觉图像自然参与训练
