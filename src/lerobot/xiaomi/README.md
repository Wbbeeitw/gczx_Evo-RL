# Xiaomi 双臂 PiPER 触觉采集脚本

这个目录里新增了一个独立采集脚本：

- `record_bi_piper_tactile.py`
- `preview_bi_tactile_heatmap.py`

它不会修改 `lerobot-record` 本体，而是在原有双臂 PiPER 遥操作采集流程上，额外加入两路 Xiaomi 触觉热力图视频。

和你原来的采集方式相比，这个脚本会保留原来的三路 RGB 视角：

- `left_top`
- `left_wrist`
- `right_wrist`

同时新增两路触觉热力图视频：

- `left_tactile`
- `right_tactile`

默认情况下，每一路触觉都对应一个独立的 Xiaomi 触觉串口控制器。脚本会把该侧控制器上的两个触觉面板热力图横向拼接成一帧 RGB 图像，再作为视频流写入数据集。

## 依赖说明

建议继续使用你原来的 LeRobot / SACM 录制环境运行。

这个脚本还需要能导入 Xiaomi 项目里的 `mibot.tactile`。如果当前环境里没有安装它，可以通过下面任一参数把 Xiaomi 项目路径临时加到 `PYTHONPATH`：

- `--xiaomi-xr0-root /path/to/Xiaomi-Robotics-0`
- `--xiaomi-xr0-root /path/to/Xiaomi-Robotics-0/xr0`

## 触觉默认配置约定

采集脚本和预览脚本的触觉热力图默认参数，已经统一对齐到你在
`F:\Xiaomi-Robotics-0\xr0\docs\tactile_toolkit.md`
里验证效果最好的这条命令：

```bash
tactile_preview_server.py --port /dev/ttyACM0 --mode auto_push --calibrate --calibration-warmup-frames 30 --calibration-samples 80 --calibration-interval 0.03 --calibration-reducer median --heatmap-vmin 0.0 --heatmap-vmax 5.0 --heatmap-colormap turbo --heatmap-gamma 0.55
```

也就是说，如果你在录制命令里不手动覆盖这些参数，默认就会使用这一套配置：

- `mode=auto_push`
- `calibrate=true`
- `calibration_warmup_frames=30`
- `calibration_samples=80`
- `calibration_interval=0.03`
- `calibration_reducer=median`
- `heatmap_vmin=0.0`
- `heatmap_vmax=5.0`
- `heatmap_colormap=turbo`
- `heatmap_gamma=0.55`

## 运行方式

在 `gczx_Evo-RL` 仓库根目录下运行：

```bash
python src/lerobot/xiaomi/record_bi_piper_tactile.py ...
```

如果你想继续沿用之前的 `conda run` 方式，可以直接参考下面这条命令：

```bash
PYTHONNOUSERSITE=1 /home/enine/miniconda3/bin/conda run --no-capture-output -n SACM env PYTHONUNBUFFERED=1 \
  python src/lerobot/xiaomi/record_bi_piper_tactile.py \
  --xiaomi-xr0-root /home/enine/xiaomi/Xiaomi-Robotics-0 \
  --robot.id piper_follower \
  --robot.left_arm_config.port can0 \
  --robot.left_arm_config.require_calibration true \
  --robot.left_arm_config.enable_on_connect true \
  --robot.left_arm_config.high_follow true \
  --robot.left_arm_config.speed_ratio 20 \
  --robot.left_arm_config.cameras '{top: {"type": "intelrealsense", "serial_number_or_name": "420222071960", "width": 640, "height": 480, "fps": 30, "warmup_s": 2}, wrist: {"type": "intelrealsense", "serial_number_or_name": "419522073184", "width": 640, "height": 480, "fps": 30, "warmup_s": 2}}' \
  --robot.right_arm_config.port can1 \
  --robot.right_arm_config.require_calibration true \
  --robot.right_arm_config.enable_on_connect true \
  --robot.right_arm_config.high_follow true \
  --robot.right_arm_config.speed_ratio 20 \
  --robot.right_arm_config.cameras '{wrist: {"type": "intelrealsense", "serial_number_or_name": "153122074100", "width": 640, "height": 480, "fps": 30, "warmup_s": 2}}' \
  --teleop.id piper_leader \
  --teleop.left_arm_config.port can2 \
  --teleop.left_arm_config.require_calibration true \
  --teleop.right_arm_config.port can3 \
  --teleop.right_arm_config.require_calibration true \
  --left-tactile-port /dev/ttyACM0 \
  --right-tactile-port /dev/ttyACM1 \
  --tactile.mode auto_push \
  --tactile.calibrate true \
  --tactile.calibration_warmup_frames 30 \
  --tactile.calibration_samples 80 \
  --tactile.calibration_interval 0.03 \
  --tactile.calibration_reducer median \
  --tactile.heatmap_vmin 0.0 \
  --tactile.heatmap_vmax 5.0 \
  --tactile.heatmap_colormap turbo \
  --tactile.heatmap_gamma 0.55 \
  --dataset.repo_id sacm/5_20_flod_towel_and_xiaomi_tactile \
  --dataset.root /home/enine/SACM/lerobot_dataset/5_20_flod_towel_and_xiaomi_tactile \
  --dataset.num_episodes 10 \
  --dataset.episode_time_s 240 \
  --dataset.reset_time_s 5 \
  --dataset.fps 30 \
  --dataset.single_task "Both arms cooperate to grasp the towel, align the corners, and fold the towel neatly into a compact rectangle." \
  --dataset.video true \
  --dataset.vcodec h264 \
  --display_data false \
  --play_sounds false \
  --enable_episode_outcome_labeling true \
  --episode_success_key s \
  --episode_failure_key f \
  --require_episode_success_label true
```

## 先看触觉热力图长什么样

如果你想在正式采集前，先单独看一下左右触觉热力图的实时效果，可以运行同级目录下这个预览脚本：

```bash
PYTHONNOUSERSITE=1 /home/enine/miniconda3/bin/conda run --no-capture-output -n SACM env PYTHONUNBUFFERED=1 \
  python src/lerobot/xiaomi/preview_bi_tactile_heatmap.py \
  --xiaomi-xr0-root /home/enine/xiaomi/Xiaomi-Robotics-0 \
  --left-tactile-port /dev/ttyACM0 \
  --right-tactile-port /dev/ttyACM1 \
  --tactile.mode auto_push \
  --tactile.calibrate true \
  --tactile.calibration_warmup_frames 30 \
  --tactile.calibration_samples 80 \
  --tactile.calibration_interval 0.03 \
  --tactile.calibration_reducer median \
  --tactile.heatmap_vmin 0.0 \
  --tactile.heatmap_vmax 5.0 \
  --tactile.heatmap_colormap turbo \
  --tactile.heatmap_gamma 0.55
```

说明：

- 预览窗口上半部分是左手触觉热力图
- 预览窗口下半部分是右手触觉热力图
- 按 `q` 或 `Esc` 退出
- 这个脚本只做实时查看，不写数据集

## 输出内容

最终生成的 LeRobot 数据集会新增下面两路 observation 视频键：

- `observation.images.left_tactile`
- `observation.images.right_tactile`

原来的三路 RGB 视角仍然保留：

- `observation.images.left_top`
- `observation.images.left_wrist`
- `observation.images.right_wrist`

机器人状态和动作仍然沿用标准的双臂 PiPER LeRobot 记录格式，不会因为新增触觉视频而改变数值部分的记录方式。

## 触觉保存格式

这个采集脚本强制要求：

- `--dataset.video true`

也就是说，触觉热力图和相机视频一样，都会作为视频视角写入数据集，并在 episode 保存后编码成 MP4。

不允许把触觉热力图按图片序列或单张图片的形式保存。

## 触觉参数说明

- `--left-tactile-port`
  - 左手触觉控制器串口，比如 `/dev/ttyACM0`
- `--right-tactile-port`
  - 右手触觉控制器串口，比如 `/dev/ttyACM1`
- `--tactile.output_size`
  - 单个触觉面板热力图的边长，默认 `256`
  - 因为每侧会把两个触觉面板横向拼接，所以单侧最终视频帧大小是 `256 x 512`
- `--tactile.calibrate`
  - 是否在启动时做零点标定，默认 `true`
- `--tactile.heatmap_vmin`
- `--tactile.heatmap_vmax`
- `--tactile.heatmap_colormap`
- `--tactile.heatmap_gamma`
  - 这些参数直接对应 Xiaomi 触觉工具链里的热力图渲染参数

## 注意事项

- 这个脚本当前支持的相机配置方式，和你原始命令里的 Intel RealSense 配置风格保持一致。
- 录制时，每一路触觉串口都应该只被这个脚本独占，不要同时再启动 `tactile_monitor.py` 或 `tactile_preview_server.py`。
- 如果某一路触觉控制器连接失败，脚本会在启动阶段直接报错退出，不会静默跳过。
- 如果你把 `--dataset.video` 设成 `false`，录制脚本会直接报错，因为这里明确要求触觉热力图必须按 MP4 视频保存。
- 如果后面你还要把这份 LeRobot 数据再转换到 XR0，比较自然的映射关系是：
  - `left_tactile -> observations.tactile_left`
  - `right_tactile -> observations.tactile_right`
