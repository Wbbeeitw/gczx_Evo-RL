# Xiaomi 双臂 PiPER 触觉采集脚本

这个目录里新增了两类脚本：

- `record_bi_piper_tactile.py`
- `preview_bi_tactile_heatmap.py`
- `serve_tactile_dataset_videos.py`

它们不会修改 `lerobot-record` 本体，而是在原有双臂 PiPER 采集流程基础上，额外加入两路 Xiaomi 触觉热力图：

- `left_tactile`
- `right_tactile`

同时保留原来的三路 RGB 相机：

- `left_top`
- `left_wrist`
- `right_wrist`

## 依赖说明

建议继续使用你原来的 LeRobot / SACM 环境运行。

这些脚本还需要能导入 Xiaomi XR0 项目里的 `mibot.tactile`。如果当前环境没有直接安装它，可以通过下面任一种路径把 XR0 源码临时加入 `PYTHONPATH`：

- `--xiaomi-xr0-root /path/to/Xiaomi-Robotics-0`
- `--xiaomi-xr0-root /path/to/Xiaomi-Robotics-0/xr0`

如果你的服务器实际目录就是 `~/xiaomi/xr0`，那就直接传：

```bash
--xiaomi-xr0-root /home/enine/xiaomi/xr0
```

## 触觉默认配置

采集脚本和预览脚本的默认触觉热力图参数，已经统一对齐到 `xr0/docs/tactile_toolkit.md` 中当前效果最好的那组配置：

```bash
tactile_preview_server.py --port /dev/ttyACM0 --mode auto_push --calibrate --calibration-warmup-frames 30 --calibration-samples 80 --calibration-interval 0.03 --calibration-reducer median --heatmap-vmin 0.0 --heatmap-vmax 5.0 --heatmap-colormap turbo --heatmap-gamma 0.55
```

也就是说，如果你不手动覆盖参数，默认就是：

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

## 录制脚本

在 `gczx_Evo-RL` 仓库根目录下运行：

```bash
python src/lerobot/xiaomi/record_bi_piper_tactile.py ...
```

如果你沿用之前的 `conda run` 方式，可以参考下面这个命令：

```bash
PYTHONNOUSERSITE=1 /home/enine/miniconda3/bin/conda run --no-capture-output -n SACM env PYTHONUNBUFFERED=1 \
  python src/lerobot/xiaomi/record_bi_piper_tactile.py \
  --xiaomi-xr0-root /home/enine/xiaomi/xr0 \
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

## 双触觉网页预览

如果你想在正式采集前，先单独查看左右手触觉热力图效果，可以运行：

```bash
PYTHONNOUSERSITE=1 /home/enine/miniconda3/bin/conda run --no-capture-output -n SACM env PYTHONUNBUFFERED=1 \
  python src/lerobot/xiaomi/preview_bi_tactile_heatmap.py \
  --xiaomi-xr0-root /home/enine/xiaomi/xr0 \
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
  --host 127.0.0.1 \
  --http-port 8765 \
  --refresh-ms 250
```

说明：

- 这个脚本现在是网页预览版，不再使用 `cv2.imshow` 弹窗。
- 启动后会在服务器上开启一个 HTTP 页面，默认地址是 `http://127.0.0.1:8765`
- 页面上半部分是左手触觉热力图，下半部分是右手触觉热力图
- 这个脚本只做实时查看，不会写数据集

如果脚本运行在远程服务器上，推荐在你本机做 SSH 端口转发：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<server>
```

然后在你本机浏览器打开：

```text
http://127.0.0.1:8765
```

## 查看已采集触觉视频

如果你已经完成了一次采集，想回看数据集里保存下来的左右手触觉 MP4，可以运行：

```bash
PYTHONNOUSERSITE=1 /home/enine/miniconda3/bin/conda run --no-capture-output -n SACM env PYTHONUNBUFFERED=1 \
  python src/lerobot/xiaomi/serve_tactile_dataset_videos.py \
  --dataset-root /home/enine/SACM/lerobot_dataset/5_20_flod_towel_and_xiaomi_tactile \
  --host 127.0.0.1 \
  --http-port 8767
```

说明：

- 这个脚本用于查看已经采集完成并落盘后的触觉视频，不读取实时串口
- 页面会并排显示左手和右手触觉视频
- 页面顶部可以切换 episode，也可以一键同时播放、暂停、重新从该 episode 起点播放
- 如果多个 episode 共用同一个 MP4 文件，页面会根据数据集元信息自动跳转到该 episode 的起止时间，而不是从整段视频头部开始看

如果脚本运行在远程服务器上，推荐在你本机做 SSH 端口转发：

```bash
ssh -L 8767:127.0.0.1:8767 <user>@<server>
```

然后在你本机浏览器打开：

```text
http://127.0.0.1:8767
```

## 输出内容

最终生成的 LeRobot 数据集会新增两路 observation 视频键：

- `observation.images.left_tactile`
- `observation.images.right_tactile`

原来的三路 RGB 视角仍然保留：

- `observation.images.left_top`
- `observation.images.left_wrist`
- `observation.images.right_wrist`

机器人状态和动作仍然沿用标准双臂 PiPER LeRobot 记录格式，不会因为新增触觉视频而改变数值部分的记录方式。

## 触觉保存格式

这个采集脚本强制要求：

- `--dataset.video true`

也就是说，触觉热力图和相机视频一样，都会作为视频视角写入数据集，并在 episode 保存后编码成 MP4。

不允许把触觉热力图按图片序列或单张图片的形式保存。

## 长度一致性

当前这套采集链路里，RGB 视角和触觉热力图都是在同一个 `record_loop` 采样步里一起写入 dataset 的。

这意味着：

- 每一个采样步都会同时写入一帧 RGB 和一帧触觉
- 最终编码出来的 RGB 视频和触觉视频，帧数会一致
- 只要 `--dataset.fps` 固定，视频时长也会一致

额外做了两层保护：

- 所有 RealSense 相机配置里的 `fps` 必须等于 `--dataset.fps`，否则脚本直接报错
- 如果触觉流在某些采样步没有拿到新帧，脚本会明确打印告警

注意：

- 触觉流如果短时间跟不上采样循环，当前实现会复用上一帧热力图，这样可以保证视频长度不漂
- 这种情况下长度仍然一致，但“有效触觉更新率”会低于目标 `fps`

## 主要参数说明

- `--left-tactile-port`
  左手触觉控制器串口，比如 `/dev/ttyACM0`
- `--right-tactile-port`
  右手触觉控制器串口，比如 `/dev/ttyACM1`
- `--tactile.output_size`
  单个触觉面板热力图边长，默认 `256`
- `--tactile.calibrate`
  是否在启动时做零点标定，默认 `true`
- `--tactile.heatmap_vmin`
- `--tactile.heatmap_vmax`
- `--tactile.heatmap_colormap`
- `--tactile.heatmap_gamma`
  这些参数直接对应 Xiaomi 触觉工具链里的热力图渲染参数
- `--host`
  网页预览监听地址，默认 `127.0.0.1`
- `--http-port`
  网页预览端口，默认 `8765`
- `--refresh-ms`
  浏览器刷新间隔，默认 `250`
- `--dataset-root`
  已采集 LeRobot 数据集根目录，供 `serve_tactile_dataset_videos.py` 回看触觉 MP4
- `--left-key`
  左手触觉视频键，默认 `observation.images.left_tactile`
- `--right-key`
  右手触觉视频键，默认 `observation.images.right_tactile`
- `--log-level`
  已采集触觉视频查看服务的日志级别，默认 `INFO`

## 注意事项

- 录制时，每一路触觉串口都应该只被一个脚本独占，不要同时再启动其他触觉监控或预览脚本
- 如果某一路触觉控制器连接失败，脚本会在启动阶段直接报错退出，不会静默跳过
- 如果把 `--dataset.video` 设成 `false`，录制脚本会直接报错，因为这里明确要求触觉热力图必须按 MP4 视频保存
- 如果后面还要把这份 LeRobot 数据再转换到 XR0，一个自然的映射关系是：
  - `left_tactile -> observations.tactile_left`
  - `right_tactile -> observations.tactile_right`
