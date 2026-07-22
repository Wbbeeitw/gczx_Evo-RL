# XR0 异步 RTC 推理

本流程使用工控机作为 robot client、GPU 服务器作为 policy server。client 将三路 RGB 和右夹爪触觉热力图压缩为 JPEG 后发送至 server；server 使用 XR0 生成带旧动作前缀的 action suffix，client 根据实测延迟替换并平滑 action queue。

## 约束

- client 和 server 必须运行同一个 Git commit。
- XR0 checkpoint 的 `chunk_size` 为 30，因此 `actions_per_chunk` 不得超过 30。
- 四路图像 key 必须完全匹配 checkpoint：

```text
observation.images.left_ego
observation.images.left_wrist
observation.images.right_wrist
observation.images.right_tactile
```

- 第一次连接真实硬件必须使用 `dry_run_actions=true`。
- 右触觉端口使用工控机当前实际存在的 `/dev/ttyACM*`，启动前重新确认。

## GPU Policy Server

```bash
cd /home/enine/gczx_Evo-RL_tactile
conda activate evo_rl_xr0

export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src
export HF_HOME=/data/xiaomi/hf_cache_clean
export HF_ENDPOINT=https://hf-mirror.com

PYTHONUNBUFFERED=1 python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=18081 \
  --fps=30 \
  --inference_latency=0 \
  --obs_queue_timeout=1
```

## 工控机 Dry-Run Client

```bash
cd ~/gczx_Evo-RL_xr0/gczx_Evo-RL
conda activate evo_rl_xr0

TASK='Use the right gripper to pick up the yellow plastic bottle on the right and place it on the blue towel on the left'
CKPT=/data/xiaomi/xr0_training/xr0_yellow_bottle_right_tactile_8ep_bs16_2000/train/checkpoints/002000/pretrained_model

PYTHONUNBUFFERED=1 python -m lerobot.async_inference.robot_client \
  --robot.type=bi_piperx_follower \
  --robot.id=piper_follower \
  --robot.left_arm_config.port=can0 \
  --robot.left_arm_config.require_calibration=false \
  --robot.left_arm_config.enable_on_connect=false \
  --robot.left_arm_config.disable_on_disconnect=false \
  --robot.left_arm_config.high_follow=true \
  --robot.left_arm_config.speed_ratio=2 \
  --robot.left_arm_config.cameras='{ego: {type: intelrealsense, serial_number_or_name: "420222071960", width: 640, height: 480, fps: 30, warmup_s: 2, read_thread_yield_s: 0.005, read_thread_poll_interval_s: 0.01}, wrist: {type: intelrealsense, serial_number_or_name: "419522073184", width: 640, height: 480, fps: 30, warmup_s: 2, read_thread_yield_s: 0.005, read_thread_poll_interval_s: 0.01}}' \
  --robot.right_arm_config.port=can1 \
  --robot.right_arm_config.require_calibration=false \
  --robot.right_arm_config.enable_on_connect=false \
  --robot.right_arm_config.disable_on_disconnect=false \
  --robot.right_arm_config.high_follow=true \
  --robot.right_arm_config.speed_ratio=2 \
  --robot.right_arm_config.cameras='{wrist: {type: intelrealsense, serial_number_or_name: "153122074100", width: 640, height: 480, fps: 30, warmup_s: 2, read_thread_yield_s: 0.005, read_thread_poll_interval_s: 0.01}, tactile: {type: tactile, port: "/dev/ttyACM0", output_size: 256, heatmap_vmin: 0.0, heatmap_vmax: 5.0, heatmap_gamma: 0.55, rgb_vmax_fz: 5.0, rgb_vmax_shear: 2.5, calibrate_on_connect: true}}' \
  --server_address=192.168.110.85:18081 \
  --policy_type=xr0 \
  --pretrained_name_or_path="$CKPT" \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=30 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --fps=30 \
  --rtc.enabled=true \
  --rtc.execution_horizon=6 \
  --rtc.prefix_attention_schedule=EXP \
  --rtc.max_guidance_weight=1.0 \
  --rtc.queue_blend_steps=3 \
  --rtc.inference_delay_multiplier=1.0 \
  --observation_image_codec=jpeg \
  --observation_jpeg_quality=90 \
  --dry_run_actions=true \
  --duration=5 \
  --task="$TASK"
```

checkpoint 路径由 GPU server 读取，因此 client 命令中的路径必须是 GPU 服务器上的路径。

dry-run 期间重点检查：

```text
JPEG encode/decode 时间与压缩率
GPU inference_time
round_trip P50/P95
RTC queue 长度
starvation/reanchor 次数
action shape 是否为 14
左臂 action 是否保持当前状态
```

所有检查通过后，才将 `enable_on_connect` 改为 `true`、将 `dry_run_actions` 改为 `false`，并继续保持 `speed_ratio=2`、`duration=5` 做首次真实动作测试。
