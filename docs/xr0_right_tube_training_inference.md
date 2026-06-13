# XR0 右臂透明管插盘任务训练与实机推理流程

本文记录本项目中已经跑通的一条 XR0 + LeRobot + Piper-X + 触觉输入闭环流程，任务为：使用右臂拿起右侧透明管，并插入左侧插盘。

适用仓库与环境：

- 本地仓库：`F:\gczx_Evo-RL`
- 服务器仓库：`/home/enine/gczx_Evo-RL_tactile`
- 分支：`feature/xr0-vla-integration`
- Conda 环境：`evo_rl_xr0`
- 服务器：`enine@192.168.110.85`
- 训练数据集：`enine/20260610_2116_right_tube_insert_tactile`
- 成功训练输出：`outputs/train/xr0_right_tube_10ep_5000_nofreq`

## 1. 硬件与数据约定

### 1.1 CAN 与设备映射

本次成功采集和推理使用的 CAN 映射如下：

| 设备 | CAN |
|---|---|
| 左从臂 left follower | `can0` |
| 右从臂 right follower | `can1` |
| 右主臂 right leader | `can2` |
| 左主臂 left leader | `can3` |

RealSense 摄像头：

| 位置 | Serial |
|---|---|
| 左腕相机 | `419522073184` |
| 基座/ego 相机 | `420222071960` |
| 右腕相机 | `153122074100` |

触觉传感器：

| 位置 | 端口 |
|---|---|
| 左触觉 | `/dev/ttyACM0` |
| 右触觉 | `/dev/ttyACM1` |

### 1.2 数据维度与控制臂

数据集保持 ALOHA 双臂 14 维格式：

```text
action: 14 = left 6 joints + left gripper + right 6 joints + right gripper
observation.state: 14 = same layout
```

但本任务只使用右臂执行，左臂 action 在数据中基本恒定。因此训练时必须使用：

```bash
--policy.action_layout=aloha14
--policy.controlled_arms=right
--policy.actions_are_delta=false
```

对应 stats 计算也必须使用：

```bash
--action-layout aloha14
--controlled-arms right
```

XR0 内部仍保持原生 32 维 state/action 结构。右臂 ALOHA 7 维被映射到 XR0 32 维中的 `20..26`，训练 loss 和 action mask 只覆盖这 7 个维度。左臂不会参与 loss，推理时左臂保持当前状态。

## 2. 成功采集数据命令

建议每次只采 10 条左右就退出并检查。此前录到第 16 条附近容易在保存 parquet/video 时触发 segfault，导致整批 parquet footer 损坏。

本次成功数据集为：

```text
/data/xiaomi/hf_cache_clean/lerobot/enine/20260610_2116_right_tube_insert_tactile
```

采集命令：

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src
export HF_HOME=/data/xiaomi/hf_cache_clean
export HF_ENDPOINT=https://hf-mirror.com

PYTHONUNBUFFERED=1 lerobot-record-piper-tactile \
  --task "Use the right arm to pick up the transparent tube on the right side and insert it into the slot tray on the left side" \
  --robot-type bi_piperx_follower \
  --left-follower-can can0 \
  --right-follower-can can1 \
  --left-leader-can can3 \
  --right-leader-can can2 \
  --top-camera 420222071960 \
  --left-wrist-camera 419522073184 \
  --right-wrist-camera 153122074100 \
  --ego-camera-side left \
  --left-tactile-port /dev/ttyACM0 \
  --right-tactile-port /dev/ttyACM1 \
  --tactile-heatmap-vmax 5.0 \
  --tactile-gamma 0.55 \
  --dataset.repo_id enine/$(date +%Y%m%d_%H%M)_right_tube_insert_tactile \
  --fps 30 \
  --follower-speed-ratio 100 \
  --follower-high-follow \
  --leader-command-speed-ratio 100 \
  --leader-command-high-follow
```

采集操作建议：

- 使用 `s` 保存成功 episode。
- 明显失败或动作不自然的 episode 用 `d` 丢弃，不要保存进训练集。
- 录到 10 条左右后，在提示 `Press ENTER to start episode 011 ...` 时退出，让数据集正常 finalize。
- 如果脚本支持 `q`，优先用 `q` 退出；`Ctrl-C` 在本次也能 finalize，但不如 `q` 温和。

## 3. 数据集完整性检查

采集完成后先确认 parquet 可读，避免拿坏数据训练：

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src
export HF_HOME=/data/xiaomi/hf_cache_clean

DATASET_NAME=20260610_2116_right_tube_insert_tactile
ROOT=/data/xiaomi/hf_cache_clean/lerobot/enine/$DATASET_NAME

python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path('/data/xiaomi/hf_cache_clean/lerobot/enine/20260610_2116_right_tube_insert_tactile')
info = json.loads((root / 'meta/info.json').read_text())
table = pq.read_table(root / 'data/chunk-000/file-000.parquet')

print('episodes:', info['total_episodes'])
print('frames:', info['total_frames'])
print('parquet rows:', table.num_rows)
print('parquet OK:', table.num_rows == info['total_frames'])
print('features:', list(info['features'].keys()))
PY
```

成功数据集应包含五路视觉输入：

```text
observation.images.left_wrist
observation.images.left_ego
observation.images.right_wrist
observation.images.left_tactile
observation.images.right_tactile
```

## 4. 计算 XR0 右臂 stats

因为本任务只训练右臂，stats 也只统计右臂 7 维。不要用 `both`，否则左臂静止维度会污染归一化。

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src
export HF_HOME=/data/xiaomi/hf_cache_clean

DATASET_NAME=20260610_2116_right_tube_insert_tactile
REPO_ID=enine/$DATASET_NAME
ROOT=/data/xiaomi/hf_cache_clean/lerobot/enine/$DATASET_NAME
STATS=outputs/xr0_stats_right_${DATASET_NAME}_minstd005.pt

python -m lerobot.policies.xr0.compute_xr0_stats \
  --repo-id $REPO_ID \
  --root $ROOT \
  --output $STATS \
  --horizon 30 \
  --action-layout aloha14 \
  --controlled-arms right \
  --min-std 0.05
```

说明：

- `actions_are_delta=false` 是默认值，表示数据集 action 是绝对关节目标。
- stats 和训练 processor 内部会自动做 `action32 - state32`，因此实际给 XR0 学的是相对 delta。
- stats 文件中只有 XR0 32 维的 `20..26` 有有效 count/mean/std，其余维度 count 为 0、std 为 1。

## 5. 官方 XR0 预训练权重

使用完整 Hugging Face 权重转换后的 checkpoint：

```text
/data/xiaomi/pretrained_ckpt/xr0_pretrained.pt
```

审计结果应接近：

```text
loaded tensors: 932
missing keys: 1
unexpected keys: 0
missing key: vlm.lm_head.weight
```

`vlm.lm_head.weight` 是文本生成 head，不参与 XR0 action head/DiT 动作预测路径；本实验可接受。

如果需要重新审计：

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src

python -m lerobot.policies.xr0.check_official_checkpoint \
  --checkpoint /data/xiaomi/pretrained_ckpt/xr0_pretrained.pt \
  --qwen-attn-implementation sdpa
```

## 6. 成功训练命令

本次成功训练设置：

- batch size: `16`
- steps: `5000`
- save frequency: `2500`，保存 step 2500 和 step 5000
- 冻结 VLM，只训练 DiT/action 相关分支
- 关闭 `enable_freq`，避免未 mask 的 frequency loss 抬高总 loss
- 右臂单臂监督：`controlled_arms=right`

完整命令：

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src
export HF_HOME=/data/xiaomi/hf_cache_clean
export HF_ENDPOINT=https://hf-mirror.com

DATASET_NAME=20260610_2116_right_tube_insert_tactile
REPO_ID=enine/$DATASET_NAME
ROOT=/data/xiaomi/hf_cache_clean/lerobot/enine/$DATASET_NAME
STATS=outputs/xr0_stats_right_${DATASET_NAME}_minstd005.pt

accelerate launch --num_processes=1 --mixed_precision=bf16 -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=$REPO_ID \
  --dataset.root=$ROOT \
  --policy.type=xr0 \
  --policy.push_to_hub=false \
  --policy.xr0_pretrained_path=/data/xiaomi/pretrained_ckpt/xr0_pretrained.pt \
  --policy.xr0_stats_path=$STATS \
  --policy.action_layout=aloha14 \
  --policy.controlled_arms=right \
  --policy.actions_are_delta=false \
  --policy.enable_freq=false \
  --policy.freeze_vlm=true \
  --policy.qwen_attn_implementation=sdpa \
  --policy.image_key_order='["observation.images.left_wrist","observation.images.left_ego","observation.images.right_wrist","observation.images.left_tactile","observation.images.right_tactile"]' \
  --policy.image_key_descriptions='{"observation.images.left_wrist":"left wrist RGB camera","observation.images.left_ego":"base ego RGB camera","observation.images.right_wrist":"right wrist RGB camera","observation.images.left_tactile":"left fingertip tactile heatmap","observation.images.right_tactile":"right fingertip tactile heatmap"}' \
  --batch_size=16 \
  --steps=5000 \
  --log_freq=10 \
  --save_freq=2500 \
  --output_dir=outputs/train/xr0_right_tube_10ep_5000_nofreq
```

本次成功训练末尾 loss 大约在 `0.05 ~ 0.06`：

```text
step:5K ... loss:0.055~0.066
Checkpoint policy after step 5000
End of training
```

如果 loss 卡在 `4.x` 左右，优先检查是否忘记加：

```bash
--policy.enable_freq=false
```

原因是当前 native XR0 的 `loss_mse` 会按 `action_mask` 只算右臂，但 `loss_freq` 未按 mask 过滤，会把 32 维全部算进去，导致右臂单臂任务的总 loss 被抬高。

## 7. 实机推理命令

### 7.1 推理前准备

使用第 5000 步 checkpoint：

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src
export HF_HOME=/data/xiaomi/hf_cache_clean
export HF_ENDPOINT=https://hf-mirror.com

CKPT=outputs/train/xr0_right_tube_10ep_5000_nofreq/checkpoints/005000/pretrained_model
test -d "$CKPT" && echo "checkpoint OK: $CKPT"
```

如果不想重新标定 Piper，从臂需要加：

```bash
--robot.left_arm_config.require_calibration=false
--robot.right_arm_config.require_calibration=false
```

如果出现 RealSense busy：

```bash
fuser -v /dev/video* 2>/dev/null
ps -ef | grep -E "lerobot|realsense|python" | grep -v grep
pkill -f lerobot-record
pkill -f "lerobot.scripts.lerobot_record"
```

如仍 busy，等待 5-10 秒或拔插对应 RealSense USB。

### 7.2 5 秒低速检查命令

第一次上机只跑 5 秒、速度 30，确认不暴起、方向正常：

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src
export HF_HOME=/data/xiaomi/hf_cache_clean
export HF_ENDPOINT=https://hf-mirror.com

CKPT=outputs/train/xr0_right_tube_10ep_5000_nofreq/checkpoints/005000/pretrained_model
ROLLOUT_NAME=eval_$(date +%Y%m%d_%H%M)_xr0_right_tube_rollout_5s

lerobot-record \
  --robot.type=bi_piperx_follower \
  --robot.left_arm_config.port=can0 \
  --robot.right_arm_config.port=can1 \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.speed_ratio=30 \
  --robot.right_arm_config.speed_ratio=30 \
  --robot.left_arm_config.high_follow=true \
  --robot.right_arm_config.high_follow=true \
  --robot.left_arm_config.cameras='{"wrist":{"type":"intelrealsense","serial_number_or_name":"419522073184","fps":30,"width":640,"height":480},"ego":{"type":"intelrealsense","serial_number_or_name":"420222071960","fps":30,"width":640,"height":480},"tactile":{"type":"tactile","port":"/dev/ttyACM0","output_size":256,"heatmap_vmax":5.0,"heatmap_gamma":0.55}}' \
  --robot.right_arm_config.cameras='{"wrist":{"type":"intelrealsense","serial_number_or_name":"153122074100","fps":30,"width":640,"height":480},"tactile":{"type":"tactile","port":"/dev/ttyACM1","output_size":256,"heatmap_vmax":5.0,"heatmap_gamma":0.55}}' \
  --policy.path=$CKPT \
  --dataset.repo_id=enine/$ROLLOUT_NAME \
  --dataset.root=/data/xiaomi/rollouts/enine/$ROLLOUT_NAME \
  --dataset.single_task="Use the right arm to pick up the transparent tube on the right side and insert it into the slot tray on the left side" \
  --dataset.fps=30 \
  --dataset.episode_time_s=5 \
  --dataset.reset_time_s=0 \
  --dataset.num_episodes=1 \
  --dataset.video=true \
  --dataset.vcodec=h264 \
  --dataset.push_to_hub=false \
  --dataset.num_image_writer_threads_per_camera=1 \
  --display_data=false
```

注意：`dataset.repo_id` 必须以 `eval_` 开头，否则 LeRobot 在提供 policy 时会拒绝记录 eval 数据集。

### 7.3 300 秒观察命令

5 秒检查通过后，可以跑 300 秒观察长时行为。速度仍保持 30：

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src
export HF_HOME=/data/xiaomi/hf_cache_clean
export HF_ENDPOINT=https://hf-mirror.com

CKPT=outputs/train/xr0_right_tube_10ep_5000_nofreq/checkpoints/005000/pretrained_model
ROLLOUT_NAME=eval_$(date +%Y%m%d_%H%M)_xr0_right_tube_rollout_300s

lerobot-record \
  --robot.type=bi_piperx_follower \
  --robot.left_arm_config.port=can0 \
  --robot.right_arm_config.port=can1 \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.speed_ratio=30 \
  --robot.right_arm_config.speed_ratio=30 \
  --robot.left_arm_config.high_follow=true \
  --robot.right_arm_config.high_follow=true \
  --robot.left_arm_config.cameras='{"wrist":{"type":"intelrealsense","serial_number_or_name":"419522073184","fps":30,"width":640,"height":480},"ego":{"type":"intelrealsense","serial_number_or_name":"420222071960","fps":30,"width":640,"height":480},"tactile":{"type":"tactile","port":"/dev/ttyACM0","output_size":256,"heatmap_vmax":5.0,"heatmap_gamma":0.55}}' \
  --robot.right_arm_config.cameras='{"wrist":{"type":"intelrealsense","serial_number_or_name":"153122074100","fps":30,"width":640,"height":480},"tactile":{"type":"tactile","port":"/dev/ttyACM1","output_size":256,"heatmap_vmax":5.0,"heatmap_gamma":0.55}}' \
  --policy.path=$CKPT \
  --dataset.repo_id=enine/$ROLLOUT_NAME \
  --dataset.root=/data/xiaomi/rollouts/enine/$ROLLOUT_NAME \
  --dataset.single_task="Use the right arm to pick up the transparent tube on the right side and insert it into the slot tray on the left side" \
  --dataset.fps=30 \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=0 \
  --dataset.num_episodes=1 \
  --dataset.video=true \
  --dataset.vcodec=h264 \
  --dataset.push_to_hub=false \
  --dataset.num_image_writer_threads_per_camera=1 \
  --display_data=false
```

安全建议：

- 首次长时测试手放急停。
- 左臂不参与控制，要摆到安全、不挡路的位置。
- 右臂初始位姿、透明管和插盘位置尽量接近采集数据。
- 如果动作方向正确但幅度偏大，先保持 `speed_ratio=30`，不要急着加速。

## 8. 推理产生文件的清理

`lerobot-record` 即使只是为了看推理效果，也会创建 rollout 目录。测试后如不需要保存，直接删除：

```bash
rm -rf -- /data/xiaomi/rollouts

test ! -e /data/xiaomi/rollouts && echo "deleted: /data/xiaomi/rollouts"
```

不要用 `--dataset.video=false` 来尝试“不保存图像”。XR0 是视觉策略，LeRobot 在 `dataset.video=false` 时不会把 image features 放进 dataset schema，policy 推理会报：

```text
KeyError: 'observation.images.left_wrist'
```

如果以后确实需要“只推理、不落盘”，建议新增一个专门的 infer-only 脚本，不走 `LeRobotDataset` 保存逻辑。

## 9. 常见问题

### 9.1 parquet 损坏

现象：

```text
Parquet magic bytes not found in footer
```

原因通常是保存 episode 时 segfault，parquet 没有正常 close。不能只删除最后一条坏 episode 来恢复，因为 tabular 主文件 footer 已损坏。建议删除该数据集并重采。

删除坏数据集示例：

```bash
ROOT=/data/xiaomi/hf_cache_clean/lerobot/enine/<bad_dataset_name>
lsof +D "$ROOT" 2>/dev/null
pkill -f lerobot-record-piper-tactile
pkill -f bi_piper_leader
rm -rf -- "$ROOT"
```

### 9.2 采集保存阶段 segfault

本任务中 5 路视频输入、30 fps、触觉图 512x512，保存压力较大。建议：

- 每批只采 10 条左右。
- 退出后检查 parquet。
- 采下一批用新的 `repo_id`。
- 如果保存 rollout，优先使用 `--dataset.vcodec=h264`，避免默认 `libsvtav1` 压力过大。

### 9.3 `image_key_descriptions` 参数格式

训练时 `image_key_descriptions` 是 `dict[str, str]`，不是 list。正确写法：

```bash
--policy.image_key_descriptions='{"observation.images.left_wrist":"left wrist RGB camera","observation.images.left_ego":"base ego RGB camera","observation.images.right_wrist":"right wrist RGB camera","observation.images.left_tactile":"left fingertip tactile heatmap","observation.images.right_tactile":"right fingertip tactile heatmap"}'
```

### 9.4 关闭 Hub 上传

训练时关闭上传使用：

```bash
--policy.push_to_hub=false
```

不是：

```bash
--push_to_hub=false
```

### 9.5 RealSense busy

现象：

```text
xioctl(VIDIOC_S_FMT) failed, errno=16
Device or resource busy
```

处理：

```bash
fuser -v /dev/video* 2>/dev/null
pkill -f lerobot-record
pkill -f "lerobot.scripts.lerobot_record"
```

如果仍 busy，等待几秒或拔插相机 USB。

### 9.6 标定提示

如果出现：

```text
No piper-follower calibration file found for 'None'. Running lerobot-calibrate flow.
```

推理命令中加入：

```bash
--robot.left_arm_config.require_calibration=false
--robot.right_arm_config.require_calibration=false
```

### 9.7 右臂单臂任务的 loss

本任务中 `enable_freq=true` 会让 total loss 被未 mask 的 frequency loss 抬高。右臂单臂训练建议固定：

```bash
--policy.enable_freq=false
```

本次成功训练在关闭 frequency loss 后，5000 step 附近 loss 约 `0.05 ~ 0.06`。