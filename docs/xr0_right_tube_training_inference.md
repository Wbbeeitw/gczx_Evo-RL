# XR0 右臂透明管插盘训练与实机推理流程

本文记录当前项目中已经跑通的 XR0 + LeRobot + Piper-X + 五路视觉/触觉输入流程。任务为：使用右臂拿起右侧透明管，并插入左侧插盘。

适用环境：

- 本地仓库：`F:\gczx_Evo-RL`
- 服务器仓库：`/home/enine/gczx_Evo-RL_tactile`
- 分支：`feature/xr0-vla-integration`
- Conda 环境：`evo_rl_xr0`
- 服务器：`enine@192.168.110.85`
- 当前数据集示例：`enine/20260610_2116_right_tube_insert_tactile`

## 1. Native Prompt 改动影响

XR0 的 VLM prompt 已经从 LeRobot 自定义风格切换到更接近小米官方 XR0 的原生风格。

现在五路输入进入 VLM 时会被组织成：

```text
<|im_start|>user
The following observations are captured from multiple views.
# Left-Wrist View
<|vision_start|><|image_pad|><|vision_end|>
# Base View
<|vision_start|><|image_pad|><|vision_end|>
# Right-Wrist View
<|vision_start|><|image_pad|><|vision_end|>
# Left-Tactile View
<|vision_start|><|image_pad|><|vision_end|>
# Right-Tactile View
<|vision_start|><|image_pad|><|vision_end|>
Generate robot actions for the task:
Use the right arm to pick up the transparent tube on the right side and insert it into the slot tray on the left side /no_cot<|im_end|>
<|im_start|>assistant
<cot></cot><|im_end|>
```

影响结论：

- 采集数据不受影响。数据集仍然只保存 task、state、action 和五路图像，不保存最终 VLM prompt。
- stats 计算不受 prompt 影响。右臂任务仍然使用 `--controlled-arms right`。
- 训练需要用新代码重新训练或至少继续微调。旧 checkpoint 是旧 prompt 风格训练的，虽然可以加载，但与现在推理 prompt 不完全同分布。
- 推理需要使用 native-prompt 训练出来的新 checkpoint，避免训练/推理 prompt 不一致。
- 新训练命令不再需要传 `--policy.image_key_descriptions`。只保留 `--policy.image_key_order` 来固定五路图像顺序。
- task 文本里不要手动加 `/no_cot`。代码会在送入 VLM 前自动补上。

## 2. 硬件与数据约定

CAN 映射：

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

数据集保持 ALOHA 双臂 14 维格式：

```text
action: 14 = left 6 joints + left gripper + right 6 joints + right gripper
observation.state: 14 = same layout
```

本任务只训练右臂，左臂虽然被采集但不参与 action loss。训练和 stats 必须使用：

```bash
--policy.action_layout=aloha14
--policy.controlled_arms=right
--policy.actions_are_delta=false
```

XR0 内部仍保持官方 32 维 state/action 结构。右臂 ALOHA 7 维会映射到 XR0 32 维中的 `20..26`，loss 和 action mask 只覆盖这 7 个维度；左臂推理时保持当前状态。

## 3. 采集命令

采集命令不需要因为 native prompt 改动而变化。注意 `--task` 只写自然语言任务，不要手动追加 `/no_cot`。

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
  --save-mode serial \
  --fps 30 \
  --follower-speed-ratio 100 \
  --follower-high-follow \
  --leader-command-speed-ratio 100 \
  --leader-command-high-follow
```

`--save-mode serial` 会让触觉采集脚本使用稳定优先的保存方式：

- 图像写盘默认只用 1 个 writer thread。
- episode 保存时五路视频逐路编码，不再按相机路数并行编码。
- 保存会更慢，但可以显著降低五路视频 + 触觉图同时保存时的 CPU/IO 峰值压力。
- 如果以后确认机器性能足够、想换速度，可以显式改成 `--save-mode parallel`。

采集后数据集应包含五路图像 key：

```text
observation.images.left_wrist
observation.images.left_ego
observation.images.right_wrist
observation.images.left_tactile
observation.images.right_tactile
```

建议每批先采 10 条左右，正常 finalize 后再训练。明显失败或动作不自然的 episode 用 `d` 丢弃，不要保存进训练集。

## 4. 数据集检查

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

root = Path("/data/xiaomi/hf_cache_clean/lerobot/enine/20260610_2116_right_tube_insert_tactile")
info = json.loads((root / "meta/info.json").read_text())
table = pq.read_table(root / "data/chunk-000/file-000.parquet")

print("episodes:", info["total_episodes"])
print("frames:", info["total_frames"])
print("parquet rows:", table.num_rows)
print("parquet OK:", table.num_rows == info["total_frames"])
print("features:")
for key in info["features"]:
    print(" ", key)
PY
```

## 5. 计算右臂 XR0 Stats

stats 不受 prompt 改动影响。因为本任务只训练右臂，必须使用 `--controlled-arms right`，避免左臂静止维度污染归一化。

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

## 6. 官方 XR0 预训练权重审计

当前使用完整 HF 权重转换后的 checkpoint：

```text
/data/xiaomi/pretrained_ckpt/xr0_pretrained.pt
```

审计命令：

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src

python -m lerobot.policies.xr0.check_official_checkpoint \
  --checkpoint /data/xiaomi/pretrained_ckpt/xr0_pretrained.pt \
  --qwen-attn-implementation sdpa
```

可接受结果：

```text
loaded tensors: 932
missing keys: 1
unexpected keys: 0
missing key: vlm.lm_head.weight
```

`vlm.lm_head.weight` 是文本生成 head，不参与 XR0 的 DiT/action 输出路径，本实验可以接受。

## 7. Native Prompt 版本训练命令

如果使用当前 native prompt 代码，建议重新训练一个新输出目录，避免和旧 prompt checkpoint 混用。

本命令配置：

- batch size: `16`
- steps: `5000`
- save frequency: `2500`
- 保存 step 2500 和 step 5000
- 冻结 VLM，只训练 DiT/action 相关分支
- 右臂单臂监督：`controlled_arms=right`
- 关闭 `enable_freq`，避免未 mask 的 frequency loss 抬高总 loss
- 只传 `image_key_order`，不再传 `image_key_descriptions`

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
  --batch_size=16 \
  --steps=5000 \
  --log_freq=10 \
  --save_freq=2500 \
  --output_dir=outputs/train/xr0_right_tube_10ep_5000_native_prompt
```

训练完成后优先使用：

```bash
outputs/train/xr0_right_tube_10ep_5000_native_prompt/checkpoints/005000/pretrained_model
```

## 8. 实机推理命令

推理必须使用 native prompt 版本训练出来的新 checkpoint。`--dataset.single_task` 仍然只写自然语言任务，不要手动加 `/no_cot`。

### 8.1 5 秒低速检查

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src
export HF_HOME=/data/xiaomi/hf_cache_clean
export HF_ENDPOINT=https://hf-mirror.com

CKPT=outputs/train/xr0_right_tube_10ep_5000_native_prompt/checkpoints/005000/pretrained_model
ROLLOUT_NAME=eval_$(date +%Y%m%d_%H%M)_xr0_right_tube_native_prompt_5s

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

### 8.2 300 秒观察命令

5 秒检查通过后，再跑 300 秒观察长时行为：

```bash
cd ~/gczx_Evo-RL_tactile
export PYTHONPATH=/home/enine/gczx_Evo-RL_tactile/src
export HF_HOME=/data/xiaomi/hf_cache_clean
export HF_ENDPOINT=https://hf-mirror.com

CKPT=outputs/train/xr0_right_tube_10ep_5000_native_prompt/checkpoints/005000/pretrained_model
ROLLOUT_NAME=eval_$(date +%Y%m%d_%H%M)_xr0_right_tube_native_prompt_300s

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

## 9. 推理文件清理

`lerobot-record` 即使只用于看推理效果，也会创建 rollout 目录。不需要保存时可以删除：

```bash
rm -rf -- /data/xiaomi/rollouts
test ! -e /data/xiaomi/rollouts && echo "deleted: /data/xiaomi/rollouts"
```

## 10. 常见问题

### 10.1 旧 checkpoint 还能不能用？

能加载，也能跑，但不建议作为最终实机测试结果。旧 checkpoint 是旧 prompt 风格训练的，现在代码推理时会使用 native prompt，存在训练/推理 prompt mismatch。推荐用相同数据重新训练 `xr0_right_tube_10ep_5000_native_prompt`。

### 10.2 还需要 `image_key_descriptions` 吗？

不需要。当前代码已经内置五路 key 到 native view heading 的映射：

```text
observation.images.left_wrist    -> # Left-Wrist View
observation.images.left_ego      -> # Base View
observation.images.right_wrist   -> # Right-Wrist View
observation.images.left_tactile  -> # Left-Tactile View
observation.images.right_tactile -> # Right-Tactile View
```

训练和推理最重要的是保留 `image_key_order`，确保五路图像顺序稳定。

### 10.3 RealSense busy

```bash
fuser -v /dev/video* 2>/dev/null
pkill -f lerobot-record
pkill -f "lerobot.scripts.lerobot_record"
```

如果仍然 busy，等待几秒或重新插拔对应 RealSense USB。

### 10.4 不想重新标定 Piper

推理命令里保留：

```bash
--robot.left_arm_config.require_calibration=false
--robot.right_arm_config.require_calibration=false
```

### 10.5 右臂单臂 loss 偏高

右臂单臂任务建议固定：

```bash
--policy.enable_freq=false
```

否则 frequency loss 可能把未 mask 的 32 维全部算进去，导致 total loss 被抬高。
