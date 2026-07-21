# XR0 单右夹爪触觉训练约定

本配置适用于三路 RGB 相机和一路右夹爪触觉热力图。数据集必须包含：

```text
observation.images.left_ego
observation.images.left_wrist
observation.images.right_wrist
observation.images.right_tactile
```

不应创建或填充 `observation.images.left_tactile`。训练与推理使用相同的四路输入。

## VLM 输入顺序与提示词

图像顺序固定为：

```json
[
  "observation.images.left_ego",
  "observation.images.left_wrist",
  "observation.images.right_wrist",
  "observation.images.right_tactile"
]
```

XR0 适配器生成的提示词为：

```text
The following observations are captured from multiple views.
# Ego View
<image>
# Left-Wrist View
<image>
# Right-Wrist View
<image>
# Right-Gripper Tactile View
<image>
Generate robot actions for the task:
<task> /no_cot
```

assistant 前缀保持为：

```text
<cot></cot>
```

前三路名称、任务段落和 `/no_cot` 遵循原生 XR0 格式；右夹爪触觉使用相同的多视图段落格式扩展。任务文本只保存自然语言内容，调用方不要手动添加 `/no_cot`。

## 训练配置

训练命令必须显式固定四路输入顺序：

```bash
--policy.image_key_order='["observation.images.left_ego","observation.images.left_wrist","observation.images.right_wrist","observation.images.right_tactile"]'
```

保持以下 XR0 参数：

```bash
--policy.action_layout=aloha14
--policy.controlled_arms=right
--policy.actions_are_delta=false
--policy.enable_freq=false
--policy.freeze_vlm=true
```

每个新数据集都要重新计算 action stats：

```bash
python -m lerobot.policies.xr0.compute_xr0_stats \
  --repo-id "$REPO_ID" \
  --root "$ROOT" \
  --output "$STATS" \
  --horizon 30 \
  --action-layout aloha14 \
  --controlled-arms right \
  --min-std 0.05
```

新训练从官方 XR0 权重初始化，不直接续训保存了五路输入配置的旧 LeRobot checkpoint：

```bash
--policy.xr0_pretrained_path=/data/xiaomi/pretrained_ckpt/xr0_pretrained.pt
--policy.xr0_stats_path="$STATS"
```

推理时仅创建三路 RGB camera 和右夹爪 tactile camera，并使用新训练得到的四路 checkpoint。不要在推理配置中创建左触觉 camera。
