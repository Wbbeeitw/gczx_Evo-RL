# XR0 双臂水瓶向水杯倒水完整流程

本文记录已经跑通的双臂倒水实验，从工控机采集双触觉数据开始，到上传服务器、合并数据、计算 XR0 action stats、训练双触觉与 RGB3 对照模型，以及使用 5000 步权重进行无 RTC 真机推理。

## 1. 固定实验定义

任务提示词在采集、训练和推理阶段必须完全一致：

```text
Use the left arm to pick up the cup on the left and the right arm to pick up the water bottle on the right. Bring both arms toward the center, then tilt the water bottle held by the right arm to pour water into the cup held by the left arm.
```

硬件映射：

| 设备 | 配置 |
| --- | --- |
| 左 Follower | `can0` |
| 右 Follower | `can1` |
| 左 Leader | `can2` |
| 右 Leader | `can3` |
| 顶部/左 ego 相机 | `420222071960` |
| 左腕相机 | `419522073184` |
| 右腕相机 | `153122074100` |
| 左触觉 | `/dev/ttyACM1` |
| 右触觉 | `/dev/ttyACM0` |

已完成实验的数据和模型：

```text
第一批：20260806_160149_bimanual_right_bottle_pour_into_left_cup_dual_tactile
        17 episodes / 11785 frames

第二批：20260806_175053_bimanual_right_bottle_pour_into_left_cup_dual_tactile
        13 episodes / 6897 frames

合并后：bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile
        30 episodes / 18682 frames
        3路RGB + 左右2路触觉
        双臂14维绝对关节action

RGB3： bimanual_right_bottle_pour_into_left_cup_30ep_rgb3
        30 episodes / 18682 frames
        3路RGB，无触觉
```

工控机仓库：

```text
~/gczx_Evo-RL_xr0/gczx_Evo-RL
```

GPU 服务器仓库：

```text
/home/enine/gczx_Evo-RL_tactile
```

服务器数据根目录：

```text
/data/xiaomi/xr0_datasets/whz
```

## 2. 开始前检查

服务器和工控机必须使用同一个 Git commit：

```bash
git rev-parse HEAD
```

本流程使用的双启动夹爪支持至少需要包含提交：

```text
f5601d0 feat(xr0): configure both startup grippers
```

工控机检查 CAN、相机和触觉设备：

```bash
ip -details link show can0
ip -details link show can1
ip -details link show can2
ip -details link show can3

ls -l /dev/ttyACM*
fuser -v /dev/ttyACM0 /dev/ttyACM1 || true

rs-enumerate-devices | grep -E 'Serial Number|Name'
```

不能同时运行采集、触觉预览和推理客户端：

```bash
pgrep -af \
  '[l]erobot_record_piper_tactile|[l]erobot.async_inference.robot_client|[l]erobot-tactile-preview|[l]erobot_tactile_preview' \
  || true
```

## 3. 工控机采集双触觉数据

下面的函数每次启动都会用当前时间创建一个新数据集。采集第二批时继续使用完全相同的函数，不要改变 CAN、相机、触觉、Follower 速度或 high-follow 配置。

```bash
collect_bimanual_bottle_pour_dual_tactile() {
  cd ~/gczx_Evo-RL_xr0/gczx_Evo-RL || return 1
  conda activate evo_rl_xr0 || return 1

  export PYTHONPATH="$PWD/src"
  export PYTHONUNBUFFERED=1

  TASK='Use the left arm to pick up the cup on the left and the right arm to pick up the water bottle on the right. Bring both arms toward the center, then tilt the water bottle held by the right arm to pour water into the cup held by the left arm.'

  TASK_SLUG='bimanual_right_bottle_pour_into_left_cup'
  DATASET_NAME="$(date +%Y%m%d_%H%M%S)_${TASK_SLUG}_dual_tactile"
  REPO_ID="whz/${DATASET_NAME}"
  DATASET_ROOT="$HOME/xr0_datasets/${DATASET_NAME}"

  LEFT_TACTILE='/dev/ttyACM1'
  RIGHT_TACTILE='/dev/ttyACM0'

  ROBOT_CALIB_DIR="$HOME/.cache/huggingface/lerobot/calibration/robots/bi_piper_follower"
  TELEOP_CALIB_DIR="$HOME/.cache/huggingface/lerobot/calibration/teleoperators/bi_piper_leader"

  LOG_DIR="$HOME/xr0_collection_logs"
  LOG="$LOG_DIR/${DATASET_NAME}.log"

  RUNNING="$(
    pgrep -af \
      '[l]erobot_record_piper_tactile|[l]erobot.async_inference.robot_client|[l]erobot-tactile-preview|[l]erobot_tactile_preview' \
      || true
  )"

  if [[ -n "$RUNNING" ]]; then
    echo "已有采集、推理或触觉预览程序运行："
    printf '%s\n' "$RUNNING"
    return 1
  fi

  for DEV in "$LEFT_TACTILE" "$RIGHT_TACTILE"; do
    if [[ ! -c "$DEV" ]]; then
      echo "找不到触觉设备：$DEV"
      ls -l /dev/ttyACM* 2>/dev/null || true
      return 1
    fi
  done

  if fuser "$LEFT_TACTILE" "$RIGHT_TACTILE" >/dev/null 2>&1; then
    echo "触觉串口被占用："
    fuser -v "$LEFT_TACTILE" "$RIGHT_TACTILE" || true
    return 1
  fi

  REQUIRED_CALIBRATIONS=(
    "$ROBOT_CALIB_DIR/piper_follower_left.json"
    "$ROBOT_CALIB_DIR/piper_follower_right.json"
    "$TELEOP_CALIB_DIR/piper_leader_left.json"
    "$TELEOP_CALIB_DIR/piper_leader_right.json"
  )

  for FILE in "${REQUIRED_CALIBRATIONS[@]}"; do
    if [[ ! -f "$FILE" ]]; then
      echo "缺少标定文件：$FILE"
      return 1
    fi
  done

  if [[ -e "$DATASET_ROOT" ]]; then
    echo "数据集目录已经存在，拒绝覆盖：$DATASET_ROOT"
    return 1
  fi

  mkdir -p "$HOME/xr0_datasets" "$LOG_DIR" || return 1

  echo "============================================================"
  echo "双臂倒水双触觉数据采集"
  echo "DATASET=$DATASET_ROOT"
  echo "左触觉=$LEFT_TACTILE"
  echo "右触觉=$RIGHT_TACTILE"
  echo "Follower速度比例=20"
  echo "FPS=30"
  echo "============================================================"

  set -o pipefail

  python -m lerobot.scripts.lerobot_record_piper_tactile \
    --task "$TASK" \
    --robot-type bi_piper_follower \
    --robot-id piper_follower \
    --teleop-id piper_leader \
    --left-follower-can can0 \
    --right-follower-can can1 \
    --left-leader-can can2 \
    --right-leader-can can3 \
    --top-camera 420222071960 \
    --left-wrist-camera 419522073184 \
    --right-wrist-camera 153122074100 \
    --ego-camera-side left \
    --width 640 \
    --height 480 \
    --camera-warmup-s 2 \
    --left-tactile-port "$LEFT_TACTILE" \
    --right-tactile-port "$RIGHT_TACTILE" \
    --tactile-output-size 256 \
    --tactile-heatmap-vmin 0.5 \
    --tactile-heatmap-vmax 5.0 \
    --tactile-colormap turbo \
    --tactile-gamma 0.55 \
    --tactile-rgb-vmax-fz 20.0 \
    --tactile-rgb-vmax-shear 20.0 \
    --tactile-dropout-policy recover \
    --tactile-hold-last-max-ms 500 \
    --tactile-reconnect-interval-s 0.25 \
    --tactile-calibrate-on-connect \
    --tactile-calibrate-after-episode \
    --follower-startup-sleep-s 0.5 \
    --leader-startup-sleep-s 0.1 \
    --follower-speed-ratio 20 \
    --follower-high-follow \
    --leader-command-speed-ratio 20 \
    --leader-command-high-follow \
    --leader-process-isolation \
    --require-calibration \
    --fps 30 \
    --episode-seconds 0 \
    --default-trajectory-type success \
    --dataset.repo_id "$REPO_ID" \
    --dataset.root "$DATASET_ROOT" \
    --dataset.vcodec h264 \
    --save-mode serial \
    --display-data \
    --display-compressed-images \
    2>&1 | tee "$LOG"

  STATUS=${PIPESTATUS[0]}

  echo "采集程序退出状态：$STATUS"
  echo "数据集：$DATASET_ROOT"
  echo "日志：$LOG"
  return "$STATUS"
}

collect_bimanual_bottle_pour_dual_tactile
```

操作要求：

1. 连接和每条 episode 结束后的触觉标定期间，两侧指尖必须完全卸载并保持静止。
2. 启动后先确认两侧 Leader 重力补偿正常，再开始 episode。
3. 一条 episode 只包含一次完整任务。成功、失败和丢弃必须按实际结果标记。
4. `speed_ratio=20` 是低层跟随设置，不是 action 跳变限幅器。

## 4. 工控机检查每批数据

将参数换成本批数据目录：

```bash
check_bimanual_dataset() {
  DATASET_ROOT="$1"

  [[ -f "$DATASET_ROOT/meta/info.json" ]] || {
    echo "缺少meta/info.json：$DATASET_ROOT"
    return 1
  }

  python - "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())

print("dataset:", root)
print("episodes:", info["total_episodes"])
print("frames:", info["total_frames"])
print("fps:", info["fps"])
print("robot_type:", info["robot_type"])
print("action shape:", info["features"]["action"]["shape"])

assert info["fps"] == 30
assert info["robot_type"] == "bi_piper_follower"
assert info["features"]["action"]["shape"] == [14]

expected_images = {
    "observation.images.left_ego",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.left_tactile",
    "observation.images.right_tactile",
}
actual_images = {
    key for key, value in info["features"].items()
    if value.get("dtype") == "video"
}
assert actual_images == expected_images, (actual_images, expected_images)
print("DATASET_METADATA_PASS")
PY
}
```

检查已经完成的两批数据：

```bash
check_bimanual_dataset \
  "$HOME/xr0_datasets/20260806_160149_bimanual_right_bottle_pour_into_left_cup_dual_tactile"

check_bimanual_dataset \
  "$HOME/xr0_datasets/20260806_175053_bimanual_right_bottle_pour_into_left_cup_dual_tactile"
```

## 5. 从工控机直接上传到服务器正确目录

上传目标必须是：

```text
/data/xiaomi/xr0_datasets/whz
```

不要再上传到 `/home/enine/xr0_workspace/datasets/whz`。

```bash
upload_bimanual_dataset() {
  DATASET_NAME="$1"

  LOCAL_ROOT="$HOME/xr0_datasets/$DATASET_NAME"
  SERVER='enine@192.168.110.85'
  REMOTE_BASE='/data/xiaomi/xr0_datasets/whz'
  REMOTE_ROOT="$REMOTE_BASE/$DATASET_NAME"

  [[ -f "$LOCAL_ROOT/meta/info.json" ]] || {
    echo "找不到本地数据集：$LOCAL_ROOT"
    return 1
  }

  ssh "$SERVER" "mkdir -p '$REMOTE_BASE' && df -h /data/xiaomi" || return 1

  if ssh "$SERVER" "test -e '$REMOTE_ROOT'"; then
    echo "远端目录已经存在，拒绝覆盖：$REMOTE_ROOT"
    return 1
  fi

  rsync -aH --info=progress2 --partial \
    "$LOCAL_ROOT/" \
    "$SERVER:$REMOTE_ROOT/" || return 1

  LOCAL_SUMS="$(mktemp)" || return 1
  REMOTE_SUMS="$(mktemp)" || {
    rm -f "$LOCAL_SUMS"
    return 1
  }

  (
    cd "$LOCAL_ROOT" || exit 1
    LC_ALL=C find . -type f -print0 | sort -z | xargs -0 sha256sum
  ) >"$LOCAL_SUMS" || return 1

  ssh "$SERVER" \
    "cd '$REMOTE_ROOT' && LC_ALL=C find . -type f -print0 | sort -z | xargs -0 sha256sum" \
    >"$REMOTE_SUMS" || return 1

  if ! diff -u "$LOCAL_SUMS" "$REMOTE_SUMS"; then
    echo "SHA-256校验失败：$DATASET_NAME"
    rm -f "$LOCAL_SUMS" "$REMOTE_SUMS"
    return 1
  fi

  rm -f "$LOCAL_SUMS" "$REMOTE_SUMS"
  echo "SHA256_PASS：$DATASET_NAME"
}

upload_bimanual_dataset \
  20260806_160149_bimanual_right_bottle_pour_into_left_cup_dual_tactile

upload_bimanual_dataset \
  20260806_175053_bimanual_right_bottle_pour_into_left_cup_dual_tactile
```

如果旧数据已经传到 `/home/enine/xr0_workspace/datasets/whz`，只在服务器确认目标不存在后移动：

```bash
mkdir -p /data/xiaomi/xr0_datasets/whz

mv -- \
  /home/enine/xr0_workspace/datasets/whz/20260806_160149_bimanual_right_bottle_pour_into_left_cup_dual_tactile \
  /data/xiaomi/xr0_datasets/whz/

mv -- \
  /home/enine/xr0_workspace/datasets/whz/20260806_175053_bimanual_right_bottle_pour_into_left_cup_dual_tactile \
  /data/xiaomi/xr0_datasets/whz/
```

## 6. 服务器合并两批数据

在服务器执行：

```bash
merge_bimanual_pour_dataset() {
  cd /home/enine/gczx_Evo-RL_tactile || return 1
  conda activate evo_rl_xr0 || return 1

  export PYTHONPATH="$PWD/src"
  export PYTHONUNBUFFERED=1
  export HF_LEROBOT_HOME='/data/xiaomi/xr0_datasets'

  BASE='/data/xiaomi/xr0_datasets/whz'
  DATASET_1='20260806_160149_bimanual_right_bottle_pour_into_left_cup_dual_tactile'
  DATASET_2='20260806_175053_bimanual_right_bottle_pour_into_left_cup_dual_tactile'
  OUTPUT_NAME='bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile'
  OUTPUT_ROOT="$BASE/$OUTPUT_NAME"

  [[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "输出目录已经存在，拒绝覆盖：$OUTPUT_ROOT"
    return 1
  }

  python - "$BASE" "$DATASET_1" "$DATASET_2" "$OUTPUT_NAME" <<'PY'
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

base = Path(sys.argv[1])
names = sys.argv[2:4]
output_name = sys.argv[4]

task = (
    "Use the left arm to pick up the cup on the left and the right arm to pick up "
    "the water bottle on the right. Bring both arms toward the center, then tilt "
    "the water bottle held by the right arm to pour water into the cup held by the left arm."
)

datasets = []
for name in names:
    dataset = LeRobotDataset(f"whz/{name}", root=base / name)
    assert list(dataset.meta.tasks.index) == [task]
    datasets.append(dataset)
    print(name, dataset.meta.total_episodes, dataset.meta.total_frames)

assert sum(ds.meta.total_episodes for ds in datasets) == 30
assert sum(ds.meta.total_frames for ds in datasets) == 18682

merged = merge_datasets(
    datasets,
    output_repo_id=f"whz/{output_name}",
    output_dir=base / output_name,
)

assert merged.meta.total_episodes == 30
assert merged.meta.total_frames == 18682
assert list(merged.meta.tasks.index) == [task]

expected_keys = {
    "observation.images.left_ego",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.left_tactile",
    "observation.images.right_tactile",
}
assert set(merged.meta.video_keys) == expected_keys

print("BIMANUAL_POUR_30EP_MERGE_PASS")
print("output:", merged.root)
print("episodes:", merged.meta.total_episodes)
print("frames:", merged.meta.total_frames)
PY
}

merge_bimanual_pour_dataset
```

## 7. 检查合并结果

检查元数据、Parquet 行数和五路视频帧数：

```bash
verify_bimanual_pour_merged_dataset() {
  ROOT='/data/xiaomi/xr0_datasets/whz/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile'
  EXPECTED_EPISODES=30
  EXPECTED_FRAMES=18682

  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())

episode_rows = sum(
    pq.read_table(path).num_rows
    for path in sorted((root / "meta/episodes").rglob("*.parquet"))
)
data_rows = sum(
    pq.read_table(path).num_rows
    for path in sorted((root / "data").rglob("*.parquet"))
)

assert info["total_episodes"] == 30
assert info["total_frames"] == 18682
assert episode_rows == 30
assert data_rows == 18682
print("MERGED_PARQUET_PASS")
PY

  for KEY in \
    observation.images.left_ego \
    observation.images.left_wrist \
    observation.images.right_wrist \
    observation.images.left_tactile \
    observation.images.right_tactile
  do
    TOTAL=0
    while IFS= read -r VIDEO; do
      FRAMES="$(
        ffprobe -v error -count_frames \
          -select_streams v:0 \
          -show_entries stream=nb_read_frames \
          -of default=nokey=1:noprint_wrappers=1 \
          "$VIDEO"
      )" || return 1
      ffmpeg -v error -i "$VIDEO" -f null - || return 1
      TOTAL=$((TOTAL + FRAMES))
      echo "$KEY：$(basename "$VIDEO")，${FRAMES}帧，解码PASS"
    done < <(find "$ROOT/videos/$KEY" -type f -name '*.mp4' | sort)

    [[ "$TOTAL" -eq "$EXPECTED_FRAMES" ]] || {
      echo "$KEY总帧数错误：$TOTAL"
      return 1
    }
  done

  echo "BIMANUAL_POUR_MERGED_DATASET_PASS"
}

verify_bimanual_pour_merged_dataset
```

## 8. 计算双臂 XR0 stats

数据集保存的是 Follower 绝对关节位置，因此这里不要添加 `--actions-are-delta`。最终 metadata 应显示 `actions_are_delta: False`。

```bash
compute_bimanual_pour_stats() {
  cd /home/enine/gczx_Evo-RL_tactile || return 1
  conda activate evo_rl_xr0 || return 1

  export PYTHONPATH="$PWD/src"
  export PYTHONUNBUFFERED=1
  export HF_LEROBOT_HOME='/data/xiaomi/xr0_datasets'

  REPO_ID='whz/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile'
  DATASET_ROOT='/data/xiaomi/xr0_datasets/whz/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile'
  RUN_ROOT='/data/xiaomi/xr0_datasets/whz/xr0_runs/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile'
  STATS="$RUN_ROOT/common/xr0_stats_both_h30_minstd005.pt"
  LOG="$RUN_ROOT/logs/compute_stats.log"

  [[ ! -e "$STATS" ]] || {
    echo "stats已经存在，拒绝覆盖：$STATS"
    return 1
  }

  mkdir -p "$RUN_ROOT/common" "$RUN_ROOT/logs" || return 1
  set -o pipefail

  python -m lerobot.policies.xr0.compute_xr0_stats \
    --repo-id "$REPO_ID" \
    --root "$DATASET_ROOT" \
    --output "$STATS" \
    --horizon 30 \
    --action-layout aloha14 \
    --controlled-arms both \
    --min-std 0.05 \
    2>&1 | tee "$LOG"

  STATUS=${PIPESTATUS[0]}
  [[ "$STATUS" -eq 0 ]] || return "$STATUS"

  python - "$STATS" <<'PY'
import sys
import torch

stats = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
action = stats["action"]
meta = stats["meta"]

assert tuple(action["mean"].shape) == (30, 32)
assert tuple(action["std"].shape) == (30, 32)
assert tuple(action["count"].shape) == (30, 32)
assert meta["controlled_arms"] == "both"
assert meta["action_layout"] == "aloha14"
assert meta["actions_are_delta"] is False
assert meta["used_samples"] == 18682

print("BIMANUAL_POUR_STATS_PASS")
print(meta)
PY

  sha256sum "$STATS"
}

compute_bimanual_pour_stats
```

已验证结果：

```text
mean shape: (30, 32)
std shape: (30, 32)
count shape: (30, 32)
controlled_arms: both
action_layout: aloha14
actions_are_delta: False
used_samples: 18682
SHA-256: 7baa615ecfe422db95a11352ab4dd08a8fe08faa8f3bdc0affece1c0bf4d4db2
```

## 9. 派生 RGB3 无触觉数据集

双触觉组和 RGB3 组使用完全相同的 action、state、episode 和任务文本。RGB3 组只删除左右触觉视频特征。

```bash
derive_bimanual_pour_rgb3() {
  cd /home/enine/gczx_Evo-RL_tactile || return 1
  conda activate evo_rl_xr0 || return 1

  export PYTHONPATH="$PWD/src"
  export PYTHONUNBUFFERED=1
  export HF_LEROBOT_HOME='/data/xiaomi/xr0_datasets'

  SOURCE_REPO='whz/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile'
  OUTPUT_REPO='whz/bimanual_right_bottle_pour_into_left_cup_30ep_rgb3'
  OUTPUT_ROOT='/data/xiaomi/xr0_datasets/whz/bimanual_right_bottle_pour_into_left_cup_30ep_rgb3'

  [[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "RGB3输出目录已经存在，拒绝覆盖：$OUTPUT_ROOT"
    return 1
  }

  python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id="$SOURCE_REPO" \
    --new_repo_id="$OUTPUT_REPO" \
    --operation.type=remove_feature \
    --operation.feature_names='["observation.images.left_tactile","observation.images.right_tactile"]' \
    --push_to_hub=false
}

derive_bimanual_pour_rgb3
```

派生工具可能把 Arrow 的 `list<float>` 重写为 `fixed_size_list<float>[14]`，导致严格 schema 比较报错。只要逐元素验证 `action` 和 `observation.state` 完全一致，该存储类型变化不会改变数值，也不影响训练和推理。

最低验证要求：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

base = Path("/data/xiaomi/xr0_datasets/whz")
source = LeRobotDataset(
    "whz/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile",
    root=base / "bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile",
)
rgb3 = LeRobotDataset(
    "whz/bimanual_right_bottle_pour_into_left_cup_30ep_rgb3",
    root=base / "bimanual_right_bottle_pour_into_left_cup_30ep_rgb3",
)

assert source.meta.total_episodes == rgb3.meta.total_episodes == 30
assert source.meta.total_frames == rgb3.meta.total_frames == 18682
assert source.meta.tasks.equals(rgb3.meta.tasks)
assert set(rgb3.meta.video_keys) == {
    "observation.images.left_ego",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
}

for key in ["action", "observation.state"]:
    left = np.asarray(source.hf_dataset[key], dtype=np.float32)
    right = np.asarray(rgb3.hf_dataset[key], dtype=np.float32)
    assert np.array_equal(left, right), key
    print("NUMERIC_FIELD_EXACT_PASS:", key)

print("BIMANUAL_POUR_RGB3_DATASET_PASS")
PY
```

RGB3 与双触觉训练共用第 8 节的 stats，因为 stats 只依赖 action、state、horizon 和受控手臂。

## 10. 训练双触觉和 RGB3 模型

下面的函数复现本次两组 5000 步训练。两组除了数据集、图像输入顺序和输出目录外，其余超参数相同。

```bash
train_bimanual_pour_5000() {
  VARIANT="$1"

  cd /home/enine/gczx_Evo-RL_tactile || return 1
  conda activate evo_rl_xr0 || return 1

  export PYTHONPATH="$PWD/src"
  export PYTHONUNBUFFERED=1
  export HF_HOME='/data/xiaomi/hf_cache_clean'
  export HF_LEROBOT_HOME='/data/xiaomi/xr0_datasets'
  export HF_ENDPOINT='https://hf-mirror.com'

  RUN_ROOT='/data/xiaomi/xr0_datasets/whz/xr0_runs/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile'
  PRETRAINED='/data/xiaomi/pretrained_ckpt/xr0_pretrained.pt'
  STATS="$RUN_ROOT/common/xr0_stats_both_h30_minstd005.pt"

  case "$VARIANT" in
    tactile)
      REPO_ID='whz/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile'
      DATASET_ROOT='/data/xiaomi/xr0_datasets/whz/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile'
      TRAIN_DIR="$RUN_ROOT/train_5view_dual_tactile_rtc_bs16_5000"
      LOG="$RUN_ROOT/logs/train_5view_dual_tactile_rtc_bs16_5000.log"
      IMAGE_KEYS='["observation.images.left_ego","observation.images.left_wrist","observation.images.right_wrist","observation.images.left_tactile","observation.images.right_tactile"]'
      ;;
    rgb3)
      REPO_ID='whz/bimanual_right_bottle_pour_into_left_cup_30ep_rgb3'
      DATASET_ROOT='/data/xiaomi/xr0_datasets/whz/bimanual_right_bottle_pour_into_left_cup_30ep_rgb3'
      TRAIN_DIR="$RUN_ROOT/train_3view_rgb_rtc_bs16_5000"
      LOG="$RUN_ROOT/logs/train_3view_rgb_rtc_bs16_5000.log"
      IMAGE_KEYS='["observation.images.left_ego","observation.images.left_wrist","observation.images.right_wrist"]'
      ;;
    *)
      echo "用法：train_bimanual_pour_5000 tactile|rgb3"
      return 2
      ;;
  esac

  LOG_DIR="$RUN_ROOT/logs"

  for FILE in "$PRETRAINED" "$STATS" "$DATASET_ROOT/meta/info.json"; do
    [[ -e "$FILE" ]] || {
      echo "缺少训练输入：$FILE"
      return 1
    }
  done

  [[ ! -e "$TRAIN_DIR" ]] || {
    echo "训练目录已经存在，拒绝覆盖：$TRAIN_DIR"
    return 1
  }

  mkdir -p "$LOG_DIR" || return 1
  df -h /data/xiaomi
  set -o pipefail

  accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=xr0 \
    --policy.push_to_hub=false \
    --policy.xr0_pretrained_path="$PRETRAINED" \
    --policy.xr0_stats_path="$STATS" \
    --policy.action_layout=aloha14 \
    --policy.controlled_arms=both \
    --policy.actions_are_delta=false \
    --policy.enable_freq=false \
    --policy.async_train=true \
    --policy.prefix_mask_prob=0.5 \
    --policy.freeze_vlm=true \
    --policy.qwen_attn_implementation=sdpa \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=5000 \
    --policy.image_key_order="$IMAGE_KEYS" \
    --seed=1000 \
    --num_workers=4 \
    --batch_size=16 \
    --steps=5000 \
    --log_freq=10 \
    --save_checkpoint=true \
    --save_freq=2500 \
    --output_dir="$TRAIN_DIR" \
    --wandb.enable=false \
    2>&1 | tee "$LOG"

  STATUS=${PIPESTATUS[0]}
  [[ "$STATUS" -eq 0 ]] || return "$STATUS"

  for STEP in 002500 005000; do
    [[ -d "$TRAIN_DIR/checkpoints/$STEP/pretrained_model" ]] || {
      echo "缺少checkpoint：$STEP"
      return 1
    }
  done

  echo "BIMANUAL_POUR_${VARIANT}_TRAIN_PASS"
}
```

依次训练，不要在同一张 GPU 上同时运行：

```bash
train_bimanual_pour_5000 tactile
train_bimanual_pour_5000 rgb3
```

本次最终 checkpoint：

```text
双触觉：
/data/xiaomi/xr0_datasets/whz/xr0_runs/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile/train_5view_dual_tactile_rtc_bs16_5000/checkpoints/005000/pretrained_model

RGB3：
/data/xiaomi/xr0_datasets/whz/xr0_runs/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile/train_3view_rgb_rtc_bs16_5000/checkpoints/005000/pretrained_model
```

虽然目录名含有 `rtc` 且训练使用 `async_train=true`，推理时仍可显式设置 `--rtc.enabled=false`。这正是本流程的无 RTC 对照方式。

## 11. 启动 GPU policy server

在服务器新终端执行并保持运行：

```bash
start_bimanual_policy_server() {
  cd /home/enine/gczx_Evo-RL_tactile || return 1
  conda activate evo_rl_xr0 || return 1

  export PYTHONPATH="$PWD/src"
  export PYTHONUNBUFFERED=1
  export HF_HOME='/data/xiaomi/hf_cache_clean'
  export HF_ENDPOINT='https://hf-mirror.com'

  if pgrep -af '[l]erobot.async_inference.policy_server'; then
    echo "已有policy server运行，请先确认端口和代码版本。"
    return 1
  fi

  python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=18081 \
    --fps=30 \
    --inference_latency=0 \
    --obs_queue_timeout=1
}

start_bimanual_policy_server
```

检查端口：

```bash
pgrep -af '[l]erobot.async_inference.policy_server'
ss -ltnp | grep ':18081'
```

checkpoint 路径由客户端发送给服务器，并由服务器读取，所以客户端参数中的 checkpoint 必须使用服务器路径。

## 12. 工控机无 RTC 真机推理，speed_ratio=10

下面一个函数同时支持 `tactile` 和 `rgb3`。两种模式都开启 Rerun 实时显示；触觉模式显示三路 RGB 和两路触觉，RGB3 模式只显示三路 RGB。

```bash
run_bimanual_pour_no_rtc_speed10() {
  VARIANT="$1"
  DURATION="${2:-5}"

  cd ~/gczx_Evo-RL_xr0/gczx_Evo-RL || return 1
  conda activate evo_rl_xr0 || return 1

  export PYTHONPATH="$PWD/src"
  export PYTHONUNBUFFERED=1

  TASK='Use the left arm to pick up the cup on the left and the right arm to pick up the water bottle on the right. Bring both arms toward the center, then tilt the water bottle held by the right arm to pour water into the cup held by the left arm.'

  RUN_ROOT='/data/xiaomi/xr0_datasets/whz/xr0_runs/bimanual_right_bottle_pour_into_left_cup_30ep_dual_tactile'
  CALIB_DIR="$HOME/.cache/huggingface/lerobot/calibration/robots/bi_piper_follower"

  LEFT_RGB_CAMERAS='{ego: {type: intelrealsense, serial_number_or_name: "420222071960", width: 640, height: 480, fps: 30, warmup_s: 2}, wrist: {type: intelrealsense, serial_number_or_name: "419522073184", width: 640, height: 480, fps: 30, warmup_s: 2}}'
  RIGHT_RGB_CAMERAS='{wrist: {type: intelrealsense, serial_number_or_name: "153122074100", width: 640, height: 480, fps: 30, warmup_s: 2}}'

  case "$VARIANT" in
    tactile)
      CKPT="$RUN_ROOT/train_5view_dual_tactile_rtc_bs16_5000/checkpoints/005000/pretrained_model"
      LEFT_TACTILE='/dev/ttyACM1'
      RIGHT_TACTILE='/dev/ttyACM0'
      LEFT_CAMERAS="{ego: {type: intelrealsense, serial_number_or_name: \"420222071960\", width: 640, height: 480, fps: 30, warmup_s: 2}, wrist: {type: intelrealsense, serial_number_or_name: \"419522073184\", width: 640, height: 480, fps: 30, warmup_s: 2}, tactile: {type: tactile, port: \"$LEFT_TACTILE\", output_size: 256, heatmap_vmin: 0.5, heatmap_vmax: 5.0, heatmap_gamma: 0.55, rgb_vmax_fz: 20.0, rgb_vmax_shear: 20.0, calibrate_on_connect: true}}"
      RIGHT_CAMERAS="{wrist: {type: intelrealsense, serial_number_or_name: \"153122074100\", width: 640, height: 480, fps: 30, warmup_s: 2}, tactile: {type: tactile, port: \"$RIGHT_TACTILE\", output_size: 256, heatmap_vmin: 0.5, heatmap_vmax: 5.0, heatmap_gamma: 0.55, rgb_vmax_fz: 20.0, rgb_vmax_shear: 20.0, calibrate_on_connect: true}}"

      for DEV in "$LEFT_TACTILE" "$RIGHT_TACTILE"; do
        [[ -c "$DEV" ]] || {
          echo "找不到触觉设备：$DEV"
          return 1
        }
      done

      if fuser "$LEFT_TACTILE" "$RIGHT_TACTILE" >/dev/null 2>&1; then
        echo "触觉串口被占用："
        fuser -v "$LEFT_TACTILE" "$RIGHT_TACTILE" || true
        return 1
      fi
      ;;
    rgb3)
      CKPT="$RUN_ROOT/train_3view_rgb_rtc_bs16_5000/checkpoints/005000/pretrained_model"
      LEFT_CAMERAS="$LEFT_RGB_CAMERAS"
      RIGHT_CAMERAS="$RIGHT_RGB_CAMERAS"
      ;;
    *)
      echo "用法：run_bimanual_pour_no_rtc_speed10 tactile|rgb3 [duration]"
      return 2
      ;;
  esac

  for FILE in \
    "$CALIB_DIR/piper_follower_left.json" \
    "$CALIB_DIR/piper_follower_right.json"
  do
    [[ -f "$FILE" ]] || {
      echo "缺少标定文件：$FILE"
      return 1
    }
  done

  RUNNING="$(
    pgrep -af \
      '[l]erobot.async_inference.robot_client|[l]erobot_record_piper_tactile|[l]erobot-tactile-preview|[l]erobot_tactile_preview' \
      || true
  )"

  if [[ -n "$RUNNING" ]]; then
    echo "已有推理、采集或触觉预览程序运行："
    printf '%s\n' "$RUNNING"
    return 1
  fi

  LOG_DIR="$HOME/xr0_inference_logs"
  LOG="$LOG_DIR/bimanual_pour_${VARIANT}_no_rtc_5000_speed10_${DURATION}s_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$LOG_DIR" || return 1

  echo "============================================================"
  echo "Variant=$VARIANT"
  echo "Checkpoint=$CKPT"
  echo "RTC=false"
  echo "left speed_ratio=10"
  echo "right speed_ratio=10"
  echo "display=true"
  echo "duration=${DURATION}s"
  echo "确认双臂已经复位且运动区域安全；异常时立即按Ctrl-C。"
  echo "============================================================"

  set -o pipefail

  python -m lerobot.async_inference.robot_client \
    --robot.type=bi_piper_follower \
    --robot.id=piper_follower \
    --robot.calibration_dir="$CALIB_DIR" \
    --robot.left_arm_config.port=can0 \
    --robot.left_arm_config.can_auto_init=true \
    --robot.left_arm_config.require_calibration=true \
    --robot.left_arm_config.enable_on_connect=true \
    --robot.left_arm_config.disable_on_disconnect=false \
    --robot.left_arm_config.high_follow=true \
    --robot.left_arm_config.speed_ratio=10 \
    --robot.left_arm_config.startup_sleep_s=0.5 \
    "--robot.left_arm_config.cameras=$LEFT_CAMERAS" \
    --robot.right_arm_config.port=can1 \
    --robot.right_arm_config.can_auto_init=true \
    --robot.right_arm_config.require_calibration=true \
    --robot.right_arm_config.enable_on_connect=true \
    --robot.right_arm_config.disable_on_disconnect=false \
    --robot.right_arm_config.high_follow=true \
    --robot.right_arm_config.speed_ratio=10 \
    --robot.right_arm_config.startup_sleep_s=0.5 \
    "--robot.right_arm_config.cameras=$RIGHT_CAMERAS" \
    --server_address=192.168.110.85:18081 \
    --policy_type=xr0 \
    --pretrained_name_or_path="$CKPT" \
    --policy_device=cuda \
    --client_device=cpu \
    --controlled_arms=both \
    --startup_left_gripper_position=100.0 \
    --startup_left_gripper_hold_s=0.0 \
    --startup_right_gripper_position=100.0 \
    --startup_right_gripper_hold_s=0.0 \
    --actions_per_chunk=30 \
    --chunk_size_threshold=0.0 \
    --aggregate_fn_name=weighted_average \
    --fps=30 \
    --rtc.enabled=false \
    --observation_image_codec=jpeg \
    --observation_jpeg_quality=90 \
    --display_data=true \
    --display_compressed_images=true \
    --dry_run_actions=false \
    --duration="$DURATION" \
    --task="$TASK" \
    2>&1 | tee "$LOG"

  STATUS=${PIPESTATUS[0]}
  echo "推理退出状态：$STATUS"
  echo "日志：$LOG"
  return "$STATUS"
}
```

双触觉 5000 步权重先测试 5 秒：

```bash
run_bimanual_pour_no_rtc_speed10 tactile 5
```

RGB3 无触觉 5000 步权重先测试 5 秒：

```bash
run_bimanual_pour_no_rtc_speed10 rgb3 5
```

只有短时测试确认关节方向、夹爪方向、相机输入和动作路径正常后，才运行较长测试：

```bash
run_bimanual_pour_no_rtc_speed10 tactile 300
run_bimanual_pour_no_rtc_speed10 rgb3 300
```

一次任务成功或明确失败后应立即停止客户端、恢复场景并复位双臂。不要把一次 300 秒连续运行当成多个独立 trial。

## 13. 无 RTC 动作块执行时序

当前配置：

```text
actions_per_chunk=30
fps=30
chunk_size_threshold=0.0
rtc.enabled=false
```

执行流程：

1. action queue 为空时，客户端采集当前双臂 state、三路 RGB，以及触觉模型需要的两路触觉图。
2. 客户端 JPEG 编码图像并通过 gRPC 发送给服务器。
3. 服务器解码、预处理、执行 XR0 推理并返回 30 个 action。
4. 客户端每 `33.3 ms` 发送一个 action，完整动作块约执行 1 秒。
5. 因为 `chunk_size_threshold=0.0`，只有队列完全清空后才采集下一次观测并请求下一块。
6. 网络与推理期间 action queue 为空，因此块之间可能出现等待。

该模式用于隔离 RTC prefix/reanchor 对动作边界的影响。它不是最低延迟配置。若后续把 `chunk_size_threshold` 改为 `0.5`，客户端会在剩余约 15 个 action 时提前请求下一块，但新块仍不是 RTC prefix-conditioned suffix，边界跳变风险依然存在。

`speed_ratio=10` 只改变 Piper 低层跟随响应，不会限制模型相邻 action 的最大关节变化。模型输出发生大跳时，机械臂仍可能快速追赶目标。真机安全最终需要额外的关节步长、速度、加速度限制以及动作块边界平滑，不能只依赖 `speed_ratio`。

## 14. 真机安全检查表

每次真机运行前确认：

- 双臂已经回到同一个保存的复位目标。
- 左右夹爪初始目标均为 `100.0`，与采集开始时的张开状态一致。
- 左臂 `can0`、右臂 `can1`，没有交换。
- 触觉模型使用左 `/dev/ttyACM1`、右 `/dev/ttyACM0`。
- 触觉连接标定期间两侧指尖完全卸载并静止。
- 杯子、水瓶、相机和光照与采集场景一致。
- 人员远离机械臂运动范围，并能够立即按 `Ctrl-C` 或物理急停。
- 第一次测试只运行 5 秒。
- 发现抖动、错误方向、突然回撤或 action chunk 边界跳变时立即停止，不继续提高速度比例。

## 15. 结果解释

训练 loss 下降只能说明优化过程没有明显发散，不能证明真机任务已经学会。当前 5000 步触觉模型曾观察到抓取和双臂靠拢，但动作块之间出现过明显关节目标跳变。判断“没有学会倒水”与“倒水 action 被执行链路跳过”时，应分别检查：

1. 服务器保存的原始 action chunk 是否包含右腕倾倒动作。
2. 相邻 action chunk 边界是否连续。
3. 客户端 action queue 是否发生 starvation。
4. `robot.send_action` 的目标和硬件反馈是否一致。
5. 同一初始状态下，无 RTC 与 RTC 的动作边界差异。

在这些检查完成前，不应仅通过调高或调低 `speed_ratio` 判断模型是否收敛。
