# 通用双臂与右臂单臂触觉采集命令

本文提供两个可复用的工控机采集函数。使用时只需要修改函数中的 `TASK`，不需要修改触觉、相机、CAN 或标定配置。

## 固定硬件映射

```text
左 Follower：can0
右 Follower：can1
左 Leader：can2
右 Leader：can3

顶部/左 ego 相机：420222071960
左腕相机：419522073184
右腕相机：153122074100

右夹爪触觉：/dev/ttyACM0
左夹爪触觉：/dev/ttyACM1
```

触觉图像范围固定为：

```text
上方两张 Fz 热力图：vmin=0.5，vmax=5.0
下方 RGB 图 Fz：vmax=20.0
下方 RGB 图 Fx/Fy：vmax=20.0
```

固定采集参数：

```text
机器人：bi_piper_follower
分辨率：640x480
FPS：30
触觉输出尺寸：256
触觉 gamma：0.55
触觉掉线策略：recover
Follower 速度比例：20
Leader command 速度比例：20
保存模式：serial
视频编码：h264
```

## 使用前检查

在工控机执行：

```bash
check_generic_collection_prerequisites() {
cd ~/gczx_Evo-RL_xr0/gczx_Evo-RL || return 1
conda activate evo_rl_xr0 || return 1
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1

for DEV in /dev/ttyACM0 /dev/ttyACM1; do
  [[ -c "$DEV" ]] || {
    echo "找不到触觉设备：$DEV"
    ls -l /dev/ttyACM* 2>/dev/null || true
    return 1
  }
done

fuser -v /dev/ttyACM0 /dev/ttyACM1 || true

pgrep -af \
  '[l]erobot_record_piper_tactile|[l]erobot.async_inference.robot_client|[l]erobot-tactile-preview|[l]erobot_tactile_preview' \
  || true
}

check_generic_collection_prerequisites
```

如果 `fuser` 或 `pgrep` 显示已有进程，先停止占用触觉串口或 CAN 的程序。连接和每条 episode 结束后的触觉标定期间，左右指尖都必须卸载、静止。

正式采集前检查四个标定文件：

```bash
CALIB_ROBOT="$HOME/.cache/huggingface/lerobot/calibration/robots/bi_piper_follower"
CALIB_TELEOP="$HOME/.cache/huggingface/lerobot/calibration/teleoperators/bi_piper_leader"

for FILE in \
  "$CALIB_ROBOT/piper_follower_left.json" \
  "$CALIB_ROBOT/piper_follower_right.json" \
  "$CALIB_TELEOP/piper_leader_left.json" \
  "$CALIB_TELEOP/piper_leader_right.json"
do
  [[ -f "$FILE" ]] || {
    echo "缺少标定文件：$FILE"
    return 1
  }
done
```

## 命令一：通用双臂触觉采集

双臂任务中，左臂和右臂都由对应 Leader 操作。只修改 `TASK` 即可换任务；数据集目录使用时间戳自动生成，不会覆盖已有数据。

```bash
collect_generic_bimanual_tactile() {
  cd ~/gczx_Evo-RL_xr0/gczx_Evo-RL || return 1
  conda activate evo_rl_xr0 || return 1

  export PYTHONPATH="$PWD/src"
  export PYTHONUNBUFFERED=1

  # 只修改这一行
  TASK='Use the left arm to pick up the cup on the left and the right arm to pick up the water bottle on the right. Bring both arms toward the center, then tilt the water bottle held by the right arm to pour water into the cup held by the left arm.'

  DATASET_NAME="$(date +%Y%m%d_%H%M%S)_generic_bimanual_dual_tactile"
  REPO_ID="whz/$DATASET_NAME"
  DATASET_ROOT="$HOME/xr0_datasets/$DATASET_NAME"

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
    [[ -c "$DEV" ]] || {
      echo "找不到触觉设备：$DEV"
      ls -l /dev/ttyACM* 2>/dev/null || true
      return 1
    }
  done

  if fuser "$LEFT_TACTILE" "$RIGHT_TACTILE" >/dev/null 2>&1; then
    echo "触觉串口被其他进程占用："
    fuser -v "$LEFT_TACTILE" "$RIGHT_TACTILE" || true
    return 1
  fi

  for FILE in \
    "$ROBOT_CALIB_DIR/piper_follower_left.json" \
    "$ROBOT_CALIB_DIR/piper_follower_right.json" \
    "$TELEOP_CALIB_DIR/piper_leader_left.json" \
    "$TELEOP_CALIB_DIR/piper_leader_right.json"
  do
    [[ -f "$FILE" ]] || {
      echo "缺少标定文件：$FILE"
      return 1
    }
  done

  [[ ! -e "$DATASET_ROOT" ]] || {
    echo "数据集目录已存在，拒绝覆盖：$DATASET_ROOT"
    return 1
  }

  mkdir -p "$HOME/xr0_datasets" "$LOG_DIR" || return 1

  echo "============================================================"
  echo "通用双臂触觉采集"
  echo "TASK=$TASK"
  echo "左触觉=$LEFT_TACTILE（ACM1）"
  echo "右触觉=$RIGHT_TACTILE（ACM0）"
  echo "上方Fz范围=0.5-5.0"
  echo "下方Fz/Fx/Fy范围=20.0"
  echo "Follower速度=20，Leader速度=20，FPS=30"
  echo "数据集=$DATASET_ROOT"
  echo "日志=$LOG"
  echo
  echo "确认左右指尖卸载且静止后再开始连接和标定。"
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
  echo "采集退出状态：$STATUS"
  echo "数据集：$DATASET_ROOT"
  echo "日志：$LOG"
  return "$STATUS"
}

collect_generic_bimanual_tactile
```

### 双臂采集时的 episode 操作

```text
ENTER：开始下一条 episode
ENTER：结束当前 episode 并选择结果
s：保存 success
f：保存 failure
o：保存 ongoing
d：丢弃当前 episode
q：退出并丢弃当前 episode
Ctrl-C：停止采集
```

## 命令二：通用右臂单臂触觉采集

当前采集脚本底层仍连接双臂并写入 14 维 `action`。因此这里的“右臂单臂”是任务层定义：

- 只由右 Leader 操作右臂；
- 左 Leader 不操作，左臂保持固定；
- 左 Follower 仍保持连接并记录状态；
- 两路触觉都采集，右触觉为 `ACM0`，左触觉为 `ACM1`。

这样可以保持通用的双臂数据 schema。训练右臂策略时再使用 `controlled_arms=right` 忽略左臂 action。不要把左臂 Leader 的误操作混入右臂单臂数据。

同样只修改 `TASK`：

```bash
collect_generic_right_arm_tactile() {
  cd ~/gczx_Evo-RL_xr0/gczx_Evo-RL || return 1
  conda activate evo_rl_xr0 || return 1

  export PYTHONPATH="$PWD/src"
  export PYTHONUNBUFFERED=1

  # 只修改这一行；任务描述应明确是右臂执行，左臂保持静止
  TASK='Use the right arm to pick up the object on the right and place it at the target location on the left. Keep the left arm stationary.'

  DATASET_NAME="$(date +%Y%m%d_%H%M%S)_generic_right_arm_dual_tactile"
  REPO_ID="whz/$DATASET_NAME"
  DATASET_ROOT="$HOME/xr0_datasets/$DATASET_NAME"

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
    [[ -c "$DEV" ]] || {
      echo "找不到触觉设备：$DEV"
      ls -l /dev/ttyACM* 2>/dev/null || true
      return 1
    }
  done

  if fuser "$LEFT_TACTILE" "$RIGHT_TACTILE" >/dev/null 2>&1; then
    echo "触觉串口被其他进程占用："
    fuser -v "$LEFT_TACTILE" "$RIGHT_TACTILE" || true
    return 1
  fi

  for FILE in \
    "$ROBOT_CALIB_DIR/piper_follower_left.json" \
    "$ROBOT_CALIB_DIR/piper_follower_right.json" \
    "$TELEOP_CALIB_DIR/piper_leader_left.json" \
    "$TELEOP_CALIB_DIR/piper_leader_right.json"
  do
    [[ -f "$FILE" ]] || {
      echo "缺少标定文件：$FILE"
      return 1
    }
  done

  [[ ! -e "$DATASET_ROOT" ]] || {
    echo "数据集目录已存在，拒绝覆盖：$DATASET_ROOT"
    return 1
  }

  mkdir -p "$HOME/xr0_datasets" "$LOG_DIR" || return 1

  echo "============================================================"
  echo "通用右臂单臂触觉采集"
  echo "TASK=$TASK"
  echo "右臂操作：can1 / can3"
  echo "左臂保持静止：can0 / can2"
  echo "左触觉=$LEFT_TACTILE（ACM1）"
  echo "右触觉=$RIGHT_TACTILE（ACM0）"
  echo "上方Fz范围=0.5-5.0"
  echo "下方Fz/Fx/Fy范围=20.0"
  echo "Follower速度=20，Leader速度=20，FPS=30"
  echo "数据集=$DATASET_ROOT"
  echo "日志=$LOG"
  echo
  echo "采集期间左手 Leader 不得移动；左臂只保持当前姿态。"
  echo "确认左右指尖卸载且静止后再开始连接和标定。"
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
  echo "采集退出状态：$STATUS"
  echo "数据集：$DATASET_ROOT"
  echo "日志：$LOG"
  return "$STATUS"
}

collect_generic_right_arm_tactile
```

## 右臂单臂采集的注意事项

右臂单臂采集不是改变机器人 action schema。脚本仍然会保存：

```text
action：[14]
observation.state：[14]
```

左臂应在整个 episode 中保持静止，否则左臂的运动会混入数据。后续训练右臂单臂策略时使用：

```text
--policy.controlled_arms=right
```

并重新计算 `controlled_arms=right` 对应的 stats；不能直接复用双臂 `controlled_arms=both` 的 stats。

## 统一数据检查

每批采集结束后至少确认：

```text
episodes 与实际保存条数一致
action shape = [14]
observation.state shape = [14]
fps = 30
图像 key 包含 left_ego、left_wrist、right_wrist、left_tactile、right_tactile
左右触觉空载时热力图接近零
触碰物体时触觉图有合理变化且没有持续饱和
所有 MP4 可以完整解码
```

采集时只需要修改 `TASK`；但如果更换相机、CAN、触觉串口或标定文件，必须同步修改命令并重新做完整元数据检查。
