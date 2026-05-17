# New Tactile Prompt Training

This directory contains an isolated training entrypoint for a simplified
thesis-style tactile experiment. It does not modify any existing file under
`src/lerobot`.

## What it does

- Assumes the dataset already contains these seven image streams:
  - `observation.images.left_top`
  - `observation.images.left_wrist`
  - `observation.images.right_wrist`
  - `observation.images.tactile_left_outer`
  - `observation.images.tactile_left_inner`
  - `observation.images.tactile_right_outer`
  - `observation.images.tactile_right_inner`
- Forces the policy image order to follow the list above.
- Rewrites the `task` prompt into a sensor-aware instruction prompt.
- Applies thesis-inspired `pi05` defaults by default:
  - `batch_size=1`
  - `steps=10000`
  - `chunk_size=30`
  - `n_action_steps=30`
  - `max_state_dim=14`
  - `max_action_dim=14`
  - `optimizer_lr=5e-5`
  - `scheduler_warmup_steps=1000`
  - `scheduler_decay_steps=6000`
  - `scheduler_decay_lr=5e-6`
  - `freeze_vision_encoder=true`

## Important limitation

This script does **not** add new tactile model heads or true cross-modal
alignment logic. It is the simplified prompt-driven version you requested:
the tactile modality is assumed to exist as four extra image streams, and the
prompt explicitly names them.

## Inference

The new inference entrypoint is:

`new/lerobot_infer_tactile_ros.py`

It is a standalone ROS1 bridge that:

- loads a trained `pi05` checkpoint
- reads dataset metadata + stats to keep normalization/action dimensions consistent
- subscribes to 3 RGB topics + 4 tactile topics + 1 state topic
- rebuilds the same sensor-aware prompt used in training
- publishes the policy output as `std_msgs/Float32MultiArray`
- can also publish a JSON action dictionary for a downstream control node

### Important deployment note

If your tactile heatmaps come from a ROS node called `tac_map`, the policy
bridge still subscribes to **topics**, not node names.

So the real workflow is:

1. Check which topics `tac_map` publishes.
2. Map those 4 topics to:
   - `observation.images.tactile_left_outer`
   - `observation.images.tactile_left_inner`
   - `observation.images.tactile_right_outer`
   - `observation.images.tactile_right_inner`
3. Also provide the 3 RGB topics and 1 robot state topic.
4. Let this script publish the action vector to another ROS topic.
5. Use a separate robot-control node to consume that action topic and send commands to the real robot.

The script intentionally does **not** directly drive the robot, so it does not
change your current control chain.

### Default topic assumptions

- RGB
  - `/camera/left_top/image_raw`
  - `/camera/left_wrist/image_raw`
  - `/camera/right_wrist/image_raw`
- tactile
  - `/tac_map/tactile_left_left`
  - `/tac_map/tactile_left_right`
  - `/tac_map/tactile_right_left`
  - `/tac_map/tactile_right_right`
- state
  - `/robot/state`
- action output
  - `/policy/action`

### Example

Run from the repo root in the same environment where ROS + PyTorch + LeRobot
are already available:

```bash
python ./new/lerobot_infer_tactile_ros.py \
  --checkpoint /path/to/checkpoint_dir \
  --dataset-repo-id whz/pour_water_tactile \
  --dataset-root /home/whz/lerobot_dataset \
  --task "The left arm grasps the metal cup, the right arm grasps the plastic bottle, and then the right arm pours water from the bottle into the cup held by the left arm." \
  --robot-type bi_piper_follower \
  --state-topic /robot/state \
  --state-msg-type float32multiarray \
  --left-top-topic /camera/left_top/image_raw \
  --left-wrist-topic /camera/left_wrist/image_raw \
  --right-wrist-topic /camera/right_wrist/image_raw \
  --tactile-left-outer-topic /tac_map/tactile_left_left \
  --tactile-left-inner-topic /tac_map/tactile_left_right \
  --tactile-right-outer-topic /tac_map/tactile_right_left \
  --tactile-right-inner-topic /tac_map/tactile_right_right \
  --action-topic /policy/action \
  --policy-rate-hz 30 \
  --print-prompt-once
```

If your state topic is `sensor_msgs/JointState`, switch to:

```bash
python ./new/lerobot_infer_tactile_ros.py \
  --checkpoint /path/to/checkpoint_dir \
  --dataset-repo-id whz/pour_water_tactile \
  --dataset-root /home/whz/lerobot_dataset \
  --task "..." \
  --state-topic /joint_states \
  --state-msg-type jointstate \
  --state-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13
```

### What to change if `tac_map` publishes a custom message

This script currently assumes tactile heatmaps arrive as standard ROS image
messages:

- `sensor_msgs/Image`, or
- `sensor_msgs/CompressedImage`

If `tac_map` publishes a custom message type instead, you only need to replace
the tactile callback decoding part so that each tactile stream becomes a
`numpy.uint8` image shaped like `H x W x 3`, then the rest of the inference
path stays the same.

## Client/Server

If you want to separate ROS acquisition from policy inference, use:

- `new/lerobot_tactile_policy_server.py`
- `new/lerobot_tactile_ros_client.py`

Architecture:

1. `ROS client` subscribes to 3 RGB topics, 4 tactile topics, and 1 state topic.
2. The client builds the same sensor-aware prompt as training.
3. The client compresses RGB/tactile images before transport:
   - RGB defaults to `jpeg`
   - tactile defaults to `png`
4. The client sends the packed observation to the remote `policy server` over the existing LeRobot `grpc` transport.
5. The server decodes the packet, restores the 7-stream observation, runs `pi05` inference, and returns an action chunk.
6. The client keeps a local action queue and publishes one action at a time to `/policy/action`.

This split is the practical version for:

- ROS and robot drivers on one machine
- GPU inference on another machine
- keeping the robot-side process light

### Server example

```bash
python ./new/lerobot_tactile_policy_server.py \
  --host 0.0.0.0 \
  --port 8080 \
  --checkpoint /path/to/checkpoint_dir \
  --dataset-repo-id whz/pour_water_tactile \
  --dataset-root /home/whz/lerobot_dataset \
  --device cuda \
  --actions-per-chunk 30
```

### Client example

```bash
python ./new/lerobot_tactile_ros_client.py \
  --server-address 192.168.1.10:8080 \
  --dataset-repo-id whz/pour_water_tactile \
  --dataset-root /home/whz/lerobot_dataset \
  --task "The left arm grasps the metal cup, the right arm grasps the plastic bottle, and then the right arm pours water from the bottle into the cup held by the left arm." \
  --robot-type bi_piper_follower \
  --state-topic /robot/state \
  --state-msg-type float32multiarray \
  --left-top-topic /camera/left_top/image_raw \
  --left-wrist-topic /camera/left_wrist/image_raw \
  --right-wrist-topic /camera/right_wrist/image_raw \
  --tactile-left-outer-topic /tac_map/tactile_left_left \
  --tactile-left-inner-topic /tac_map/tactile_left_right \
  --tactile-right-outer-topic /tac_map/tactile_right_left \
  --tactile-right-inner-topic /tac_map/tactile_right_right \
  --action-topic /policy/action \
  --policy-rate-hz 30 \
  --actions-per-chunk 30
```

### Practical note

The client/server version is intentionally better suited than the single-process
version for network deployment, because it compresses the 7 image streams before
sending them to the inference server. Sending raw uncompressed images would be
too expensive for a real 30Hz setup.

### BiPiper direct client

If you want the client side to directly connect to a dual-arm PiPER follower and
send actions to:

- left arm: `can0`
- right arm: `can1`

use:

- `new/lerobot_tactile_bipiper_client.py`

This version:

- still subscribes to RGB/tactile ROS topics
- reads robot state directly from `bi_piper_follower`
- sends the predicted action directly to the two follower arms
- keeps `/policy/action` and `/policy/action_json` only as debug output topics

Topic mapping in the current implementation is:

- `observation.images.tactile_left_outer` <= `/tac_map/tactile_left_left`
- `observation.images.tactile_left_inner` <= `/tac_map/tactile_left_right`
- `observation.images.tactile_right_outer` <= `/tac_map/tactile_right_left`
- `observation.images.tactile_right_inner` <= `/tac_map/tactile_right_right`

This means the dataset schema still keeps the original assumed four tactile
feature keys, while the live ROS side uses your new topic names.

### Local loopback demo

If you want a teacher-facing demo on the server without ROS or the real robot,
use:

- `new/lerobot_tactile_policy_server.py`
- `new/lerobot_tactile_mock_loop_client.py`

This mode keeps everything on one machine:

1. The local `policy server` loads the trained checkpoint.
2. The `mock client` reads one episode from the `tac` dataset.
3. RGB frames come from the dataset.
4. The four tactile streams are turned into synthetic dynamic heatmaps so the
   input changes over time.
5. The previous policy action updates the next simulated `observation.state`,
   so the loop is not just static replay.
6. The client sends the packed observation to `127.0.0.1:<port>` and prints the
   returned action chunk.
7. Optionally, it saves a per-step composite PNG panel and a `summary.jsonl`
   file for offline presentation material.

Server example:

```bash
python ./new/lerobot_tactile_policy_server.py \
  --host 127.0.0.1 \
  --port 8080 \
  --checkpoint /home/enine/SACM/outputs/4_5_tactile_projector_expert_pi05_base/checkpoints/last/pretrained_model \
  --dataset-repo-id tac \
  --dataset-root /home/enine/SACM/lerobot_dataset/tac \
  --device cuda \
  --actions-per-chunk 30
```

Mock client example:

```bash
python ./new/lerobot_tactile_mock_loop_client.py \
  --server-address 127.0.0.1:8080 \
  --dataset-repo-id tac \
  --dataset-root /home/enine/SACM/lerobot_dataset/tac \
  --task "The left arm grasps the metal cup, the right arm grasps the plastic bottle, and then the right arm pours water from the bottle into the cup held by the left arm." \
  --episode-index 0 \
  --steps 30 \
  --policy-rate-hz 2 \
  --save-dir /home/enine/SACM/outputs/tactile_loopback_demo
```

## Example

Run from the repo root:

```powershell
python .\new\lerobot_train_tactile_prompt.py `
  --dataset.repo_id=local/my_tactile_dataset `
  --dataset.root=E:\path\to\dataset `
  --policy.type=pi05 `
  --policy.path=lerobot/pi05 `
  --wandb.enable=false
```

If your dataset uses different image key names, override the expected order:

```powershell
python .\new\lerobot_train_tactile_prompt.py `
  --dataset.repo_id=local/my_tactile_dataset `
  --dataset.root=E:\path\to\dataset `
  --policy.type=pi05 `
  --policy.path=lerobot/pi05 `
  --tactile_prompt.ordered_image_keys=observation.images.top,observation.images.left_wrist,observation.images.right_wrist,observation.images.tactile_left_outer,observation.images.tactile_left_inner,observation.images.tactile_right_outer,observation.images.tactile_right_inner
```
