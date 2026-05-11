# SACM 算法流程说明

本文档总结当前代码库中 SACM 方法的完整运行流程。

SACM 指 **Stage-Aware Chunk Mining**，即“阶段感知的优势动作片段挖掘”。在当前代码实现里，它是一个用于 VLA policy post-training 的离线方法：

1. 先用 value model 给离线数据集的每一帧打价值分数；
2. 再根据价值曲线把 episode 划分成若干阶段；
3. 用固定长度窗口滑动，计算每个 action chunk 的优势；
4. 在阶段内部和阶段边界处筛选优势 chunk；
5. 把优势 chunk 展开成帧级二值标签；
6. 训练 VLA policy 时把标签转换成 `Advantage: positive` 或 `Advantage: negative` prompt；
7. 推理时手动输入 `Advantage: positive`，让 policy 按优势条件输出动作。

需要强调：**当前实现不是在线 RL**。推理阶段不会重新跑 value model，也不会在线挖 chunk，更不会根据真实环境 reward 更新 policy。当前 SACM 更准确的定位是：

```text
value-guided offline VLA post-training
```

也就是：

```text
价值模型引导的离线 VLA 后训练方法
```

---

## 1. 总体流程

完整流程如下：

```text
LeRobot 数据集
  |
  | 1. 训练或复用 value model
  v
pistar06 value model checkpoint
  |
  | 2. 对数据集每一帧做 value inference
  v
数据集 + complementary_info.value_*
  |
  | 3. 运行 SACM 阶段感知 chunk mining
  v
数据集 + complementary_info.vgsacm_*.indicator
  |
  | 4. 用 Advantage prompt 注入训练 VLA policy
  v
SACM post-trained policy checkpoint
  |
  | 5. 推理时输入 task + "Advantage: positive"
  v
优势条件动作 chunk 推理
```

核心思想可以概括为：

```text
value model 负责判断哪些帧更接近任务完成；
SACM 负责从 value 曲线中挖出高优势的 K-step 动作片段；
被选中的 chunk 内所有帧标为 positive；
训练 policy 时，positive 帧的 task 后面拼 Advantage: positive；
negative 帧的 task 后面拼 Advantage: negative；
最后得到一个 advantage-conditioned VLA policy。
```

---

## 2. 数据集要求

输入数据需要是 LeRobot 格式的数据集。对于 `pi05` 这类 VLA policy，数据集至少需要包含：

```text
observation.state
observation.images.*
action
task
episode_index
frame_index
task_index
```

对于 value training 和 SACM，还需要 episode 成功/失败信息：

```text
episode_success
```

`episode_success` 会影响两个地方：

1. value model 的训练目标；
2. SACM 对成功 episode 和失败 episode 的阶段筛选逻辑。

当前 ALOHA 双臂任务的数据形状是：

```text
observation.state: 14 维
action: 14 维

left arm: 6 维关节 + 1 维夹爪
right arm: 6 维关节 + 1 维夹爪
```

当前相机 key 是：

```text
observation.images.left_wrist
observation.images.right_top
observation.images.right_wrist
```

当前任务文本是：

```text
The left arm grasps the metal cup, the right arm grasps the plastic bottle, and then the right arm pours water from the bottle into the cup held by the left arm.
```

---

## 3. Value Model 训练

value model 的训练入口是：

```text
src/lerobot/scripts/lerobot_value_train.py
```

配置文件相关代码是：

```text
src/lerobot/configs/value_train.py
```

当前使用的 value model 类型是：

```text
pistar06
```

核心实现位于：

```text
src/lerobot/values/pistar06/modeling_pistar06.py
```

### 3.1 value target 是怎么来的

当前 value model 的监督信号不是人工标注的 reward，而是根据 episode 长度和成功/失败标签自动构造的 dense target。

对于某个 episode 中的第 `t` 帧：

```text
remaining_steps = episode_length - t - 1
g = -remaining_steps
```

如果该 episode 是失败 episode，则额外加失败惩罚：

```text
c_fail = task_max_length * c_fail_coef
g = g - c_fail
```

然后归一化：

```text
g_norm = g / (task_max_length + c_fail)
target = clip(g_norm, -1, 0)
```

所以对成功 episode 来说：

```text
越早的帧 -> value 越接近 -1
越晚的帧 -> value 越接近 0
最后的帧 -> value 最接近 0
```

这意味着，如果所有 episode 都被标成 success，那么 value model 更像是在学习“任务进度”，而不是严格意义上学习“哪些动作导致成功”。如果想让 value model 学到失败/错误动作的区别，需要加入失败 episode 或伪失败数据。

### 3.2 value model 输出

value model 训练完成后，对每一帧输出一个标量 value：

```text
V_t
```

理论上它大致处在：

```text
[-1, 0]
```

越接近 0，表示模型认为当前帧越接近任务完成。

---

## 4. Value Inference：给每帧打分

value inference 的入口是：

```text
src/lerobot/scripts/lerobot_value_infer.py
```

它会加载 value model checkpoint，对数据集每一帧预测 value，然后写回数据集 parquet。

典型输出字段是：

```text
complementary_info.value_sacm
```

示例命令结构：

```bash
lerobot-value-infer \
  --dataset.repo_id=<DATASET_REPO_ID> \
  --dataset.root=<DATASET_ROOT> \
  --inference.checkpoint_path=<VALUE_CHECKPOINT_DIR> \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=false \
  --acp.value_field=complementary_info.value_sacm \
  --output_dir=outputs/value_infer/<RUN_NAME>
```

这一步结束后，数据集里会多出：

```text
complementary_info.value_sacm
```

此时还没有运行 SACM，只有每帧 value。

---

## 5. SACM 阶段感知 Chunk Mining

SACM 的命令入口是：

```text
src/lerobot/scripts/lerobot_stage_chunk_mine.py
```

核心代码在：

```text
src/lerobot/rl/stage_chunk_mining/pipeline.py
src/lerobot/rl/stage_chunk_mining/selection.py
src/lerobot/rl/stage_chunk_mining/stages.py
src/lerobot/rl/stage_chunk_mining/advantage.py
src/lerobot/rl/stage_chunk_mining/boundary.py
src/lerobot/rl/stage_chunk_mining/report.py
```

典型命令结构：

```bash
lerobot-stage-chunk-mine \
  --dataset.repo_id=<DATASET_REPO_ID> \
  --dataset.root=<DATASET_ROOT> \
  --dataset.success_field=episode_success \
  --dataset.default_success=success \
  --mining.value_field=complementary_info.value_sacm \
  --mining.output_prefix=complementary_info.vgsacm_sacm_threshold_r015_chunkmask \
  --mining.value_normalization=episode_minmax \
  --mining.stage_aware=true \
  --mining.num_stages=5 \
  --mining.chunk_size=50 \
  --mining.stage_top_ratio=0.15 \
  --mining.boundary_top_k=1 \
  --mining.boundary_mode=unique_stage_boundary \
  --mining.failure_max_stage=2 \
  --output_dir=outputs/stage_chunk_mining/sacm_threshold_r015_chunkmask
```

主要参数含义：

```text
num_stages / M:
  把 value 进度分成多少个阶段。

chunk_size / K:
  每个动作 chunk 的长度。

stage_top_ratio:
  每个 episode-stage 组内保留多少比例的 intra-stage 高优势 chunk。

stage_aware:
  是否启用阶段感知筛选。true 使用 SACM 当前的分阶段和边界逻辑；false 则关闭分阶段筛选，改成每条 episode 内全局滑窗 top-ratio 筛选。

global_top_ratio:
  stage_aware=false 时，每条 episode 所有合法滑动窗口中保留前多少比例的高 advantage chunk。

global_top_k:
  stage_aware=false 时，每条 episode 固定保留多少个全局最高 advantage chunk；如果大于 0，优先于 global_top_ratio。

boundary_top_k:
  每个阶段边界附近保留几个最好的 boundary chunk。

value_normalization:
  value 归一化方式。

failure_max_stage:
  对失败 episode，只允许挖到的最高阶段。
```

---

## 6. Value 归一化

SACM 先对每个 episode 的 value 曲线做归一化。

支持三种模式：

```text
episode_minmax
clip
none
```

### 6.1 episode_minmax

这是当前实验推荐使用的模式。

对一个 episode 内的 value：

```text
V_min = min_t V_t
V_max = max_t V_t
```

归一化为：

```text
V_norm(t) = (V_t - V_min) / (V_max - V_min) - 1
```

然后 clip 到：

```text
[-1, 0]
```

也就是说，每条 episode 内部：

```text
最小 value -> -1
最大 value -> 0
```

这种方式的好处是，不同 episode 之间 value 标定不完全一致时，也可以根据每条 episode 自己的 value 范围做阶段划分。

### 6.2 clip

`clip` 模式直接做：

```text
V_norm(t) = clip(V_t, -1, 0)
```

如果 value model 本身已经输出稳定、全局可比的 `[-1, 0]` value，可以用这个模式。

### 6.3 none

`none` 不做归一化，直接使用原始 value。一般用于调试或自定义 value。

---

## 7. 阶段划分

归一化后：

```text
V_norm(t) in [-1, 0]
```

先转成 completion：

```text
C_t = clip(V_norm(t) + 1, 0, 1)
```

再划分成 `M` 个阶段：

```text
raw_stage_t = floor(C_t * M)
raw_stage_t = clamp(raw_stage_t, 0, M - 1)
```

例如：

```text
M = 5
```

阶段编号就是：

```text
0, 1, 2, 3, 4
```

---

## 8. 成功 Episode 的阶段逻辑

成功 episode 使用“首次到达阈值”的单调阶段逻辑。

代码会对 raw stage 做 running max：

```text
running_stage_t = max(raw_stage_0, raw_stage_1, ..., raw_stage_t)
```

然后减去初始阶段：

```text
stage_t = running_stage_t - raw_stage_0
stage_t = clamp(stage_t, 0, M - 1)
```

所以成功 episode 有几个特点：

```text
1. 一定从 stage 0 开始；
2. 只有第一次达到更高阈值时才跳阶段；
3. 不会因为 value 抖动而阶段下降；
4. 如果一开始不是最低 value，也不会直接从高阶段开始。
```

例子：

```text
raw_stage:      2 1 2 3 4
success_stage:  0 0 0 1 2
```

这符合我们想要的语义：

```text
成功演示是一个整体完成过程，阶段应该表示首次达到新的进度阈值。
```

---

## 9. 失败 Episode 的阶段逻辑

失败 episode 不强制做单调首次到达。

失败 episode 保留原始阶段变化，例如：

```text
0 -> 1 -> 2 -> 1 -> 0
```

然后 SACM 找第一次下降：

```text
第一次出现 stage_t < running_max_stage 的位置
```

只在下降之前的阶段中挖优势 chunk，并且受：

```text
failure_max_stage
```

限制。

例如：

```text
failure_max_stage = 2
```

那么失败 episode 只会在前面的：

```text
0, 1, 2
```

这些上升阶段中挖 chunk，一旦阶段开始下降，后面的片段不再参与优势 chunk 筛选。

这个逻辑对应的是：

```text
失败 episode 中仍可能包含早期有用动作，但一旦进入回退/失败趋势，就不把后续动作当成优势动作。
```

---

## 10. Chunk 滑窗

SACM 用固定长度 `K` 的窗口从 episode 开始滑到结束。

一个从 `s` 开始的 chunk 覆盖：

```text
[s, s + K)
```

并用 `s + K` 位置的 value 做 bootstrap。

因此合法 chunk start 需要满足：

```text
s + K < episode_length
```

如果：

```text
K = 50
```

那么一个 chunk 包含 50 帧动作：

```text
s, s+1, ..., s+49
```

---

## 11. Chunk Advantage 公式

chunk advantage 的代码在：

```text
src/lerobot/rl/stage_chunk_mining/advantage.py
```

**advantage update**：当前实现使用 GAE-style 的 single-step TD error 累加形式，而不是只看 chunk 起点和终点：

```text
rho = 1 / L_max
delta_{s+i} = -rho + V_norm(s+i+1) - V_norm(s+i)
A_chunk(s) = sum_{i=0}^{K-1} lambda^i * delta_{s+i}
```

默认：

```text
lambda = 0.95
gamma = 1
```

其中：

```text
K:
  chunk 长度

L_max:
  当前 task 的 episode 长度尺度

rho:
  每一步的时间惩罚

lambda:
  GAE 衰减系数，控制后续 TD error 的权重
```

直观解释：

```text
V_norm(s+i+1) - V_norm(s+i)
```

表示第 `i` 步带来的 value 改善。

```text
-rho
```

是每一步的时间惩罚。

所以 chunk advantage 不再只依赖：

```text
V_norm(s)
V_norm(s + K)
```

而是利用 chunk 内部每一步的 TD error。这样对 critic 在起点或终点的单点噪声更不敏感。

当：

```text
lambda = 1
```

时，公式会退化为旧的端点差值形式：

```text
A_chunk(s) = V_norm(s + K) - V_norm(s) - K / L_max
```

当：

```text
lambda = 0
```

时，只使用第一步 TD error：

```text
A_chunk(s) = V_norm(s + 1) - V_norm(s) - 1 / L_max
```

所以一个 chunk 只有在：

```text
chunk 内持续产生足够高的 value 改善
```

时，才会有较高 advantage。

---

## 12. L_max 的含义

`L_max` 用于定义“正常任务长度尺度”。

支持的模式包括：

```text
task_p95
task_max
global_p95
global_max
task_success_max
```

默认是：

```text
task_p95
```

也就是对每个 task，用 episode 长度的 95 分位数作为长度尺度。

代码还会做保护：

```text
safe_l_max = max(L_max, K + 1)
```

保证分母合法。

---

## 13. Chunk 类型

每个 chunk 会根据窗口内的阶段变化被分类。

代码检查：

```text
stage[s : s + K + 1]
```

然后分成：

```text
CHUNK_TYPE_INTRA_STAGE
CHUNK_TYPE_FORWARD_TRANSITION
CHUNK_TYPE_REGRESSION
CHUNK_TYPE_INVALID
```

规则是：

```text
窗口内 stage 完全不变:
  INTRA_STAGE

窗口内 stage 只上升、不下降:
  FORWARD_TRANSITION

窗口内出现 stage 下降:
  REGRESSION

失败 episode 中不满足可挖条件:
  INVALID
```

---

## 14. 边界 Chunk 筛选

边界 chunk 是跨越阶段边界的 chunk。

当前默认边界模式是：

```text
unique_stage_boundary
```

它会找到每个新阶段第一次被达到的位置。

例如：

```text
0 -> 1 的第一次边界
1 -> 2 的第一次边界
2 -> 3 的第一次边界
3 -> 4 的第一次边界
```

对于每个边界位置 `b`，候选 chunk start 需要满足：

```text
s < b < s + K
```

也就是说，这个 chunk 必须覆盖边界。

然后按：

```text
A_chunk(s)
```

排序，并经过 temporal NMS，保留：

```text
boundary_top_k
```

个最好的边界 chunk。

当前常用设置是：

```text
boundary_top_k = 1
```

也就是每个阶段边界只保留一个最佳 chunk。

---

## 15. 阶段内部 Chunk 筛选

阶段内部 chunk 指的是：

```text
CHUNK_TYPE_INTRA_STAGE
```

这些 chunk 不跨阶段，只在某个阶段内部发生。

SACM 会按照：

```text
episode_index
task_index
stage
```

进行分组。

每个组内按：

```text
A_chunk(s)
```

从高到低排序。

然后保留：

```text
ceil(num_candidates * stage_top_ratio)
```

个 chunk。

例如：

```text
stage_top_ratio = 0.15
```

表示每个 episode-stage 组内保留 top 15% 的 intra-stage chunk。

如果设置了：

```text
stage_top_k > 0
```

则优先使用固定 top-k。

---

## 16. 可选实现：全局 Chunk 筛选

除了默认的 stage-aware SACM，当前代码还支持一个更简单的对照分支：

```bash
lerobot-stage-chunk-mine \
  ... \
  --mining.stage_aware=false \
  --mining.global_top_ratio=0.15
```

该模式下仍然会：

```text
1. 使用 value model 输出每帧 value。
2. 对每条 episode 的 value 做归一化。
3. 对所有合法 K-step 滑窗计算 GAE-style chunk advantage。
```

但它不会使用：

```text
episode-stage 分组
boundary chunk
failure episode 的阶段上升约束
```

而是对每条 episode 内所有合法 chunk 起点直接按照：

```text
A_chunk(s)
```

从高到低排序，然后保留：

```text
ceil(num_episode_chunks * global_top_ratio)
```

个 chunk。

如果设置：

```text
global_top_k > 0
```

则每条 episode 固定保留 top-k 个全局最高 advantage chunk。

这个模式适合作为 baseline / ablation：它只验证“value-based chunk advantage 筛选”本身，不引入分阶段均衡和阶段边界偏置。

---

## 17. 最新实现：优势 Chunk 展开为帧级标签

这是当前最重要的实现语义。

SACM 现在区分两个字段：

```text
chunk_start_indicator
indicator
```

### 17.1 chunk_start_indicator

`chunk_start_indicator` 只标记被选中的 chunk 起点。

如果选中了从 `s` 开始的 chunk：

```text
chunk_start_indicator[s] = 1
```

其他帧为 0。

这个字段只用于：

```text
统计
可视化
分析 selected chunk start
```

不应该直接用于 policy 训练。

### 17.2 indicator

`indicator` 是真正用于训练的帧级二值标签。

如果选中了从 `s` 开始、长度为 `K` 的 chunk，则：

```text
indicator[s : s + K] = 1
```

也就是：

```text
indicator[s]      = 1
indicator[s + 1]  = 1
...
indicator[s+K-1]  = 1
```

如果：

```text
K = 50
```

那么这个 chunk 内 50 帧全部标 1。

这才是正确的训练语义：

```text
优势 chunk 内的所有帧 -> Advantage: positive
其他帧                 -> Advantage: negative
```

如果多个 selected chunk 重叠，则 `indicator` 是这些 chunk 的并集。

因此：

```text
positive_frames <= selected_chunks * K
```

因为重叠区域不会重复计数。

---

## 18. SACM 输出字段

SACM 会在数据集里写入一组字段，前缀由：

```text
--mining.output_prefix
```

指定。

例如使用：

```text
complementary_info.vgsacm_sacm_threshold_r015_chunkmask
```

则输出字段包括：

```text
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.normalized_value
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.completion
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.stage
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.chunk_advantage
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.chunk_type
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.chunk_stage
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.chunk_start_indicator
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.chunk_start_role
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.boundary_id
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.indicator
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.weight
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.selection_role
```

policy 训练时应该使用：

```text
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.indicator
```

不要用：

```text
complementary_info.vgsacm_sacm_threshold_r015_chunkmask.chunk_start_indicator
```

否则会退回到旧问题：只有 chunk 起点是 positive，chunk 内其他帧仍然是 negative。

---

## 19. SACM Report

SACM 会输出报告：

```text
outputs/stage_chunk_mining/<RUN_NAME>/stage_chunk_mining_report.json
```

重要字段包括：

```text
selected_chunks
positive_frames
positive_frame_ratio
valid_chunk_starts
selected_ratio
selection_mode
intra_candidates
intra_selected
global_candidates
global_selected
boundary_count
boundary_candidates
boundary_selected
per_task_stage
chunk_start_role_counts
selection_role_counts
```

含义如下：

```text
selected_chunks:
  被选中的 chunk 起点数量。

positive_frames:
  被 selected chunks 覆盖到的正样本帧数量。

positive_frame_ratio:
  positive_frames / total_frames。

valid_chunk_starts:
  合法 chunk 起点数量。

selected_ratio:
  selected_chunks / valid_chunk_starts。

selection_mode:
  当前筛选模式。stage_aware 表示分阶段 SACM；global_top 表示每条 episode 内全局 top-ratio 筛选。

global_candidates:
  stage_aware=false 时参与全局排序的合法 chunk 起点数量。

global_selected:
  stage_aware=false 时被全局 top-ratio / top-k 选中的 chunk 起点数量。
```

最新实现后，训练标签密度主要看：

```text
positive_frame_ratio
```

如果它太小，`Advantage: positive` 信号可能太弱。  
如果它太大，说明优势 chunk 覆盖了太多数据，positive/negative 区分可能变弱。

---

## 20. ACP Prompt 注入训练

policy 训练入口是：

```text
src/lerobot/scripts/lerobot_train.py
```

ACP prompt 注入逻辑在：

```text
src/lerobot/rl/acp_hook.py
src/lerobot/rl/acp_tags.py
```

训练时打开：

```text
--acp.enable=true
```

并指定：

```text
--acp.indicator_field=<SACM_INDICATOR_FIELD>
```

示例：

```bash
lerobot-train \
  --dataset.repo_id=<DATASET_REPO_ID> \
  --dataset.root=<DATASET_ROOT> \
  --policy.path=<BASE_POLICY_CHECKPOINT> \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.use_amp=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --steps=5000 \
  --batch_size=16 \
  --save_freq=2500 \
  --eval_freq=0 \
  --wandb.enable=false \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.vgsacm_sacm_threshold_r015_chunkmask.indicator \
  --acp.indicator_dropout_prob=0.0 \
  --output_dir=outputs/train/sacm_threshold_r015_chunkmask_pi05_5000
```

ACP hook 会在 policy preprocessor 之前修改 batch 中的 `task` 字段。

如果：

```text
indicator = 1
```

则：

```text
task = task + "\nAdvantage: positive"
```

如果：

```text
indicator = 0
```

则：

```text
task = task + "\nAdvantage: negative"
```

如果设置：

```text
--acp.indicator_dropout_prob=0.3
```

那么 30% 概率会不加 advantage tag，保留原始 task。

诊断阶段建议先用：

```text
--acp.indicator_dropout_prob=0.0
```

这样训练样本只有两类：

```text
Advantage: positive
Advantage: negative
```

更容易判断模型是否真的学到了 advantage 条件。

---

## 21. PI05 如何接收 Advantage Prompt

`pi05` 的文本处理逻辑在：

```text
src/lerobot/policies/pi05/processor_pi05.py
```

它会把 task 和 robot state 拼成：

```text
Task: <task text>, State: <discretized robot state>;
Action:
```

如果 ACP 注入了：

```text
Advantage: positive
```

最终 prompt 类似：

```text
Task: The left arm grasps ... Advantage: positive, State: ...
Action:
```

所以 `Advantage: positive` 确实会进入语言 token。

当前 pi05 训练配置通常是：

```text
freeze_vision_encoder=true
train_expert_only=true
```

含义是：

```text
冻结 PaliGemma/VLM 主体；
训练 action expert 和动作相关 projection。
```

所以 Advantage tag 虽然进入了语言条件，但它只是一个文本条件。如果正负样本区分不明显、positive 标签太少或太多，模型可能仍然会忽略这个 tag。

---

## 22. 推理阶段

推理阶段不会运行 SACM。

推理阶段也不会运行 value model。

推理只做：

```text
当前观测 + task + Advantage: positive
  -> policy
  -> action chunk
```

async 推理链路是：

```text
robot_client -> policy_server -> pi05 policy -> action chunk
```

推理时 task 应该写成：

```text
The left arm grasps the metal cup, the right arm grasps the plastic bottle, and then the right arm pours water from the bottle into the cup held by the left arm.
Advantage: positive
```

这叫：

```text
advantage-conditioned inference
```

也就是优势条件推理。

它不是：

```text
在线 value inference
在线 SACM mining
在线 RL
```

---

## 23. 建议的诊断实验

为了判断 Advantage 条件是否真的被模型学到，建议比较同一个 checkpoint 下三种 prompt：

```text
1. 原始 task，不加 Advantage
2. task + Advantage: positive
3. task + Advantage: negative
```

可以做两类诊断。

### 22.1 离线 action sensitivity

固定同一个 observation，分别输入：

```text
no Advantage
Advantage: positive
Advantage: negative
```

比较输出 action chunk 的差异：

```text
L2 distance
夹爪维度差异
左右臂末端方向差异
前几步动作方向
```

如果离线 action 差异都很小，说明模型基本忽略了 Advantage tag。

### 22.2 真机 rollout 对比

在机器人上分别测试：

```text
no Advantage
Advantage: positive
Advantage: negative
```

观察是否出现明显行为差异。

如果三者现象一样，可能原因包括：

```text
positive_frame_ratio 太低或太高；
positive/negative chunk 行为差异不明显；
value model 只学到了进度，没有学到成功因果；
所有 episode 都是 success，缺少失败对照；
VLM 冻结，Advantage tag 条件较弱；
训练用了 indicator dropout，no Advantage 变成混合条件；
动作维度、夹爪方向或相机 key 存在问题。
```

---

## 24. 当前实现中的关键字段语义

当前实现中，几个字段的语义如下：

```text
indicator:
  帧级 positive/negative 标签。
  这是 policy training 应该使用的字段。

chunk_start_indicator:
  被选中 chunk 的起点。
  只用于统计、分析和可视化。

selection_role:
  帧级 selection role。
  chunk 展开后，chunk 内帧都会带对应 role。

chunk_start_role:
  起点级 selection role。
  只描述被选中的 chunk start 来自 boundary 还是 intra-stage。

weight:
  当前被选中帧权重。
  目前 selected chunk 覆盖帧为 1，其余为 0。
```

训练命令中应该使用：

```text
--acp.indicator_field=<output_prefix>.indicator
```

不要使用：

```text
--acp.indicator_field=<output_prefix>.chunk_start_indicator
```

---

## 25. 修改 SACM 标签后哪些步骤要重跑

如果 value model 已经训练好，且数据集里已经有：

```text
complementary_info.value_sacm
```

那么修改 SACM label 逻辑后，不需要重训 value model。

需要重跑：

```text
1. lerobot-stage-chunk-mine
2. lerobot-train
3. 推理/评估
```

不需要重跑：

```text
1. lerobot-value-train
2. lerobot-value-infer
```

除非你换了 value model 或 value 字段。

推荐使用新的 output prefix，避免和旧的“只标 chunk 起点”的实验混在一起：

```text
complementary_info.vgsacm_sacm_threshold_r015_chunkmask
```

然后训练时使用：

```text
--acp.indicator_field=complementary_info.vgsacm_sacm_threshold_r015_chunkmask.indicator
```

---

## 26. 方法总结

SACM 当前实现可以总结为：

```text
1. 训练或复用 pistar06 value model。
2. 对数据集每一帧预测 value。
3. 对每条 episode 的 value 做归一化。
4. 根据 value completion 划分 M 个阶段。
5. 成功 episode 使用首次达到阈值的单调阶段。
6. 失败 episode 只在首次下降前的上升阶段挖 chunk。
7. 用 K-step 滑窗枚举 action chunks。
8. 用 GAE-style 的 TD error 累加计算 chunk advantage。
9. 默认 stage_aware=true 时，阶段边界处保留 boundary_top_k 个优势 chunk。
10. 默认 stage_aware=true 时，每个 episode-stage 内保留 top stage_top_ratio 的 intra-stage chunk。
11. 可选 stage_aware=false 时，每条 episode 内全局保留 top global_top_ratio 的高优势 chunk。
12. 把被选中 chunk 的 K 帧全部标为 indicator=1。
13. 训练 policy 时把 indicator=1 转成 Advantage: positive。
14. 把 indicator=0 转成 Advantage: negative。
15. 推理时输入 Advantage: positive，得到优势条件动作 chunk。
```

最关键的最新语义是：

```text
选中 chunk 起点 s:
  chunk_start_indicator[s] = 1
  indicator[s : s + K] = 1
```

也就是说，policy 训练看到的是完整优势动作片段，而不是只有片段第一帧。
