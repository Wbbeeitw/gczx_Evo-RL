#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import pickle  # nosec B403: internal trusted transport
import sys
import time
from concurrent import futures
from contextlib import nullcontext
from pathlib import Path
from queue import Empty, Queue

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import grpc
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, receive_bytes_in_chunks
from lerobot.utils.constants import ACTION
from lerobot.utils.control_utils import prepare_observation_for_inference

from tactile_client_server_common import (
    DEFAULT_ORDERED_IMAGE_KEYS,
    EncodedTactileObservation,
    POLICY_POSTPROCESSOR_FILENAME,
    POLICY_PREPROCESSOR_FILENAME,
    TactilePolicyRequestConfig,
    TactileTimedAction,
    configure_policy_features_for_tactile_metadata,
    decode_observation_packet,
    parse_csv_items,
    resolve_dataset_root,
    resolve_device,
)

# 解析服务端启动参数，包括模型、数据集、监听地址和每次返回的动作数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone tactile pi05 policy server using the existing LeRobot gRPC transport.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory containing policy config + weights.")
    parser.add_argument("--dataset-repo-id", required=True, help="Dataset repo id used during training.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root or dataset parent root.")
    parser.add_argument("--device", default="auto", help="Inference device: auto/cpu/cuda/mps.")
    parser.add_argument(
        "--ordered-image-keys",
        default=",".join(DEFAULT_ORDERED_IMAGE_KEYS),
        help="Comma-separated image key order expected by the policy.",
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=0,
        help="Default action count returned per request. 0 means use policy.config.n_action_steps.",
    )
    parser.add_argument(
        "--obs-queue-timeout-s",
        type=float,
        default=1.0,
        help="How long GetActions waits for an incoming observation.",
    )
    parser.add_argument("--rebuild-processors", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


# 远程触觉策略推理服务端：接收观测包，执行推理，并返回动作块。
class TactilePolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    # 加载 checkpoint、数据集元信息、处理器和策略，并初始化单槽位观测队列。
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = resolve_device(args.device)
        self.device_type = torch.device(self.device).type
        self.ordered_image_keys = parse_csv_items(args.ordered_image_keys)
        # 这里只保留一个最新观测，优先降低端到端延迟，而不是保证每帧都参与推理。
        self.observation_queue: Queue[EncodedTactileObservation] = Queue(maxsize=1)
        self.actions_per_chunk = 0
        self.use_amp = self.device_type == "cuda" and not args.disable_amp

        # 根据数据集元信息重建与训练阶段一致的输入特征顺序。
        dataset_root = resolve_dataset_root(args.dataset_root, args.dataset_repo_id)
        logging.info("Resolved dataset root: %s", dataset_root)
        self.ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)

        policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
        if policy_cfg.type != "pi05":
            raise ValueError(f"This server currently supports only pi05, got '{policy_cfg.type}'.")
        policy_cfg.pretrained_path = args.checkpoint
        policy_cfg.device = self.device
        configure_policy_features_for_tactile_metadata(policy_cfg, self.ds_meta, self.ordered_image_keys)

        self.policy = make_policy(policy_cfg, ds_meta=self.ds_meta)
        self.preprocessor, self.postprocessor = self._make_processors(policy_cfg)
        self.policy.eval()
        self.max_actions_per_chunk = int(self.policy.config.chunk_size)
        requested_actions_per_chunk = args.actions_per_chunk or int(self.policy.config.n_action_steps)
        self.default_actions_per_chunk = min(requested_actions_per_chunk, self.max_actions_per_chunk)
        self.actions_per_chunk = self.default_actions_per_chunk

        logging.info(
            "Policy server ready | device=%s actions_per_chunk=%d max_chunk=%d",
            self.device,
            self.actions_per_chunk,
            self.max_actions_per_chunk,
        )

    # 优先复用 checkpoint 里保存的处理器；必要时再用配置和统计信息重建。
    def _make_processors(
        self,
        policy_cfg: PreTrainedConfig,
    ) -> tuple[
        PolicyProcessorPipeline[dict, dict],
        PolicyProcessorPipeline[PolicyAction, PolicyAction],
    ]:
        checkpoint_dir = Path(self.args.checkpoint)
        has_saved_processors = (
            (checkpoint_dir / POLICY_PREPROCESSOR_FILENAME).exists()
            and (checkpoint_dir / POLICY_POSTPROCESSOR_FILENAME).exists()
        )
        processor_overrides = {"device_processor": {"device": self.device}}
        if has_saved_processors and not self.args.rebuild_processors:
            logging.info("Loading saved pre/post processors from checkpoint.")
            return make_pre_post_processors(
                policy_cfg,
                pretrained_path=str(checkpoint_dir),
                preprocessor_overrides=processor_overrides,
                postprocessor_overrides=processor_overrides,
                dataset_stats=self.ds_meta.stats,
            )

        logging.info("Rebuilding pre/post processors from config and dataset stats.")
        return make_pre_post_processors(policy_cfg, dataset_stats=self.ds_meta.stats)

    # 在新客户端准备就绪时重置服务端状态，避免沿用上一个会话的残留队列和配置。
    def _reset_state(self) -> None:
        self.observation_queue = Queue(maxsize=1)
        self.actions_per_chunk = self.default_actions_per_chunk

    # RPC：客户端声明“我已准备好”，服务端据此重置内部状态。
    def Ready(self, request, context):  # noqa: N802
        logging.info("Client ready: %s", context.peer())
        self._reset_state()
        return services_pb2.Empty()

    # RPC：接收客户端发送的策略指令，目前主要用于覆盖每次返回的动作块长度。
    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not request.data:
            return services_pb2.Empty()
        try:
            setup = pickle.loads(request.data)  # nosec B301: same workspace controlled scripts
        except Exception as exc:
            logging.warning("Failed to deserialize policy setup from %s: %s", context.peer(), exc)
            return services_pb2.Empty()

        if not isinstance(setup, TactilePolicyRequestConfig):
            logging.warning("Ignoring unsupported setup payload type: %s", type(setup).__name__)
            return services_pb2.Empty()

        if setup.actions_per_chunk is not None and setup.actions_per_chunk > 0:
            self.actions_per_chunk = min(int(setup.actions_per_chunk), self.max_actions_per_chunk)
        logging.info("Updated per-client actions_per_chunk=%d", self.actions_per_chunk)
        return services_pb2.Empty()

    # RPC：接收分块上传的观测字节流，反序列化后放入单槽位观测队列。
    def SendObservations(self, request_iterator, context):  # noqa: N802
        received_bytes = receive_bytes_in_chunks(
            request_iterator,
            queue=None,
            shutdown_event=_NoOpShutdownEvent(),
            log_prefix="[SERVER] observation",
        )
        if not received_bytes:
            return services_pb2.Empty()

        packet = pickle.loads(received_bytes)  # nosec B301: same workspace controlled scripts
        if not isinstance(packet, EncodedTactileObservation):
            raise TypeError(f"Expected EncodedTactileObservation, got {type(packet).__name__}")

        # 如果队列已满，则丢弃旧观测，只保留最新的一帧。
        if self.observation_queue.full():
            try:
                self.observation_queue.get_nowait()
            except Empty:
                pass
        self.observation_queue.put(packet)
        logging.debug("Queued observation #%d", packet.get_timestep())
        return services_pb2.Empty()

    # RPC：取出最新观测执行推理，并把动作块序列化后返回给客户端。
    def GetActions(self, request, context):  # noqa: N802
        try:
            packet = self.observation_queue.get(timeout=self.args.obs_queue_timeout_s)
        except Empty:
            return services_pb2.Actions(data=b"")

        start_t = time.perf_counter()
        action_chunk = self._predict_action_chunk(packet)
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        logging.info(
            "Observation #%d -> action chunk %d | latency=%.2fms",
            packet.get_timestep(),
            len(action_chunk),
            latency_ms,
        )
        return services_pb2.Actions(data=pickle.dumps(action_chunk))  # nosec B301: internal transport only

    # 完成一次完整的服务端推理：解码观测、预处理、策略预测、后处理和封装返回值。
    def _predict_action_chunk(self, packet: EncodedTactileObservation) -> list[TactileTimedAction]:
        observation_np = decode_observation_packet(packet)

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device_type) if self.use_amp else nullcontext(),
        ):
            # 先把原始 numpy 观测整理成策略推理期所需的输入格式。
            observation = prepare_observation_for_inference(
                observation=observation_np,
                device=torch.device(self.device),
                task=packet.task_prompt,
                robot_type=packet.robot_type,
            )
            observation = self.preprocessor(observation)
            action_tensor = self.policy.predict_action_chunk(observation)

            # 统一动作张量形状，并按当前配置截取本次需要返回的动作数量。
            if action_tensor.ndim != 3:
                action_tensor = action_tensor.unsqueeze(0)
            action_tensor = action_tensor[:, : self.actions_per_chunk, :]

            # 对动作块中的每一步动作分别做后处理，恢复到可执行动作空间。
            processed_actions = []
            for i in range(action_tensor.shape[1]):
                processed_actions.append(self.postprocessor(action_tensor[:, i, :]))
            action_chunk = torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu().numpy().astype("float32")

        return [
            TactileTimedAction(
                timestamp=packet.get_timestamp(),
                timestep=packet.get_timestep() + index,
                action=action,
            )
            for index, action in enumerate(action_chunk)
        ]


# 为分块接收工具提供一个永远不会触发关闭的占位事件对象。
class _NoOpShutdownEvent:
    # 始终返回 False，表示当前接收过程不需要被外部中断。
    def is_set(self) -> bool:
        return False


# 进程入口：构建 gRPC 服务端并开始监听推理请求。
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    server_impl = TactilePolicyServer(args)

    # 使用线程池承载 gRPC 请求处理，并注册推理服务实现。
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=grpc_channel_options(),
    )
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_impl, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logging.info("Tactile policy server listening on %s:%d", args.host, args.port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
