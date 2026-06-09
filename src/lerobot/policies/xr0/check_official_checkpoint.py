#!/usr/bin/env python

"""Audit official Xiaomi XR0 checkpoint loading in the LeRobot wrapper.

Example:
    python -m lerobot.policies.xr0.check_official_checkpoint \
        --checkpoint F:/Xiaomi-Robotics-0/xr0/pretrained_ckpt/xr0_pretrained.pt
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from lerobot.policies.xr0.configuration_xr0 import XR0Config
from lerobot.policies.xr0.native import XR0 as NativeXR0


def _load_official_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if "module" in payload:
        state_dict = payload["module"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    out = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key.removeprefix("model.")
        elif key.startswith("_forward_module.model."):
            key = key.removeprefix("_forward_module.model.")
        out[key] = value
    return out


def _prefix_counts(keys: list[str]) -> dict[str, int]:
    return dict(Counter(key.split(".", 1)[0] for key in keys))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--qwen-variant", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--qwen-attn-implementation", default="flash_attention_2")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config = XR0Config(
        qwen_variant=args.qwen_variant,
        qwen_attn_implementation=args.qwen_attn_implementation,
        dtype=args.dtype,
    )
    model = NativeXR0(
        state_shape=(1, config.max_state_dim),
        action_shape=(config.chunk_size, config.max_action_dim),
        dit_num_layers=config.dit_num_layers,
        dit_hidden_size=config.dit_hidden_size,
        num_steps=config.num_inference_steps,
        flow_sampling=config.flow_sampling,
        training_repeat=config.training_repeat,
        enable_freq=config.enable_freq,
        prefix_mask_prob=config.prefix_mask_prob,
        async_train=config.async_train,
        qwen_variant=config.qwen_variant,
        qwen_attn_implementation=config.qwen_attn_implementation,
        dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float32,
    )

    state_dict = _load_official_state_dict(args.checkpoint)
    info = model.load_state_dict(state_dict, strict=args.strict)

    print(f"checkpoint: {args.checkpoint}")
    print(f"loaded tensors: {len(state_dict)}")
    print(f"missing keys: {len(info.missing_keys)}")
    print(f"unexpected keys: {len(info.unexpected_keys)}")
    print(f"missing prefixes: {_prefix_counts(info.missing_keys)}")
    print(f"unexpected prefixes: {_prefix_counts(info.unexpected_keys)}")
    if info.missing_keys:
        print("first missing keys:")
        for key in info.missing_keys[:20]:
            print(f"  {key}")
    if info.unexpected_keys:
        print("first unexpected keys:")
        for key in info.unexpected_keys[:20]:
            print(f"  {key}")


if __name__ == "__main__":
    main()
