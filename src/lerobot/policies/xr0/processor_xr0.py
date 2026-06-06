#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre- and post-processor pipelines for XR0 policy (Qwen3-VL + DiT + Flow Matching).

The preprocessor handles:
1. Feature renaming
2. Batch dimension addition
3. State and action normalization (images kept as IDENTITY - raw [0,1])
4. Moving to device

The Qwen3-VL-specific processing (image encoding, chat template, tokenization)
is handled inside the XR0Model, following the same pattern as Wall-X.
"""

from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.xr0.configuration_xr0 import XR0Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_xr0_pre_post_processors(
    config: XR0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct pre- and post-processor pipelines for XR0 policy.

    The pre-processing pipeline:
    1. Renames features to match pretrained configurations
    2. Adds a batch dimension
    3. Normalizes state (MEAN_STD) and action (MEAN_STD); images are IDENTITY
    4. Moves to device

    The Qwen3-VL-specific processing (chat template, tokenization, image encoding)
    is handled inside XR0Model, following the Wall-X pattern.

    The post-processing pipeline:
    1. Unnormalizes actions back to original scale
    2. Moves to CPU

    Args:
        config: XR0Config with normalization settings.
        dataset_stats: Dataset statistics for normalization.

    Returns:
        Tuple of (pre_processor_pipeline, post_processor_pipeline).
    """
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
