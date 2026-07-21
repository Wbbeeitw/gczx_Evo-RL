from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.xr0.modeling_xr0 import XR0Policy


RIGHT_TACTILE_IMAGE_KEYS = (
    "observation.images.left_ego",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.right_tactile",
)


class CapturingProcessor:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }


def make_prompt_policy(image_keys=RIGHT_TACTILE_IMAGE_KEYS, image_key_order=None):
    policy = XR0Policy.__new__(XR0Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        image_features=dict.fromkeys(image_keys, object()),
        image_key_order=image_key_order,
        image_key_descriptions={},
    )
    policy.processor = CapturingProcessor()
    policy.register_parameter(
        "_test_device_parameter",
        torch.nn.Parameter(torch.zeros(()), requires_grad=False),
    )
    return policy


def test_right_tactile_default_image_order_matches_native_xr0_views():
    policy = make_prompt_policy()

    assert policy._get_image_keys() == list(RIGHT_TACTILE_IMAGE_KEYS)


def test_explicit_image_order_rejects_missing_left_tactile_input():
    policy = make_prompt_policy(
        image_key_order=[
            *RIGHT_TACTILE_IMAGE_KEYS,
            "observation.images.left_tactile",
        ]
    )

    with pytest.raises(ValueError, match="observation.images.left_tactile"):
        policy._get_image_keys()


def test_right_tactile_prompt_uses_native_xr0_format():
    policy = make_prompt_policy()
    batch = {
        image_key: torch.full((1, 3, 2, 2), fill_value=index / 10)
        for index, image_key in enumerate(RIGHT_TACTILE_IMAGE_KEYS, start=1)
    }
    batch["task"] = [
        "Use the right gripper to pick up the yellow plastic bottle on the right "
        "and place it on the blue towel on the left"
    ]

    inputs = policy._prepare_vlm_inputs(batch)

    assert inputs["input_ids"].device.type == "cpu"
    assert policy.processor.kwargs == {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "padding": True,
        "images_kwargs": {"do_resize": False},
    }
    assert len(policy.processor.messages) == 1
    user_message, assistant_message = policy.processor.messages[0]
    assert user_message["role"] == "user"
    assert assistant_message == {
        "role": "assistant",
        "content": [{"type": "text", "text": "<cot></cot>"}],
    }

    content = user_message["content"]
    assert [item["type"] for item in content] == [
        "text",
        "text",
        "image",
        "text",
        "text",
        "image",
        "text",
        "text",
        "image",
        "text",
        "text",
        "image",
        "text",
        "text",
    ]
    assert [item["text"] for item in content if item["type"] == "text"] == [
        "The following observations are captured from multiple views.\n",
        "# Ego View\n",
        "\n",
        "# Left-Wrist View\n",
        "\n",
        "# Right-Wrist View\n",
        "\n",
        "# Right-Gripper Tactile View\n",
        "\n",
        "Generate robot actions for the task:\n"
        "Use the right gripper to pick up the yellow plastic bottle on the right "
        "and place it on the blue towel on the left /no_cot",
    ]
