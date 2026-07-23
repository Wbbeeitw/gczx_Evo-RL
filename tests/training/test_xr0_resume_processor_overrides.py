from types import SimpleNamespace

import torch

from lerobot.scripts.lerobot_train import _make_pretrained_processor_overrides


def _make_config(*, policy_type: str, resume: bool):
    return SimpleNamespace(
        policy=SimpleNamespace(type=policy_type),
        resume=resume,
        rename_map={"camera": "observation.images.camera"},
    )


def _make_policy():
    return SimpleNamespace(
        config=SimpleNamespace(
            input_features={"observation.state": object()},
            output_features={"action": object()},
            normalization_mapping={"ACTION": "MEAN_STD"},
        )
    )


def test_xr0_resume_only_overrides_steps_saved_by_xr0_processors():
    stats = {"action": {"mean": torch.zeros(1), "std": torch.ones(1)}}

    preprocessor_overrides, postprocessor_overrides = _make_pretrained_processor_overrides(
        _make_config(policy_type="xr0", resume=True),
        _make_policy(),
        torch.device("cuda"),
        stats,
    )

    assert preprocessor_overrides == {
        "device_processor": {"device": "cuda"},
        "rename_observations_processor": {
            "rename_map": {"camera": "observation.images.camera"}
        },
    }
    assert postprocessor_overrides == {}


def test_other_pretrained_policies_keep_generic_normalization_overrides():
    stats = {"action": {"mean": torch.zeros(1), "std": torch.ones(1)}}

    preprocessor_overrides, postprocessor_overrides = _make_pretrained_processor_overrides(
        _make_config(policy_type="pi05", resume=True),
        _make_policy(),
        torch.device("cuda"),
        stats,
    )

    assert preprocessor_overrides["normalizer_processor"]["stats"] is stats
    assert postprocessor_overrides["unnormalizer_processor"]["stats"] is stats
