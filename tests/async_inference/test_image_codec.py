import numpy as np
import pytest

from lerobot.async_inference.helpers import (
    EncodedImage,
    decode_observation_images,
    encode_observation_images,
)


def test_jpeg_observation_round_trip_preserves_shape_and_dtype():
    horizontal = np.linspace(0, 255, 64, dtype=np.uint8)
    image = np.stack(
        [
            np.tile(horizontal, (48, 1)),
            np.tile(horizontal[::-1], (48, 1)),
            np.full((48, 64), 127, dtype=np.uint8),
        ],
        axis=-1,
    )
    observation = {"right_tactile": image, "joint": 1.25, "task": "test"}

    encoded, encode_stats = encode_observation_images(observation, codec="jpeg", jpeg_quality=95)

    assert isinstance(encoded["right_tactile"], EncodedImage)
    assert encoded["joint"] == observation["joint"]
    assert encoded["task"] == observation["task"]
    assert encode_stats.image_count == 1
    assert encode_stats.raw_bytes == image.nbytes
    assert encode_stats.encoded_bytes < image.nbytes
    assert observation["right_tactile"] is image

    decoded, decode_stats = decode_observation_images(encoded)

    decoded_image = decoded["right_tactile"]
    assert decoded_image.shape == image.shape
    assert decoded_image.dtype == image.dtype
    assert np.abs(decoded_image.astype(np.int16) - image.astype(np.int16)).mean() < 8.0
    assert decode_stats.image_count == 1
    assert decode_stats.raw_bytes == image.nbytes


def test_raw_observation_transport_is_a_noop():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {"camera": image}

    encoded, stats = encode_observation_images(observation, codec="raw")

    assert encoded is observation
    assert stats.image_count == 0


@pytest.mark.parametrize("quality", [0, 101])
def test_jpeg_quality_is_validated(quality):
    with pytest.raises(ValueError, match="jpeg_quality"):
        encode_observation_images({}, codec="jpeg", jpeg_quality=quality)
