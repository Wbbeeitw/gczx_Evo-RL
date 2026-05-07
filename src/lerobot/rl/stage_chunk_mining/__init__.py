#!/usr/bin/env python

"""Value-guided stage-aware chunk mining for VLA post-training."""

from __future__ import annotations

__all__ = [
    "StageChunkDatasetConfig",
    "StageChunkMinePipelineConfig",
    "StageChunkMiningConfig",
    "mine_stage_chunks",
    "run_stage_chunk_mining",
]


def __getattr__(name: str):
    if name in {"StageChunkDatasetConfig", "StageChunkMinePipelineConfig", "StageChunkMiningConfig"}:
        from lerobot.rl.stage_chunk_mining.config import (
            StageChunkDatasetConfig,
            StageChunkMinePipelineConfig,
            StageChunkMiningConfig,
        )

        return {
            "StageChunkDatasetConfig": StageChunkDatasetConfig,
            "StageChunkMinePipelineConfig": StageChunkMinePipelineConfig,
            "StageChunkMiningConfig": StageChunkMiningConfig,
        }[name]
    if name == "mine_stage_chunks":
        from lerobot.rl.stage_chunk_mining.selection import mine_stage_chunks

        return mine_stage_chunks
    if name == "run_stage_chunk_mining":
        from lerobot.rl.stage_chunk_mining.pipeline import run_stage_chunk_mining

        return run_stage_chunk_mining
    raise AttributeError(name)
