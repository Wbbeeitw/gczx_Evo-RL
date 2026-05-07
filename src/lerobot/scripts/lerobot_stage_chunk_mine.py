#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Mine stage-aware high-advantage action chunks from value-scored LeRobot datasets."""

from lerobot.configs import parser
from lerobot.rl.stage_chunk_mining.config import StageChunkMinePipelineConfig
from lerobot.rl.stage_chunk_mining.pipeline import run_stage_chunk_mining
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@parser.wrap()
def stage_chunk_mine(cfg: StageChunkMinePipelineConfig):
    init_logging()
    return run_stage_chunk_mining(cfg)


def main():
    register_third_party_plugins()
    stage_chunk_mine()


if __name__ == "__main__":
    main()

