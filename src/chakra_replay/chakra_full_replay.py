# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

from chakra_replay.config import FullReplayConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Chakra replayer (communication & computation)")
    parser.add_argument("--config", type=Path, required=True, help="Path to a TOML configuration file.")
    args = parser.parse_args()

    config = FullReplayConfig.load_from_toml(args.config)
    full_replayer = ChakraFullReplayer(config)
    full_replayer.run()


if __name__ == "__main__":
    main()
