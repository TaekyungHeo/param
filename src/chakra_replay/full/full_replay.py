# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from enum import Enum

from param_bench.train.compute.python.lib import pytorch as lib_pytorch
from param_bench.train.compute.python.lib.init_helper import load_modules
from param_bench.train.compute.python.workloads import pytorch as workloads_pytorch

# grid and split_scan_grid are dynamically loaded
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid  # noqa


class TensorAllcationMode(Enum):
    """
    Enum to represent the tensor allocation mode
    """

    # Allocate input tensors that can not be generated when replaying the trace
    # at the beginning and reuse them for all iterations.
    PRE_ALLOCATE = 1

    # Allocate tensors on the fly and free them after they are out of scope
    LAZY_ALLOCATE = 2


def main():
    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    replay_manager = ExgrReplayManager()
    replay_manager.readComputeArgs()
    replay_manager.initBench()
    benchmark_result = replay_manager.benchTime()
    if benchmark_result["execution finished"]:
        logging.info("Replay finished successfully.")
        sys.exit(0)
    else:
        logging.info("Replay failed.")
        sys.exit(-1)


if __name__ == "__main__":
    main()
