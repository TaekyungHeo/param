# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import List, Tuple, Union

import torch

from chakra_replay.common import ChakraCommTraceParser
from chakra_replay.common.utils import param_to_comm_name, read_comms_env_vars
from chakra_replay.config import CommReplayConfig

from .backend import BackendContext, PyTorchDistBackend, PyTorchTPUBackend
from .comm_op import CommOp
from .comm_utils import CollArgs
from .tensor_allocator import TensorAllocator

SUPPORTED_P2P_OPS = ["send", "recv", "isend", "irecv"]


class ChakraCommReplayer:
    """Replay collective communication operations for performance evaluation."""

    def __init__(self, config: CommReplayConfig) -> None:
        self.config = config
        self.coll_args = CollArgs()
        self.init_val = 1
        self.shrink = False
        self.reuse_tensors = False

        self.comm_ops = self.load_trace_operations(config.trace.directory)

        backend_ctx = read_comms_env_vars()
        self.backend = self._initialize_backend(config.replay.backend.backend, backend_ctx)
        self.backend.say_hello()

        self.tensor_allocator = TensorAllocator(self.backend, self.init_val, self.coll_args)

    def _initialize_backend(self, backend_name: str, backend_ctx: BackendContext):
        if backend_name == "pytorch-dist":
            return PyTorchDistBackend(backend_ctx)
        elif backend_name == "pytorch-tpu":
            return PyTorchTPUBackend(backend_ctx)
        logging.error(f"Unsupported backend: {backend_name}")
        raise ValueError(f"Unsupported backend: {backend_name}")

    def load_trace_operations(self, dir_path: Path) -> List[CommOp]:
        try:
            parser = ChakraCommTraceParser(self.backend.context.world_size, self.backend.context.global_rank, dir_path)
            ops = parser.parse()
            logging.info(f"Loaded {len(ops)} trace operations.")
            return ops
        except Exception as e:
            logging.exception("Failed to load trace operations.")
            raise RuntimeError("Failed to load trace operations.") from e

    def run(self) -> None:
        self.reset_communications()
        for idx in range(self.config.replay.num_replays):
            if self.backend.context.global_rank == 0:
                logging.info(f"Starting replay #{idx}")
            self.process_operations()
            self.reset_communications()
            self.backend.sync_barrier(self.coll_args)
        self.backend.clear_memory(self.coll_args)
        self.backend.barrier_all_ranks()

    def reset_communications(self) -> None:
        self.coll_args.group = self.backend.get_default_group()

    def process_operations(self) -> None:
        for idx, op in enumerate(self.comm_ops):
            self.process_single_operation(op, idx)

    def process_single_operation(self, op: CommOp, idx: int) -> None:
        op_name = param_to_comm_name(op.name)
        self.get_communication_group(op)
        inp, out = self.prepare_communication(op, regenerate_tensors=not self.reuse_tensors)
        self.coll_args.input_tensor = inp
        self.coll_args.output_tensor = out
        self.execute_collective(op_name, op)

    def get_communication_group(self, op: CommOp) -> Tuple[int, str]:
        group = (
            self.coll_args.groups.get(op.pg_id)
            if op.pg_id is not None and not self.shrink
            else self.backend.get_default_group()
        )
        return self.backend.get_group_rank(group)

    def prepare_communication(
        self, op: CommOp, regenerate_tensors: bool = True
    ) -> Tuple[torch.Tensor, Union[List[torch.Tensor], torch.Tensor]]:
        self._update_comm_group(op)
        return self.tensor_allocator.generate_io_tensors(op, regenerate_tensors)

    def _update_comm_group(self, op: CommOp) -> None:
        if op.pg_id is not None and not self.shrink:
            self.coll_args.group = self.coll_args.groups[op.pg_id]
            self.coll_args.world_size = op.world_size
        else:
            self.coll_args.group = self.backend.get_default_group()
            self.coll_args.world_size = self.world_size

    def execute_collective(self, op_name: str, op: CommOp) -> None:
        try:
            collective_method = getattr(self.backend, op_name)
        except AttributeError as err:
            logging.error(f"Collective operation {op_name} not implemented in backend.")
            raise NotImplementedError(f"Collective operation {op_name} not implemented.") from err

        try:
            collective_method(self.coll_args)
        except Exception as e:
            logging.exception(f"Operation {op_name} failed.")
            raise RuntimeError(f"Operation {op_name} failed.") from e
