# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from chakra_replay.comm.comms_utils import CollArgs, CommArgs
from chakra_replay.comm.tensor_allocator import TensorAllocator


class DummyBackend:
    def __init__(self) -> None:
        from chakra_replay.comm.backend.backend_context import BackendContext

        self.context = BackendContext(world_size=4, global_rank=0, local_size=2, local_rank=0)
        self.context.comm_params = type("CommParams", (), {"device": "cpu"})
        self.context.memory_params = type("MemoryParams", (), {"init_value": 1, "dtype": torch.float32})

    def alloc_ones(self, shape: list[int], device: str, dtype: torch.dtype, scale: float) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype, device=device) * scale

    def alloc_random(self, shape: list[int], device: str, dtype: torch.dtype, scale: float) -> torch.Tensor:
        return torch.rand(shape, dtype=dtype, device=device) / scale


@pytest.fixture
def dummy_coll_args() -> CollArgs:
    ca = CollArgs()
    ca.world_size = 4
    ca.et_to_tensors = {}
    ca.groups = {}
    return ca


@pytest.fixture
def dummy_backend() -> DummyBackend:
    return DummyBackend()


@pytest.fixture
def tensor_allocator(dummy_backend: DummyBackend, dummy_coll_args: CollArgs) -> TensorAllocator:
    return TensorAllocator(dummy_backend, init_val=1.0, coll_args=dummy_coll_args)


class DummyCommArgs(CommArgs):
    def __init__(self) -> None:
        self.id = True
        self.comms = "all_gather"
        self.in_msg_size = 16
        self.out_msg_size = 16
        self.dtype = "float32"
        self.pg_id = None
        self.in_split = None
        self.out_split = None
        self.src_rank = 0
        self.dst_rank = 1
        self.req = None
        self.world_size = 4
        self.root = 0
        self.use_batch_p2p = False


@pytest.fixture
def dummy_comm_args() -> DummyCommArgs:
    return DummyCommArgs()


def test_compute_operation_hash(tensor_allocator: TensorAllocator, dummy_comm_args: DummyCommArgs) -> None:
    h = tensor_allocator.compute_operation_hash(dummy_comm_args)
    assert isinstance(h, int)


def test_create_io_tensors(tensor_allocator: TensorAllocator, dummy_comm_args: DummyCommArgs) -> None:
    inp, out = tensor_allocator.create_io_tensors(dummy_comm_args, regenerate=True)
    assert isinstance(inp, torch.Tensor)
    assert isinstance(out, (list, torch.Tensor))
