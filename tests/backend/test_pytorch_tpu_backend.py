# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from chakra_replay.comm.backend.pytorch_tpu_backend import PyTorchTPUBackend


class DummyXM:
    def __init__(self) -> None:
        self.called = False

    def rendezvous(self, tag: str) -> None:
        self.called = True

    def mark_step(self) -> None:
        self.called = True

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return tensor

    def xla_device(self) -> str:
        return "xla:0"

    def _xla_real_device(self, device: str) -> str:
        return "TPU0"

    def get_local_ordinal(self) -> int:
        return 0

    def get_ordinal(self) -> int:
        return 0

    def xrt_world_size(self) -> int:
        return 8

    REDUCE_SUM = "SUM"
    REDUCE_MAX = "MAX"


@pytest.fixture
def dummy_xm(monkeypatch):
    dummy = DummyXM()
    monkeypatch.setattr("torch_xla.core.xla_model", dummy)
    monkeypatch.setattr(
        "torch_xla.distributed.xla_multiprocessing", type("dummy", (), {"spawn": lambda fn, args, nprocs: None})
    )
    return dummy


@pytest.fixture
def tpu_backend(backend_context, dummy_xm) -> PyTorchTPUBackend:
    from chakra_replay.comm.backend.pytorch_tpu_backend import PyTorchTPUBackend

    return PyTorchTPUBackend(backend_context)


def test_alloc_methods(tpu_backend: PyTorchTPUBackend) -> None:
    t_empty = tpu_backend.alloc_empty([2, 3], torch.float32, "cpu")
    t_ones = tpu_backend.alloc_ones([2, 2], torch.float32, "cpu")
    t_rand = tpu_backend.alloc_random([3, 3], torch.float32, "cpu")
    assert t_empty.shape == (2, 3)
    assert t_ones.shape == (2, 2) and torch.all(t_ones == 1)
    assert t_rand.shape == (3, 3)


def test_barrier(tpu_backend: PyTorchTPUBackend, dummy_xm) -> None:
    dummy_xm.called = False
    tpu_backend.barrier()
    assert dummy_xm.called


def test_complete_accel_ops(tpu_backend: PyTorchTPUBackend, dummy_xm) -> None:
    dummy_xm.called = False
    tpu_backend.complete_accel_ops()
    assert dummy_xm.called


def test_device_info(tpu_backend: PyTorchTPUBackend, dummy_xm) -> None:
    assert tpu_backend.device_str() == str(dummy_xm.xla_device())
    assert tpu_backend.hw_device() == dummy_xm._xla_real_device(dummy_xm.xla_device())
