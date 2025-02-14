# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from chakra_replay.comm.backend.pytorch_dist_backend import PyTorchDistBackend


@pytest.mark.parametrize("shape,dtype", [([2, 2], torch.float32), ([3, 3], torch.int32)])
def test_alloc_methods(pytorch_dist_backend: PyTorchDistBackend, shape: List[int], dtype: torch.dtype) -> None:
    t_empty = pytorch_dist_backend.alloc_empty(shape, dtype, "cpu")
    t_ones = pytorch_dist_backend.alloc_ones(shape, dtype, "cpu")
    t_rand = pytorch_dist_backend.alloc_random(shape, dtype, "cpu")
    assert t_empty.shape == tuple(shape)
    assert t_ones.shape == tuple(shape) and torch.all(t_ones == 1)
    assert t_rand.shape == tuple(shape)


def test_alloc_embedding_tables(pytorch_dist_backend: PyTorchDistBackend) -> None:
    emb = pytorch_dist_backend.alloc_embedding_tables(10, 3, torch.float32, "cpu")
    from torch.nn import EmbeddingBag

    assert isinstance(emb, EmbeddingBag)
    assert emb.weight.shape == (10, 3)


def test_clear_memory(pytorch_dist_backend: PyTorchDistBackend) -> None:
    # We assume clear_memory calls torch.cuda.empty_cache on GPU; on CPU, nothing happens.
    with patch("torch.cuda.empty_cache") as mock_empty:
        pytorch_dist_backend.clear_memory()
        if torch.cuda.is_available():
            mock_empty.assert_called_once()


def test_barrier_and_sync(pytorch_dist_backend: PyTorchDistBackend) -> None:
    with (
        patch.object(dist, "barrier") as mock_barrier,
        patch.object(pytorch_dist_backend, "complete_accel_ops") as mock_complete,
    ):
        pytorch_dist_backend.sync_barrier()
        mock_complete.assert_called()
        mock_barrier.assert_called_once()


def test_broadcast(pytorch_dist_backend: PyTorchDistBackend, dummy_tensor: torch.Tensor) -> None:
    with patch.object(dist, "broadcast") as mock_broadcast:
        pytorch_dist_backend.broadcast(dummy_tensor, root=0)
        mock_broadcast.assert_called_once()


def test_scatter(pytorch_dist_backend: PyTorchDistBackend, dummy_tensor: torch.Tensor) -> None:
    # Ensure scatter asserts when input is None.
    with pytest.raises(AssertionError):
        pytorch_dist_backend.scatter(None, root=0)


def test_get_device(pytorch_dist_backend: PyTorchDistBackend) -> None:
    d = pytorch_dist_backend.get_device()
    if torch.cuda.is_available():
        assert d.startswith("cuda:")
    else:
        assert d == "cpu"


def test_get_reduce_op(pytorch_dist_backend: PyTorchDistBackend) -> None:
    from torch.distributed import ReduceOp

    assert pytorch_dist_backend.get_reduce_op("sum") == ReduceOp.SUM
    assert pytorch_dist_backend.get_reduce_op("max") == ReduceOp.MAX
