# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from .backend_context import BackendContext
from .base_backend import BaseBackend


class PyTorchTPUBackend(BaseBackend):
    """PyTorch TPU Backend using `torch_xla` for distributed computation."""

    def __init__(self, context: BackendContext):
        super().__init__(context)

    # --- Memory Management ---
    def alloc_empty(self, shape: list[int], dtype: torch.dtype, device: str) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)

    def alloc_ones(self, shape: list[int], dtype: torch.dtype, device: str) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype, device=device)

    def alloc_random(self, shape: list[int], dtype: torch.dtype, device: str) -> torch.Tensor:
        if dtype in (torch.int32, torch.long):
            return torch.randint(0, 1000, shape, device=device, dtype=dtype)
        return torch.rand(shape, dtype=dtype, device=device)

    def alloc_embedding_tables(
        self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype, device: str
    ) -> torch.nn.EmbeddingBag:
        embedding = nn.EmbeddingBag(num_embeddings, embedding_dim, mode="sum", sparse=True)
        weight = np.random.uniform(
            low=-np.sqrt(1 / num_embeddings), high=np.sqrt(1 / num_embeddings), size=(num_embeddings, embedding_dim)
        ).astype(np.float32)
        embedding.weight.data = torch.tensor(weight, dtype=dtype, device=device, requires_grad=True)
        return embedding

    def clear_memory(self) -> None:
        pass

    # --- Synchronization & Initialization ---
    def barrier(self) -> None:
        xm.rendezvous("global_barrier")

    def sync_barrier(self) -> None:
        """Ensure all outstanding operations complete before the next barrier."""
        self.complete_accel_ops()
        self.barrier()
        self.complete_accel_ops()

    def say_hello(self) -> None:
        logging.info(
            f"[TPU Rank {self.global_rank}] Host: {os.uname().nodename}, "
            f"Device: {self.device_str()} ({self.hw_device()}), Local Rank: {self.local_rank}, "
            f"World Size: {self.world_size}"
        )

    # --- Collective Communication ---
    def broadcast(self, input_tensor: torch.Tensor, root: int) -> torch.Tensor:
        return xm.all_gather(input_tensor, dim=0)

    def scatter(self, input_tensor: Optional[torch.Tensor], root: int) -> torch.Tensor:
        raise NotImplementedError("scatter is not implemented on TPU")

    def gather(self, input_tensor: torch.Tensor, root: int) -> Optional[torch.Tensor]:
        raise NotImplementedError("gather is not implemented on TPU")

    def reduce(self, input_tensor: torch.Tensor, root: int, op: str) -> Optional[torch.Tensor]:
        raise NotImplementedError("reduce is not implemented on TPU")

    def reduce_scatter(self, input_tensor: torch.Tensor, op: str) -> torch.Tensor:
        raise NotImplementedError("reduce_scatter is not implemented on TPU")

    def all_reduce(self, input_tensor: torch.Tensor, op: str) -> torch.Tensor:
        return xm.all_reduce(self.get_reduce_op(op), [input_tensor])

    def all_gather(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return xm.all_gather(input_tensor, dim=0)

    def all_to_all(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return xm.all_to_all(input_tensor, 0, 0, self.world_size)

    # --- TPU-Specific Utilities ---
    def complete_accel_ops(self) -> None:
        """Mark a TPU execution step to flush pending operations."""
        xm.mark_step()

    def get_reduce_op(self, op: str):
        """Convert string-based reduction operations to `torch_xla` ReduceOp."""
        return {"sum": xm.REDUCE_SUM, "max": xm.REDUCE_MAX}.get(op, xm.REDUCE_SUM)

    def get_mem_size(self, tensor: torch.Tensor) -> int:
        """Return the memory size (in bytes) of a given tensor."""
        return tensor.nelement() * tensor.element_size()

    def tensor_list_to_numpy(self, tensor_list: torch.Tensor) -> np.ndarray:
        """Convert a PyTorch tensor to a NumPy array."""
        return tensor_list.cpu().detach().numpy()

    # --- TPU-Specific Information ---
    @property
    def local_rank(self) -> int:
        return xm.get_local_ordinal()

    @property
    def global_rank(self) -> int:
        return xm.get_ordinal()

    @property
    def world_size(self) -> int:
        return xm.xrt_world_size()

    def device_str(self) -> str:
        """Get TPU device as a string."""
        return str(xm.xla_device())

    def hw_device(self) -> str:
        """Get the real hardware device ID."""
        return xm._xla_real_device(xm.xla_device())

    # --- TPU Backend Initialization ---
    def initialize_backend(self, master_ip: str, master_port: str, backend: str = "xla") -> None:
        pass

    def benchmark_comms(self, bench_time, comms_params) -> None:
        xmp.spawn(
            fn=bench_time,
            args=(comms_params, self),
            nprocs=self.world_size,
        )
