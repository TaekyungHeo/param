# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .backend_context import BackendContext
from .base_backend import BaseBackend


class PyTorchDistBackend(BaseBackend):
    """PyTorch Distributed Backend using `torch.distributed` API."""

    def __init__(self, context: BackendContext):
        super().__init__(context)
        self.groups: Dict[int, dist.ProcessGroup] = {}

        # Initialize default process group
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        default_group = dist.new_group()
        if default_group is None:
            logging.warning("dist.new_group() returned None, using dist.group.WORLD instead.")
            default_group = dist.group.WORLD

        assert default_group is not None
        self.groups[0] = default_group

        self.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        self.master_port = os.environ.get("MASTER_PORT", "29500")

    # --- Memory Management ---
    def alloc_empty(self, shape: List[int], dtype: torch.dtype, device: str) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)

    def alloc_ones(self, shape: List[int], dtype: torch.dtype, device: str) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype, device=device)

    def alloc_random(self, shape: List[int], dtype: torch.dtype, device: str) -> torch.Tensor:
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
        torch.cuda.empty_cache()

    # --- Synchronization & Initialization ---
    def barrier(self) -> None:
        dist.barrier()

    def sync_barrier(self) -> None:
        """Ensure all outstanding operations complete before the next barrier."""
        self.complete_accel_ops()
        self.barrier()
        self.complete_accel_ops()

    def say_hello(self) -> None:
        logging.info(
            f"[Rank {self.global_rank}] Host: {os.uname().nodename}, "
            f"Device: {self.get_device()}, Local Rank: {self.local_rank}, World Size: {self.world_size}"
        )

    # --- Collective Communication ---
    def broadcast(self, input_tensor: torch.Tensor, root: int) -> torch.Tensor:
        dist.broadcast(input_tensor, root)
        return input_tensor

    def scatter(self, input_tensor: Optional[torch.Tensor], root: int) -> torch.Tensor:
        assert input_tensor is not None, "Scatter operation received None input."
        output_tensor = torch.empty_like(input_tensor)
        dist.scatter(output_tensor, scatter_list=input_tensor if self.global_rank == root else None, src=root)
        return output_tensor

    def gather(self, input_tensor: torch.Tensor, root: int) -> Optional[torch.Tensor]:
        gathered_tensors: List[torch.Tensor] = []
        if self.global_rank == root:
            gathered_tensors = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.gather(input_tensor, gather_list=gathered_tensors if self.global_rank == root else None, dst=root)
        return torch.cat(gathered_tensors) if self.global_rank == root else None

    def reduce(self, input_tensor: torch.Tensor, root: int, op: str) -> Optional[torch.Tensor]:
        dist.reduce(input_tensor, dst=root, op=self.get_reduce_op(op))
        return input_tensor if self.global_rank == root else None

    def reduce_scatter(self, input_tensor: torch.Tensor, op: str) -> torch.Tensor:
        output_tensor = torch.empty_like(input_tensor)
        dist.reduce_scatter(output_tensor, input_tensor, op=self.get_reduce_op(op))
        return output_tensor

    def all_reduce(self, input_tensor: torch.Tensor, op: str) -> torch.Tensor:
        dist.all_reduce(input_tensor, op=self.get_reduce_op(op))
        return input_tensor

    def all_gather(self, input_tensor: torch.Tensor) -> torch.Tensor:
        gathered_tensors = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, input_tensor)
        return torch.cat(gathered_tensors)

    def all_to_all(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output_tensor = torch.empty_like(input_tensor)
        dist.all_to_all_single(output_tensor, input_tensor)
        return output_tensor

    # --- Point-to-Point Communication ---
    def send(self, input_tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        dist.send(input_tensor, dst, tag=tag)

    def recv(self, output_tensor: torch.Tensor, src: int, tag: int = 0) -> None:
        dist.recv(output_tensor, src, tag=tag)

    def isend(self, input_tensor: torch.Tensor, dst: int, tag: int = 0) -> dist.Work:
        work = dist.isend(input_tensor, dst, tag=tag)
        assert work is not None, "isend() returned None, which is unexpected."
        return work

    def irecv(self, output_tensor: torch.Tensor, src: int, tag: int = 0) -> dist.Work:
        work = dist.irecv(output_tensor, src, tag=tag)
        assert work is not None, "irecv() returned None, which is unexpected."
        return work

    # --- Utilities ---
    def get_device(self) -> str:
        return f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"

    def get_reduce_op(self, op: str):
        return {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}.get(op, dist.ReduceOp.SUM)

    def complete_accel_ops(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
