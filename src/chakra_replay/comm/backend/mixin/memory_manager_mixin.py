# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List

import torch


class MemoryManagerMixin(ABC):
    """Provide memory management functions for backends."""

    @abstractmethod
    def alloc_empty(self, shape: List[int], dtype: torch.dtype, device: str) -> torch.Tensor:
        """Allocate an empty tensor (uninitialized memory)."""
        pass

    @abstractmethod
    def alloc_ones(self, shape: List[int], dtype: torch.dtype, device: str) -> torch.Tensor:
        """Allocate a tensor filled with ones."""
        pass

    @abstractmethod
    def alloc_random(self, shape: List[int], dtype: torch.dtype, device: str) -> torch.Tensor:
        """Allocate a tensor with random values."""
        pass

    @abstractmethod
    def alloc_embedding_tables(
        self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype, device: str
    ) -> torch.nn.EmbeddingBag:
        """Allocate an embedding table for deep learning workloads."""
        pass

    @abstractmethod
    def clear_memory(self) -> None:
        """Free memory and clear caches (if applicable)."""
        pass

    @abstractmethod
    def get_mem_size(self, tensor: torch.Tensor) -> int:
        """Return the memory size (in bytes) of the given tensor."""
        pass
