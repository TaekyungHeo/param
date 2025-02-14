# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional

import torch


class CollectivesMixin(ABC):
    """Provide collective communication operations for distributed computing."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all processes."""
        pass

    @abstractmethod
    def broadcast(self, input_tensor: torch.Tensor, root: int) -> torch.Tensor:
        """Broadcast a tensor from the root process to all other processes."""
        pass

    @abstractmethod
    def scatter(self, input_tensor: Optional[torch.Tensor], root: int) -> torch.Tensor:
        """Scatter a tensor from the root process to all other processes."""
        pass

    @abstractmethod
    def gather(self, input_tensor: torch.Tensor, root: int) -> Optional[torch.Tensor]:
        """Gather tensors from all processes to the root process."""
        pass

    @abstractmethod
    def reduce(self, input_tensor: torch.Tensor, root: int, op: str) -> Optional[torch.Tensor]:
        """Reduce a tensor from all processes to the root process using an operation (e.g., SUM, MAX)."""
        pass

    @abstractmethod
    def reduce_scatter(self, input_tensor: torch.Tensor, op: str) -> torch.Tensor:
        """Reduce and scatter a tensor among all processes."""
        pass

    @abstractmethod
    def all_reduce(self, input_tensor: torch.Tensor, op: str) -> torch.Tensor:
        """Perform an all-reduce operation where all processes contribute and receive the reduced result."""
        pass

    @abstractmethod
    def all_gather(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Gather data from all processes to all processes."""
        pass

    @abstractmethod
    def all_to_all(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform an all-to-all collective communication operation."""
        pass
