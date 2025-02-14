# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .backend_context import BackendContext
from .mixin import CollectivesMixin, MemoryManagerMixin


class BaseBackend(CollectivesMixin, MemoryManagerMixin):
    """Abstract base class defining the standard backend API."""

    def __init__(self, context: BackendContext):
        """Initialize the backend with the given context."""
        super().__init__()
        self.context = context

    @property
    def world_size(self) -> int:
        """Get the world size (total number of processes)."""
        return self.context.world_size

    @property
    def global_rank(self) -> int:
        """Get the global rank of the process."""
        return self.context.global_rank

    @property
    def local_rank(self) -> int:
        """Get the local rank of the process within the node."""
        return self.context.local_rank

    def say_hello(self) -> None:
        """Print startup information for debugging or logging."""
        pass

    def get_device(self) -> str:
        """Return the string representation of the current device."""
        return ""
