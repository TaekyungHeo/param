# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class BackendContext:
    """Holds the overall backend context, ensuring all configuration settings are explicitly provided."""

    world_size: int
    global_rank: int
    local_size: int
    local_rank: int

    def __post_init__(self) -> None:
        """Validate that all fields are explicitly set and properly initialized."""
        if any(v is None for v in (self.world_size, self.global_rank, self.local_size, self.local_rank)):
            raise ValueError(
                "All fields (world_size, global_rank, local_size, local_rank) must be explicitly provided."
            )

        if not all(
            isinstance(v, int) and v >= 0 for v in (self.world_size, self.global_rank, self.local_size, self.local_rank)
        ):
            raise ValueError("All fields must be non-negative integers.")
