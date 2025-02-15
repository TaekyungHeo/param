# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class ChakraReplayTimer:
    """Timer utility for tracking elapsed time in Chakra Replay."""

    elapsed_time_ns: float = 0.0

    def reset(self, new_time_ns: float = 0.0) -> None:
        """Reset the timer to a specific time (default is 0 nanoseconds)."""
        self.elapsed_time_ns = new_time_ns

    def increment(self, time_ns: float) -> None:
        """Increment the elapsed time by a given nanosecond value."""
        self.elapsed_time_ns += time_ns

    def elapsed_us(self) -> float:
        """Return the elapsed time in microseconds."""
        return self.elapsed_time_ns / 1e3

    def elapsed_ns(self) -> float:
        """Return the elapsed time in nanoseconds."""
        return self.elapsed_time_ns
