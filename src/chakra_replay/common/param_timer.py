# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class ParamTimer:
    """Timer for ParamProfiler."""

    elapsed_time_ns: float = 0.0  # Store elapsed time in nanoseconds

    def reset(self, new_time: float = 0.0) -> None:
        """Reset the timer to a new time (default is 0)."""
        self.elapsed_time_ns = new_time

    def increment_time_ns(self, time_ns: float) -> None:
        """Increment the elapsed time by a given nanosecond value."""
        self.elapsed_time_ns += time_ns

    def get_time_us(self) -> float:
        """Return elapsed time in microseconds."""
        return self.elapsed_time_ns / 1e3

    def get_time_ns(self) -> float:
        """Return elapsed time in nanoseconds."""
        return self.elapsed_time_ns
