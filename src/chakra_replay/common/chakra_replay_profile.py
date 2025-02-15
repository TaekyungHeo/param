# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any, Optional, Type

from torch.autograd.profiler import record_function

from .chakra_replay_timer import ChakraReplayTimer


class ChakraReplayProfiler(record_function):
    """Profiler for measuring execution time while leveraging PyTorch's autograd profiler."""

    def __init__(self, timer: Optional[ChakraReplayTimer] = None, description: str = "") -> None:
        """Initialize the profiler with an optional timer and description."""
        super().__init__(name=description)
        self.description: str = description
        self.timer: Optional[ChakraReplayTimer] = timer
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._elapsed_ns: float = 0.0

    def __enter__(self) -> "ChakraReplayProfiler":
        """Start profiling and record the start time."""
        super().__enter__()
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        """End profiling, calculate elapsed time, and update the timer if provided."""
        self._end_time = time.perf_counter()
        self._elapsed_ns = (self._end_time - self._start_time) * 1e9  # Convert to nanoseconds

        if self.timer is not None:
            self.timer.increment(self._elapsed_ns)

        logging.debug("%s took %.0f ns", self.description, self._elapsed_ns)
        super().__exit__(exc_type, exc_value, traceback)
