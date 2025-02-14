# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any

from torch.autograd.profiler import record_function


class ParamProfiler(record_function):
    """Inherit from PyTorch profiler to enable autograd profiling while measuring execution time."""

    def __init__(self, timer: Optional[ParamTimer] = None, description: str = "") -> None:
        super().__init__(name=description)
        self.description = description
        self.timer = timer
        self.start = 0.0
        self.end = 0.0
        self.interval_ns = 0.0

    def __enter__(self) -> ParamProfiler:
        super().__enter__()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.end = time.perf_counter()
        self.interval_ns = (self.end - self.start) * 1e9  # Convert seconds to nanoseconds

        # If provided, update the timer with the measured time interval
        if isinstance(self.timer, ParamTimer):
            self.timer.increment_time_ns(self.interval_ns)

        logging.debug("%s took %d ns", self.description, self.interval_ns)
        super().__exit__(exc_type, exc_value, traceback)
