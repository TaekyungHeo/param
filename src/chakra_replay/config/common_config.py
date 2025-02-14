# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="ERROR")


class TraceConfig(BaseModel):
    """Traue configuration."""

    directory: Path = Field(default=Path("./"))
