# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional

import toml
from pydantic import BaseModel, field_validator

from .common_config import LoggingConfig


class ExecutionConfig(BaseModel):
    """ExecutionConfig."""

    warmup_iter: int = 5
    iter: int = 10
    delay: int = 0


class TraceConfig(BaseModel):
    """TraceConfig."""

    input: Optional[Path] = None
    trace_path: Optional[Path] = None
    subgraph: str = ""
    skip_node_file: Optional[Path] = None
    update_skip_node_file: bool = False

    @field_validator("trace_path", "input", "skip_node_file", mode="before")
    @classmethod
    def validate_paths(cls, v):
        return Path(v) if v else None


class ProfilingConfig(BaseModel):
    """ProfilingConfig."""

    profile_replay: bool = False
    profile_memory: bool = False
    debug: bool = False


class ComputationConfig(BaseModel):
    """ComputationConfig."""

    compute: bool = False
    cpu: bool = False
    tf32: bool = False


class CodeGenConfig(BaseModel):
    """CodeGenConfig."""

    generator: bool = False
    dump: bool = False
    dump_path: Path = Path("./")

    @field_validator("dump_path", mode="before")
    @classmethod
    def validate_dump_path(cls, v):
        return Path(v) if v else Path("./")


class MemoryConfig(BaseModel):
    """MemoryConfig."""

    enable_lazy_tensor_allocation: bool = False
    device_memory_threshold: float = 1.0
    cuda: int = -1


class FullReplayConfig(BaseModel):
    """FullReplayConfig."""

    logging: LoggingConfig = LoggingConfig()
    trace: TraceConfig = TraceConfig()
    execution: ExecutionConfig = ExecutionConfig()
    profiling: ProfilingConfig = ProfilingConfig()
    computation: ComputationConfig = ComputationConfig()
    codegen: CodeGenConfig = CodeGenConfig()
    memory: MemoryConfig = MemoryConfig()

    @classmethod
    def load_from_toml(cls, path: Path) -> "FullReplayConfig":
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        config_dict = toml.load(path)

        return cls(**config_dict)
