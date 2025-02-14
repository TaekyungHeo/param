# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import toml
import torch
from pydantic import BaseModel, Field, field_validator

from .common_config import LoggingConfig, TraceConfig


class BackendConfig(BaseModel):
    """Defines backend settings for communication replay."""

    master_ip: str = Field(default_factory=lambda: os.getenv("MASTER_ADDR", "127.0.0.1"))
    master_port: str = Field(default_factory=lambda: os.getenv("MASTER_PORT", "29500"))
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    backend: str = "nccl"

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed_devices = {"cpu", "cuda"}
        if v not in allowed_devices:
            raise ValueError(f"Invalid device '{v}'. Allowed values: {allowed_devices}")
        return v


class ReplayConfig(BaseModel):
    """Configuration for replaying communication traces."""

    dry_run: bool = False
    num_replays: int = 1
    backend: BackendConfig = BackendConfig()


class ProfilerConfig(BaseModel):
    """Configuration for profiling replay performance."""

    enabled: bool = False
    num_replays_start: int = 0
    num_replays: int = 1


class CommReplayConfig(BaseModel):
    """Aggregates all configurations for the Chakra communication replayer."""

    trace: TraceConfig = TraceConfig()
    replay: ReplayConfig = ReplayConfig()
    profiler: ProfilerConfig = ProfilerConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def load_from_toml(cls, path: Path) -> "CommReplayConfig":
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        config_dict = toml.load(path)
        return cls(**config_dict)
