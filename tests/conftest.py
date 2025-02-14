# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from chakra_replay.comm.backend.backend_context import BackendContext


@pytest.fixture
def backend_context() -> BackendContext:
    return BackendContext(world_size=1, global_rank=0, local_size=1, local_rank=0)


@pytest.fixture
def dummy_dist_group() -> Any:
    return MagicMock(name="DummyProcessGroup")


@pytest.fixture
def pytorch_dist_backend(backend_context: BackendContext, dummy_dist_group: Any) -> Any:
    from chakra_replay.comm.backend import PyTorchDistBackend

    backend = PyTorchDistBackend(backend_context)
    backend.groups = {0: dummy_dist_group}
    return backend


@pytest.fixture
def pytorch_tpu_backend(backend_context: BackendContext) -> Any:
    from chakra_replay.comm.backend import PyTorchTPUBackend

    return PyTorchTPUBackend(backend_context)


@pytest.fixture
def dummy_tensor() -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0])
