# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pytest import raises

from chakra_replay.comm.backend.backend_context import BackendContext


def test_backend_context_valid() -> None:
    context = BackendContext(world_size=2, global_rank=1, local_size=1, local_rank=0)
    assert context.world_size == 2
    assert context.global_rank == 1
    assert context.local_size == 1
    assert context.local_rank == 0


def test_backend_context_missing_field() -> None:
    with raises(TypeError):
        BackendContext(world_size=2, global_rank=1, local_size=1)  # Missing local_rank
