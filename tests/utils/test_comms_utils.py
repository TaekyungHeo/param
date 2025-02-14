# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import pytest

from chakra_replay.comm.comms_utils import param_to_comm_name


@pytest.mark.parametrize(
    "input_name, expected", [("alltoall", "all_to_all"), ("allreduce", "all_reduce"), ("unknown", "unknown")]
)
def test_param_to_comm_name(input_name: str, expected: str) -> None:
    result = param_to_comm_name(input_name)
    assert result == expected


def test_param_to_comm_name_with_supported():
    supported: List[str] = ["all_to_all", "all_reduce"]
    result = param_to_comm_name("alltoall", supported_comms=supported)
    assert result in supported
