# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BaseOp:
    """Base class for all operations."""

    id: Optional[int] = None
    name: Optional[str] = None


@dataclass
class BaseCommOp(BaseOp):
    """Base class for all communication operations."""

    dtype: Optional[str] = None
    process_group_id: Optional[int] = None
    group_ranks: Optional[List[int]] = None
    world_size: Optional[int] = None


@dataclass
class CollCommOp(BaseCommOp):
    """Represents collective communication operations (e.g., all_reduce, broadcast)."""

    root: Optional[int] = None
    in_msg_size: Optional[int] = None
    out_msg_size: Optional[int] = None
    input_splits: Optional[List[int]] = None
    output_splits: Optional[List[int]] = None


@dataclass
class PointToPointCommOp(BaseCommOp):
    """Represents point-to-point communication operations (e.g., send, recv)."""

    src_rank: Optional[int] = None
    dst_rank: Optional[int] = None
    msg_size: Optional[int] = None
