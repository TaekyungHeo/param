# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CommOp:
    """Store parameters for collective communication operations and compute kernels."""

    # Identifiers & Core Metadata
    trace_id: Optional[int] = None
    request_id: Optional[int] = None
    comms: Optional[str] = None
    dtype: Optional[str] = None

    # Communication Details
    in_msg_size: Optional[int] = None
    out_msg_size: Optional[int] = None
    in_split: Optional[List[int]] = None
    out_split: Optional[List[int]] = None

    # Process Group Information
    process_group_id: Optional[int] = None
    group_ranks: Optional[List[int]] = None
    world_size: Optional[int] = None

    # Role-Specific Data
    root: Optional[int] = None
    src_rank: Optional[int] = None
    dst_rank: Optional[int] = None
