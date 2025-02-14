# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import List, Tuple, Union

import torch

from .backend import BaseBackend
from .comms_utils import CollArgs, CommArgs, param_to_comm_name

SUPPORTED_P2P_OPS: List[str] = ["send", "recv", "isend", "irecv"]

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "byte": torch.uint8,
    "char": torch.int8,
    "double": torch.double,
    "float": torch.float32,
    "float16": torch.half,
    "float32": torch.float32,
    "float64": torch.double,
    "half": torch.half,
    "int": torch.int32,
    "int32": torch.int32,
    "int8": torch.int8,
    "long": torch.long,
    "short": torch.short,
    "signed char": torch.int8,
    "uint8": torch.uint8,
    "unsigned char": torch.uint8,
}


class TensorAllocator:
    """Handles allocation of input/output tensors for collective operations."""

    def __init__(self, backend: BaseBackend, init_val: float, coll_args: CollArgs) -> None:
        self.backend: BaseBackend = backend
        self.init_val: float = init_val
        self.coll_args: CollArgs = coll_args

    def create_io_tensors(
        self, op: CommArgs, regenerate: bool
    ) -> Tuple[torch.Tensor, Union[List[torch.Tensor], torch.Tensor]]:
        if not op.id or regenerate:
            return self.allocate_operation_tensors(op, allocate=True)
        else:
            op_hash: int = self.compute_operation_hash(op)
            if op_hash in self.coll_args.et_to_tensors:
                self.allocate_operation_tensors(op, allocate=False)
                inp, out = self.coll_args.et_to_tensors[op_hash]
            else:
                inp, out = self.allocate_operation_tensors(op, allocate=True)
                self.coll_args.et_to_tensors[op_hash] = (inp, out)
            return inp, out

    def compute_operation_hash(self, op: CommArgs) -> int:
        if op.comms in SUPPORTED_P2P_OPS:
            tup = (op.comms, op.src_rank, op.dst_rank, op.in_msg_size, op.out_msg_size)
        elif op.in_split or op.out_split:
            tup = (
                op.comms,
                op.pg_id,
                op.in_msg_size,
                op.out_msg_size,
                tuple(op.in_split or []),
                tuple(op.out_split or []),
            )
        else:
            tup = (op.comms, op.pg_id, op.in_msg_size, op.out_msg_size)
        return hash(tup)

    def allocate_operation_tensors(
        self, op: CommArgs, allocate: bool = True
    ) -> Tuple[torch.Tensor, Union[List[torch.Tensor], torch.Tensor]]:
        comm_name: str = param_to_comm_name(op.comms or "")
        if comm_name in ("wait", "barrier"):
            return torch.Tensor(), torch.Tensor()
        in_elems: int = op.in_msg_size
        out_elems: int = op.out_msg_size
        ws: int = self.coll_args.world_size
        dtype: torch.dtype = DTYPE_MAP.get(op.dtype, torch.float32)
        device: str = self.backend.context.comm_params.device
        input_tensor: torch.Tensor = torch.Tensor()
        if allocate:
            input_tensor = self.backend.alloc_ones([in_elems], dtype, device)
        dispatch: OrderedDict[str, callable] = OrderedDict(
            [
                ("scatter", self._allocate_scatter_tensors),
                ("gather", self._allocate_gather_tensors),
                ("reduce_scatter", self._allocate_reduce_scatter_tensors),
                ("all_gather", self._allocate_all_gather_tensors),
                ("all_to_all_single", self._allocate_all_to_all_single_tensors),
                ("all_to_allv", self._allocate_all_to_allv_tensors),
                ("all_to_all", self._allocate_all_to_all_tensors),
                ("pt2pt", self._allocate_pt2pt_tensors),
            ]
        )
        func = dispatch.get(comm_name)
        if func is None:
            raise NotImplementedError(f"Unsupported collective operation: {comm_name}")
        inp, out = func(input_tensor, op, in_elems, out_elems, ws, dtype, device, allocate)
        return inp, out

    def _allocate_scatter_tensors(
        self,
        inp: torch.Tensor,
        op: CommArgs,
        in_elems: int,
        out_elems: int,
        ws: int,
        device: str,
        dtype: torch.dtype,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        tensors: List[torch.Tensor] = []
        if allocate:
            for _ in range(ws):
                tensors.append(self.backend.alloc_random([in_elems // ws], dtype, device))
        return inp, tensors

    def _allocate_gather_tensors(
        self,
        inp: torch.Tensor,
        op: CommArgs,
        in_elems: int,
        out_elems: int,
        ws: int,
        device: str,
        dtype: torch.dtype,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        tensors: List[torch.Tensor] = []
        if allocate:
            inp = self.backend.alloc_random([in_elems], dtype, device)
            for _ in range(ws):
                tensors.append(self.backend.alloc_random([in_elems], dtype, device))
        return inp, tensors

    def _allocate_reduce_scatter_tensors(
        self,
        inp: torch.Tensor,
        op: CommArgs,
        in_elems: int,
        out_elems: int,
        ws: int,
        device: str,
        dtype: torch.dtype,
        allocate: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        tensors: List[torch.Tensor] = []
        in_elems = (out_elems // ws) if not self.backend.context.comm_params.size_from_trace else (in_elems // ws)
        if allocate:
            for _ in range(ws):
                tensors.append(self.backend.alloc_random([in_elems], device, self.backend.context.comm_params.dtype))
            out_tensor: torch.Tensor = self.backend.alloc_random([out_elems], dtype, device)
        else:
            out_tensor = torch.Tensor()
        return tensors, out_tensor

    def _allocate_all_gather_tensors(
        self,
        inp: torch.Tensor,
        op: CommArgs,
        in_elems: int,
        out_elems: int,
        ws: int,
        device: str,
        dtype: torch.dtype,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        tensors: List[torch.Tensor] = []
        in_elems = (in_elems // ws) if not self.backend.context.comm_params.size_from_trace else in_elems
        if allocate:
            inp = self.backend.alloc_random([in_elems], dtype, device)
            for _ in range(ws):
                tensors.append(self.backend.alloc_random([in_elems], dtype, device))
        return inp, tensors

    def _allocate_all_to_all_single_tensors(
        self,
        inp: torch.Tensor,
        op: CommArgs,
        in_elems: int,
        out_elems: int,
        ws: int,
        device: str,
        dtype: torch.dtype,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if allocate:
            inp = self.backend.alloc_ones([in_elems], dtype, device)
            out: torch.Tensor = self.backend.alloc_random([out_elems], dtype, device)
        else:
            inp, out = torch.Tensor(), torch.Tensor()
        return inp, out

    def _allocate_all_to_allv_tensors(
        self,
        inp: torch.Tensor,
        op: CommArgs,
        in_elems: int,
        out_elems: int,
        ws: int,
        device: str,
        dtype: torch.dtype,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if allocate:
            out: torch.Tensor = self.backend.alloc_random([out_elems], dtype, device)
        else:
            out = torch.Tensor()
        self.coll_args.output_tensor_split = op.out_split if op.out_split else [out_elems // ws for _ in range(ws)]
        self.coll_args.input_tensor_split = op.in_split if op.in_split else [in_elems // ws for _ in range(ws)]
        return inp, out

    def _allocate_all_to_all_tensors(
        self,
        inp: torch.Tensor,
        op: CommArgs,
        in_elems: int,
        out_elems: int,
        ws: int,
        device: str,
        dtype: torch.dtype,
        allocate: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        inp_list: List[torch.Tensor] = []
        out_list: List[torch.Tensor] = []
        if allocate:
            for _ in range(ws):
                inp_list.append(self.backend.alloc_ones([in_elems // ws], dtype, device))
            for _ in range(ws):
                out_list.append(self.backend.alloc_random([out_elems // ws], dtype, device))
        return inp_list, out_list

    def _allocate_pt2pt_tensors(
        self,
        inp: torch.Tensor,
        op: CommArgs,
        in_elems: int,
        out_elems: int,
        ws: int,
        device: str,
        dtype: torch.dtype,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if allocate:
            out: torch.Tensor = self.backend.alloc_random([out_elems], dtype, device)
        else:
            out = torch.Tensor()
        return inp, out
