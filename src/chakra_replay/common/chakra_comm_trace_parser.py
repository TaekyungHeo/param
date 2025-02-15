# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

from chakra_replay.comm.comm_op import CollCommOp, PointToPointCommOp, BaseCommOp
from chakra_replay.common.execution_trace import ExecutionTrace
from chakra_replay.common.utils import param_to_comm_name


class ChakraCommTraceParser:
    """Parse Chakra execution traces and converts them into structured communication ops."""

    SUPPORTED_P2P_OPS: ClassVar[set[str]] = {"send", "recv", "isend", "irecv"}

    def __init__(self, num_ranks: int, rank: int, trace_directory: Path) -> None:
        self.num_ranks = num_ranks
        self.rank = rank
        self.trace_directory = trace_directory
        logging.debug(f"Initialized ChakraCommTraceParser with num_ranks={num_ranks}, rank={rank}")

    def parse_trace(self) -> List[BaseCommOp]:
        trace_file_path = self.trace_directory / f"rank-{self.rank}.json"
        if not trace_file_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file_path}")

        logging.info(f"Loading execution trace from {trace_file_path}")

        with trace_file_path.open("r") as file:
            trace_data = json.load(file)

        execution_trace = ExecutionTrace(trace_data)
        if execution_trace.schema_pytorch() < (1, 0, 3):
            raise ValueError(f"Incompatible trace version: {execution_trace.schema}. Expected >1.0.3.")

        process_group_ranks, process_group_descriptions = self._parse_process_groups(execution_trace)
        comm_ops = self._parse_comm_ops(execution_trace, process_group_ranks, process_group_descriptions)

        logging.info(f"Extracted {len(comm_ops)} communication ops from {trace_file_path}")
        return comm_ops

    def _parse_process_groups(
        self, execution_trace: ExecutionTrace
    ) -> Tuple[Dict[int, Dict[int, List[int]]], Dict[int, Dict[int, str]]]:
        process_group_ranks: Dict[int, Dict[int, List[int]]] = {}
        process_group_descriptions: Dict[int, Dict[int, str]] = {}

        for node in execution_trace.nodes.values():
            if "process_group:init" not in node.name:
                continue

            try:
                process_group_data = json.loads(node.inputs[0])
            except json.JSONDecodeError:
                logging.warning("Skipping process group initialization due to invalid JSON input.")
                continue

            process_group_ranks[node.id] = {}
            process_group_descriptions[node.id] = {}

            for group in process_group_data:
                if not group["pg_name"].isdigit():
                    logging.warning(f"Skipping unsupported process group name '{group['pg_name']}' in node {node.id}.")
                    continue

                group_id = int(group["pg_name"])
                process_group_ranks[node.id][group_id] = group.get("ranks", list(range(group["group_size"])))
                process_group_descriptions[node.id][group_id] = group["pg_desc"]

            break  # Only process the first process group initialization node

        return process_group_ranks, process_group_descriptions

    def _parse_comm_ops(
        self,
        execution_trace: ExecutionTrace,
        process_group_ranks: Dict[int, Dict[int, List[int]]],
        process_group_descriptions: Dict[int, Dict[int, str]],
    ) -> List[BaseCommOp]:
        comm_ops: List[BaseCommOp] = []
        flattened_process_group_ranks = {pg_id: ranks for pg_map in process_group_ranks.values() for pg_id, ranks in pg_map.items()}

        for node in execution_trace.nodes.values():
            if node.name != "record_param_comms":
                continue

            comm_op = self._initialize_comm_op(node)
            if comm_op is None:
                continue

            if node.commArgs.pg_name and node.commArgs.pg_name.isdigit():
                comm_op.process_group_id = int(node.commArgs.pg_name)
                comm_op.group_ranks = flattened_process_group_ranks.get(comm_op.process_group_id, [])
                comm_op.world_size = len(comm_op.group_ranks)

            if isinstance(comm_op, CollCommOp):
                comm_op.in_msg_size = node.commArgs.in_msg_nelems
                comm_op.out_msg_size = node.commArgs.out_msg_nelems
                comm_op.dtype = node.commArgs.dtype.lower()

            if isinstance(comm_op, PointToPointCommOp):
                self._configure_p2p_op(comm_op, node.rank)

            if isinstance(comm_op, CollCommOp) and comm_op.name == "all_to_allv":
                comm_op = self._configure_all_to_allv_op(comm_op, node)

            comm_ops.append(comm_op)

        logging.debug(f"Extracted {len(comm_ops)} communication ops.")
        return comm_ops

    def _initialize_comm_op(self, node) -> Optional[BaseCommOp]:
        comms_name = param_to_comm_name(node.commArgs.collective_name.lower())
        if comms_name == "init":
            return None

        if comms_name in self.SUPPORTED_P2P_OPS:
            return PointToPointCommOp(id=node.id, name=comms_name)
        return CollCommOp(id=node.id, name=comms_name)

    def _configure_p2p_op(self, comm_op: PointToPointCommOp, recorded_rank: int) -> None:
        if "send" in comm_op.name:
            comm_op.src_rank = self.rank
            comm_op.dst_rank = comm_op.group_ranks[recorded_rank]
        elif "recv" in comm_op.name:
            comm_op.src_rank = comm_op.group_ranks[recorded_rank]
            comm_op.dst_rank = self.rank

    def _configure_all_to_allv_op(self, comm_op: CollCommOp, node) -> CollCommOp:
        comm_op.world_size = comm_op.world_size or self.num_ranks
        comm_op.input_splits = json.loads(node.commArgs.in_split_size) or [comm_op.in_msg_size // comm_op.world_size] * comm_op.world_size
        comm_op.output_splits = json.loads(node.commArgs.out_split_size) or [comm_op.out_msg_size // comm_op.world_size] * comm_op.world_size
        return comm_op
