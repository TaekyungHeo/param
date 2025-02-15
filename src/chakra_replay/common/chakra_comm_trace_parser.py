# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chakra_replay.comm import CommOp
from chakra_replay.common import ExecutionTrace
from chakra_replay.common.utils import param_to_comm_name

SUPPORTED_P2P_OPS = {"send", "recv", "isend", "irecv"}


class ChakraCommTraceParser:
    """Parses Chakra execution traces and converts them into communication operations."""

    def __init__(self, num_ranks: int, rank: int, directory_path: Path) -> None:
        self.num_ranks = num_ranks
        self.rank = rank
        self.directory_path = directory_path
        logging.debug(f"Initialized ChakraCommTraceParser with num_ranks={num_ranks}, rank={rank}")

    def parse(self) -> List[CommOp]:
        """Parse communication operations from Chakra execution traces."""
        trace_file = self.directory_path / f"rank-{self.rank}.json"
        if not trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

        with trace_file.open("r") as file:
            in_trace = json.load(file)

        execution_trace = ExecutionTrace(in_trace)
        if execution_trace.schema_pytorch() < (1, 0, 3):
            raise ValueError(f"Trace version >1.0.3 is required, but found {execution_trace.schema}")

        pg_ranks_map, pg_desc_map = self._parse_process_groups(execution_trace)
        parsed_ops = self._parse_communication_ops(execution_trace, pg_ranks_map, pg_desc_map)

        logging.info(f"Parsed {len(parsed_ops)} communication operations from {trace_file}")
        return parsed_ops

    def _parse_process_groups(
        self, execution_trace: ExecutionTrace
    ) -> Tuple[Dict[int, Dict[int, List[int]]], Dict[int, Dict[int, str]]]:
        """Extract process group information from the execution trace."""
        pg_ranks_map: Dict[int, Dict[int, List[int]]] = {}
        pg_desc_map: Dict[int, Dict[int, str]] = {}

        for node in execution_trace.nodes.values():
            if "process_group:init" not in node.name:
                continue

            try:
                pg_objs = json.loads(node.inputs[0])
            except json.JSONDecodeError:
                logging.warning("Skipping truncated JSON input for process group initialization.")
                continue

            pg_ranks_map[node.id] = {}
            pg_desc_map[node.id] = {}

            for pg in pg_objs:
                if not pg["pg_name"].isdigit():
                    logging.warning(f"Skipping unsupported process group name '{pg['pg_name']}' in node {node.id}.")
                    continue

                pg_id = int(pg["pg_name"])
                pg_ranks_map[node.id][pg_id] = pg.get("ranks", list(range(pg["group_size"])))
                pg_desc_map[node.id][pg_id] = pg["pg_desc"]

            break  # Only process the first process_group init node

        return pg_ranks_map, pg_desc_map

    def _parse_communication_ops(
        self,
        execution_trace: ExecutionTrace,
        pg_ranks_map: Dict[int, Dict[int, List[int]]],
        pg_desc_map: Dict[int, Dict[int, str]],
    ) -> List[CommOp]:
        """Extract communication operations from the execution trace."""
        comms_op_list: List[CommOp] = []
        pg_ranks_flatten = {pg_id: ranks for pg_map in pg_ranks_map.values() for pg_id, ranks in pg_map.items()}

        for node in execution_trace.nodes.values():
            if node.name != "record_param_comms":
                continue

            comm_op = self._initialize_comm_op(node)
            if comm_op is None:
                continue

            if node.commArgs.pg_name and node.commArgs.pg_name.isdigit():
                comm_op.process_group_id = int(node.commArgs.pg_name)
                comm_op.group_ranks = pg_ranks_flatten.get(comm_op.process_group_id, [])
                comm_op.world_size = len(comm_op.group_ranks)

            if comm_op.comms not in ("wait", "barrier"):
                comm_op.in_msg_size = node.commArgs.in_msg_nelems
                comm_op.out_msg_size = node.commArgs.out_msg_nelems
                comm_op.dtype = node.commArgs.dtype.lower()

            if comm_op.comms in SUPPORTED_P2P_OPS:
                self._handle_p2p_communication(comm_op, node.rank)
            elif comm_op.comms in {"reduce", "broadcast", "gather", "scatter"}:
                comm_op.root = comm_op.group_ranks[node.rank]

            if comm_op.comms == "all_to_allv":
                comm_op = self._handle_all_to_allv(comm_op, node)

            comms_op_list.append(comm_op)

        logging.debug(f"Extracted {len(comms_op_list)} communication operations.")
        return comms_op_list

    def _initialize_comm_op(self, node) -> Optional[CommOp]:
        """Initialize a communication operation from a trace node."""
        req_id = node.inputs[0] if isinstance(node.inputs[0], (int, list)) else node.inputs[1]
        recorded_rank = node.inputs[2] if isinstance(node.inputs[0], (int, list)) else node.inputs[3]

        comms_name = param_to_comm_name(node.commArgs.collective_name.lower())
        if comms_name == "init":
            return None

        return CommOp(
            trace_id=node.id,
            comms=comms_name,
            request_id=(req_id, False) if isinstance(req_id, int) else req_id,
        )

    def _handle_p2p_communication(self, comm_op: CommOp, recorded_rank: int) -> None:
        """Handle point-to-point communication operations."""
        if "send" in comm_op.comms:
            comm_op.src_rank = self.rank
            comm_op.dst_rank = comm_op.group_ranks[recorded_rank]
        elif "recv" in comm_op.comms:
            comm_op.src_rank = comm_op.group_ranks[recorded_rank]
            comm_op.dst_rank = self.rank

    def _handle_all_to_allv(self, comm_op: CommOp, node) -> CommOp:
        """Handle all-to-all variable communication operations."""
        comm_op.world_size = comm_op.world_size or self.num_ranks
        comm_op.in_split = (
            json.loads(node.commArgs.in_split_size) or [comm_op.in_msg_size // comm_op.world_size] * comm_op.world_size
        )
        comm_op.out_split = (
            json.loads(node.commArgs.out_split_size)
            or [comm_op.out_msg_size // comm_op.world_size] * comm_op.world_size
        )
        return comm_op
