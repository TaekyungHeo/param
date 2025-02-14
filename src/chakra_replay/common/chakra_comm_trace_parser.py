# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from et_replay import ExecutionTrace
from et_replay.comm import comms_utils
from et_replay.comm.backend.base_backend import supportedP2pOps
from et_replay.comm.comms_utils import CommsArgs


class ChakraTraceParser:
    """Parser for Chakra execution traces, making them compatible with PARAM replay-mode."""

    def __init__(self, num_ranks: int, rank: int) -> None:
        self.num_ranks = num_ranks
        self.rank = rank
        logging.debug(f"Initialized ChakraTraceParser with num_ranks={num_ranks}, rank={rank}")

    def parse_trace(self, trace_file_path: Path, in_trace: List[dict]) -> List[CommsArgs]:
        logging.debug(f"Parsing execution trace from {trace_file_path}")

        execution_trace = ExecutionTrace(in_trace)
        if execution_trace.schema_pytorch() < (1, 0, 3):
            logging.error(f"Unsupported trace version: {execution_trace.schema.split('-')[0]}")
            raise ValueError(f"Trace version >1.0.3 is required, but found {execution_trace.schema.split('-')[0]}")

        pg_ranks_map, pg_desc_map = self._parse_process_group_info(execution_trace)
        parsed_ops = self._parse_comms_op_nodes(execution_trace, pg_ranks_map, pg_desc_map)

        logging.info(f"Finished parsing {trace_file_path}. Extracted {len(parsed_ops)} communication operations.")
        return parsed_ops

    def _parse_process_group_info(
        self, in_trace: ExecutionTrace
    ) -> Tuple[Dict[int, Dict[int, List[int]]], Dict[int, Dict[int, str]]]:
        logging.debug("Extracting process group information.")

        pg_ranks_map: Dict[int, Dict[int, List[int]]] = {}
        pg_desc_map: Dict[int, Dict[int, str]] = {}

        for node in in_trace.nodes.values():
            if "process_group:init" not in node.name:
                continue

            try:
                pg_objs = json.loads(node.inputs[0])
            except json.decoder.JSONDecodeError:
                logging.warning("Skipping process group initialization due to truncated JSON input.")
                break

            pg_ranks_map[node.id] = {}
            pg_desc_map[node.id] = {}

            for pg in pg_objs:
                if not pg["pg_name"].isdecimal():
                    logging.warning(f"Skipping unsupported process group name '{pg['pg_name']}' in node {node.id}.")
                    continue

                pg_id = int(pg["pg_name"])
                pg_ranks_map[node.id][pg_id] = pg["ranks"] or list(range(pg["group_size"]))
                pg_desc_map[node.id][pg_id] = pg["pg_desc"]

            logging.debug(f"Processed process group initialization for node {node.id}.")
            break  # Only one process_group init node per trace

        return pg_ranks_map, pg_desc_map

    def _parse_comms_op_nodes(
        self,
        in_trace: ExecutionTrace,
        pg_ranks_map: Dict[int, Dict[int, List[int]]],
        pg_desc_map: Dict[int, Dict[int, str]],
    ) -> List[CommsArgs]:
        logging.debug("Parsing communication operations from the execution trace.")
        comms_op_list: List[CommsArgs] = []

        for node_id, pg_info in pg_ranks_map.items():
            for pg_id, ranks in pg_info.items():
                comms_op_list.append(
                    self._create_pg_init_node(node_id, pg_id, ranks, pg_desc_map[node_id][pg_id], len(ranks))
                )

        pg_ranks_flatten = {pg_id: ranks for pg_map in pg_ranks_map.values() for pg_id, ranks in pg_map.items()}

        for node in in_trace.nodes.values():
            if node.name != "record_param_comms":
                continue

            req_id, recorded_rank, comm_args = self._initialize_comm_args(node)

            if comm_args is None:
                continue  # Skip "init" node

            if node.commArgs.pg_name and node.commArgs.pg_name.isdecimal():
                comm_args.process_group_id = int(node.commArgs.pg_name)
                comm_args.group_ranks = pg_ranks_flatten.get(comm_args.process_group_id, [])
                comm_args.world_size = len(comm_args.group_ranks)

            if comm_args.comms not in ("wait", "barrier"):
                comm_args.in_msg_size = node.commArgs.in_msg_nelems
                comm_args.out_msg_size = node.commArgs.out_msg_nelems
                comm_args.dtype = node.commArgs.dtype.lower()

            if comm_args.comms in supportedP2pOps:
                self._handle_p2p_communication(comm_args, recorded_rank)
            elif comm_args.comms in ["reduce", "broadcast", "gather", "scatter"]:
                comm_args.root = comm_args.group_ranks[recorded_rank]

            if comm_args.comms == "all_to_allv":
                comm_args = self._handle_all_to_allv(comm_args, node)

            comms_op_list.append(comm_args)

        logging.debug(f"Extracted {len(comms_op_list)} communication operations.")
        return comms_op_list

    def _initialize_comm_args(self, node) -> Tuple[int, int, CommsArgs]:
        req_id = node.inputs[0] if isinstance(node.inputs[0], (int, list)) else node.inputs[1]
        recorded_rank = node.inputs[2] if isinstance(node.inputs[0], (int, list)) else node.inputs[3]

        comms_name = comms_utils.paramToCommName(node.commArgs.collective_name.lower())
        if comms_name == "init":
            return None  # Skip initialization nodes

        return (
            req_id,
            recorded_rank,
            CommsArgs(
                trace_id=node.id,
                comms=comms_name,
                request_id=(req_id, False) if isinstance(req_id, int) else req_id,
            ),
        )

    def _handle_p2p_communication(self, comm_args: CommsArgs, recorded_rank: int) -> None:
        if "send" in comm_args.comms:
            comm_args.src_rank = self.rank
            comm_args.dst_rank = comm_args.group_ranks[recorded_rank]
        elif "recv" in comm_args.comms:
            comm_args.src_rank = comm_args.group_ranks[recorded_rank]
            comm_args.dst_rank = self.rank

    def _handle_all_to_allv(self, comm_args: CommsArgs, node) -> CommsArgs:
        if not comm_args.world_size:
            comm_args.world_size = self.num_ranks

        comm_args.in_split = (
            json.loads(node.commArgs.in_split_size)
            or [comm_args.in_msg_size // comm_args.world_size] * comm_args.world_size
        )
        comm_args.out_split = (
            json.loads(node.commArgs.out_split_size)
            or [comm_args.out_msg_size // comm_args.world_size] * comm_args.world_size
        )

        return comm_args
