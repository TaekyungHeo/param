# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List, Tuple

import pytest
import toml
import torch

from chakra_replay.comm.backend import BaseBackend
from chakra_replay.comm.chakra_comm_replayer import ChakraCommReplayer, param_to_comm_name
from chakra_replay.comm.comms_utils import CollArgs, CommArgs
from chakra_replay.config.backend_context import BackendContext
from chakra_replay.config.comm_replay_config import CommReplayConfig


class DummyBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(BackendContext(world_size=4, global_rank=0, local_size=2, local_rank=0))
        self.context.comm_params = type("CommParams", (), {"device": "cpu"})
        self.context.memory_params = type("MemoryParams", (), {"init_value": 1, "dtype": torch.float32})

    def get_default_group(self) -> str:
        return "dummy_group"

    def get_group_rank(self, group: str) -> Tuple[int, str]:
        return (0, "dummy")

    def say_hello(self) -> None:
        pass

    def alloc_ones(self, shape: List[int], device: str, dtype: torch.dtype, scale: float = 1.0) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype, device=device) * scale

    def alloc_random(self, shape: List[int], device: str, dtype: torch.dtype, scale: float = 1.0) -> torch.Tensor:
        return torch.rand(shape, dtype=dtype, device=device)

    def clear_memory(self, coll_args: CollArgs) -> None:
        pass

    def barrier_all_ranks(self) -> None:
        pass

    def sync_barrier(self, coll_args: CollArgs) -> None:
        pass

    def broadcast(self, coll_args: CollArgs) -> None:
        pass

    def scatter(self, coll_args: CollArgs) -> None:
        pass

    def gather(self, coll_args: CollArgs) -> None:
        pass

    def reduce(self, coll_args: CollArgs) -> None:
        pass

    def reduce_scatter(self, coll_args: CollArgs) -> None:
        pass

    def all_reduce(self, coll_args: CollArgs) -> None:
        pass

    def all_gather(self, coll_args: CollArgs) -> None:
        pass

    def all_to_all(self, coll_args: CollArgs) -> None:
        pass

    def p2p_op(self, coll_args: CollArgs) -> None:
        pass

    def complete_accel_ops(self, coll_args: CollArgs) -> None:
        pass


class DummyCommArgs(CommArgs):
    def __init__(self) -> None:
        self.id = True
        self.comms = "all_gather"
        self.in_msg_size = 16
        self.out_msg_size = 16
        self.dtype = "float32"
        self.pg_id = None
        self.in_split = None
        self.out_split = None
        self.src_rank = 0
        self.dst_rank = 1
        self.req = None
        self.world_size = 4
        self.root = 0
        self.use_batch_p2p = False


@pytest.fixture
def dummy_config(tmp_path: Path) -> CommReplayConfig:
    config_dict = {
        "trace": {"directory": str(tmp_path / "trace")},
        "replay": {
            "dry_run": False,
            "num_replays": 1,
            "backend": {"master_ip": "127.0.0.1", "master_port": "29500", "device": "cpu", "backend": "nccl"},
        },
        "profiler": {"enabled": False, "num_replays_start": 0, "num_replays": 1},
        "logging": {"level": "ERROR"},
    }
    config_file = tmp_path / "config.toml"
    config_file.write_text(toml.dumps(config_dict))
    return CommReplayConfig.load_from_toml(config_file)


@pytest.fixture
def dummy_replayer(dummy_config: CommReplayConfig) -> ChakraCommReplayer:
    dummy_backend = DummyBackend()
    replayer = ChakraCommReplayer(dummy_config)
    replayer.backend = dummy_backend
    replayer.coll_args.groups = {"dummy": "dummy_group"}
    replayer.global_rank = 0
    replayer.world_size = 4
    return replayer


def test_load_trace_operations(dummy_replayer: ChakraCommReplayer, tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "dummy.json").write_text("[]")
    ops: List[CommArgs] = dummy_replayer.load_trace_operations(trace_dir)
    assert isinstance(ops, list)


def test_process_single_operation(dummy_replayer: ChakraCommReplayer) -> None:
    dummy_op = DummyCommArgs()
    dummy_replayer.process_single_operation(dummy_op, 0)
    assert dummy_replayer.coll_args.input_tensor is not None


def test_execute_collective(dummy_replayer: ChakraCommReplayer) -> None:
    dummy_op = DummyCommArgs()
    op_name = param_to_comm_name(dummy_op.comms)
    # Since our dummy backend methods do nothing, we simply call execute_collective
    dummy_replayer.execute_collective(op_name, dummy_op)
