# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) NVIDIA Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Optional, Sequence

from chakra_replay.comm.backend import BackendContext


def get_first_valid_env_int(env_vars: Sequence[str]) -> Optional[int]:
    return next((int(value) for env in env_vars if (value := os.getenv(env)) and value.isdigit()), None)


def read_comms_env_vars() -> BackendContext:
    """Reads communication-related environment variables and returns a BackendContext instance."""
    env_mapping = {
        "world_size": ["MV2_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE", "SLURM_NTASKS"],
        "global_rank": ["MV2_COMM_WORLD_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK", "SLURM_PROCID"],
        "local_size": [
            "LOCAL_SIZE",
            "MPI_LOCALNRANKS",
            "MV2_COMM_WORLD_LOCAL_SIZE",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
            "SLURM_NTASKS_PER_NODE",
        ],
        "local_rank": [
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID",
        ],
    }

    values = {}

    for key, env_vars in env_mapping.items():
        value = get_first_valid_env_int(env_vars)
        if value is None:
            logging.error(f"Failed to retrieve '{key}' from environment variables: {env_vars}")
            raise ValueError(f"Missing required environment variable for '{key}'")
        values[key] = value

    return BackendContext(**values)
