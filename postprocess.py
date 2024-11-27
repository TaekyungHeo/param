import os
import json
import argparse
import logging
from typing import Dict, Any, Optional, List, DefaultDict
from collections import defaultdict
import numpy as np
from scipy.stats import gmean

dtype_size_map: Dict[str, int] = {
    "Long": 8,
    "Float": 4,
    "Double": 8,
    "Int": 4,
}

name_standardization_map: Dict[str, str] = {
    "all_to_all": "all_to_all",
    "all_to_allv": "all_to_all",
    "all_to_all_single": "all_to_all",
    "all_reduce": "all_reduce",
    "all_gather": "all_gather",
    "all_gather_base": "all_gather",
    "reduce_scatter": "reduce_scatter",
    "reduce_scatter_base": "reduce_scatter",
    "_allgather_base": "all_gather",
    "_reduce_scatter_base": "reduce_scatter",
    "send": "send",
    "recv": "recv",
    "broadcast": "broadcast",
    "reduce": "reduce",
}

correction_factors: Dict[str, Any] = {
    "all_to_all": (lambda ws: (ws - 1) / ws),
    "all_reduce": (lambda ws: 2 * (ws - 1) / ws),
    "all_gather": (lambda ws: (ws - 1) / ws),
    "reduce_scatter": (lambda ws: (ws - 1) / ws),
    "send": (lambda ws: 1),
    "recv": (lambda ws: 1),
    "broadcast": (lambda ws: 1),
    "reduce": (lambda ws: 1),
}

def get_in_msg_size(event: Dict[str, Any]) -> int:
    args = event.get("args", {})
    in_msg_nelems = args.get("In msg nelems")
    dtype = args.get("dtype")
    if dtype not in dtype_size_map:
        raise ValueError(f"Missing or unsupported dtype in event: {event}")
    element_size = dtype_size_map[dtype]
    if in_msg_nelems is None:
        raise ValueError(f"Missing 'In msg nelems' in event: {event}")
    return in_msg_nelems * element_size

def calculate_algbw(event: Dict[str, Any]) -> float:
    total_bytes = get_in_msg_size(event)
    duration_seconds = event.get("dur", 0) / 1e6
    if duration_seconds <= 0:
        raise ValueError(f"Invalid or missing duration in event: {event}")
    return round((total_bytes / duration_seconds) / 1e9, 2)

def calculate_bus_bw(algbw: float, coll_name: str, world_size: int) -> float:
    standardized_name = name_standardization_map.get(coll_name)
    if standardized_name is None:
        raise ValueError(f"Unsupported collective operation nickname: {coll_name}")
    correction_factor_func = correction_factors.get(standardized_name)
    if correction_factor_func is None:
        raise ValueError(f"Unsupported collective operation: {standardized_name}")
    correction_factor = correction_factor_func(world_size)
    return round(algbw * correction_factor, 2)

def calculate_relative_perf(
    standardized_name: Optional[str],
    msg_size: int,
    busbw: float,
    nccl_test_results: Dict[str, Dict[int, float]]
) -> Optional[float]:
    if not standardized_name or standardized_name not in nccl_test_results:
        return None
    sol_busbw = nccl_test_results[standardized_name]
    interpolated_sol_busbw = interpolate_busbw(sol_busbw, msg_size)
    if interpolated_sol_busbw > 0:
        return round(busbw / interpolated_sol_busbw, 2)
    return None

def parse_nccl_test_file(filepath: str) -> Dict[int, float]:
    results = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            try:
                message_size = int(parts[0])
                busbw = float(parts[-2])
                results[message_size] = busbw
            except (IndexError, ValueError):
                continue
    return results

def load_nccl_test_results(directory: Optional[str]) -> Dict[str, Dict[int, float]]:
    if not directory or not os.path.isdir(directory):
        return {}
    collective_map = {}
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        collective_name = os.path.splitext(filename)[0]
        filepath = os.path.join(directory, filename)
        collective_map[collective_name] = parse_nccl_test_file(filepath)
    return collective_map

def interpolate_busbw(measured_bw: Dict[int, float], size: int) -> float:
    sizes = sorted(measured_bw.keys())
    bw_values = [measured_bw[s] for s in sizes]
    return float(np.interp(size, sizes, bw_values))

def process_trace_with_nccl_results(
    trace: Dict[str, Any],
    nccl_test_results: Dict[str, Dict[int, float]],
    logger: logging.Logger
) -> Dict[str, Any]:
    filtered_events = []
    world_size = trace.get("distributedInfo", {}).get("world_size")
    if world_size is None:
        raise ValueError("Missing 'world_size' in 'distributedInfo'")
    logger.info("Processing trace with world_size=%d", world_size)

    relative_perf_values: List[float] = []
    per_collective_perf: DefaultDict[str, List[float]] = defaultdict(list)

    for event in trace.get("traceEvents", []):
        if event.get("name", "").startswith("ncclDevKernel"):
            try:
                algbw = calculate_algbw(event)
                coll_name = event["args"].get("Collective name")
                if not coll_name:
                    raise ValueError(f"Missing 'Collective name' in event: {event}")
                busbw = calculate_bus_bw(algbw, coll_name, world_size)
                event["args"]["algbw (GB/sec)"] = algbw
                event["args"]["busbw (GB/sec)"] = busbw
                standardized_name = name_standardization_map.get(coll_name)
                msg_size = get_in_msg_size(event)
                relative_perf = calculate_relative_perf(standardized_name, msg_size, busbw, nccl_test_results)
                if relative_perf is not None:
                    event["args"]["relative perf"] = relative_perf
                    relative_perf_values.append(relative_perf)
                    if standardized_name:
                        per_collective_perf[standardized_name].append(relative_perf)
                filtered_events.append(event)
                logger.debug("Processed event: %s", event)
            except ValueError as e:
                logger.error("Error processing event: %s", e)
    trace["filteredTraceEvents"] = filtered_events

    if relative_perf_values:
        global_geomean = gmean(relative_perf_values)
        print(f"Global geomean of relative performance: {global_geomean:.2f}")

    for coll, values in per_collective_perf.items():
        if values:
            collective_geomean = gmean(values)
            print(f"Geomean of relative performance for {coll}: {collective_geomean:.2f}")

    return trace

def main() -> None:
    parser = argparse.ArgumentParser(description="Process and enhance trace data.")
    parser.add_argument("trace_file", type=str, help="Path to the input trace file.")
    parser.add_argument("output_file", type=str, help="Path to the output trace file.")
    parser.add_argument("nccl_test_dir", type=str, nargs="?", help="Path to the directory containing NCCL test results.")
    parser.add_argument(
        "--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
        help="Set the logging level."
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    try:
        logger.info("Loading NCCL test results from directory: %s", args.nccl_test_dir)
        nccl_test_results = load_nccl_test_results(args.nccl_test_dir)
        with open(args.trace_file, "r") as f:
            trace_data = json.load(f)
        logger.info("Loaded trace file: %s", args.trace_file)
        processed_trace = process_trace_with_nccl_results(trace_data, nccl_test_results, logger)
        with open(args.output_file, "w") as f:
            json.dump(processed_trace, f, indent=2)
        logger.info("Processed trace saved to: %s", args.output_file)
    except Exception as e:
        logger.error("An error occurred: %s", e)
        raise

if __name__ == "__main__":
    main()
