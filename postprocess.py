import json
import argparse
import logging
from typing import Dict, Any, Optional

dtype_size_map: Dict[str, int] = {
    "Long": 8,
    "Float": 4,
    "Double": 8,
    "Int": 4,
}

nickname_map: Dict[str, str] = {
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

correction_factors: Dict[str, Optional[float]] = {
    "all_to_all": (lambda ws: (ws - 1) / ws),
    "all_reduce": (lambda ws: 2 * (ws - 1) / ws),
    "all_gather": (lambda ws: (ws - 1) / ws),
    "reduce_scatter": (lambda ws: (ws - 1) / ws),
    "send": (lambda ws: 1),
    "recv": (lambda ws: 1),
    "broadcast": (lambda ws: 1),
    "reduce": (lambda ws: 1),
}

def calculate_alg_bw(event: Dict[str, Any]) -> float:
    args = event.get("args", {})
    dtype = args.get("dtype")
    if dtype not in dtype_size_map:
        raise ValueError(f"Missing or unsupported dtype in event: {event}")
    element_size = dtype_size_map[dtype]
    in_msg_nelems = args.get("In msg nelems")
    if in_msg_nelems is None:
        raise ValueError(f"Missing 'In msg nelems' in event: {event}")
    total_bytes = in_msg_nelems * element_size
    duration_seconds = event.get("dur", 0) / 1e6
    if duration_seconds <= 0:
        raise ValueError(f"Invalid or missing duration in event: {event}")
    return round((total_bytes / duration_seconds) / 1e9, 2)

def calculate_bus_bw(alg_bw: float, coll_name: str, world_size: int) -> float:
    standardized_name = nickname_map.get(coll_name)
    if standardized_name is None:
        raise ValueError(f"Unsupported collective operation nickname: {coll_name}")
    correction_factor_func = correction_factors.get(standardized_name)
    if correction_factor_func is None:
        raise ValueError(f"Unsupported collective operation: {standardized_name}")
    correction_factor = correction_factor_func(world_size)
    return round(alg_bw * correction_factor, 2)

def process_trace(trace: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    filtered_events = []
    world_size = trace.get("distributedInfo", {}).get("world_size")
    if world_size is None:
        raise ValueError("Missing 'world_size' in 'distributedInfo'")
    logger.info("Processing trace with world_size=%d", world_size)
    for event in trace.get("traceEvents", []):
        if event.get("name", "").startswith("ncclDevKernel"):
            try:
                alg_bw = calculate_alg_bw(event)
                coll_name = event["args"].get("Collective name")
                if not coll_name:
                    raise ValueError(f"Missing 'Collective name' in event: {event}")
                bus_bw = calculate_bus_bw(alg_bw, coll_name, world_size)
                event["args"]["algbw (GB/sec)"] = alg_bw
                event["args"]["busbw (GB/sec)"] = bus_bw
                filtered_events.append(event)
                logger.debug("Processed event: %s", event)
            except ValueError as e:
                logger.error("Error processing event: %s", e)
    trace["filteredTraceEvents"] = filtered_events
    logger.info("Trace processing completed with %d events", len(filtered_events))
    return trace

def main() -> None:
    parser = argparse.ArgumentParser(description="Process and enhance trace data.")
    parser.add_argument("input_file", type=str, help="Path to the input trace file.")
    parser.add_argument("output_file", type=str, help="Path to the output trace file.")
    parser.add_argument(
        "--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
        help="Set the logging level."
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    try:
        with open(args.input_file, "r") as f:
            trace_data = json.load(f)
        logger.info("Loaded trace file: %s", args.input_file)

        processed_trace = process_trace(trace_data, logger)

        with open(args.output_file, "w") as f:
            json.dump(processed_trace, f, indent=2)
        logger.info("Processed trace saved to: %s", args.output_file)
    except Exception as e:
        logger.error("An error occurred: %s", e)
        raise

if __name__ == "__main__":
    main()
