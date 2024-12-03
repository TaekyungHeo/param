import json
import argparse
import logging
from typing import Dict, Any, List, DefaultDict
from collections import defaultdict

dtype_size_map: Dict[str, int] = {
    "BFloat16": 2,
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


def calculate_algbw(event: Dict[str, Any], coll_name: str, group_size: int) -> float:
    total_size = calculate_s(event, coll_name, group_size)
    duration_seconds = event.get("dur", 0) / 1e6
    if duration_seconds <= 0:
        raise ValueError(f"Invalid or missing duration in event: {event}")
    return round((total_size / duration_seconds) / 1e9, 2)


def get_in_msg_nelems(event: Dict[str, Any]) -> int:
    args = event.get("args", {})
    in_msg_nelems = args.get("In msg nelems")
    if in_msg_nelems is None:
        raise ValueError(f"Missing 'In msg nelems' in event: {event}")
    return in_msg_nelems

def get_out_msg_nelems(event: Dict[str, Any]) -> int:
    args = event.get("args", {})
    out_msg_nelems = args.get("Out msg nelems")
    if out_msg_nelems is None:
        raise ValueError(f"Missing 'Out msg nelems' in event: {event}")
    return out_msg_nelems

def calculate_s(event: Dict[str, Any], coll_name: str, num_ranks: int) -> int:
    args = event.get("args", {})
    dtype = args.get("dtype")
    if dtype not in dtype_size_map:
        raise ValueError(f"Missing or unsupported dtype in event: {event}")
    element_size = dtype_size_map[dtype]

    if coll_name == "all_reduce" or coll_name == "reduce_scatter":
        return get_in_msg_nelems(event) * element_size * num_ranks

    elif coll_name == "all_to_all":
        in_msg_nelems = get_in_msg_nelems(event)
        out_msg_nelems = get_out_msg_nelems(event)
        if in_msg_nelems != out_msg_nelems:
            raise ValueError(
                f"Mismatch between 'In msg nelems' ({in_msg_nelems}) and 'Out msg nelems' ({out_msg_nelems}) for AlltoAll"
            )
        return in_msg_nelems * element_size * num_ranks

    elif coll_name == "all_gather":
        return get_in_msg_nelems(event) * element_size * num_ranks

    elif coll_name == "broadcast":
        return get_in_msg_nelems(event) * element_size

    elif coll_name == "reduce":
        return get_in_msg_nelems(event) * element_size

    else:
        raise ValueError(f"Unsupported or unknown collective operation: {coll_name}")


def calculate_bus_bw(algbw: float, standard_coll_name: str, world_size: int) -> float:
    correction_factor_func = correction_factors.get(standard_coll_name)
    if correction_factor_func is None:
        raise ValueError(f"Unsupported collective operation: {standard_coll_name}")
    correction_factor = correction_factor_func(world_size)
    return round(algbw * correction_factor, 2)


def process_trace(
    trace: Dict[str, Any], logger: logging.Logger
) -> Dict[str, Any]:
    filtered_events = []
    for event in trace.get("traceEvents", []):
        if event.get("name", "").startswith("ncclDevKernel"):
            try:
                coll_name = event["args"].get("Collective name")
                if not coll_name:
                    raise ValueError(f"Missing 'Collective name' in event: {event}")
                standard_coll_name = name_standardization_map.get(coll_name)

                group_size = event["args"].get("Group size")
                if group_size is None or group_size <= 0:
                    raise ValueError(f"Invalid or missing 'Group size' in event: {event}")

                algbw = calculate_algbw(event, standard_coll_name, group_size)
                busbw = calculate_bus_bw(algbw, standard_coll_name, group_size)
                event["args"]["algbw (GB/sec)"] = algbw
                event["args"]["busbw (GB/sec)"] = busbw
                filtered_events.append(event)
                logger.debug("Processed event: %s", event)
            except ValueError as e:
                logger.error("Error processing event: %s", e)
    trace["filteredTraceEvents"] = filtered_events

    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Process and enhance trace data.")
    parser.add_argument("trace_file", type=str, help="Path to the input trace file.")
    parser.add_argument("output_file", type=str, help="Path to the output trace file.")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    try:
        with open(args.trace_file, "r") as f:
            trace_data = json.load(f)
        logger.info("Loaded trace file: %s", args.trace_file)
        processed_trace = process_trace(trace_data, logger)
        with open(args.output_file, "w") as f:
            json.dump(processed_trace, f, indent=2)
        logger.info("Processed trace saved to: %s", args.output_file)
    except Exception as e:
        logger.error("An error occurred: %s", e)
        raise


if __name__ == "__main__":
    main()
