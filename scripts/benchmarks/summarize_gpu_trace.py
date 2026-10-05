"""Summarize offline JAX Perfetto GPU traces without counting overlaps twice."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics


def summarize(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    devices = {e["pid"]: e["args"]["name"] for e in events
               if e.get("name") == "process_name"
               and "GPU" in e.get("args", {}).get("name", "")}
    result = {"trace": str(path), "devices": []}
    for pid, name in devices.items():
        activities = [e for e in events if e.get("ph") == "X"
                      and e.get("pid") == pid and e.get("dur", 0) > 0]
        if not activities:
            continue
        intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in activities)
        start, end = intervals[0]
        busy = 0
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                busy += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        busy += end - start
        span = max(end for _, end in intervals) - intervals[0][0]
        kernels = [e for e in activities if not e["name"].startswith(("Memcpy", "Memset"))]
        aggregate = defaultdict(lambda: {"count": 0, "total_ms": 0})
        for event in activities:
            row = aggregate[event["name"]]
            row["count"] += 1
            row["total_ms"] += event["dur"] / 1000
        result["devices"].append({
            "name": name, "activity_count": len(activities),
            "kernel_count": len(kernels), "span_ms": span / 1000,
            "active_union_ms": busy / 1000, "active_fraction": busy / span,
            "median_kernel_us": statistics.median(e["dur"] for e in kernels) if kernels else None,
            "kernels_under_5us_fraction": sum(e["dur"] < 5 for e in kernels) / len(kernels) if kernels else None,
            "top_activities": [{"name": name, **values} for name, values in
                               sorted(aggregate.items(), key=lambda item: -item[1]["total_ms"])[:15]],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"note": "Trace timings include profiler overhead; active fraction is not SM utilization.",
              "traces": [summarize(path) for path in args.traces]}
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
