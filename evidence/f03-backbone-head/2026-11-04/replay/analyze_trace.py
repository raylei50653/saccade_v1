#!/usr/bin/env python3
"""Map the isolated F03 CUDA Graph trace to its exact engine inspector order.

This is a strict analysis of this capture, not a heuristic kernel-name join.
Every expected kernel node must be consumed and repeated once per graph launch.
Only SQLite, JSON and CSV are read; this script never submits GPU work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def require(condition, explanation):
    if not condition:
        raise RuntimeError(explanation)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def identity(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def region_of(layer, producers):
    name, metadata = layer["Name"], layer.get("Metadata", "")
    match = re.match(r"p([345])\.", name)
    if match:
        return "p" + match.group(1), "explicit-head-layer-name"
    if "ONNX" in metadata or "/layers." in name:
        return "backbone-neck", "onnx-metadata-or-layer-name"
    io = layer.get("Inputs", []) + layer.get("Outputs", [])
    if any(t["Name"] == "images" for t in io):
        return "backbone-neck", "model-images-input-cast"
    named = {"p" + m.group(1) for t in io if (m := re.match(r"p([345])\.", t["Name"]))}
    if len(named) == 1:
        return named.pop(), "head-io-name"
    inherited = {producers[t["Name"]] for t in layer.get("Inputs", []) if t["Name"] in producers}
    if len(inherited) == 1:
        return inherited.pop(), "input-producer-region"
    return "unknown", f"named={sorted(named)},inherited={sorted(inherited)}"


def kind_of(layer, kernel, operation):
    if operation == "scan":
        return "scan-core" if "scan_kernel<" in kernel else "scan-checks"
    if operation in ("expand", "merge"):
        return "cross-scan"
    # Pooling also has "xmma" in its tactic name; it is not a convolution/GEMM.
    if "xmma" in kernel and "pooling" not in kernel:
        return "conv-gemm"
    return "layout-pointwise"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    database = root / "f03-graph.sqlite"
    inspector = root / "graph-layer-profile/engine-layers.json"
    layers = json.loads(inspector.read_text())["Layers"]
    require(len(layers) == 286, f"expected 286 exact-engine layers, got {len(layers)}")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    events = [dict(r) for r in connection.execute("""
        SELECT k.*, s.value AS kernel
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS s ON s.id=k.demangledName
        ORDER BY k.start,k.end
    """)]
    graph_launches = connection.execute("""
        SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r JOIN StringIds AS s ON s.id=r.nameId
        WHERE s.value LIKE 'cudaGraphLaunch%'
    """).fetchone()[0]
    require(graph_launches == 20, f"expected 20 captured graph launches, got {graph_launches}")
    require(len(events) == 6040, f"expected 6040 kernel calls, got {len(events)}")
    streams = sorted({e["streamId"] for e in events})
    require(len(streams) == 1, f"expected one GPU stream, got {streams}")
    interval_overlaps_ns = [a["end"] - b["start"] for a, b in zip(events, events[1:]) if a["end"] > b["start"]]
    grouped = defaultdict(list)
    for event in events:
        grouped[event["graphNodeId"]].append(event)
    nodes = sorted(grouped.values(), key=lambda group: min(e["start"] for e in group))
    require(len(nodes) == 302, f"expected 302 graph kernel nodes, got {len(nodes)}")
    require(all(len(group) == graph_launches for group in nodes), "not every graph node has exactly 20 calls")
    node_order = [group[0]["graphNodeId"] for group in nodes]
    require([e["graphNodeId"] for e in events] == node_order * graph_launches,
            "kernel order is not the same 302-node order for all 20 graph replays")
    cursor, rows, skipped, unknown, producers = 0, [], [], [], {"images": "backbone-neck"}
    for index, layer in enumerate(layers):
        name, tactic = layer["Name"], layer.get("TacticName", "")
        plugin = re.match(r"^p[345]\.(expand|scan|merge)(?:_|$)", name)
        operation = plugin.group(1) if plugin else None
        if not tactic and not plugin:
            require(index in (0, 285) and not layer.get("Outputs"), f"unexplained kernel-free layer {index}: {name}")
            require(all(t["Format/Datatype"] in ("UInt8", "UInt64") for t in layer.get("Inputs", [])),
                    f"kernel-free layer {index} is not the expected L2 policy control")
            skipped.append({"inspector_index": index, "layer": name, "reason": "L2 policy control; no GPU kernel"})
            continue
        region, reason = region_of(layer, producers)
        if region == "unknown":
            unknown.append({"index": index, "layer": name, "reason": reason})
        for tensor in layer.get("Outputs", []):
            producers[tensor["Name"]] = region
        expected_functions = ["check_finite", operation + "_kernel", "finalize"] if plugin else [None]
        for expected in expected_functions:
            require(cursor < len(nodes), f"ran out of nodes at layer {index}: {name}")
            group = nodes[cursor]
            first = group[0]
            kernel = first["kernel"]
            require(len({e["kernel"] for e in group}) == 1, f"node {cursor} changes kernel name")
            require(len({tuple(e[k] for k in ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ")) for e in group}) == 1,
                    f"node {cursor} changes launch geometry")
            if plugin:
                require("saccade::" in kernel and expected in kernel,
                        f"plugin order mismatch layer {index} {name}: want {expected}, got {kernel}")
            else:
                wanted = tactic.replace("_by_fusion_tactic", "_fused")
                require(kernel.startswith(wanted), f"tactic mismatch layer {index} {name}: want {wanted}, got {kernel}")
            rows.append({
                "inspector_index": index, "layer": name, "region": region,
                "kind": kind_of(layer, kernel, operation), "graph_node_id": first["graphNodeId"],
                "kernel": kernel, "calls": len(group),
                "mean_us": sum(e["end"] - e["start"] for e in group) / len(group) / 1000,
                "grid": "x".join(str(first["grid" + axis]) for axis in "XYZ"),
                "block": "x".join(str(first["block" + axis]) for axis in "XYZ"),
                "region_evidence": reason,
            })
            cursor += 1
    require(cursor == len(nodes), f"{len(nodes) - cursor} kernel nodes remain unconsumed")
    require(not unknown, f"unknown layer regions: {unknown}")
    by_region_kind, by_region, by_kind, by_layer = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    layer_kinds, layer_nodes = defaultdict(set), Counter()
    for row in rows:
        region, kind, elapsed, index = row["region"], row["kind"], row["mean_us"], row["inspector_index"]
        by_region_kind[(region, kind)] += elapsed
        by_region[region] += elapsed
        by_kind[kind] += elapsed
        by_layer[index] += elapsed
        layer_kinds[index].add(kind)
        layer_nodes[index] += 1
    busy = sum(row["mean_us"] for row in rows)
    span = (events[-1]["end"] - events[0]["start"]) / 1000 / graph_launches
    replay_spans = [(events[(i + 1) * len(nodes) - 1]["end"] - events[i * len(nodes)]["start"]) / 1000
                    for i in range(graph_launches)]
    top_layers = [{"inspector_index": index, "layer": layers[index]["Name"],
                   "region": next(row["region"] for row in rows if row["inspector_index"] == index),
                   "kinds": sorted(layer_kinds[index]), "kernel_nodes": layer_nodes[index],
                   "mean_us": elapsed, "percent_gpu_busy": 100 * elapsed / busy}
                  for index, elapsed in sorted(by_layer.items(), key=lambda pair: pair[1], reverse=True)]
    captured_identity = identity(root / "nsys-model/identity.txt")
    summary = {
        "status": "PASS", "capture_kind": "development-optimization", "formal_acceptance": False,
        "metric": "CUDA Graph replay processing intervals and summed GPU kernel busy; not end-to-end frame latency",
        "model": {"scope": "full F03 backbone-neck plus P3/P4/P5 head; input0 held fixed",
                  "engine_sha256": captured_identity["engine_sha256"], "images_sha256": captured_identity["images_sha256"],
                  "profile": captured_identity["profile"], "tensor_precision": "FP16 tactics; FP32 scan and model I/O",
                  "plugin_build_commit": captured_identity["build_commit"], "timing_input": 0},
        "sources": {"sqlite": str(database), "sqlite_sha256": sha256(database),
                    "engine_inspector": str(inspector), "engine_inspector_sha256": sha256(inspector)},
        "graph_launches": graph_launches, "inspector_layers": len(layers), "kernel_nodes": len(nodes), "kernel_calls": len(events),
        "single_stream": True, "stream_ids": streams, "graph_ids": sorted({e["graphId"] for e in events}),
        "adjacent_recorded_interval_overlaps": {"count": len(interval_overlaps_ns),
                                                "max_ns": max(interval_overlaps_ns, default=0),
                                                "sum_ns": sum(interval_overlaps_ns)},
        "all_nodes_have_20_calls": True, "all_kernel_nodes_aligned_to_inspector": True,
        "all_kernel_nodes_consumed": True, "unknown_count": 0, "unknown": [], "kernel_free_layers": skipped,
        "sum_kernel_busy_us_per_replay": busy, "trace_span_us_per_replay": span,
        "mean_within_graph_envelope_us": sum(replay_spans) / graph_launches,
        "sum_kernel_busy_us_per_frame": busy, "trace_span_us_per_frame": span,
        "by_region_kind": [{"region": region, "kind": kind, "us": elapsed} for (region, kind), elapsed in sorted(by_region_kind.items())],
        "by_region": [{"region": region, "us": elapsed, "percent_gpu_busy": 100 * elapsed / busy} for region, elapsed in sorted(by_region.items())],
        "by_kind": [{"kind": kind, "us": elapsed, "percent_gpu_busy": 100 * elapsed / busy} for kind, elapsed in sorted(by_kind.items())],
        "top_layers": top_layers,
        "limitations": ["Twenty instrumented fixed-input replays on one stream; no performance distribution or production single/queued session claim.",
                        "Summed kernel duration is GPU busy; trace span divided by twenty includes inter-replay gaps and is a processing interval, not latency.",
                        "Recorded adjacent kernel intervals have small timestamp overlaps; summed durations are reported as recorded, without inferring physical concurrent execution.",
                        "layout-pointwise includes pooling, softmax/reductions, activations, casts and permutations; it is not all memory copying.",
                        "p3/p4/p5 regions are the three head scales. The model-input cast belongs to backbone-neck.",
                        "No GPU work, model evaluation, GT access or quality scoring occurs during this SQLite analysis."],
    }
    with (root / "kernel-layer-map.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "kernel-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "kernel_nodes", "kernel_calls", "unknown_count", "sum_kernel_busy_us_per_replay", "trace_span_us_per_replay", "by_region", "by_kind")}, indent=2))


if __name__ == "__main__":
    main()
