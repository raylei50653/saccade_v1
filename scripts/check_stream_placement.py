"""Stream-placement gate of docs/TRACKING.md §17.3 on two nsys sqlite exports.

NODE is a ``--cuda-graph-trace=node`` capture and GRAPH a
``--cuda-graph-trace=graph`` capture of the same binary, input and settings.

From the first F02 kernel on, every kernel of NODE gets exactly one category.
Kernels outside CUDA Graphs keep the original rules: kernels launched inside a
TensorRT ``ExecutionContext::enqueueV3`` NVTX range are the engine (including
the saccade scan plugins); every other kernel must match one name rule below.
D is the stream of the F02 kernels and M the stream of the tracking kernels
outside graphs, and every category must run on its own stream; nvJPEG's
kernels are recorded wherever they run and are the only ones allowed off D and
M.

Node tracing gives graph kernels synthetic profiler streams, so graph kernels
are checked by content instead: every graph must hold only D categories (a D
graph, where kernels matching no rule are the engine) or only M categories (an
M graph, where every kernel must match a rule). The real stream of each graph
launch comes from GRAPH, which records one row per launch with the stream that
cudaGraphLaunch used. Both captures replay the same launch sequence: the i-th
cudaGraphLaunch of NODE and of GRAPH must be the same graph role, and every D
graph launch must be on GRAPH's D stream and every M graph launch on its M
stream. Graph IDs are matched one to one between the two captures.

usage: python3 -I check_stream_placement.py NODE.sqlite GRAPH.sqlite OUTPUT.txt
"""

import re
import sqlite3
import sys

U = r"saccade::<unnamed>::"
S = r"saccade::detail::<unnamed>::"
# (category, stream, demangled-name regex); "engine" is the NVTX rule.
RULES = [
    ("F02", "D", U + r"(horizontal|vertical)\("),
    ("F05", "D", U + r"(image_kernel|image_header)\("),
    ("producer", "D", U + r"(initialize\(saccade::PostprocessProducerStatus \*|finite\(|aggregate\()"),
    ("F04", "D", U + r"(initialize\(const saccade::PostprocessProducerStatus \*|decode\(|adaptive_prefix\(|dense_merge\(|finalize\(saccade::DetectionCandidates \*)"),
    ("cuFFT", "M", r"void regular_fft(_r2c|_c2r)?<"),
    ("F06", "M", U + r"(reset_numeric|motion_stage|deliver|cross_power|reduce_correlation|context_kernel)\("),
    ("tracking", "M", U + r"(predict_kernel|batch_kernel|update_kernel)\("),
    ("session control", "M", S + r"(initialize|begin|stage|candidates|commit)\("),
    ("packet", "M", S + r"packet\("),
    ("nvJPEG", None, r"void nvjpeg::"),
]
ENGINE = ("TensorRT engine", "D")
ENQUEUE = "ExecutionContext::enqueueV3"
LAUNCH = "cudaGraphLaunch%"


def rule(name):
    return [(c, s) for c, s, pattern in RULES if re.match(pattern, name)]


def launches(db, table, graph_column):
    """cudaGraphLaunch calls in host call order: (correlationId, graphId, streamId)."""
    return db.execute(
        f"SELECT r.correlationId, t.{graph_column}, t.streamId FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id = r.nameId "
        f"JOIN (SELECT correlationId, MIN({graph_column}) AS {graph_column}, MIN(streamId) AS streamId, "
        f"COUNT(DISTINCT {graph_column}) AS graphs FROM {table} "
        f"WHERE {graph_column} IS NOT NULL GROUP BY correlationId) t ON t.correlationId = r.correlationId "
        "WHERE s.value LIKE ? AND t.graphs = 1 ORDER BY r.start", (LAUNCH,)).fetchall()


def side_streams(kernels, failures, label):
    """D = stream of F02 kernels, M = stream of tracking kernels, both outside graphs."""
    streams = {}
    for category, side in (("F02", "D"), ("tracking", "M")):
        found = {stream for _, stream, name, *_ in kernels if (category, side) in rule(name)}
        if len(found) != 1:
            failures.append(f"{label}: {category} kernels outside graphs on {len(found)} streams")
        else:
            streams[side] = found.pop()
    if len(streams) == 2 and streams["D"] == streams["M"]:
        failures.append(f"{label}: D and M are the same stream")
    return streams


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    node = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    graph = sqlite3.connect(f"file:{sys.argv[2]}?mode=ro", uri=True)
    kernels = node.execute(
        "SELECT k.start, k.streamId, d.value, r.globalTid, r.start, k.graphId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds d ON d.id = k.demangledName "
        "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId "
        "ORDER BY k.start").fetchall()
    ranges = node.execute(
        "SELECT n.globalTid, n.start, n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId "
        "WHERE coalesce(n.text, s.value) = ? AND n.end IS NOT NULL", (ENQUEUE,)).fetchall()
    lines, failures = [], []
    boundary = next((k[0] for k in kernels if re.match(RULES[0][2], k[2])), None)
    if boundary is None or not ranges:
        failures.append("no F02 kernel" if boundary is None else f"no {ENQUEUE} NVTX range")
    kept = [k for k in kernels if boundary is not None and k[0] >= boundary]
    plain = [k for k in kept if k[5] is None]
    in_graph = [k for k in kept if k[5] is not None]

    # Kernels outside graphs: original rules and streams.
    table = {}
    for start, stream, name, tid, api_start, _ in plain:
        found = []
        if tid is not None and any(t == tid and a <= api_start <= b for t, a, b in ranges):
            found.append(ENGINE)
        found += rule(name)
        if len(found) != 1:
            failures.append(f"{'unclassified' if not found else 'ambiguous'} kernel on stream {stream}: {name}")
            continue
        entry = table.setdefault(found[0], {})
        entry.setdefault(stream, {}).setdefault(name, 0)
        entry[stream][name] += 1
    streams = side_streams(plain, failures, "node")

    # Graph kernels: one role per graph, from its content.
    roles, graph_table = {}, {}
    for graph_id in sorted({k[5] for k in in_graph}):
        members = [k for k in in_graph if k[5] == graph_id]
        sides = {s for k in members for _, s in rule(k[2])}
        if sides not in ({"D"}, {"M"}):
            failures.append(f"graph {graph_id}: categories from sides {sorted(map(str, sides))}")
            continue
        role = sides.pop()
        roles[graph_id] = role
        for _, stream, name, *_ in members:
            found = rule(name) or ([ENGINE] if role == "D" else [])
            if len(found) != 1 or found[0][1] != role:
                failures.append(f"graph {graph_id} ({role}): {'unclassified' if not found else 'misplaced'} kernel {name}")
                continue
            entry = graph_table.setdefault((graph_id, role, found[0][0]), {})
            entry[name] = entry.get(name, 0) + 1
    for role, needed in (("D", ("producer", "F04", "TensorRT engine")), ("M", ("tracking", "packet"))):
        for graph_id in [g for g, r in roles.items() if r == role]:
            missing = [c for c in needed if (graph_id, role, c) not in graph_table]
            if missing:
                failures.append(f"graph {graph_id} ({role}): no {', '.join(missing)} kernels")

    # Launch streams from the graph-level capture, matched by launch order.
    node_launches = launches(node, "CUPTI_ACTIVITY_KIND_KERNEL", "graphId")
    graph_launches = launches(graph, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "graphId")
    graph_kernels = graph.execute(
        "SELECT k.start, k.streamId, d.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds d ON d.id = k.demangledName WHERE k.graphId IS NULL ORDER BY k.start").fetchall()
    launch_streams = side_streams(graph_kernels, failures, "graph capture")
    if len(node_launches) != len(graph_launches):
        failures.append(f"{len(node_launches)} graph launches in the node capture, {len(graph_launches)} in the graph capture")
    pairing, launch_table = {}, {}
    for (_, node_id, _), (_, graph_id, stream) in zip(node_launches, graph_launches):
        role = roles.get(node_id)
        if role is None:
            failures.append(f"launch of node graph {node_id} has no role")
            continue
        if pairing.setdefault(node_id, graph_id) != graph_id:
            failures.append(f"node graph {node_id} matches graph-capture graphs {pairing[node_id]} and {graph_id}")
        key = (node_id, graph_id, role, stream)
        launch_table[key] = launch_table.get(key, 0) + 1
        if len(launch_streams) == 2 and stream != launch_streams[role]:
            failures.append(f"{role} graph {node_id}/{graph_id} launched on stream {stream}, required {role} = {launch_streams[role]}")
    if len(set(pairing.values())) != len(pairing):
        failures.append("graph IDs do not match one to one between the captures")
    if in_graph and not node_launches:
        failures.append("graph kernels without cudaGraphLaunch correlation")

    lines.append(f"boundary (first F02 kernel start) = {boundary}")
    lines.append(f"kernels from boundary = {len(kept)} ({len(plain)} outside graphs, {len(in_graph)} in graphs); {ENQUEUE} ranges = {len(ranges)}")
    lines.append(f"D = stream {streams.get('D')}; M = stream {streams.get('M')}")
    lines.append("")
    lines.append("outside graphs")
    lines.append("category\trequired\tstream\tkernels\tdistinct names")
    for (category, side), by_stream in sorted(table.items(), key=lambda e: (e[0][1] or "~", e[0][0])):
        for stream, names in sorted(by_stream.items()):
            lines.append(f"{category}\t{side or '-'}\t{stream}\t{sum(names.values())}\t{len(names)}")
            if side is not None and len(streams) == 2 and stream != streams[side]:
                failures.append(f"{category} kernels on stream {stream}, required {side} = {streams[side]}")
    lines.append("")
    lines.append("graph content (node capture)")
    lines.append("graph\trole\tcategory\tkernels\tdistinct names")
    for (graph_id, role, category), names in sorted(graph_table.items(), key=lambda e: (e[0][0], e[0][2])):
        lines.append(f"{graph_id}\t{role}\t{category}\t{sum(names.values())}\t{len(names)}")
    lines.append("")
    lines.append(f"graph launches (graph capture): D = stream {launch_streams.get('D')}; M = stream {launch_streams.get('M')}; "
                 f"{len(node_launches)} node / {len(graph_launches)} graph launches")
    lines.append("node graph\tgraph-capture graph\trole\tlaunch stream\tlaunches")
    for (node_id, graph_id, role, stream), count in sorted(launch_table.items()):
        lines.append(f"{node_id}\t{graph_id}\t{role}\t{stream}\t{count}")
    lines.append("")
    lines += [f"FAIL: {f}" for f in dict.fromkeys(failures)] or ["stream placement: PASS"]
    with open(sys.argv[3], "w", encoding="ascii") as out:
        out.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
