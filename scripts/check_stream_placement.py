"""Stream-placement gate of docs/TRACKING.md §17.3 on an nsys sqlite export.

From the first F02 kernel on, every kernel gets exactly one category: kernels
launched inside a TensorRT ``ExecutionContext::enqueueV3`` NVTX range are the
engine (including the saccade scan plugins); every other kernel must match one
name rule below. D is the stream of the F02 kernels and M the stream of the
tracking kernels. Every category must run on its own stream; nvJPEG's kernels
are recorded wherever they run and are the only ones allowed off D and M.

usage: python3 -I check_stream_placement.py SQLITE OUTPUT.txt
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


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    kernels = db.execute(
        "SELECT k.start, k.streamId, d.value, r.globalTid, r.start FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds d ON d.id = k.demangledName "
        "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId "
        "ORDER BY k.start").fetchall()
    ranges = db.execute(
        "SELECT n.globalTid, n.start, n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId "
        "WHERE coalesce(n.text, s.value) = ? AND n.end IS NOT NULL", (ENQUEUE,)).fetchall()
    lines, failures = [], []
    boundary = next((k[0] for k in kernels if re.match(RULES[0][2], k[2])), None)
    if boundary is None or not ranges:
        failures.append("no F02 kernel" if boundary is None else f"no {ENQUEUE} NVTX range")
    kept = [k for k in kernels if boundary is not None and k[0] >= boundary]
    table = {}
    for start, stream, name, tid, api_start in kept:
        found = []
        if tid is not None and any(t == tid and a <= api_start <= b for t, a, b in ranges):
            found.append(ENGINE)
        found += [(c, s) for c, s, pattern in RULES if re.match(pattern, name)]
        if len(found) != 1:
            failures.append(f"{'unclassified' if not found else 'ambiguous'} kernel on stream {stream}: {name}")
            continue
        entry = table.setdefault(found[0], {})
        entry.setdefault(stream, {}).setdefault(name, 0)
        entry[stream][name] += 1
    streams = {}
    for category, side in (("F02", "D"), ("tracking", "M")):
        found = set(table.get((category, side), {}))
        if len(found) != 1:
            failures.append(f"{category} kernels on {len(found)} streams")
        else:
            streams[side] = found.pop()
    if len(streams) == 2 and streams["D"] == streams["M"]:
        failures.append("D and M are the same stream")
    lines.append(f"boundary (first F02 kernel start) = {boundary}")
    lines.append(f"kernels from boundary = {len(kept)}; {ENQUEUE} ranges = {len(ranges)}")
    lines.append(f"D = stream {streams.get('D')}; M = stream {streams.get('M')}")
    lines.append("")
    lines.append("category\trequired\tstream\tkernels\tdistinct names")
    for (category, side), by_stream in sorted(table.items(), key=lambda e: (e[0][1] or "~", e[0][0])):
        for stream, names in sorted(by_stream.items()):
            lines.append(f"{category}\t{side or '-'}\t{stream}\t{sum(names.values())}\t{len(names)}")
            if side is not None and len(streams) == 2 and stream != streams[side]:
                failures.append(f"{category} kernels on stream {stream}, required {side} = {streams[side]}")
    lines.append("")
    lines += [f"FAIL: {f}" for f in dict.fromkeys(failures)] or ["stream placement: PASS"]
    with open(sys.argv[2], "w", encoding="ascii") as out:
        out.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
