"""Replay real profiler databases with the old and fused F02 name rules.

The fused fixtures are controlled edits of the old profiler capture, not a new
GPU capture: remove the horizontal kernels and rename the vertical kernels to
the exact fused F02 demangled name. The parent MOT run supplies real fused
profiler evidence separately.
"""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

root = Path.cwd()
node, graph, out = (Path(arg).resolve() for arg in sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
checker = root / "scripts/check_stream_placement.py"
legacy_runner = root / "evidence/cuda-graph/2026-11-04-gate-d21a144/placement-checker/mutations.sh"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
inputs = {str(path): digest(path) for path in (node, graph, checker, legacy_runner)}
(out / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
legacy = subprocess.run(
    ["bash", str(legacy_runner), str(node), str(graph), str(out / "legacy-work"), str(out / "legacy")],
    text=True, capture_output=True, check=True,
)
(out / "legacy-results.tsv").write_text(legacy.stdout)
(out / "legacy-stderr.txt").write_text(legacy.stderr)
rows = list(csv.reader(legacy.stdout.splitlines(), delimiter="\t"))[1:]
assert len(rows) == 9, rows
for name, code, *_ in rows:
    assert int(code) == (0 if name == "unchanged" else 1), (name, code)

fused_node, fused_graph = out / "fused-node.sqlite", out / "fused-graph.sqlite"
for original, transformed in ((node, fused_node), (graph, fused_graph)):
    shutil.copyfile(original, transformed)
    with sqlite3.connect(transformed) as db:
        count = db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName IN (SELECT id FROM StringIds WHERE value LIKE 'saccade::<unnamed>::horizontal(%')").rowcount
        assert count == 30, count
        count = db.execute("UPDATE StringIds SET value='saccade::<unnamed>::tiled_hr(const unsigned char *, float *)' WHERE value LIKE 'saccade::<unnamed>::vertical(%'").rowcount
        assert count == 1, count

results = []


def case(name, expected, which=None, sql=None, diagnostic=None):
    n, g = out / "case-node.sqlite", out / "case-graph.sqlite"
    shutil.copyfile(fused_node, n)
    shutil.copyfile(fused_graph, g)
    if sql:
        with sqlite3.connect(n if which == "node" else g) as db:
            changed = db.execute(sql).rowcount
            assert changed > 0, (name, changed)
    output = out / (name + ".txt")
    run = subprocess.run([sys.executable, "-I", str(checker), str(n), str(g), str(output)], text=True, capture_output=True)
    (out / (name + ".stdout")).write_text(run.stdout)
    (out / (name + ".stderr")).write_text(run.stderr)
    verdict = output.read_text()
    assert run.returncode == expected, (name, run.returncode, verdict)
    if diagnostic:
        assert diagnostic in verdict, (name, diagnostic, verdict)
    results.append({"case": name, "expected_exit": expected, "actual_exit": run.returncode, "verdict": "PASS"})


case("fused-unchanged", 0, diagnostic="stream placement: PASS")
wrong_stream = "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId=(SELECT k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE 'saccade::<unnamed>::predict_kernel%' LIMIT 1) WHERE rowid=(SELECT k.rowid FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE 'saccade::<unnamed>::tiled_hr(%' ORDER BY k.start DESC LIMIT 1)"
case("fused-node-wrong-stream", 1, "node", wrong_stream, "F02 kernels outside graphs on 2 streams")
case("fused-graph-wrong-stream", 1, "graph", wrong_stream, "F02 kernels outside graphs on 2 streams")
missing = "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName IN (SELECT id FROM StringIds WHERE value LIKE 'saccade::<unnamed>::tiled_hr(%')"
case("fused-missing", 1, "node", missing, "no F02 kernel")
unknown = "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=(SELECT id FROM StringIds WHERE value LIKE 'saccade::<unnamed>::tiled_hr_unknown(%' LIMIT 1) WHERE rowid=(SELECT k.rowid FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE 'saccade::<unnamed>::tiled_hr(%' ORDER BY k.start DESC LIMIT 1)"
# Reuse the deleted horizontal name's StringId for one unrecognized kernel;
# keep the earlier tiled_hr kernels to ensure the checker reaches that row.
with sqlite3.connect(fused_node) as db:
    db.execute("UPDATE StringIds SET value='saccade::<unnamed>::tiled_hr_unknown(const unsigned char *, float *)' WHERE value LIKE 'saccade::<unnamed>::horizontal(%'")
case("fused-unknown-kernel", 1, "node", unknown, "unclassified kernel")

for path, expected in inputs.items():
    assert digest(Path(path)) == expected, path
(out / "results.json").write_text(json.dumps({"legacy_cases": 9, "fused_cases": results, "total_cases": 14, "status": "PASS"}, indent=2) + "\n")
with (out / "results.tsv").open("w") as stream:
    writer = csv.writer(stream, delimiter="\t")
    writer.writerow(["case", "expected_exit", "actual_exit", "verdict"])
    for name, code, *_ in rows:
        writer.writerow(["legacy-" + name, 0 if name == "unchanged" else 1, int(code), "PASS"])
    for row in results:
        writer.writerow(row.values())
print("PASS: 9 legacy replay cases and 5 fused-name replay cases")
