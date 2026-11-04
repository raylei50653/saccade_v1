#!/usr/bin/env bash
# Negative controls for scripts/check_stream_placement.py: each case edits a
# copy of the node or graph sqlite export and must make the checker exit 1;
# the unchanged copy must pass.
# usage: mutations.sh NODE.sqlite GRAPH.sqlite WORKDIR OUTDIR
set -u
node=$1 graph=$2 work=$3 out=$4
mkdir -p "$work" "$out"
mut() {
  local name=$1 which=$2 sql=$3 code
  cp "$node" "$work/n.sqlite"; cp "$graph" "$work/g.sqlite"
  sqlite3 "$work/$which.sqlite" "$sql" > /dev/null
  python3 -I scripts/check_stream_placement.py "$work/n.sqlite" "$work/g.sqlite" "$out/$name.txt" > /dev/null 2>&1
  code=$?
  printf '%s\t%s\t%s\n' "$name" "$code" "$(grep -m1 '^FAIL\|^stream placement' "$out/$name.txt")"
}
printf 'case\texit\tfirst verdict line\n'
mut d-graph-on-M g "UPDATE CUPTI_ACTIVITY_KIND_GRAPH_TRACE SET streamId=(SELECT max(streamId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE) WHERE rowid=(SELECT rowid FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE streamId=(SELECT min(streamId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE) LIMIT 1 OFFSET 5);"
mut m-graph-on-D g "UPDATE CUPTI_ACTIVITY_KIND_GRAPH_TRACE SET streamId=(SELECT min(streamId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE) WHERE graphId=(SELECT max(graphId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE streamId=(SELECT max(streamId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE));"
mut trt-in-M-graph n "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphId=(SELECT k.graphId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE 'saccade::detail::<unnamed>::packet%' AND k.graphId IS NOT NULL LIMIT 1) WHERE rowid IN (SELECT k.rowid FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.graphId IS NOT NULL AND s.value LIKE 'sm80_xmma%' LIMIT 1);"
mut tracking-in-D-graph n "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphId=(SELECT k.graphId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE 'sm80_xmma%' AND k.graphId IS NOT NULL LIMIT 1) WHERE rowid IN (SELECT k.rowid FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.graphId IS NOT NULL AND s.value LIKE 'saccade::<unnamed>::batch_kernel%' LIMIT 1);"
mut launch-order-swap g "UPDATE CUPTI_ACTIVITY_KIND_GRAPH_TRACE SET graphId=(SELECT max(graphId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE streamId=(SELECT min(streamId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE)) + (SELECT min(graphId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE streamId=(SELECT min(streamId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE)) - graphId WHERE rowid=(SELECT rowid FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE streamId=(SELECT min(streamId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE) ORDER BY start LIMIT 1);"
mut missing-launch g "DELETE FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE rowid=(SELECT max(rowid) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE);"
mut F02-on-M n "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId=(SELECT k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE 'saccade::<unnamed>::predict_kernel%' LIMIT 1) WHERE rowid IN (SELECT k.rowid FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE 'saccade::<unnamed>::vertical%' ORDER BY k.start DESC LIMIT 1);"
mut predict-on-D n "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId=(SELECT k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE 'saccade::<unnamed>::vertical%' LIMIT 1) WHERE rowid IN (SELECT k.rowid FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE 'saccade::<unnamed>::predict_kernel%' ORDER BY k.start DESC LIMIT 1);"
mut unchanged n "SELECT 1;"
