#!/usr/bin/env python3
"""Reconstruct the archived G1 GPU schedule from read-only SQLite snapshots.

This is an analysis of a 2026-11-05 profiler capture, not a new GPU run.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics
import tempfile


HERE = Path(__file__).resolve().parent
STEADY = list(range(2, 30))
EXAMPLE = [20, 21, 22]
U = "saccade::<unnamed>::"
S = "saccade::detail::<unnamed>::"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        assert a < b
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def duration(intervals):
    return sum(b - a for a, b in union(intervals))


def intersection(left, right):
    left, right = union(left), union(right)
    return union([(max(a, c), min(b, d)) for a, b in left for c, d in right
                  if max(a, c) < min(b, d)])


def envelope(events):
    return min(k['start'] for k in events), max(k['end'] for k in events)


def intervals(events):
    return [(k['start'], k['end']) for k in events]


def summary(values):
    return {'samples': len(values), 'mean_ms': statistics.mean(values) / 1e6,
            'median_ms': statistics.median(values) / 1e6,
            'min_ms': min(values) / 1e6, 'max_ms': max(values) / 1e6}


def classify_name(name):
    if name.startswith(U):
        name = name[len(U):].split('(', 1)[0]
        if name in ['tiled_hr', 'horizontal', 'vertical']:
            return 'F02'
        if name == 'flow_gray':
            return 'F05'
        if name == 'flow_mask':
            return 'flow_mask'
        if name.startswith('flow_'):
            return 'F06'
        if name == 'predict_kernel':
            return 'predict'
    return None


def load_db(path):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def extract(node_path, graph_path):
    node, graph = load_db(node_path), load_db(graph_path)
    kernels = [dict(r) for r in node.execute(
        'SELECT k.start,k.end,k.streamId AS raw_stream,k.graphId AS graph_id,'
        'k.correlationId AS correlation_id,s.value AS name,r.start AS api_start '
        'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName '
        'LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId ORDER BY k.start')]
    gray = sorted([k for k in kernels if classify_name(k['name']) == 'F05'], key=lambda k: k['api_start'])
    resize = sorted([k for k in kernels if classify_name(k['name']) == 'F02'], key=lambda k: k['api_start'])
    assert len(gray) == len(resize) == 30
    assert {k['raw_stream'] for k in gray + resize} == {13}
    graph_ids = {k['graph_id'] for k in kernels if k['graph_id'] is not None}
    roles = {}
    for gid in graph_ids:
        members = [k for k in kernels if k['graph_id'] == gid]
        has_packet = any(k['name'].startswith(S + 'packet(') for k in members)
        has_decode = any(k['name'].startswith(U + 'decode(') for k in members)
        assert has_packet != has_decode
        roles[gid] = 'M' if has_packet else 'D'
    assert roles == {2: 'M', 5: 'M', 8: 'D', 11: 'D'}
    node_launches = [dict(r) for r in node.execute(
        'SELECT r.start,k.graphId AS graph_id,MIN(k.streamId) AS stream '
        'FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId '
        'JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId '
        'WHERE s.value LIKE "cudaGraphLaunch%" GROUP BY r.correlationId ORDER BY r.start')]
    graph_launches = [dict(r) for r in graph.execute(
        'SELECT r.start,g.graphId AS graph_id,g.streamId AS stream '
        'FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId '
        'JOIN CUPTI_ACTIVITY_KIND_GRAPH_TRACE g ON g.correlationId=r.correlationId '
        'WHERE s.value LIKE "cudaGraphLaunch%" ORDER BY r.start')]
    assert len(node_launches) == len(graph_launches) == 60
    pairing = {}
    for n, g in zip(node_launches, graph_launches):
        assert g['stream'] == {'D': 13, 'M': 14}[roles[n['graph_id']]]
        assert pairing.setdefault(n['graph_id'], g['graph_id']) == g['graph_id']
    assert len(set(pairing.values())) == 4
    copies = [dict(r) for r in node.execute(
        'SELECT c.start,c.end,c.streamId AS raw_stream,c.bytes,r.start AS api_start '
        'FROM CUPTI_ACTIVITY_KIND_MEMCPY c JOIN CUPTI_ACTIVITY_KIND_RUNTIME r '
        'ON r.correlationId=c.correlationId WHERE c.copyKind=2 AND c.streamId=14 '
        'AND c.bytes IN (56,12368) ORDER BY c.start')]
    assert len(copies) == 60
    frames = []
    for index, g in enumerate(gray):
        lower = g['api_start']
        upper = gray[index + 1]['api_start'] if index + 1 < len(gray) else float('inf')
        members = [k for k in kernels if k['api_start'] is not None and lower <= k['api_start'] < upper]
        dg = [k for k in members if roles.get(k['graph_id']) == 'D']
        mg = [k for k in members if roles.get(k['graph_id']) == 'M']
        assert len(dg) == 310 and len(mg) == 9
        assert len({k['correlation_id'] for k in dg}) == len({k['correlation_id'] for k in mg}) == 1
        assert resize[index]['api_start'] < lower and resize[index]['end'] <= g['start']
        f04_names = ['initialize(const saccade::PostprocessProducerStatus *', 'decode(',
                     'adaptive_prefix(', 'dense_merge(', 'finalize(saccade::DetectionCandidates *']
        f04 = [k for k in dg if any(k['name'].startswith(U + name) for name in f04_names)]
        producer = [k for k in dg if k not in f04 and
                    any(k['name'].startswith(U + name) for name in ['initialize(saccade::PostprocessProducerStatus *', 'finite(', 'aggregate('])]
        f03 = [k for k in dg if k not in f04 + producer]
        assert len(f03) == 302 and len(producer) == 3 and len(f04) == 5
        plain = [k for k in members if k['graph_id'] is None]
        flow_input, = [k for k in plain if k['name'].startswith(U + 'flow_input(')]
        flow_context, = [k for k in plain if k['name'].startswith(U + 'flow_context(')]
        flow_output, = [k for k in plain if k['name'].startswith(U + 'flow_output(')]
        flow_cells, = [k for k in plain if k['name'].startswith(U + 'flow_cells(')]
        flow_mask, = [k for k in plain if k['name'].startswith(U + 'flow_mask(')]
        predict = [k for k in plain if classify_name(k['name']) == 'predict']
        assert len(predict) == 2
        f06 = [k for k in plain if k['raw_stream'] == 14 and
               flow_input['start'] <= k['start'] < k['end'] <= flow_output['end']]
        assert all(classify_name(k['name']) == 'F06' or k['name'].startswith('void cub::') or
                   k['name'] in ['Convert_PL2BL', 'Convert_BL2PL', 'Gaussian_Subsample2x2_Luma_BL2BL_v2'] for k in f06)
        assert flow_context['end'] <= flow_output['start'] < flow_output['end'] <= envelope(predict)[0]
        assert envelope(predict)[1] <= flow_mask['start']
        assert flow_mask['end'] <= envelope(mg)[0]
        d2h = [k for k in copies if lower <= k['api_start'] < upper]
        assert len(d2h) == 2 and sorted(k['bytes'] for k in d2h) == [56, 12368]
        assert envelope(mg)[1] <= envelope(d2h)[0]
        frames.append({'frame': index, 'source_frame': index + 1,
                       'stages': {'F02': [resize[index]], 'F05': [g], 'F03': f03,
                                  'producer': producer, 'F04': f04, 'F06': f06,
                                  'predict': predict, 'flow_mask': [flow_mask],
                                  'tracking_tail': mg, 'D2H': d2h},
                       'D': [resize[index], g] + dg,
                       'ofa_bound': (flow_input['end'], flow_cells['start'])})
    return frames, roles, pairing


def main():
    provenance = json.loads((HERE / 'source-inputs.json').read_text())
    for source in provenance['supporting_snapshots']:
        raw = (HERE / source['snapshot']).read_bytes()
        assert len(raw) == source['bytes'] and digest(raw) == source['sha256']
    expected = {Path(line.split(maxsplit=1)[1]).name: line.split()[0]
                for line in (HERE / 'inputs/nsys.sha256').read_text().splitlines()}
    with tempfile.TemporaryDirectory(prefix='saccade-schedule-sqlite-') as temp:
        paths = {}
        for source in provenance['sqlite_inputs']:
            packed = HERE / source['gzip_path']
            assert digest(packed.read_bytes()) == source['gzip_sha256']
            raw = gzip.decompress(packed.read_bytes())
            assert len(raw) == source['bytes'] and digest(raw) == source['sha256'] == expected[source['name']]
            path = Path(temp) / source['name']; path.write_bytes(raw); paths[source['name']] = path
        frames, roles, pairing = extract(paths['track.sqlite'], paths['track-graph.sqlite'])
    rows = []
    stage_segments = []
    for f in frames:
        stages = f['stages']; f03 = envelope(stages['F03']); f06 = envelope(stages['F06'])
        tail_kernels = stages['flow_mask'] + stages['tracking_tail']
        tail = envelope(tail_kernels)
        graph_tail = envelope(stages['tracking_tail'])
        bound = f['ofa_bound']
        row = {'frame': f['frame'], 'source_frame': f['source_frame'], 'steady': f['frame'] in STEADY,
               'f03_start_ns': f03[0], 'f03_end_ns': f03[1], 'f03_envelope_ns': f03[1] - f03[0],
               'f03_kernel_union_ns': duration(intervals(stages['F03'])),
               'f06_start_ns': f06[0], 'f06_end_ns': f06[1], 'f06_envelope_ns': f06[1] - f06[0],
               'f06_cuda_kernel_union_ns': duration(intervals(stages['F06'])),
               'f06_envelope_contained_in_f03': f03[0] <= f06[0] < f06[1] <= f03[1],
               'ofa_bound_start_ns': bound[0], 'ofa_bound_end_ns': bound[1],
               'ofa_bound_ns': bound[1] - bound[0],
               'ofa_bound_contained_in_f03': f03[0] <= bound[0] < bound[1] <= f03[1],
               'tail_start_ns': tail[0], 'tail_end_ns': tail[1],
               'tail_envelope_ns': tail[1] - tail[0],
               'tail_kernel_union_ns': duration(intervals(tail_kernels)),
               'M_graph_tail_start_ns': graph_tail[0], 'M_graph_tail_end_ns': graph_tail[1],
               'M_graph_tail_kernel_union_ns': duration(intervals(stages['tracking_tail']))}
        if f['frame'] < len(frames) - 1:
            nxt = frames[f['frame'] + 1]
            busy_overlap = intersection(intervals(tail_kernels), intervals(nxt['D']))
            env_overlap = intersection([tail], [envelope(nxt['D'])])
            next_f03_busy = intersection(intervals(tail_kernels), intervals(nxt['stages']['F03']))
            row.update({'next_frame': nxt['frame'], 'next_source_frame': nxt['source_frame'],
                        'cross_tail_nextD_envelope_overlap_ns': duration(env_overlap),
                        'cross_tail_nextD_kernel_union_overlap_ns': duration(busy_overlap),
                        'cross_tail_nextF03_kernel_union_overlap_ns': duration(next_f03_busy),
                        'cross_M_graph_nextD_kernel_union_overlap_ns': duration(intersection(
                            intervals(stages['tracking_tail']), intervals(nxt['D'])))})
            for stage in ['F02', 'F05', 'F03', 'producer', 'F04']:
                row[f'cross_tail_next_{stage}_kernel_union_overlap_ns'] = duration(intersection(
                    intervals(tail_kernels), intervals(nxt['stages'][stage])))
        rows.append(row)
        if f['frame'] in EXAMPLE:
            for name, events in stages.items():
                a, b = envelope(events); role = 'D' if name in ['F02', 'F05', 'F03', 'producer', 'F04'] else 'M'
                stage_segments.append({'frame': f['frame'], 'source_frame': f['source_frame'],
                                       'stage': name, 'role': role, 'start_ns': a, 'end_ns': b,
                                       'interval_kind': 'stage_envelope',
                                       'logical_stream_id': {'D': 13, 'M': 14}[role],
                                       'raw_profiler_stream_ids': sorted({k['raw_stream'] for k in events}),
                                       'activity_count': len(events),
                                       'busy_ns': duration(intervals(events)),
                                       'busy_intervals_ns': union(intervals(events))})
    steady_rows = [r for r in rows if r['steady']]
    pairs = [r for r in steady_rows if r.get('next_frame') in STEADY]
    assert len(steady_rows) == 28 and len(pairs) == 27
    stats = {
        'schema': 'saccade-archived-schedule-analysis/1', 'analysis_date': '2026-11-12',
        'capture_source': provenance['capture_source'], 'new_GPU_capture': False,
        'total_frames': 30, 'steady_frame_indices': STEADY, 'steady_frames': 28,
        'steady_source_frames': [i + 1 for i in STEADY], 'adjacent_steady_pairs': 27,
        'frame_numbering': 'frame is zero-based enqueue index; source_frame is one-based JPEG frame',
        'logical_streams': {'D': 13, 'M': 14}, 'node_graph_roles': roles,
        'node_graph_to_graph_capture_ids': pairing,
        'f06_envelope_fully_inside_same_frame_f03': sum(r['f06_envelope_contained_in_f03'] for r in steady_rows),
        'ofa_bound_fully_inside_same_frame_f03': sum(r['ofa_bound_contained_in_f03'] for r in steady_rows),
        'f03_envelope': summary([r['f03_envelope_ns'] for r in steady_rows]),
        'f06_envelope': summary([r['f06_envelope_ns'] for r in steady_rows]),
        'ofa_indirect_bound': summary([r['ofa_bound_ns'] for r in steady_rows]),
        'tail_kernel_union': summary([r['tail_kernel_union_ns'] for r in steady_rows]),
        'cross_frame_pairs_with_actual_kernel_overlap': sum(r['cross_tail_nextD_kernel_union_overlap_ns'] > 0 for r in pairs),
        'cross_tail_nextD_envelope_overlap': summary([r['cross_tail_nextD_envelope_overlap_ns'] for r in pairs]),
        'cross_tail_nextD_kernel_union_overlap': summary([r['cross_tail_nextD_kernel_union_overlap_ns'] for r in pairs]),
        'cross_tail_nextF03_kernel_union_overlap': summary([r['cross_tail_nextF03_kernel_union_overlap_ns'] for r in pairs]),
        'methods': {
            'frame_mapping': 'One ordered F02/F05 pair per frame; each F05 host API window has exactly one D graph launch (310 kernels) and one M tail graph launch (9 kernels). Both graph roles and actual D/M stream IDs verified by matching all 60 node/graph capture launch orders. Frame0 has extra TensorRT warmup/capture work and is retained but excluded from steady statistics; frame1 is first Execute.',
            'same_frame_F06': 'flow_input start through flow_output end, including the final motion-image write; excludes later flow_mask and predict. Full interval includes CUDA and indirect fixed-function work/waits.',
            'OFA_bound': 'flow_input end through flow_cells start. This is an indirect enclosing interval, not OFA engine activity or exact latency.',
            'D_next_activity': 'CUDA kernels of F02/F05/F03/producer/F04 for n+1; excludes nvJPEG private stream, JPEG hardware, H2D and OFA hardware.',
            'M_tail_activity': 'Post-candidate flow_mask plus all 9 real CUDA kernels of the n-th M graph, ending at packet; excludes earlier F06/predict and subsequent D2H. Separate M_graph fields omit flow_mask.',
            'actual_overlap': 'Intersect union of actual kernel start/end intervals across both roles; no stage gap or engine bound counted as kernel busy.',
            'envelope_overlap': 'Intersect minimum-start/maximum-end bounding intervals; includes gaps and is reported separately.'},
        'limits': ['Historical clean fae5cb7 capture, driver 617.42, G1, same fixed HR engine; not a fresh trace of 3cbaa518f0dc.',
                   'Nsight Systems node instrumentation changes execution timing; these numbers are schedule evidence, not unprofiled latency or current FPS.',
                   'OFA fixed-function engine is not directly visible; F06/OFA intervals do not measure OFA utilization.',
                   'Kernel activity overlap does not establish occupancy, speedup or absence of races; original sanitizer OPEN/FAIL remains.']}
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (HERE / 'per-frame.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    timeline = {'schema': 'saccade-report-schedule-timeline/1', 'capture_source': provenance['capture_source'],
                'frame_numbering': stats['frame_numbering'], 'frames': EXAMPLE,
                'source_frames': [i + 1 for i in EXAMPLE],
                'start_ns': min(s['start_ns'] for s in stage_segments),
                'end_ns': max(s['end_ns'] for s in stage_segments),
                'segments': sorted(stage_segments, key=lambda s: (s['start_ns'], s['role'])),
                'tail_definition': 'flow_mask plus tracking_tail (the 9-kernel M graph). These remain separate drawable stages.',
                'representative_pairs': [r for r in pairs if r['frame'] in EXAMPLE[:-1]],
                'interval_kind': 'stage envelopes; busy_intervals_ns records actual CUDA activity separately',
                'all_values_from': 'per-frame.csv and read-only archived SQLite; no new GPU capture',
                'limits': stats['limits']}
    for name, value in [('summary.json', stats), ('timeline.json', timeline)]:
        (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'steady_frames': 28, 'pairs': 27,
                      'same_frame_F06_contained': stats['f06_envelope_fully_inside_same_frame_f03'],
                      'actual_overlap_pairs': stats['cross_frame_pairs_with_actual_kernel_overlap'],
                      'actual_overlap_ms': stats['cross_tail_nextD_kernel_union_overlap'],
                      'F03_next_actual_overlap_ms': stats['cross_tail_nextF03_kernel_union_overlap'],
                      'f03_ms': stats['f03_envelope'], 'f06_ms': stats['f06_envelope']}, indent=2))


if __name__ == '__main__':
    main()
