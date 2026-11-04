#!/usr/bin/env python3
"""Exact fixed-shape span accounting; logical payload, not DRAM traffic."""
from collections import Counter
from fractions import Fraction
import json


def span(index, size, scale):
    center = scale * (Fraction(index) + Fraction(1, 2))
    support = max(scale, Fraction(1))
    return (max(int(center - support + Fraction(1, 2)), 0),
            min(int(center + support + Fraction(1, 2)), size))


xs = [span(x, 1920, Fraction(3, 2)) for x in range(1280)]
ys = [span(y, 1080, Fraction(45, 32)) for y in range(768)]
tiles = []
for y in range(0, 768, 8):
    begin, end = ys[y][0], ys[y + 7][1]
    assert all(begin <= a <= b <= end for a, b in ys[y:y + 8])
    tiles.append(dict(output_y=y, source_begin=begin, source_end=end, rows=end-begin))
assert max(t['rows'] for t in tiles) == 13
x_taps = sum(b-a for a, b in xs)
y_taps = sum(b-a for a, b in ys)
source_rows = 3 * sum(t['rows'] for t in tiles)
result = dict(
    kind='source-level-logical-payload-model-not-DRAM',
    input_shape=[3, 1080, 1920], output_shape=[3, 768, 1280],
    x_tap_counts=dict(Counter(b-a for a, b in xs)),
    y_tap_counts=dict(Counter(b-a for a, b in ys)),
    tile_row_counts=dict(Counter(t['rows'] for t in tiles)),
    baseline_horizontal_values=3*1080*1280,
    tiled_horizontal_values=source_rows*1280,
    baseline_rgb_logical_reads=3*1080*x_taps,
    baseline_scratch_logical_writes=3*1080*1280*4,
    baseline_scratch_logical_reads=3*1280*y_taps*4,
    output_logical_writes=3*768*1280*4,
    tiled_rgb_logical_reads=source_rows*x_taps,
    tiled_scratch_global_reads=0, tiled_scratch_global_writes=0,
    workspace_allocation_bytes_still_reserved=3*1080*1280*4,
    shared_bytes_per_block=16*32*4, tiles=tiles,
)
result['baseline_global_logical_payload'] = sum(result[k] for k in (
    'baseline_rgb_logical_reads', 'baseline_scratch_logical_writes',
    'baseline_scratch_logical_reads', 'output_logical_writes'))
result['tiled_global_logical_payload'] = (
    result['tiled_rgb_logical_reads'] + result['output_logical_writes'])
print(json.dumps(result, indent=2))
