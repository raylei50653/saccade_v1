#!/usr/bin/env node
// Dependency-free tests for the formal v1-mot/1 display reconstruction.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const gapFill = require('../gap-fill.js');
const checks = [];
function check(name, fn) { fn(); checks.push({ name, status: 'PASS' }); }
function row(id, uid, frame, flags = 1) {
  return [id, uid, frame, 2 * frame, 3 * frame + 1, 4 * frame + 2, 0.5, 0.7, flags];
}
function fixture(times, length = 46) {
  const frames = Array.from({ length }, () => ({ rows: [] }));
  for (const frame of times) frames[frame].rows.push(row(4, 40, frame));
  return frames;
}
function deepFreeze(value) {
  Object.freeze(value);
  for (const child of Object.values(value)) if (child && typeof child === 'object') deepFreeze(child);
  return value;
}
check('35 missing source frames fill; 36 missing source frames do not', () => {
  const index = gapFill.build(fixture([2, 3, 4, 5, 41]));
  assert.equal(index.stats.intervals, 1);
  assert.equal(index.stats.filledRows, 35);
  assert.equal(index.byFrame[6][0].missingFrames, 35);
  for (let frame = 6; frame < 41; frame++) {
    const sample = gapFill.sample(index, frame)[0];
    assert.deepEqual(sample.xyxy, [frame, 2 * frame, 3 * frame + 1, 4 * frame + 2]);
    assert.equal(sample.confidence, -1);
    assert.equal(sample.method, 'v1-mot/1');
  }
  assert.equal(gapFill.build(fixture([2, 3, 4, 5, 42])).stats.filledRows, 0);
});
check('four input rows do not fill; five input rows qualify', () => {
  assert.equal(gapFill.build(fixture([2, 3, 4, 40])).stats.filledRows, 0);
  assert.equal(gapFill.build(fixture([2, 3, 4, 5, 40])).stats.filledRows, 34);
});
check('leading, trailing and existing source rows are never synthesized', () => {
  const index = gapFill.build(fixture([2, 3, 4, 5, 41]));
  for (const frame of [-1, 0, 1, 2, 3, 4, 5, 41, 42, 45, 46, 1.5]) {
    assert.deepEqual(gapFill.sample(index, frame, 0.5), []);
  }
});
check('formal MOT rule allows UID changes while retaining their provenance', () => {
  const frames = fixture([2, 3, 4, 5, 10]);
  frames[10].rows[0][1] = 99;
  const index = gapFill.build(frames), sample = gapFill.sample(index, 7)[0];
  assert.equal(index.stats.filledRows, 4);
  assert.equal(sample.outputId, 4); assert.equal(sample.motId, 1);
  assert.equal(sample.fromUid, 40); assert.equal(sample.toUid, 99);
  assert.deepEqual(sample.xyxy, [7, 14, 22, 30]);
});
check('different output IDs never merge even when the internal UID is the same', () => {
  const frames = fixture([0, 1, 2, 3, 4, 5], 8);
  for (const frame of [3, 4, 5]) frames[frame].rows[0][0] = 8;
  const index = gapFill.build(frames);
  assert.equal(index.stats.outputIds, 2);
  assert.equal(index.stats.filledRows, 0);
});
check('all packet row flags count and may form endpoints as in SequenceOutput::add', () => {
  const frames = fixture([2, 3, 4, 5, 10]);
  for (const frame of [2, 3, 4, 5, 10]) frames[frame].rows[0][8] = frame === 10 ? 2 : 0;
  const index = gapFill.build(frames);
  assert.equal(index.stats.inputRows, 5);
  assert.equal(index.stats.filledRows, 4);
  assert.deepEqual(gapFill.sample(index, 7)[0].xyxy, [7, 14, 22, 30]);
});
check('formal rule adds no geometry filter beyond the original packet rows', () => {
  const frames = fixture([2, 3, 4, 5, 10]);
  frames[5].rows[0].splice(2, 4, 9, 8, 3, 2);
  frames[10].rows[0].splice(2, 4, 19, 18, 13, 12);
  const sample = gapFill.sample(gapFill.build(frames), 7)[0];
  assert.deepEqual(sample.xyxy, [13, 12, 7, 6]);
});
check('MOT IDs follow global first appearance and packet order, not numeric order', () => {
  const frames = [{ rows: [row(70, 100, 0), row(2, 7, 0)] },
    { rows: [row(99, 7, 1), row(70, 101, 1)] }];
  const index = gapFill.build(frames);
  assert.deepEqual({ ...index.motIdByOutputId }, { 2: 2, 70: 1, 99: 3 });
});
check('binary64 fractions use the same arithmetic order as formal interpolation', () => {
  const frames = [{ rows: [[7, 70, 0, 1, 2, 3, 0.5, 0.7, 1]] }, { rows: [] }, { rows: [] },
    { rows: [[7, 71, 1, 3, 5, 7, 0.5, 0.7, 1]] }];
  const index = gapFill.build(frames, { maxMissingFrames: 2, minObservations: 2 });
  const first = gapFill.sample(index, 1)[0], second = gapFill.sample(index, 2)[0];
  assert.deepEqual(first.xyxy, [1 / 3, 1 + (1 / 3) * 2, 3, 3 + (1 / 3) * 4]);
  assert.equal(second.xyxy[0], 2 / 3);
  assert.equal(first.fraction, 1 / 3); assert.equal(second.fraction, 2 / 3);
});
check('phase is display-only, clamped, and gives exact interval midpoints', () => {
  const index = gapFill.build(fixture([0, 1, 2, 3, 6], 8));
  assert.deepEqual(gapFill.sample(index, 4, 0.5)[0].xyxy, [4.5, 9, 14.5, 20]);
  assert.deepEqual(gapFill.sample(index, 4, -1), gapFill.sample(index, 4, 0));
  assert.deepEqual(gapFill.sample(index, 4, NaN), gapFill.sample(index, 4, 0));
  assert.deepEqual(gapFill.sample(index, 4, Infinity), gapFill.sample(index, 4, 0));
  assert.deepEqual(gapFill.sample(index, 4, 5), gapFill.sample(index, 4, 1));
  assert.deepEqual(gapFill.sample(index, 5, 1)[0].xyxy, [6, 12, 19, 26]);
});
check('input rows stay immutable and sampled coordinates do not alias the index', () => {
  const frames = deepFreeze(fixture([2, 3, 4, 5, 10]));
  const before = JSON.stringify(frames), index = gapFill.build(frames);
  const sample = gapFill.sample(index, 7)[0]; sample.xyxy[0] = 12345;
  assert.equal(JSON.stringify(frames), before);
  assert.equal(gapFill.sample(index, 7)[0].xyxy[0], 7);
});
check('fill descriptors are ordered by formal MOT ID within each source frame', () => {
  const frames = fixture([0, 1, 2, 3, 7], 9);
  for (const frame of [0, 1, 2, 3, 7]) frames[frame].rows.unshift(row(20, 200, frame));
  const index = gapFill.build(frames);
  assert.deepEqual(gapFill.sample(index, 4).map(r => [r.outputId, r.motId]), [[20, 1], [4, 2]]);
});
check('empty inputs and zero gap configuration emit no synthetic rows', () => {
  const empty = gapFill.build([]);
  assert.deepEqual(empty.byFrame, []); assert.equal(empty.stats.filledRows, 0);
  assert.deepEqual(gapFill.sample(empty, 0), []);
  assert.equal(gapFill.build(fixture([0, 1, 2, 3, 7]), { maxMissingFrames: 0 }).stats.filledRows, 0);
});
check('module is available both as CommonJS and a buildless browser global', () => {
  assert.equal(globalThis.SaccadeGapFill, gapFill);
  const context = vm.createContext({ window: {} });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../gap-fill.js'), 'utf8'), context);
  assert.equal(typeof context.window.SaccadeGapFill.build, 'function');
  assert.equal(typeof context.window.SaccadeGapFill.sample, 'function');
});
const datasets = [];
for (const [name, filledRows] of [['data.json', 934], ['data-02.json', 2040], ['data-04.json', 2053]]) {
  check(`${name} reproduces its retained formal interpolation row count`, () => {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, '../assets', name), 'utf8'));
    const index = gapFill.build(data.frames);
    assert.equal(index.stats.filledRows, filledRows);
    datasets.push({ sequence: data.meta.sequence, ...index.stats });
  });
}
process.stdout.write(JSON.stringify({ schema: 'saccade-gap-fill-validation/1', status: 'PASS', checks, datasets }, null, 2) + '\n');
