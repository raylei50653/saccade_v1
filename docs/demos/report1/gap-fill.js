// Offline display reconstruction of SequenceOutput::finish(), profile v1-mot/1.
// Input row flags and internal UID do not participate in the formal MOT rule.
(function (root) {
  'use strict';
  function configValue(value, fallback, name) {
    const result = value === undefined ? fallback : value;
    if (!Number.isInteger(result) || result < 0 || result > 0xffffffff) {
      throw new RangeError(`${name} must be a uint32`);
    }
    return result;
  }
  function build(frames, options = {}) {
    if (!Array.isArray(frames)) throw new TypeError('frames must be an array');
    const maxMissingFrames = configValue(options.maxMissingFrames, 35, 'maxMissingFrames');
    const minObservations = configValue(options.minObservations, 5, 'minObservations');
    const groups = new Map();
    const motIdByOutputId = Object.create(null);
    let inputRows = 0;
    // Map insertion order matches SequenceOutput::add(): global first appearance,
    // including the order of rows within a packet, rather than numeric ID order.
    for (let frame = 0; frame < frames.length; frame++) {
      for (const row of frames[frame].rows || []) {
        const outputId = row[0];
        if (!groups.has(outputId)) {
          const motId = groups.size + 1;
          groups.set(outputId, { motId, rows: [] });
          motIdByOutputId[outputId] = motId;
        }
        groups.get(outputId).rows.push({ frame, uid: row[1], xyxy: row.slice(2, 6) });
        inputRows++;
      }
    }
    const byFrame = Array.from({ length: frames.length }, () => []);
    let intervals = 0, filledRows = 0;
    for (const [outputId, group] of groups) {
      if (group.rows.length < minObservations) continue;
      for (let i = 1; i < group.rows.length; i++) {
        const a = group.rows[i - 1], b = group.rows[i];
        const distance = b.frame - a.frame;
        if (distance <= 1 || distance > maxMissingFrames + 1) continue;
        const descriptor = {
          outputId, motId: group.motId, fromUid: a.uid, toUid: b.uid,
          fromFrame: a.frame, toFrame: b.frame, missingFrames: distance - 1,
          from: a.xyxy.slice(), to: b.xyxy.slice()
        };
        for (let frame = a.frame + 1; frame < b.frame; frame++) {
          byFrame[frame].push(descriptor);
          filledRows++;
        }
        intervals++;
      }
    }
    return { byFrame, motIdByOutputId, stats: {
      intervals, filledRows, inputRows, outputIds: groups.size, frameCount: frames.length,
      maxMissingFrames, minObservations, profile: 'v1-mot/1'
    } };
  }
  function sample(index, frame, phase = 0) {
    if (!Number.isInteger(frame) || frame < 0 || frame >= index.byFrame.length) return [];
    const number = Number(phase);
    const subframe = Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 0;
    return index.byFrame[frame].map((gap) => {
      const fraction = (frame + subframe - gap.fromFrame) / (gap.toFrame - gap.fromFrame);
      return {
        outputId: gap.outputId, motId: gap.motId, fromUid: gap.fromUid, toUid: gap.toUid,
        xyxy: gap.from.map((value, i) => value + fraction * (gap.to[i] - value)),
        fromFrame: gap.fromFrame, toFrame: gap.toFrame, missingFrames: gap.missingFrames,
        fraction, confidence: -1, method: 'v1-mot/1'
      };
    });
  }
  const api = Object.freeze({ build, sample });
  root.SaccadeGapFill = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window === 'object' ? window : globalThis);
