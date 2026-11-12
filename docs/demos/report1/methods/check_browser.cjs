#!/usr/bin/env node
// Optional real-browser QA; the replay itself has no Node dependencies.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const candidates = [process.env.SACCADE_PLAYWRIGHT_MODULE, 'playwright',
  path.join(os.homedir(), '.cache/ms-playwright-go/1.57.0/package')].filter(Boolean);
let playwright;
for (const candidate of candidates) { try { playwright = require(candidate); break; } catch {} }
if (!playwright) throw new Error('Set SACCADE_PLAYWRIGHT_MODULE to an installed Playwright package.');
const demo = path.resolve(__dirname, '..');
const root = path.resolve(demo, '../../..');
const out = process.env.SACCADE_DEMO_QA_OUTPUT || path.join(root, 'build/demo-report1-qa');
fs.mkdirSync(out, { recursive: true });
const result = { schema: 'saccade-demo-browser-validation/1', checks: [],
  browsers: [], externalRequests: [], pageErrors: [], screenshots: [] };
let browser;
async function check(name, fn) {
  await fn(); result.checks.push({ name, status: 'PASS' });
  process.stdout.write(`PASS ${name}\n`);
}
async function waitFrame(page, index) {
  await page.waitForFunction(i => window.demoAPI.getState().frame === i &&
    !document.querySelector('video').seeking && document.querySelector('video').readyState >= 2, index);
}
async function state(page) { return page.evaluate(() => window.demoAPI.getState()); }
async function waitSequence(page, key) {
  await page.waitForFunction(sequence => {
    const s = window.demoAPI.getState(), v = document.querySelector('video');
    return s.sequence === sequence && s.loaded && s.frame === 220 && !v.seeking && v.readyState >= 2;
  }, key);
}
async function gapWitness(page, sequence, frame, outputId) {
  if ((await state(page)).sequence !== sequence) {
    await page.evaluate(key => window.demoAPI.switchSequence(key), sequence);
    await waitSequence(page, sequence);
  }
  await page.evaluate(index => window.demoAPI.seekFrame(index), frame);
  await waitFrame(page, frame);
  const snapshot = await page.evaluate(({ frame, outputId }) => {
    const api = window.demoAPI, frames = window.SACCADE_DEMO.frames;
    const gap = api.sampleGapRows(frame).find(row => row.outputId === outputId);
    if (!gap) return { gap: null };
    const seen = new Set(), outputIds = [];
    for (const source of frames) for (const row of source.rows) {
      if (!seen.has(row[0])) { seen.add(row[0]); outputIds.push(row[0]); }
    }
    return { gap, half: api.sampleGapRows(frame, 0.5).find(row => row.outputId === outputId),
      from: frames[gap.fromFrame].rows.find(row => row[0] === outputId),
      to: frames[gap.toFrame].rows.find(row => row[0] === outputId),
      expectedMotId: outputIds.indexOf(outputId) + 1,
      state: api.getState(), raw: frames[frame].rows };
  }, { frame, outputId });
  assert(snapshot.gap, `${sequence} output ID ${outputId} missing at frame ${frame}`);
  const { gap, from, to } = snapshot;
  assert.equal(gap.outputId, outputId); assert.equal(gap.motId, snapshot.expectedMotId);
  assert.equal(gap.fromUid, from[1]); assert.equal(gap.toUid, to[1]);
  assert.equal(gap.missingFrames, gap.toFrame - gap.fromFrame - 1);
  assert(gap.fromFrame < frame && frame < gap.toFrame && gap.missingFrames <= 35);
  assert.equal(gap.confidence, -1); assert.equal(gap.method, 'v1-mot/1');
  assert.equal(snapshot.raw.some(row => row[0] === outputId), false);
  assert.equal(snapshot.state.playing, false); assert.equal(snapshot.state.interpolationPhase, 0);
  assert.deepEqual(snapshot.state.displayRows, snapshot.raw);
  assert.equal(snapshot.state.observedRows, snapshot.raw.length);
  for (const [sample, phase] of [[gap, 0], [snapshot.half, 0.5]]) {
    assert(sample);
    const fraction = (frame + phase - gap.fromFrame) / (gap.toFrame - gap.fromFrame);
    assert(Math.abs(sample.fraction - fraction) < 1e-12);
    for (let i = 0; i < 4; i++) {
      const expected = from[i + 2] + fraction * (to[i + 2] - from[i + 2]);
      assert(Math.abs(sample.xyxy[i] - expected) < 1e-10);
    }
  }
  assert.deepEqual(snapshot.state.gapRows.find(row => row.outputId === outputId), gap);
  return snapshot;
}
async function main() {
  browser = await playwright.chromium.launch({ executablePath: process.env.SACCADE_CHROMIUM_BIN || '/usr/bin/chromium',
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  result.browsers.push(await browser.version());
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  page.on('pageerror', e => result.pageErrors.push(e.message));
  page.on('request', r => { if (/^https?:/.test(r.url()) && !/^https?:\/\/(127\.0\.0\.1|localhost)(:|\/)/.test(r.url())) result.externalRequests.push(r.url()); });
  await page.goto(pathToFileURL(path.join(demo, 'index.html')).href);
  await page.waitForFunction(() => window.demoAPI?.getState().loaded);
  await waitFrame(page, 220);
  await check('file:// opens with real decoded H.264 and correct initial frame', async () => {
    const v = await page.locator('video').evaluate(v => ({ width: v.videoWidth, height: v.videoHeight, duration: v.duration }));
    assert.equal(v.width, 1920); assert.equal(v.height, 1080); assert.equal(v.duration, 21.8);
    assert.equal((await state(page)).observedRows, 21);
    assert.equal(await page.locator('#frame-input').inputValue(), '221');
    assert.equal(await page.locator('#metric-single').textContent(), '6.982');
    assert.equal(await page.locator('#metric-queued').textContent(), '21.181');
  });
  await check('display interpolation defaults on but paused coordinates stay exact', async () => {
    assert.equal(await page.locator('#show-interpolation').isChecked(), true);
    const snapshot = await page.evaluate(() => ({ state: window.demoAPI.getState(), raw: window.SACCADE_DEMO.frames[220].rows }));
    assert.equal(snapshot.state.interpolation, true);
    assert.equal(snapshot.state.interpolationActive, false);
    assert.equal(snapshot.state.interpolationPhase, 0);
    assert.deepEqual(snapshot.state.displayRows, snapshot.raw);
  });
  await check('adjacent observed boxes interpolate coordinates only without mutating data', async () => {
    const sample = await page.evaluate(() => {
      const before = JSON.stringify(window.SACCADE_DEMO.frames);
      return { raw: window.SACCADE_DEMO.frames[220].rows, next: window.SACCADE_DEMO.frames[221].rows,
        sampled: window.demoAPI.sampleDisplayRows(220, 0.5),
        unchanged: before === JSON.stringify(window.SACCADE_DEMO.frames) };
    });
    assert.equal(sample.unchanged, true);
    assert.equal(sample.sampled.length, sample.raw.length);
    let moved = 0;
    for (const [i, original] of sample.raw.entries()) {
      const next = sample.next.find(r => r[0] === original[0] && r[1] === original[1]);
      const expected = [...original];
      if (next && original[8] === 1 && next[8] === 1) {
        for (let k = 2; k < 6; k++) expected[k] = (original[k] + next[k]) / 2;
      }
      for (let k = 0; k < expected.length; k++) assert(Math.abs(sample.sampled[i][k] - expected[k]) < 1e-10);
      if (sample.sampled[i].slice(2, 6).some((v, j) => v !== original[j + 2])) moved++;
    }
    assert(moved > 0);
  });
  await check('interpolation does not bridge gaps, identities, predicted rows or births', async () => {
    const guards = await page.evaluate(() => {
      const frames = window.SACCADE_DEMO.frames, api = window.demoAPI;
      const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
      const missing = frames[5].rows.find(r => r[0] === 14);
      const gap = equal(api.sampleDisplayRows(5, 0.75).find(r => r[0] === 14), missing);
      const noBirths = api.sampleDisplayRows(1, 0.9).length === 0 && frames[2].rows.length > 0;
      const current = frames[220].rows, following = frames[221].rows;
      const row = current[0], target = following.find(r => r[0] === row[0] && r[1] === row[1]);
      const results = { gap, noBirths };
      try {
        for (const [name, field, value] of [['uid', 1, target[1] + 10000], ['outputId', 0, target[0] + 10000], ['predictedNext', 8, 2]]) {
          frames[221].rows = following.map(r => { const copy = [...r]; if (r === target) copy[field] = value; return copy; });
          results[name] = equal(api.sampleDisplayRows(220, 0.5)[0], row);
        }
        frames[221].rows = following;
        frames[220].rows = current.map((r, i) => { const copy = [...r]; if (i === 0) copy[8] = 2; return copy; });
        results.predictedCurrent = equal(api.sampleDisplayRows(220, 0.5)[0], frames[220].rows[0]);
      } finally { frames[220].rows = current; frames[221].rows = following; }
      return results;
    });
    for (const [name, passed] of Object.entries(guards)) assert.equal(passed, true, name);
  });
  await check('terminal frames and phase limits never extrapolate coordinates', async () => {
    const cases = await page.evaluate(() => {
      const api = window.demoAPI, frames = window.SACCADE_DEMO.frames;
      return { first: api.sampleDisplayRows(220, -1), raw: frames[220].rows,
        high: api.sampleDisplayRows(220, 5), endpoint: api.sampleDisplayRows(220, 1),
        nan: api.sampleDisplayRows(220, NaN), end: api.sampleDisplayRows(653, 0.5), last: frames[653].rows };
    });
    assert.deepEqual(cases.first, cases.raw); assert.deepEqual(cases.nan, cases.raw);
    assert.deepEqual(cases.high, cases.endpoint); assert.deepEqual(cases.end, cases.last);
  });
  await check('ID selection retains output ID and distinct internal UID', async () => {
    await page.locator('[data-track-id="71"]').click();
    assert.equal((await state(page)).selectedId, 71);
    assert.equal(await page.locator('#selected-uid').textContent(), '106');
    assert.equal(await page.locator('#selected-state').textContent(), '本幀有觀測');
  });
  await page.screenshot({ path: path.join(demo, 'preview.png'), fullPage: true });
  result.screenshots.push('preview.png');
  await check('next/previous buttons move exactly one source frame', async () => {
    await page.locator('#next-button').click(); await waitFrame(page, 221);
    await page.locator('#previous-button').click(); await waitFrame(page, 220);
  });
  await check('typing a source frame uses one-based JPEG numbering', async () => {
    await page.locator('#frame-input').fill('401'); await page.locator('#frame-input').press('Enter');
    await page.locator('#frame-input').blur(); await waitFrame(page, 400);
    assert.equal(await page.locator('#frame-label').textContent(), '401');
  });
  await check('missing selected ID stays selected and reports no observation', async () => {
    await page.evaluate(() => window.demoAPI.seekFrame(0)); await waitFrame(page, 0);
    assert.equal((await state(page)).selectedId, 71);
    assert.equal((await state(page)).observedRows, 0);
    assert.equal(await page.locator('#selected-state').textContent(), '本幀未輸出');
    assert.equal(await page.locator('#selected-uid').textContent(), '—');
    assert.equal(await page.locator('#motion-x').textContent(), '—');
    assert.equal(await page.locator('#motion-state').textContent(), '初始化');
  });
  await check('first/second/last source frame boundaries and controls', async () => {
    assert.equal(await page.locator('#previous-button').isDisabled(), true);
    await page.evaluate(() => window.demoAPI.seekFrame(1)); await waitFrame(page, 1);
    assert.equal((await state(page)).observedRows, 0);
    await page.evaluate(() => window.demoAPI.seekFrame(653)); await waitFrame(page, 653);
    assert.equal(await page.locator('#next-button').isDisabled(), true);
    assert.equal(await page.locator('#frame-input').inputValue(), '654');
  });
  await check('rapid seeks finish on the last requested frame', async () => {
    await page.evaluate(() => Promise.all([12, 220, 400, 653, 60].map(i => window.demoAPI.seekFrame(i))));
    await waitFrame(page, 60);
  });
  await check('raw / overlay / side-by-side view buttons work', async () => {
    for (const [selector, view] of [['#view-raw', 'raw'], ['#view-side', 'sideBySide'], ['#view-overlay', 'overlay']]) {
      await page.locator(selector).click(); assert.equal((await state(page)).view, view);
      assert.equal(await page.locator(selector).getAttribute('aria-pressed'), 'true');
    }
  });
  await check('box and trail toggles change the visible layer state', async () => {
    await page.locator('#show-boxes').uncheck(); assert.equal((await state(page)).boxes, false);
    await page.locator('#show-trails').uncheck(); assert.equal((await state(page)).trails, false);
    await page.locator('#show-boxes').check(); await page.locator('#show-trails').check();
  });
  await check('native video avoids pixel copies and comparison reuses its paused image', async () => {
    const sample = await page.evaluate(async () => {
      const api = window.demoAPI, video = document.querySelector('video');
      await api.seekFrame(220); api.setView('overlay'); api.setSelected(null);
      const proto = CanvasRenderingContext2D.prototype, original = proto.drawImage;
      const copies = [];
      proto.drawImage = function (...args) { if (args[0] === video) copies.push(this.canvas.id); return original.apply(this, args); };
      try {
        const native = {opacity:getComputedStyle(video).opacity,fit:getComputedStyle(video).objectFit,
          background:getComputedStyle(document.getElementById('video-canvas')).visibility};
        await video.play(); await new Promise(resolve => setTimeout(resolve, 400)); video.pause();
        const nativeCopies = copies.length;
        api.setView('sideBySide');
        const comparisonCopies = copies.length - nativeCopies;
        const comparisonOpacity = getComputedStyle(video).opacity;
        const row = window.SACCADE_DEMO.frames[api.getState().frame].rows[0];
        api.setSelected(row[0]); api.setSelected(null); api.setSelected(row[0]);
        return {native,nativeCopies,comparisonCopies,comparisonOpacity,
          selectionCopies:copies.length-nativeCopies-comparisonCopies};
      } finally { proto.drawImage = original; video.pause(); api.setView('overlay'); api.setSelected(null); }
    });
    assert.equal(sample.native.opacity, '1'); assert.equal(sample.native.fit, 'contain');
    assert.equal(sample.native.background, 'hidden'); assert.equal(sample.nativeCopies, 0);
    assert.equal(sample.comparisonOpacity, '0'); assert.equal(sample.comparisonCopies, 2);
    assert.equal(sample.selectionCopies, 0);
  });
  await check('ID controls retain their DOM nodes across source frame updates', async () => {
    const retained = await page.evaluate(async () => {
      const api = window.demoAPI; await api.seekFrame(220);
      const id = window.SACCADE_DEMO.frames[220].rows.find(row => window.SACCADE_DEMO.frames[221].rows.some(next => next[0] === row[0]))[0];
      const node = document.querySelector(`[data-track-id="${id}"]`);
      await api.seekFrame(221);
      return node === document.querySelector(`[data-track-id="${id}"]`);
    });
    assert.equal(retained, true);
  });
  await check('clicking a visible bbox selects its output ID', async () => {
    await page.evaluate(() => { window.demoAPI.setSelected(null); return window.demoAPI.seekFrame(220); });
    await waitFrame(page, 220);
    const point = await page.evaluate(() => {
      const row = window.SACCADE_DEMO.frames[220].rows.find(r => r[0] === 218);
      const rect = document.querySelector('#replay-canvas').getBoundingClientRect();
      return { x: ((row[2]+row[4])/2)/1920*rect.width, y: ((row[3]+row[5])/2)/1080*rect.height };
    });
    await page.locator('#replay-canvas').click({ position: point }); assert.equal((await state(page)).selectedId, 218);
  });
  await check('keyboard arrows step and Escape clears selected ID', async () => {
    await page.locator('#replay-canvas').focus(); await page.keyboard.press('ArrowRight'); await waitFrame(page, 221);
    await page.keyboard.press('ArrowLeft'); await waitFrame(page, 220);
    await page.keyboard.press('Escape'); assert.equal((await state(page)).selectedId, null);
  });
  await check('rate selection, actual playback and pause work', async () => {
    await page.locator('#playback-rate').selectOption('0.5'); assert.equal((await state(page)).rate, 0.5);
    await page.locator('#playback-rate').selectOption('2');
    await page.locator('#play-button').click();
    await page.waitForFunction(() => window.demoAPI.getState().frame > 226);
    assert.equal((await state(page)).playing, true);
    await page.locator('#play-button').click(); assert.equal((await state(page)).playing, false);
    await page.locator('#playback-rate').selectOption('1');
  });
  await check('real display refreshes move boxes within a held source frame at 0.25x', async () => {
    await page.evaluate(() => { window.demoAPI.setInterpolation(true); window.demoAPI.setSelected(6); return window.demoAPI.seekFrame(220); });
    await page.locator('#playback-rate').selectOption('0.25');
    await page.locator('#play-button').click();
    await page.waitForFunction(() => window.demoAPI.getState().interpolationActive);
    const samples = await page.evaluate(() => new Promise(resolve => {
      const records = [];
      const sample = () => {
        const s = window.demoAPI.getState(), raw = window.SACCADE_DEMO.frames[s.frame].rows;
        records.push({ frame: s.frame, phase: s.interpolationPhase, active: s.interpolationActive,
          display: s.displayRows, raw, next: window.SACCADE_DEMO.frames[s.frame + 1]?.rows || [],
          selected: s.selectedRow, sidebarBox: document.querySelector('#selected-box').textContent,
          modeLabel: document.querySelector('#interpolation-status').textContent });
        if (records.length >= 36) resolve(records); else requestAnimationFrame(sample);
      };
      requestAnimationFrame(sample);
    }));
    let subframeMovement = false, active = 0;
    for (const [i, s] of samples.entries()) {
      if (s.active && s.phase > 0 && s.phase < 1) active++;
      assert.equal(s.display.length, s.raw.length);
      assert.equal(s.modeLabel, '顯示插值');
      const selected = s.raw.find(r => r[0] === 6) || null;
      assert.deepEqual(s.selected, selected);
      assert.equal(s.sidebarBox, selected ? selected.slice(2, 6).map(v => v.toFixed(1)).join(', ') : '—');
      for (const [j, row] of s.display.entries()) {
        const original = s.raw[j], target = s.next.find(r => r[0] === original[0] && r[1] === original[1]);
        for (const k of [0, 1, 6, 7, 8]) assert.equal(row[k], original[k]);
        for (let k = 2; k < 6; k++) {
          const expected = target && original[8] === 1 && target[8] === 1 ? original[k] + (target[k] - original[k]) * s.phase : original[k];
          assert(Math.abs(row[k] - expected) < 1e-10);
        }
      }
      const prior = samples[i - 1];
      if (prior && prior.frame === s.frame && prior.phase !== s.phase && s.display.some((r, j) => r.slice(2, 6).some((v, k) => v !== prior.display[j][k + 2]))) subframeMovement = true;
    }
    assert(active > 0); assert.equal(subframeMovement, true);
    result.interpolationObservations = { rafSamples: samples.length, activeSubframeSamples: active, sameFrameMovementObserved: subframeMovement };
  });
  await check('turning interpolation off during playback immediately restores raw geometry', async () => {
    await page.locator('#show-interpolation').uncheck();
    const snapshots = await page.evaluate(() => new Promise(resolve => {
      const rows = [];
      const sample = () => { const s = window.demoAPI.getState(); rows.push({ state: s, raw: window.SACCADE_DEMO.frames[s.frame].rows });
        if (rows.length === 8) resolve(rows); else requestAnimationFrame(sample); };
      requestAnimationFrame(sample);
    }));
    for (const s of snapshots) {
      assert.equal(s.state.playing, true); assert.equal(s.state.interpolation, false);
      assert.equal(s.state.interpolationActive, false); assert.equal(s.state.interpolationPhase, 0);
      assert.deepEqual(s.state.displayRows, s.raw);
    }
  });
  await check('pause, stall, rate change and exact stepping reset the interpolation clock', async () => {
    await page.locator('#show-interpolation').check();
    const resets = await page.evaluate(() => {
      const v = document.querySelector('video'), records = [];
      for (const name of ['waiting', 'stalled', 'ratechange', 'pause']) {
        v.dispatchEvent(new Event(name)); records.push(window.demoAPI.getState());
      }
      v.pause(); return records;
    });
    for (const s of resets) { assert.equal(s.interpolationPhase, 0); assert.equal(s.interpolationActive, false); }
    await page.evaluate(() => window.demoAPI.seekFrame(220)); await waitFrame(page, 220);
    await page.locator('#next-button').click(); await waitFrame(page, 221);
    const paused = await page.evaluate(() => ({ state: window.demoAPI.getState(), raw: window.SACCADE_DEMO.frames[221].rows }));
    assert.equal(paused.state.interpolation, true); assert.equal(paused.state.playing, false);
    assert.equal(paused.state.interpolationPhase, 0); assert.deepEqual(paused.state.displayRows, paused.raw);
    await page.locator('#playback-rate').selectOption('1');
  });
  await check('fullscreen works with the canvas stage', async () => {
    await page.locator('#fullscreen-button').click();
    await page.waitForFunction(() => !!document.fullscreenElement);
    assert.equal(await page.evaluate(() => document.fullscreenElement.id), 'stage');
    await page.evaluate(() => document.exitFullscreen());
  });
  await check('320px and 390px viewports reflow without horizontal overflow', async () => {
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('#view-side').click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      const size = await page.locator('#replay-canvas').boundingBox(); assert(size.height > size.width);
      await page.screenshot({ path: path.join(out, `mobile-${width}.png`), fullPage: true });
    }
  });
  await check('sequence selector exposes all three complete replay datasets', async () => {
    assert.deepEqual(await page.locator('#sequence-select option').evaluateAll(options => options.map(o => o.value)), ['MOT17-10-SDP', 'MOT17-02-SDP', 'MOT17-04-SDP']);
    const catalog = await page.evaluate(() => Object.fromEntries(Object.entries(window.demoAPI.allCatalog).map(([key, value]) => [key, { frames: value.frames.length, fps: value.meta.fps, metrics: value.metrics }])));
    assert.equal(catalog['MOT17-10-SDP'].frames, 654); assert.equal(catalog['MOT17-02-SDP'].frames, 600);
    assert.equal(catalog['MOT17-02-SDP'].fps, 30); assert.equal(catalog['MOT17-02-SDP'].metrics, null);
    assert.equal(catalog['MOT17-04-SDP'].frames, 1050);
    assert.equal(catalog['MOT17-04-SDP'].fps, 30); assert.equal(catalog['MOT17-04-SDP'].metrics, null);
  });
  await check('selecting MOT17-02 resets identity while retaining display preferences', async () => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('#show-boxes').uncheck(); await page.locator('#show-trails').uncheck();
    await page.locator('#show-interpolation').uncheck(); await page.locator('#playback-rate').selectOption('0.5');
    await page.evaluate(() => window.demoAPI.setSelected(6));
    await page.locator('#sequence-select').selectOption('MOT17-02-SDP'); await waitSequence(page, 'MOT17-02-SDP');
    const s = await state(page);
    assert.equal(s.selectedId, null); assert.equal(s.playing, false); assert.equal(s.totalFrames, 600);
    assert.equal(s.view, 'sideBySide'); assert.equal(s.boxes, false); assert.equal(s.trails, false);
    assert.equal(s.interpolation, false); assert.equal(s.rate, 0.5); assert.equal(s.observedRows, 19);
    assert.equal(await page.locator('#frame-total').textContent(), '600');
    assert.equal(await page.locator('#duration-label').textContent(), '00:20.000');
    assert.equal(await page.locator('#frame-slider').getAttribute('max'), '599');
    assert.equal(await page.locator('#backup-download').getAttribute('href'), 'assets/tracking-demo-02.mp4');
    assert((await page.locator('video').evaluate(v => v.currentSrc)).endsWith('/assets/source-02.mp4'));
    assert.equal(await page.locator('video').evaluate(v => v.duration), 20);
  });
  await check('MOT17-02 does not borrow MOT17-10 benchmark results or sample counts', async () => {
    for (const id of ['metric-single', 'metric-queued', 'metric-throughput', 'metric-offline']) assert.equal(await page.locator('#' + id).textContent(), '—');
    const text = await page.locator('.measurement-section').textContent();
    assert(text.includes('MOT17-02-SDP')); assert(text.includes('尚未量測'));
    assert(!text.includes('480')); assert(!text.includes('654')); assert(!text.includes('6.982'));
  });
  await check('MOT17-02 frame bounds, raw rows and selected identity are source-specific', async () => {
    await page.locator('#show-boxes').check(); await page.locator('#show-trails').check();
    await page.locator('#show-interpolation').check(); await page.locator('#view-overlay').click();
    const id = await page.evaluate(() => window.SACCADE_DEMO.frames[220].rows[0][0]);
    await page.locator(`[data-track-id="${id}"]`).click();
    const snapshot = await page.evaluate(() => ({ state: window.demoAPI.getState(), raw: window.SACCADE_DEMO.frames[220].rows }));
    assert.deepEqual(snapshot.state.displayRows, snapshot.raw);
    assert.deepEqual(snapshot.state.selectedRow, snapshot.raw.find(r => r[0] === id));
    await page.screenshot({ path: path.join(demo, 'preview-02.png'), fullPage: true });
    result.screenshots.push('preview-02.png');
    await page.evaluate(() => window.demoAPI.seekFrame(0)); await waitFrame(page, 0);
    assert.equal((await state(page)).observedRows, 0);
    await page.evaluate(() => window.demoAPI.seekFrame(599)); await waitFrame(page, 599);
    assert.equal(await page.locator('#frame-input').inputValue(), '600');
    assert.equal(await page.locator('#next-button').isDisabled(), true);
    await page.locator('#next-button').evaluate(button => button.click()); assert.equal((await state(page)).frame, 599);
  });
  await check('MOT17-02 actual slow playback interpolates only its own source rows', async () => {
    await page.evaluate(() => window.demoAPI.seekFrame(220)); await waitFrame(page, 220);
    await page.locator('#playback-rate').selectOption('0.25'); await page.locator('#play-button').click();
    await page.waitForFunction(() => window.demoAPI.getState().interpolationActive);
    const samples = await page.evaluate(() => new Promise(resolve => {
      const records = [];
      const sample = () => { const s = window.demoAPI.getState(); records.push({ state: s,
        raw: window.SACCADE_DEMO.frames[s.frame].rows, next: window.SACCADE_DEMO.frames[s.frame + 1]?.rows || [] });
        if (records.length === 24) resolve(records); else requestAnimationFrame(sample); };
      requestAnimationFrame(sample);
    }));
    let active = 0;
    for (const { state: s, raw, next } of samples) {
      assert.equal(s.sequence, 'MOT17-02-SDP'); assert.equal(s.displayRows.length, raw.length);
      if (s.interpolationActive) active++;
      for (const [i, row] of s.displayRows.entries()) {
        const from = raw[i], to = next.find(r => r[0] === from[0] && r[1] === from[1]);
        for (let k = 2; k < 6; k++) {
          const expected = to && from[8] === 1 && to[8] === 1 ? from[k] + (to[k] - from[k]) * s.interpolationPhase : from[k];
          assert(Math.abs(row[k] - expected) < 1e-10);
        }
      }
    }
    assert(active > 0); result.mot02InterpolationObservations = { rafSamples: samples.length, activeSamples: active };
  });
  await check('switching a playing source back to MOT17-10 pauses and restores its metrics', async () => {
    await page.locator('#sequence-select').selectOption('MOT17-10-SDP'); await waitSequence(page, 'MOT17-10-SDP');
    const s = await state(page);
    assert.equal(s.playing, false); assert.equal(s.selectedId, null); assert.equal(s.observedRows, 21);
    assert.equal(s.totalFrames, 654); assert.equal(s.rate, 0.25); assert.equal(s.interpolationPhase, 0);
    assert.equal(await page.locator('#metric-single').textContent(), '6.982');
    assert.equal(await page.locator('#metric-queued').textContent(), '21.181');
    assert((await page.locator('#metric-single-note').textContent()).includes('480'));
    assert.equal(await page.locator('#backup-download').getAttribute('href'), 'assets/tracking-demo.mp4');
  });
  await check('rapid sequence changes cancel stale readiness, seeks and playback callbacks', async () => {
    await page.evaluate(() => Promise.race([
      Promise.all(['MOT17-02-SDP', 'MOT17-10-SDP', 'MOT17-02-SDP', 'MOT17-10-SDP', 'MOT17-02-SDP'].map(key => window.demoAPI.switchSequence(key))),
      new Promise((_, reject) => setTimeout(() => reject(new Error('rapid switches did not settle')), 10000))
    ]));
    await waitSequence(page, 'MOT17-02-SDP');
    await page.evaluate(() => Promise.race([
      Promise.all([window.demoAPI.seekFrame(599), window.demoAPI.switchSequence('MOT17-10-SDP')]),
      new Promise((_, reject) => setTimeout(() => reject(new Error('source/seek cancellation did not settle')), 10000))
    ]));
    await waitSequence(page, 'MOT17-10-SDP');
    assert.equal((await state(page)).error, null); assert.equal((await state(page)).observedRows, 21);
  });
  await check('MOT17-02 selector and split view fit a narrow viewport', async () => {
    await page.evaluate(() => window.demoAPI.switchSequence('MOT17-02-SDP')); await waitSequence(page, 'MOT17-02-SDP');
    await page.setViewportSize({ width: 320, height: 844 }); await page.locator('#view-side').click();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: path.join(out, 'mobile-mot02-320.png'), fullPage: true });
  });
  await check('MOT17-04 loads 1050-frame media with source-specific rows, links and unmeasured metrics', async () => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.evaluate(() => window.demoAPI.setSelected(2));
    const before = await state(page);
    await page.locator('#sequence-select').selectOption('MOT17-04-SDP'); await waitSequence(page, 'MOT17-04-SDP');
    const s = await state(page);
    assert.equal(s.selectedId, null); assert.equal(s.playing, false); assert.equal(s.totalFrames, 1050);
    for (const key of ['view', 'boxes', 'trails', 'interpolation', 'rate']) assert.equal(s[key], before[key], key);
    assert.equal(await page.locator('#frame-total').textContent(), '1050');
    assert.equal(await page.locator('#frame-input').getAttribute('max'), '1050');
    assert.equal(await page.locator('#frame-slider').getAttribute('max'), '1049');
    assert.equal(await page.locator('#duration-label').textContent(), '00:35.000');
    assert.equal(await page.locator('video').evaluate(v => v.duration), 35);
    assert((await page.locator('video').evaluate(v => v.currentSrc)).endsWith('/assets/source-04.mp4'));
    assert.equal(await page.locator('#backup-download').getAttribute('href'), 'assets/tracking-demo-04.mp4');
    for (const id of ['metric-single', 'metric-queued', 'metric-throughput', 'metric-offline']) assert.equal(await page.locator('#' + id).textContent(), '—');
    const text = await page.locator('.measurement-section').textContent();
    assert(text.includes('MOT17-04-SDP') && text.includes('尚未量測') && text.includes('1050'));
    assert(!text.includes('480') && !text.includes('654') && !text.includes('6.982'));
    const snapshot = await page.evaluate(() => ({ state: window.demoAPI.getState(), raw: window.SACCADE_DEMO.frames[220].rows }));
    assert.deepEqual(snapshot.state.displayRows, snapshot.raw); assert.equal(s.observedRows, snapshot.raw.length);
    const id = snapshot.raw[0][0]; await page.locator(`[data-track-id="${id}"]`).click();
    assert.deepEqual((await state(page)).selectedRow, snapshot.raw[0]);
    await page.locator('#view-overlay').click();
    await page.screenshot({ path: path.join(demo, 'preview-04.png'), fullPage: true }); result.screenshots.push('preview-04.png');
  });
  await check('MOT17-04 exact steps, terminal frame and scene bookmarks cover the full 35 seconds', async () => {
    await page.locator('#next-button').click(); await waitFrame(page, 221);
    await page.locator('#previous-button').click(); await waitFrame(page, 220);
    await page.locator('[data-bookmark="525"]').click(); await waitFrame(page, 525);
    await page.locator('[data-bookmark="996"]').click(); await waitFrame(page, 996);
    await page.evaluate(() => window.demoAPI.seekFrame(1049)); await waitFrame(page, 1049);
    assert.equal(await page.locator('#frame-input').inputValue(), '1050');
    assert.equal(await page.locator('#next-button').isDisabled(), true);
    const end = await page.evaluate(() => ({ shown: window.demoAPI.sampleDisplayRows(1049, 0.9), raw: window.SACCADE_DEMO.frames[1049].rows }));
    assert.deepEqual(end.shown, end.raw);
    await page.locator('#restart-button').click(); await waitFrame(page, 0);
    assert.equal((await state(page)).observedRows, 0);
  });
  await check('MOT17-04 real slow playback interpolates only matching packet identities', async () => {
    await page.evaluate(() => { window.demoAPI.setInterpolation(true); return window.demoAPI.seekFrame(220); });
    await page.locator('#playback-rate').selectOption('0.25'); await page.locator('#play-button').click();
    await page.waitForFunction(() => window.demoAPI.getState().interpolationActive);
    const snapshots = await page.evaluate(() => new Promise(resolve => {
      const records = [];
      const sample = () => { const s = window.demoAPI.getState(); records.push({ state: s,
        raw: window.SACCADE_DEMO.frames[s.frame].rows, next: window.SACCADE_DEMO.frames[s.frame + 1]?.rows || [] });
        if (records.length === 24) resolve(records); else requestAnimationFrame(sample); };
      requestAnimationFrame(sample);
    }));
    let active = 0;
    for (const { state: s, raw, next } of snapshots) {
      assert.equal(s.sequence, 'MOT17-04-SDP'); assert.equal(s.displayRows.length, raw.length);
      if (s.interpolationActive) active++;
      for (const [i, row] of s.displayRows.entries()) {
        const from = raw[i], to = next.find(r => r[0] === from[0] && r[1] === from[1]);
        for (const k of [0, 1, 6, 7, 8]) assert.equal(row[k], from[k]);
        for (let k = 2; k < 6; k++) {
          const expected = to && from[8] === 1 && to[8] === 1 ? from[k] + (to[k] - from[k]) * s.interpolationPhase : from[k];
          assert(Math.abs(row[k] - expected) < 1e-10);
        }
      }
    }
    assert(active > 0); result.mot04InterpolationObservations = { rafSamples: snapshots.length, activeSamples: active };
  });
  await check('three-source rapid switching cancels old clocks and restores each source frame range', async () => {
    await page.locator('#sequence-select').selectOption('MOT17-10-SDP'); await waitSequence(page, 'MOT17-10-SDP');
    assert.equal((await state(page)).playing, false); assert.equal((await state(page)).selectedId, null);
    assert.equal(await page.locator('#metric-single').textContent(), '6.982');
    await page.evaluate(() => Promise.race([
      Promise.all(['MOT17-04-SDP', 'MOT17-02-SDP', 'MOT17-10-SDP', 'MOT17-04-SDP'].map(key => window.demoAPI.switchSequence(key))),
      new Promise((_, reject) => setTimeout(() => reject(new Error('three-source switches did not settle')), 10000))
    ]));
    await waitSequence(page, 'MOT17-04-SDP');
    await page.evaluate(() => Promise.race([
      Promise.all([window.demoAPI.seekFrame(1049), window.demoAPI.switchSequence('MOT17-02-SDP')]),
      new Promise((_, reject) => setTimeout(() => reject(new Error('04 terminal seek cancellation did not settle')), 10000))
    ]));
    await waitSequence(page, 'MOT17-02-SDP'); assert.equal((await state(page)).totalFrames, 600);
    await page.evaluate(() => window.demoAPI.switchSequence('MOT17-04-SDP')); await waitSequence(page, 'MOT17-04-SDP');
    assert.equal((await state(page)).totalFrames, 1050); assert.equal((await state(page)).error, null);
  });
  await check('MOT17-04 selector and split view fit 320px without horizontal overflow', async () => {
    await page.setViewportSize({ width: 320, height: 844 }); await page.locator('#view-side').click();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: path.join(out, 'mobile-mot04-320.png'), fullPage: true });
  });
  await check('formal MOT gap filling defaults on and reproduces all three source totals without changing packets', async () => {
    assert.equal(await page.locator('#show-gap-fill').isChecked(), true);
    assert.equal((await state(page)).gapFill, true);
    const totals = [];
    for (const [sequence, expected] of [['MOT17-10-SDP', 934], ['MOT17-02-SDP', 2040], ['MOT17-04-SDP', 2053]]) {
      await page.evaluate(key => window.demoAPI.switchSequence(key), sequence); await waitSequence(page, sequence);
      const inventory = await page.evaluate(() => {
        const api = window.demoAPI, frames = window.SACCADE_DEMO.frames;
        const before = JSON.stringify(frames);
        let count = 0, valid = true;
        for (let frame = 0; frame < frames.length; frame++) {
          const rows = api.sampleGapRows(frame); count += rows.length;
          valid &&= new Set(rows.map(row => row.outputId)).size === rows.length;
          for (const row of rows) valid &&= row.fromFrame < frame && frame < row.toFrame &&
            row.missingFrames >= 1 && row.missingFrames <= 35 && row.confidence === -1 && row.method === 'v1-mot/1' &&
            !frames[frame].rows.some(raw => raw[0] === row.outputId);
        }
        return { count, valid, stats: api.getState().gapStats, unchanged: before === JSON.stringify(frames) };
      });
      assert.equal(inventory.count, expected, sequence); assert.equal(inventory.stats.filledRows, expected, sequence);
      assert.equal(inventory.valid, true); assert.equal(inventory.unchanged, true);
      totals.push({ sequence, filledRows: inventory.count });
    }
    result.gapFillTotals = totals;
  });
  await check('same-UID formal gaps show exact endpoint interpolation while paused and stepping', async () => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('#view-overlay').click(); await page.locator('#show-boxes').check(); await page.locator('#show-trails').check();
    for (const [sequence, frame, id] of [['MOT17-10-SDP', 6, 14], ['MOT17-02-SDP', 20, 29], ['MOT17-04-SDP', 30, 23], ['MOT17-04-SDP', 221, 61]]) {
      const snapshot = await gapWitness(page, sequence, frame, id);
      assert.equal(snapshot.gap.fromUid, snapshot.gap.toUid);
      assert.equal(await page.locator('#gap-fill-status').isHidden(), false);
      assert.equal(await page.locator('#gap-fill-status').textContent(), `補框 ${snapshot.state.gapRows.length}`);
      const dashed = await page.evaluate(() => {
        const proto = CanvasRenderingContext2D.prototype, original = proto.setLineDash;
        let observed = false;
        proto.setLineDash = function (segments) { observed ||= segments.some(value => value > 0); return original.call(this, segments); };
        try { window.demoAPI.setGapFill(false); window.demoAPI.setGapFill(true); return observed; }
        finally { proto.setLineDash = original; }
      });
      assert.equal(dashed, true);
      await page.locator('#next-button').click(); await waitFrame(page, frame + 1);
      const stepped = await page.evaluate(() => ({ state: window.demoAPI.getState(), expected: window.demoAPI.sampleGapRows(window.demoAPI.getState().frame),
        raw: window.SACCADE_DEMO.frames[window.demoAPI.getState().frame].rows }));
      assert.deepEqual(stepped.state.gapRows, stepped.expected); assert.deepEqual(stepped.state.displayRows, stepped.raw);
      assert.equal(stepped.state.playing, false); assert.equal(stepped.state.interpolationPhase, 0);
    }
  });
  await check('formal output-ID gaps may cross UIDs and remain explicitly separate from observations', async () => {
    for (const [sequence, frame, id, from, to] of [['MOT17-10-SDP', 90, 71, 86, 101], ['MOT17-02-SDP', 220, 12, 211, 238], ['MOT17-04-SDP', 240, 20, 229, 256]]) {
      const snapshot = await gapWitness(page, sequence, frame, id);
      assert.notEqual(snapshot.gap.fromUid, snapshot.gap.toUid);
      assert.equal(snapshot.gap.fromFrame, from); assert.equal(snapshot.gap.toFrame, to);
      await page.evaluate(outputId => window.demoAPI.setSelected(outputId), id);
      assert.equal((await state(page)).selectedRow, null);
      assert.equal(await page.locator('#selected-state').textContent(), '缺幀插值（非觀測）');
      assert.equal(await page.locator('#selected-uid').textContent(), `${snapshot.gap.fromUid} → ${snapshot.gap.toUid}`);
      for (const selector of ['#selected-score', '#selected-association', '#selected-box']) assert.equal(await page.locator(selector).textContent(), '—');
      assert.equal(await page.locator('#selected-flags').textContent(), 'interpolated');
      assert.equal(await page.locator('#selected-gap-coordinates').isHidden(), false);
      assert.equal(await page.locator('#selected-gap-box').textContent(), snapshot.gap.xyxy.map(value => value.toFixed(1)).join(', '));
      const history = await page.locator('#selected-history').textContent();
      assert(history.includes(String(from + 1)) && history.includes(String(to + 1)));
      const counts = await page.locator('#gap-count-note').textContent();
      assert(counts.includes(String(snapshot.raw.length)) && counts.includes(String(snapshot.state.gapRows.length)));
      assert.equal(await page.locator('#observed-count').textContent(), String(snapshot.raw.length));
    }
  });
  await check('gap controls preserve raw rows and preferences while raw or hidden layers suppress the badge', async () => {
    const snapshot = await gapWitness(page, 'MOT17-02-SDP', 220, 12);
    await page.evaluate(() => window.demoAPI.setSelected(12));
    await page.locator('#show-gap-fill').uncheck();
    const off = await state(page);
    assert.equal(off.gapFill, false); assert.deepEqual(off.gapRows, []);
    assert.equal(off.selectedId, 12); assert.equal(off.selectedRow, null); assert.deepEqual(off.displayRows, snapshot.raw);
    assert.equal(off.observedRows, snapshot.raw.length); assert.equal(await page.locator('#gap-fill-status').isHidden(), true);
    assert.deepEqual(await page.evaluate(() => window.demoAPI.sampleGapRows(220)), snapshot.state.gapRows);
    assert.equal(await page.locator('#selected-gap-coordinates').isHidden(), true);
    await page.evaluate(() => window.demoAPI.switchSequence('MOT17-04-SDP')); await waitSequence(page, 'MOT17-04-SDP');
    assert.equal((await state(page)).gapFill, false); assert.equal(await page.locator('#show-gap-fill').isChecked(), false);
    await page.locator('#show-gap-fill').check();
    await gapWitness(page, 'MOT17-10-SDP', 6, 14);
    assert.equal((await state(page)).gapFill, true); assert.equal(await page.locator('#show-gap-fill').isChecked(), true);
    await page.locator('#view-raw').click(); assert.equal(await page.locator('#gap-fill-status').isHidden(), true);
    await page.locator('#view-overlay').click(); assert.equal(await page.locator('#gap-fill-status').isHidden(), false);
    await page.locator('#show-boxes').uncheck(); await page.locator('#show-trails').uncheck();
    assert.equal(await page.locator('#gap-fill-status').isHidden(), true);
    await page.locator('#show-boxes').check(); await page.locator('#show-trails').check();
    assert.equal(await page.locator('#gap-fill-status').isHidden(), false);
    const restored = await state(page);
    const restoredRaw = await page.evaluate(() => window.SACCADE_DEMO.frames[window.demoAPI.getState().frame].rows);
    assert.deepEqual(restored.displayRows, restoredRaw); assert.equal(restored.observedRows, restoredRaw.length);
    assert.equal(restored.interpolation, snapshot.state.interpolation);
  });
  await check('gap bounds never extrapolate and source-switch races cannot retain another source table', async () => {
    const before = await gapWitness(page, 'MOT17-10-SDP', 90, 71);
    await page.evaluate(() => window.demoAPI.setSelected(71));
    await page.evaluate(() => Promise.race([
      Promise.all(['MOT17-04-SDP', 'MOT17-10-SDP', 'MOT17-04-SDP', 'MOT17-02-SDP'].map(key => window.demoAPI.switchSequence(key))),
      new Promise((_, reject) => setTimeout(() => reject(new Error('gap source switches did not settle')), 10000))
    ]));
    await waitSequence(page, 'MOT17-02-SDP');
    const after = await gapWitness(page, 'MOT17-02-SDP', 220, 12);
    assert.equal(after.state.selectedId, null); assert.equal(after.state.gapStats.filledRows, 2040);
    await page.evaluate(() => Promise.race([
      Promise.all([window.demoAPI.seekFrame(599), window.demoAPI.switchSequence('MOT17-04-SDP')]),
      new Promise((_, reject) => setTimeout(() => reject(new Error('gap source/seek cancellation did not settle')), 10000))
    ]));
    await waitSequence(page, 'MOT17-04-SDP');
    const final = await gapWitness(page, 'MOT17-04-SDP', 221, 61);
    assert.equal(final.state.gapStats.filledRows, 2053); assert.equal(final.state.error, null);
    assert.deepEqual(final.state.gapRows, await page.evaluate(() => window.demoAPI.sampleGapRows(221)));
    for (const sequence of ['MOT17-10-SDP', 'MOT17-02-SDP', 'MOT17-04-SDP']) {
      await page.evaluate(key => window.demoAPI.switchSequence(key), sequence); await waitSequence(page, sequence);
      const edges = await page.evaluate(() => {
        const api = window.demoAPI, last = window.SACCADE_DEMO.frames.length - 1;
        return [-100, 0, last, last + 100].map(frame => api.sampleGapRows(frame, 0.9));
      });
      assert.deepEqual(edges, [[], [], [], []]);
      for (const frame of [0, (await state(page)).totalFrames - 1]) {
        await page.evaluate(index => window.demoAPI.seekFrame(index), frame); await waitFrame(page, frame);
        assert.deepEqual((await state(page)).gapRows, []); assert.equal(await page.locator('#gap-fill-status').isHidden(), true);
      }
    }
    result.gapSwitchObservations = [before, after, final].map(sample => ({ sequence: sample.state.sequence,
      frame: sample.state.frame, gapRows: sample.state.gapRows, filledRows: sample.state.gapStats.filledRows }));
  });
  await check('local HTTP delivery loads identical data and video', async () => {
    const base = process.argv[2];
    if (!base) throw new Error('Pass the optional local HTTP server URL for this delivery check.');
    const http = await browser.newPage(); http.on('pageerror', e => result.pageErrors.push(e.message));
    const response = await http.goto(base); assert.equal(response.status(), 200);
    await http.waitForFunction(() => window.demoAPI?.getState().loaded);
    await waitFrame(http, 220);
    assert.equal((await state(http)).observedRows, 21);
    await http.locator('#sequence-select').selectOption('MOT17-02-SDP'); await waitSequence(http, 'MOT17-02-SDP');
    assert.equal((await state(http)).observedRows, 19);
    await http.evaluate(() => window.demoAPI.seekFrame(599)); await waitFrame(http, 599);
    assert.equal((await state(http)).totalFrames, 600);
    await http.locator('#sequence-select').selectOption('MOT17-04-SDP'); await waitSequence(http, 'MOT17-04-SDP');
    assert.equal((await state(http)).totalFrames, 1050);
    await http.evaluate(() => window.demoAPI.seekFrame(1049)); await waitFrame(http, 1049);
    assert.equal(await http.locator('video').evaluate(v => v.duration), 35);
    await http.close();
  });
  await check('no page errors or external network services', async () => {
    assert.deepEqual(result.pageErrors, []); assert.deepEqual(result.externalRequests, []);
    assert.equal((await state(page)).error, null);
  });
  result.status = 'PASS';
}
main().catch(e => { result.status = 'FAIL'; result.error = e.stack; process.stderr.write(e.stack + '\n'); process.exitCode = 1; })
  .finally(async () => {
    if (browser) await browser.close();
    result.created_utc = new Date().toISOString();
    fs.writeFileSync(path.join(out, 'browser-validation.json'), JSON.stringify(result, null, 2) + '\n');
  });
