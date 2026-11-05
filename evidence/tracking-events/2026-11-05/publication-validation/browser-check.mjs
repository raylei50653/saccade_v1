// Run after starting Chromium with --remote-debugging-port=9339.
// Optional argv[2]: URL of the generated viewer; defaults to this repo's publication viewer.
import fs from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../../../../', import.meta.url));
const url = process.argv[2] || pathToFileURL(path.join(root, 'build/tracking-events/publication-viewer/index.html')).href;
const screenshotPath = path.join(root, 'build/tracking-events/publication-viewer.png');
const pages = await (await fetch('http://localhost:9339/json')).json();
const page = pages.find(p => p.type === 'page');
if (!page) throw new Error('No Chromium page target on CDP port 9339');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let id = 0;
const pending = new Map();
const errors = [];
ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.id) {
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    clearTimeout(p.timeout);
    m.error ? p.reject(m.error) : p.resolve(m.result);
  } else if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails);
};
const call = (method, params = {}) => new Promise((resolve, reject) => {
  const k = ++id;
  const timeout = setTimeout(() => { pending.delete(k); reject(new Error(`CDP timeout: ${method}`)); }, 15000);
  pending.set(k, { resolve, reject, timeout });
  ws.send(JSON.stringify({ id: k, method, params }));
});
const evaluate = async expression => {
  const x = await call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (x.exceptionDetails) throw x.exceptionDetails;
  return x.result.value;
};
const assert = (condition, description, detail) => {
  if (!condition) throw new Error(`${description}: ${JSON.stringify(detail)}`);
};
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

try {
  await call('Runtime.enable');
  await call('Page.enable');
  await call('Emulation.setDeviceMetricsOverride', { width: 1500, height: 1100, deviceScaleFactor: 1, mobile: false });
  await call('Page.navigate', { url });
  let ready = false;
  for (let i = 0; i < 150; i++) {
    ready = await evaluate('typeof D!=="undefined" && document.getElementById("detail").textContent.length>0');
    if (ready) break;
    await pause(100);
  }
  assert(ready, 'Viewer did not finish rendering', { url });
  const initial = await evaluate('({events:D.events.length,options:$("events").options.length,frames:D.frames.length,detail:$("detail").textContent.length,frame})');
  assert(initial.events === 169 && initial.options === 169 && initial.frames === 654 && initial.detail > 0,
    'Initial viewer coverage', initial);

  const filters = {};
  for (const [name, expected] of [['idsw',56],['clear_fragment',150],['gap_recovery',150],['unrecovered_tail',10]]) {
    filters[name] = await evaluate(`$("filter").value=${JSON.stringify(name)};listing();$("events").options.length`);
    assert(filters[name] === expected, 'Filter count', { name, expected, actual: filters[name] });
  }
  await evaluate('$("filter").value="all";listing();choose(15);frame=event.frame;draw()');
  const atEvent = await evaluate('({event:event.index,frame,detail:JSON.parse($("detail").textContent)})');
  assert(atEvent.event === 15 && atEvent.frame === 30 && atEvent.detail.frame === 30 && atEvent.detail.gt?.id === 26,
    'Event 15 rendered detail', atEvent);
  for (const key of ['motion','matches','predicted','packet_rows','after','batch','cost_calls','event_evidence']) {
    assert(key in atEvent.detail, 'Missing detail field', key);
  }
  for (const key of ['matches','predicted','packet_rows','after','batch','cost_calls','event_evidence']) {
    assert(atEvent.detail[key].length > 0, 'Empty event 15 detail', key);
  }
  const frameFields = await evaluate(`(() => {
    let rows=0, freshRows=0;
    for (const tr of D.frames) for (const group of ['before','predicted','after']) for (const row of tr[group]) {
      rows++;
      if (!Number.isInteger(row.birth_frame) || !Number.isInteger(row.last_observed_frame) ||
          row.birth_frame<1 || row.birth_frame>row.last_observed_frame || row.last_observed_frame>tr.frame ||
          ('birth' in row) || ('last_observed' in row)) return {valid:false,frame:tr.frame,group,row};
      if (group==='after' && row.age===0) {
        freshRows++;
        if (row.last_observed_frame!==tr.frame) return {valid:false,frame:tr.frame,group,row};
      }
    }
    const first=D.frames[0].after;
    return {valid:first.length>0 && first.every(r=>r.birth_frame===1 && r.last_observed_frame===1),rows,freshRows};
  })()`);
  assert(frameFields.valid && frameFields.freshRows > 0, 'One-based state frame fields', frameFields);

  for (const layer of ['gt','mot','batch','candidates','predicted']) {
    await evaluate(`document.querySelector('[data-layer=${layer}]').checked=true;draw()`);
  }
  const layers = await evaluate('Array.from(document.querySelectorAll("[data-layer]")).map(x=>({layer:x.dataset.layer,checked:x.checked}))');
  assert(layers.length === 5 && layers.every(x => x.checked), 'Layer controls', layers);
  const screenshot = await call('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(screenshotPath, Buffer.from(screenshot.data, 'base64'));
  await evaluate('$("next").click();draw()');
  assert(await evaluate('frame') === 31, 'Next frame control', await evaluate('frame'));
  await evaluate('$("prev").click();draw()');
  assert(await evaluate('frame') === 30, 'Previous frame control', await evaluate('frame'));
  await evaluate('$("slider").value=29;$("slider").dispatchEvent(new Event("input"));draw()');
  assert(await evaluate('JSON.parse($("detail").textContent).frame') === 29, 'Slider detail synchronization');
  await evaluate('$("focus").checked=false;draw()');
  await evaluate('$("focus").checked=true;draw()');
  await evaluate('$("play").click()');
  await pause(240);
  const playbackFrame = await evaluate('frame');
  await evaluate('$("play").click();draw()');
  assert(playbackFrame > 29 && await evaluate('timer===null'), 'Playback start/stop', playbackFrame);

  await evaluate('$("filter").value="unrecovered_tail";listing();choose(102);frame=event.frame;draw()');
  const tail = await evaluate('({event:event.index,frame,info:$("info").textContent,detail:JSON.parse($("detail").textContent)})');
  assert(tail.event === 102 && tail.frame === 296 && tail.detail.frame === 296 &&
      tail.info.includes('末段未接回 f296–335') && tail.info.includes('最後配對 f295') &&
      tail.info.includes('40 個有 GT 標註的未配對幀') && !tail.info.includes('→'),
    'Tail span must not imply recovery', tail);
  assert(errors.length === 0, 'Browser JavaScript exceptions', errors);
  console.log(JSON.stringify({ url, initial, filters, event:atEvent.event, eventFrame:atEvent.frame,
    frameFields, layers, playbackFrame, tail:{event:tail.event,frame:tail.frame,info:tail.info},
    screenshot:screenshotPath, errors }, null, 2));
} finally {
  await call('Browser.close');
}
