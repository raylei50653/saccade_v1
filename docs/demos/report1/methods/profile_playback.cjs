#!/usr/bin/env node
// Optional instrumented viewer profile; not an inference benchmark.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
let playwright;
for (const candidate of [process.env.SACCADE_PLAYWRIGHT_MODULE, 'playwright',
  path.join(os.homedir(), '.cache/ms-playwright-go/1.57.0/package')].filter(Boolean)) {
  try { playwright = require(candidate); break; } catch {}
}
if (!playwright) throw new Error('Set SACCADE_PLAYWRIGHT_MODULE to an installed Playwright package.');
const {chromium} = playwright;
const [url, source, output] = process.argv.slice(2);
if (!url || !source || !output) throw new Error('Usage: node profile_playback.cjs URL APP_JS OUTPUT_JSON');
const original = fs.readFileSync(source, 'utf8');
const instrumented = original.replace('  drawPlaceholder(); updateFrameUI();', `
  window.playbackProfile = {render: [], trails: [], ui: [], images: [], chips: 0};
  for (const entry of [
    ['render', () => render, (fn) => { render = fn; }],
    ['trails', () => trailFor, (fn) => { trailFor = fn; }],
    ['ui', () => updateFrameUI, (fn) => { updateFrameUI = fn; }]
  ]) {
    const fn = entry[1]();
    entry[2](function (...args) { const t = performance.now(); try { return fn(...args); }
      finally { window.playbackProfile[entry[0]].push(performance.now() - t); } });
  }
  const originalIds = updateIds;
  updateIds = function (...args) { window.playbackProfile.chips++; return originalIds(...args); };
  const imageContext = typeof videoCtx === 'undefined' ? ctx : videoCtx;
  const originalImage = imageContext.drawImage.bind(imageContext);
  imageContext.drawImage = function (...args) { const t = performance.now(); try { return originalImage(...args); }
    finally { window.playbackProfile.images.push(performance.now() - t); } };
  drawPlaceholder(); updateFrameUI();`);
const stats = values => { const a = [...values].sort((x,y)=>x-y); return {count:a.length,
  meanMs:a.reduce((s,v)=>s+v,0)/(a.length||1), p95Ms:a[Math.min(a.length-1,Math.floor(a.length*.95))]||0,
  maxMs:a[a.length-1]||0, totalMs:a.reduce((s,v)=>s+v,0)}; };
(async () => {
  const browser = await chromium.launch({executablePath:process.env.SACCADE_CHROMIUM_BIN || '/usr/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  const page = await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:2});
  await page.route('**/app.js', r=>r.fulfill({contentType:'text/javascript',body:instrumented}));
  await page.goto(url);
  await page.waitForFunction(()=>window.demoAPI?.getState().loaded && window.demoAPI.getState().frame===220);
  const result = {scope:'Headless Chromium local viewer; instrumented JS wall time, not GPU inference or screen presentation guarantee',browser:await browser.version(),viewport:{width:1440,height:1000,dpr:2},cases:[]};
  const cases = [
    {name:'04-overlay-trails-interpolation',sequence:'MOT17-04-SDP',view:'overlay',trails:true,interpolation:true},
    {name:'04-raw',sequence:'MOT17-04-SDP',view:'raw',trails:true,interpolation:true},
    {name:'04-overlay-no-trails',sequence:'MOT17-04-SDP',view:'overlay',trails:false,interpolation:true},
    {name:'04-overlay-no-interpolation',sequence:'MOT17-04-SDP',view:'overlay',trails:true,interpolation:false},
    {name:'10-overlay-trails-interpolation',sequence:'MOT17-10-SDP',view:'overlay',trails:true,interpolation:true}
  ];
  for (const c of cases) {
    await page.evaluate(async c=>{
      await window.demoAPI.switchSequence(c.sequence); await window.demoAPI.seekFrame(220);
      window.demoAPI.setView(c.view); window.demoAPI.setInterpolation(c.interpolation);
      const trails=document.getElementById('show-trails'); trails.checked=c.trails; trails.dispatchEvent(new Event('change'));
      document.querySelector('video').playbackRate=1; await document.querySelector('video').play();
    },c);
    await page.waitForTimeout(700);
    await page.evaluate(()=>{
      window.playbackProfile={render:[],trails:[],ui:[],images:[],chips:0};
      const v=document.querySelector('video'),q=v.getVideoPlaybackQuality();
      window.profileStart={now:performance.now(),total:q.totalVideoFrames,dropped:q.droppedVideoFrames,mediaTime:v.currentTime};
      window.rafTimes=[]; window.profileRunning=true;
      function tick(t){if(!window.profileRunning)return;window.rafTimes.push(t);requestAnimationFrame(tick);}requestAnimationFrame(tick);
    });
    await page.waitForTimeout(3500);
    const sample = await page.evaluate(()=>{
      window.profileRunning=false;const v=document.querySelector('video'),q=v.getVideoPlaybackQuality();
      const value={...window.playbackProfile,elapsedMs:performance.now()-window.profileStart.now,
        mediaSeconds:v.currentTime-window.profileStart.mediaTime,
        decodedFrames:q.totalVideoFrames-window.profileStart.total,droppedFrames:q.droppedVideoFrames-window.profileStart.dropped,
        rafIntervals:window.rafTimes.slice(1).map((t,i)=>t-window.rafTimes[i])};v.pause();return value;
    });
    const row={...c,...sample};
    for(const key of ['render','trails','ui','images','rafIntervals'])row[key]=stats(sample[key]);
    row.rendersPerSecond=row.render.count*1000/row.elapsedMs;result.cases.push(row);
    process.stdout.write(`${c.name}: ${row.rendersPerSecond.toFixed(1)} render/s, p95 ${row.render.p95Ms.toFixed(2)}ms, dropped ${row.droppedFrames}/${row.decodedFrames}\n`);
  }
  fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
