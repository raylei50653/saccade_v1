(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  let data = window.SACCADE_DEMO;
  const video = $('source-video');
  const canvas = $('replay-canvas');
  const ctx = canvas.getContext('2d');
  const videoCanvas = $('video-canvas');
  const videoCtx = videoCanvas.getContext('2d', { alpha: false });
  const stage = $('stage');
  if (!data || !data.meta || !Array.isArray(data.frames) || !data.frames.length || !window.SaccadeGapFill) {
    $('loading-state').hidden = true;
    $('error-state').hidden = false;
    $('error-state').textContent = '找不到完整回放資料。請保持 assets 與 gap-fill.js 在同一份展示包內。';
    return;
  }
  const catalog = { ...(window.SACCADE_DEMOS || {}) };
  catalog[data.meta.sequence || 'MOT17-10-SDP'] = data;
  window.SACCADE_DEMOS = catalog;
  let meta = data.meta;
  let fps = Number(meta.fps);
  let total = data.frames.length;
  let activeSequence = meta.sequence || 'MOT17-10-SDP';
  const state = { frame: 0, selectedId: null, view: 'overlay', boxes: true, trails: true, interpolation: true, gapFill: true, interpolationActive: false, interpolationPhase: 0, loaded: false, error: null };
  const gapIndexes = new WeakMap();
  let gapIndex = null;
  let gapFrameRows = [];
  let displayedGaps = [];
  let panels = [];
  let lastSidebarFrame = -1;
  let lastSidebarSelection = undefined;
  let lastSidebarLoaded = undefined;
  let lastSidebarGapFill = undefined;
  let stageRect = null;
  let videoPaintKey = '';
  let lastTrailFrame = -1;
  let trailCache = new Map();
  const textWidths = new Map();
  let seekGeneration = 0;
  let sourceGeneration = 0;
  let loadedSourceGeneration = -1;
  let expectedSourceUrl = '';
  let videoFrameCallbackId = null;
  const pendingSeeks = new Set();
  // rVFC identifies the decoded source frame; one rAF loop paints subframe overlays.
  // Its clock may move boxes within k..k+1, but can never advance the source frame.
  const hasVideoFrameClock = 'requestVideoFrameCallback' in video;
  let displayAnchor = null;
  let lastDecodedFrame = null;
  let waitingForVideo = true;
  let displayedRows = [];
  let resolveReady = () => {};
  let ready = Promise.resolve(false);
  const palette = ['#79efce', '#ffbd73', '#9cbcff', '#f1a3ee', '#f5e17c', '#75d7f3', '#b7e98c', '#f9959f'];
  const frameByIndex = (frame) => data.frames[clampFrame(frame)];
  const rowsAt = (frame) => frameByIndex(frame).rows || [];
  const color = (id) => palette[Math.abs(Number(id) * 7) % palette.length];
  function clampFrame(frame) { return Math.max(0, Math.min(total - 1, Math.round(Number(frame) || 0))); }
  function sourceMatches() { return Boolean(expectedSourceUrl) && (video.currentSrc || video.src) === expectedSourceUrl; }
  function clampPhase(phase) { return Number.isFinite(Number(phase)) ? Math.max(0, Math.min(1, Number(phase))) : 0; }
  function sampleDisplayRows(frame, phase) {
    const index = clampFrame(frame);
    const fraction = clampPhase(phase);
    const current = rowsAt(index);
    const next = index < total - 1 ? rowsAt(index + 1) : [];
    const following = new Map(next.map((row) => [`${row[0]}:${row[1]}`, row]));
    return current.map((row) => {
      const result = row.slice();
      const target = following.get(`${row[0]}:${row[1]}`);
      // No new rows, gap filling, identity merges, predicted-row interpolation or extrapolation.
      if (fraction > 0 && row[8] === 1 && target && target[8] === 1) {
        for (let coordinate = 2; coordinate < 6; coordinate++) result[coordinate] = row[coordinate] + (target[coordinate] - row[coordinate]) * fraction;
      }
      return result;
    });
  }
  function sampleGapRows(frame, phase = 0) {
    return window.SaccadeGapFill.sample(gapIndex, clampFrame(frame), phase);
  }
  function resetDisplayClock() {
    displayAnchor = null;
    state.interpolationActive = false;
    state.interpolationPhase = 0;
  }
  function phaseAt(now, frame = state.frame) {
    if (!state.interpolation || !hasVideoFrameClock || !displayAnchor || waitingForVideo || video.paused || video.ended || video.seeking || video.readyState < 2 || frame === total - 1 || displayAnchor.frame !== frame || state.view === 'raw' || (!state.boxes && !state.trails)) return 0;
    return clampPhase((now - displayAnchor.expectedDisplayTime) * displayAnchor.playbackRate * fps / 1000);
  }
  function reanchorPlayback(now = performance.now()) {
    resetDisplayClock();
    if (hasVideoFrameClock && lastDecodedFrame && lastDecodedFrame.frame === state.frame && !video.paused && !video.ended && !video.seeking && !waitingForVideo) {
      displayAnchor = { frame: state.frame, mediaTime: lastDecodedFrame.mediaTime, expectedDisplayTime: now, playbackRate: video.playbackRate };
    }
  }
  function formatTime(seconds) {
    const ms = Math.max(0, Math.round(seconds * 1000));
    return `${String(Math.floor(ms / 60000)).padStart(2, '0')}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}.${String(ms % 1000).padStart(3, '0')}`;
  }
  function setError(message) {
    resetDisplayClock();
    state.error = message;
    $('error-state').textContent = message;
    $('error-state').hidden = false;
    $('loading-state').hidden = true;
    updatePlaybackStatus();
  }
  function drawPlaceholder() {
    ctx.fillStyle = '#080e17';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  function layouts(width, height) {
    const vertical = window.matchMedia('(max-width:620px)').matches;
    const aspect = Number(meta.width) / Number(meta.height);
    const contentAspect = state.view !== 'sideBySide' ? aspect : vertical ? aspect / 2 : aspect * 2;
    const w = Math.min(width, height * contentAspect), h = w / contentAspect;
    const x = (width - w) / 2, y = (height - h) / 2;
    if (state.view !== 'sideBySide') return [{ x, y, w, h, tracks: state.view === 'overlay', title: state.view === 'raw' ? '原始影像' : '追蹤觀測' }];
    if (vertical) return [{ x, y, w, h: h / 2, tracks: false, title: '原始影像' }, { x, y: y + h / 2, w, h: h / 2, tracks: true, title: '追蹤觀測' }];
    return [{ x, y, w: w / 2, h, tracks: false, title: '原始影像' }, { x: x + w / 2, y, w: w / 2, h, tracks: true, title: '追蹤觀測' }];
  }
  function paintLabel(text, x, y, fill, textColor, fontSize = 11) {
    ctx.font = `600 ${fontSize}px "Noto Sans TC", "Noto Sans CJK TC", system-ui, sans-serif`;
    const key = `${ctx.font}:${text}`;
    if (!textWidths.has(key)) textWidths.set(key, ctx.measureText(text).width);
    const textWidth = textWidths.get(key);
    ctx.fillStyle = fill;
    ctx.fillRect(x, y, textWidth + 12, fontSize + 10);
    ctx.fillStyle = textColor;
    ctx.fillText(text, x + 6, y + fontSize + 2);
    return textWidth + 12;
  }
  function trailFor(row, frame, filled = false) {
    const id = row[0], uid = row[1];
    if (lastTrailFrame !== frame) { lastTrailFrame = frame; trailCache = new Map(); }
    const key = `${id}:${uid}`;
    let history = trailCache.get(key);
    if (history) return [...history, [(row[2] + row[4]) / 2, row[5], filled]];
    const points = [];
    for (let previous = frame - 1; previous >= Math.max(0, frame - 29); previous--) {
      const found = rowsAt(previous).find((candidate) => candidate[0] === id && (state.gapFill || candidate[1] === uid));
      const gap = !found && state.gapFill ? gapFrameRows[previous].get(id) : null;
      if (!found && !gap) break;
      const box = found ? found.slice(2, 6) : gap.xyxy;
      points.push([(box[0] + box[2]) / 2, box[3], Boolean(gap)]);
    }
    points.reverse();
    trailCache.set(key, points);
    return [...points, [(row[2] + row[4]) / 2, row[5], filled]];
  }
  function paintPanel(panel) {
    ctx.save();
    ctx.beginPath(); ctx.rect(panel.x, panel.y, panel.w, panel.h); ctx.clip();
    const sx = panel.w / Number(meta.width);
    const sy = panel.h / Number(meta.height);
    const currentRows = [
      ...displayedGaps.map(gap => ({ row: [gap.outputId, gap.fromUid, ...gap.xyxy], gap })),
      ...displayedRows.map(row => ({ row, gap: null }))
    ];
    if (panel.tracks) {
      for (const { row, gap } of currentRows) {
        const chosen = state.selectedId === row[0];
        const dimmed = state.selectedId !== null && !chosen;
        const stroke = color(row[0]);
        ctx.setLineDash(gap ? [5, 4] : []);
        if (state.trails) {
          const points = trailFor(row, state.frame, Boolean(gap));
          if (points.length > 1) {
            ctx.globalAlpha = dimmed ? 0.22 : chosen ? 1 : 0.7;
            ctx.strokeStyle = stroke; ctx.lineWidth = chosen ? 2.7 : 1.4;
            let dashed = null;
            for (let i = 1; i < points.length; i++) {
              const inferred = points[i - 1][2] || points[i][2];
              if (inferred !== dashed) {
                if (dashed !== null) ctx.stroke();
                ctx.setLineDash(inferred ? [5, 4] : []);
                ctx.beginPath();
                ctx.moveTo(panel.x + points[i - 1][0] * sx, panel.y + points[i - 1][1] * sy);
                dashed = inferred;
              }
              ctx.lineTo(panel.x + points[i][0] * sx, panel.y + points[i][1] * sy);
            }
            ctx.stroke();
          }
        }
        if (!state.boxes) continue;
        ctx.setLineDash(gap ? [5, 4] : []);
        const x = panel.x + row[2] * sx, y = panel.y + row[3] * sy;
        const w = (row[4] - row[2]) * sx, h = (row[5] - row[3]) * sy;
        ctx.globalAlpha = dimmed ? 0.28 : 1;
        ctx.strokeStyle = stroke;
        ctx.lineWidth = chosen ? 3 : 1.6;
        if (chosen) {
          ctx.fillStyle = stroke;
          ctx.globalAlpha = 0.10;
          ctx.fillRect(x, y, w, h);
          ctx.globalAlpha = 1;
        }
        ctx.strokeRect(x, y, w, h);
        const label = `ID ${row[0]}${gap ? ' 補' : ''}`;
        const fontSize = panel.w < 520 ? 9 : 11;
        ctx.font = `600 ${fontSize}px "Noto Sans TC", system-ui, sans-serif`;
        const labelW = ctx.measureText(label).width + 12;
        const labelX = Math.max(panel.x, Math.min(panel.x + panel.w - labelW, x));
        const labelY = Math.max(panel.y, Math.min(panel.y + panel.h - fontSize - 10, y - fontSize - 10));
        paintLabel(label, labelX, labelY, stroke, '#071d21', fontSize);
      }
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    paintLabel(panel.title, panel.x + 10, panel.y + 10, '#0b1525d9', '#eaf3fa', 10);
    const frameText = `F ${String(state.frame + 1).padStart(3, '0')}`;
    ctx.font = '500 9px "DejaVu Sans Mono", monospace';
    const labelWidth = ctx.measureText(frameText).width + 12;
    paintLabel(frameText, panel.x + panel.w - labelWidth - 10, panel.y + 10, '#0b1525d9', '#d3e8ed', 9);
    ctx.restore();
  }
  function render(frame = state.frame, phase = null) {
    state.frame = clampFrame(frame);
    const fraction = phase === null ? phaseAt(performance.now(), state.frame) : clampPhase(phase);
    displayedRows = sampleDisplayRows(state.frame, fraction);
    displayedGaps = state.gapFill ? sampleGapRows(state.frame, fraction) : [];
    state.interpolationPhase = fraction;
    state.interpolationActive = fraction > 0 && displayedRows.some((row, i) => row.slice(2, 6).some((coordinate, j) => coordinate !== rowsAt(state.frame)[i][j + 2]));
    const rect = stageRect || (stageRect = stage.getBoundingClientRect());
    const width = rect.width;
    const height = rect.height;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (!width || !height) return;
    const targetWidth = Math.round(width * dpr), targetHeight = Math.round(height * dpr);
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) { canvas.width = targetWidth; canvas.height = targetHeight; }
    if (videoCanvas.width !== targetWidth || videoCanvas.height !== targetHeight) { videoCanvas.width = targetWidth; videoCanvas.height = targetHeight; videoPaintKey = ''; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    panels = layouts(width, height);
    // Normal views use the browser's native video compositor. Only comparison
    // mode needs a pixel copy to show the same decoded image in two panels.
    const imageKey = `${sourceGeneration}:${state.frame}:${state.loaded}:${state.view}:${width}:${height}:${dpr}`;
    if (state.view === 'sideBySide' && videoPaintKey !== imageKey) {
      videoCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      videoCtx.fillStyle = '#050b13'; videoCtx.fillRect(0, 0, width, height);
      if (state.loaded && sourceMatches() && video.readyState >= 2) {
        for (const panel of panels) videoCtx.drawImage(video, panel.x, panel.y, panel.w, panel.h);
        videoPaintKey = imageKey;
      } else videoPaintKey = '';
    }
    if (state.loaded && sourceMatches() && video.readyState >= 2) {
      panels.forEach(paintPanel);
      if (state.view === 'sideBySide') {
        ctx.strokeStyle = '#0a1420'; ctx.lineWidth = 2;
        ctx.beginPath();
        if (window.matchMedia('(max-width:620px)').matches) { ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); }
        else { ctx.moveTo(width / 2, 0); ctx.lineTo(width / 2, height); }
        ctx.stroke();
      }
    }
    updateFrameUI();
    updateInterpolationStatus();
    const gapVisible = state.gapFill && state.view !== 'raw' && (state.boxes || state.trails) && displayedGaps.length > 0;
    const gapBadge = $('gap-fill-status');
    gapBadge.hidden = !gapVisible;
    const gapLabel = `補框 ${displayedGaps.length}`;
    if (gapBadge.textContent !== gapLabel) gapBadge.textContent = gapLabel;
  }
  function updateInterpolationStatus() {
    const badge = $('interpolation-status');
    // Describe the selected playback mode, not the instant phase: phase is
    // deliberately zero at every decoded-frame boundary and must not flicker.
    const enabledForPlayback = state.interpolation && hasVideoFrameClock && state.loaded && !waitingForVideo && !video.paused && !video.ended && !video.seeking && video.readyState >= 2 && state.view !== 'raw' && (state.boxes || state.trails);
    const label = enabledForPlayback ? '顯示插值' : state.gapFill && state.view !== 'raw' ? '逐幀補框' : '原始逐幀';
    if (badge.textContent !== label) badge.textContent = label;
    badge.classList.toggle('active', enabledForPlayback);
    const observedMode = !hasVideoFrameClock ? '此瀏覽器沒有影片影格時鐘，觀測框以來源逐幀顯示。' : state.interpolation ? '播放時平滑相鄰同身份觀測框；暫停／逐幀時觀測框為原始座標。' : '觀測框目前使用原始逐幀座標。';
    badge.title = observedMode + (state.gapFill ? 'MOT 缺幀補框已開啟，虛線為離線回補。' : 'MOT 缺幀補框已關閉。');
  }
  function updateFrameUI() {
    if (lastSidebarFrame === state.frame && lastSidebarSelection === state.selectedId && lastSidebarLoaded === state.loaded && lastSidebarGapFill === state.gapFill) return;
    $('frame-label').textContent = String(state.frame + 1).padStart(3, '0');
    $('frame-slider').value = String(state.frame);
    if (document.activeElement !== $('frame-input')) $('frame-input').value = String(state.frame + 1);
    $('time-label').textContent = formatTime(state.frame / fps);
    $('observed-count').textContent = String(rowsAt(state.frame).length);
    $('gap-count-note').textContent = state.gapFill ? `原始觀測 ${rowsAt(state.frame).length} 筆 ＋ 插值補框 ${displayedGaps.length} 筆；虛線非新觀測。` : 'MOT 缺幀補框已關閉；只顯示原始觀測。';
    $('previous-button').disabled = !state.loaded || state.frame === 0;
    $('next-button').disabled = !state.loaded || state.frame === total - 1;
    if (lastSidebarFrame !== state.frame || lastSidebarSelection !== state.selectedId || lastSidebarGapFill !== state.gapFill) {
      updateIds(); updateSelection(); updateMotion();
      lastSidebarFrame = state.frame; lastSidebarSelection = state.selectedId;
    }
    lastSidebarLoaded = state.loaded;
    lastSidebarGapFill = state.gapFill;
  }
  function updateIds() {
    const list = $('id-list');
    const rows = rowsAt(state.frame);
    $('empty-observations').hidden = rows.length !== 0;
    const sorted = [...rows.map(row => ({id:row[0], filled:false})), ...displayedGaps.map(gap => ({id:gap.outputId, filled:true}))].sort((a, b) => a.id - b.id);
    const missing = state.selectedId !== null && !sorted.some(item => item.id === state.selectedId);
    const signature = `${sorted.map(item => `${item.id}:${item.filled}`).join(',')}|${missing ? state.selectedId : ''}`;
    if (list.dataset.signature === signature) {
      for (const button of list.querySelectorAll('[data-track-id]')) {
        const selected = Number(button.dataset.trackId) === state.selectedId;
        button.classList.toggle('selected', selected);
        button.setAttribute('aria-pressed', String(selected));
      }
      return;
    }
    list.dataset.signature = signature;
    const existing = new Map([...list.querySelectorAll('[data-track-id]')].map(button => [Number(button.dataset.trackId), button]));
    const buttons = [];
    for (const item of sorted) {
      const id = item.id;
      let button = existing.get(id);
      if (!button) {
        button = document.createElement('button');
        button.addEventListener('click', () => setSelected(id));
      }
      button.type = 'button'; button.className = `id-chip${item.filled ? ' gap-chip' : ''}${state.selectedId === id ? ' selected' : ''}`;
      button.style.setProperty('--track-color', color(id));
      button.textContent = `${id}${item.filled ? ' 補' : ''}`;
      button.dataset.trackId = String(id);
      button.dataset.interpolated = String(item.filled);
      button.setAttribute('aria-label', `選取輸出 ID ${id}${item.filled ? '，缺幀插值' : ''}`);
      button.setAttribute('aria-pressed', String(state.selectedId === id));
      buttons.push(button);
    }
    if (missing) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'id-chip missing';
      button.style.setProperty('--track-color', color(state.selectedId));
      button.textContent = `ID ${state.selectedId} · 本幀未輸出`;
      button.addEventListener('click', () => setSelected(null));
      button.title = '清除選取'; buttons.push(button);
    }
    list.replaceChildren(...buttons);
  }
  function findAppearance(direction) {
    if (state.selectedId === null) return null;
    for (let frame = state.frame + direction; frame >= 0 && frame < total; frame += direction) {
      if (rowsAt(frame).some((row) => row[0] === state.selectedId)) return frame;
    }
    return null;
  }
  function updateSelection() {
    const selected = state.selectedId !== null;
    $('selection-empty').hidden = selected;
    $('selection-content').hidden = !selected;
    $('clear-selection').hidden = !selected;
    $('selected-title').textContent = selected ? `輸出 ID ${state.selectedId}` : '選取軌跡';
    if (!selected) return;
    const row = rowsAt(state.frame).find((candidate) => candidate[0] === state.selectedId);
    const gap = state.gapFill ? gapFrameRows[state.frame].get(state.selectedId) : null;
    $('selected-state').textContent = row ? '本幀有觀測' : gap ? '缺幀插值（非觀測）' : '本幀未輸出';
    $('selected-state').classList.toggle('missing', !row);
    $('selected-state').classList.toggle('interpolated', Boolean(gap));
    $('selected-uid').textContent = row ? String(row[1]) : gap ? gap.fromUid === gap.toUid ? String(gap.fromUid) : `${gap.fromUid} → ${gap.toUid}` : '—';
    $('selected-score').textContent = row ? Number(row[6]).toFixed(3) : '—';
    $('selected-association').textContent = row ? Number(row[7]).toFixed(3) : '—';
    $('selected-flags').textContent = row ? (row[8] === 1 ? 'observed' : row[8] === 2 ? 'predicted' : String(row[8])) : gap ? 'interpolated' : '—';
    $('selected-box').textContent = row ? row.slice(2, 6).map((value) => Number(value).toFixed(1)).join(', ') : '—';
    $('selected-gap-coordinates').hidden = !gap;
    $('selected-gap-box').textContent = gap ? gap.xyxy.map(value => value.toFixed(1)).join(', ') : '—';
    const before = findAppearance(-1), after = findAppearance(1);
    $('previous-appearance').disabled = before === null;
    $('next-appearance').disabled = after === null;
    $('selected-history').textContent = row ? '框為 Kalman 狀態估計；座標保留原始數值。' : gap ? `來源影格 ${gap.fromFrame + 1} → ${gap.toFrame + 1}，依相同輸出 ID 線性回補 ${gap.missingFrames} 個缺幀；MOT confidence = −1。` : before === null ? '截至目前影格，此 ID 尚未輸出。' : `前次輸出：來源影格 ${before + 1}。`;
  }
  function updateMotion() {
    const motion = frameByIndex(state.frame).motion;
    const valid = motion && Number(motion.status) === 2;
    $('motion-state').textContent = valid ? '有效' : motion && Number(motion.status) === 1 ? '初始化' : '無有效位移';
    $('motion-state').classList.toggle('valid', Boolean(valid));
    const signed = (value) => `${value > 0 ? '+' : ''}${Number(value).toFixed(2)}`;
    $('motion-x').textContent = valid ? signed(motion.tx) : '—';
    $('motion-y').textContent = valid ? signed(motion.ty) : '—';
  }
  function updatePlaybackStatus() {
    const playing = !video.paused && !video.ended;
    $('play-label').textContent = playing ? '暫停' : '播放';
    $('play-icon').textContent = playing ? 'Ⅱ' : '▶';
    $('play-button').setAttribute('aria-label', playing ? '暫停回放' : '播放回放');
    document.querySelector('.status-dot').classList.toggle('playing', playing);
    $('player-status').textContent = state.error ? '載入失敗' : !state.loaded ? '準備回放' : video.ended ? '序列回放完成' : playing ? `觀測結果回放中 · ${video.playbackRate}×` : `已暫停 · 原始 ${fps} FPS 時間軸`;
  }
  async function seekFrame(frame, options = {}) {
    const target = clampFrame(frame);
    const generation = ++seekGeneration;
    const source = sourceGeneration;
    const sourceReady = ready;
    const loaded = await sourceReady;
    if (!loaded || source !== sourceGeneration || generation !== seekGeneration || !sourceMatches()) return getState();
    const keepPlaying = Boolean(options.keepPlaying);
    waitingForVideo = true;
    resetDisplayClock();
    if (!keepPlaying) video.pause();
    const targetTime = Math.min((target + 0.1) / fps, Math.max(0, video.duration - 0.0001));
    if (Math.abs(video.currentTime - targetTime) > 0.000001) {
      await new Promise((resolve) => {
        const finish = () => { video.removeEventListener('seeked', finish); clearTimeout(timeout); pendingSeeks.delete(finish); resolve(); };
        const timeout = setTimeout(finish, 5000);
        pendingSeeks.add(finish);
        video.addEventListener('seeked', finish);
        video.currentTime = targetTime;
      });
    }
    if (source === sourceGeneration && generation === seekGeneration && sourceMatches()) {
      videoPaintKey = '';
      lastDecodedFrame = { frame: target, mediaTime: target / fps };
      waitingForVideo = false;
      render(target, 0);
      if (keepPlaying) reanchorPlayback();
    }
    updatePlaybackStatus();
    return getState();
  }
  function setSelected(id) {
    state.selectedId = id === null || id === undefined || id === '' ? null : Number(id);
    lastSidebarSelection = undefined;
    render();
    return getState();
  }
  function setView(view) {
    if (!['raw', 'overlay', 'sideBySide'].includes(view)) throw new Error(`Unknown view: ${view}`);
    state.view = view;
    stageRect = null;
    stage.classList.toggle('side-by-side', view === 'sideBySide');
    for (const button of document.querySelectorAll('[data-view]')) {
      const active = button.dataset.view === view;
      button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
    }
    render(); return getState();
  }
  function getState() {
    return { ...state, sequence: activeSequence, playing: state.loaded && !video.paused && !video.ended, playbackRate: video.playbackRate, rate: video.playbackRate, totalFrames: total, observedRows: rowsAt(state.frame).length, mediaTime: video.currentTime, selectedRow: rowsAt(state.frame).find((row) => row[0] === state.selectedId) || null, displayRows: displayedRows.map((row) => row.slice()), gapRows: displayedGaps.map(gap => ({...gap, xyxy:gap.xyxy.slice()})), gapStats: gapIndex ? {...gapIndex.stats} : null };
  }
  function setGapFill(enabled) {
    state.gapFill = Boolean(enabled);
    $('show-gap-fill').checked = state.gapFill;
    lastTrailFrame = -1;
    trailCache = new Map();
    render();
    return getState();
  }
  function setInterpolation(enabled) {
    state.interpolation = Boolean(enabled);
    $('show-interpolation').checked = state.interpolation;
    reanchorPlayback();
    render(state.frame, 0);
    return getState();
  }
  async function togglePlayback() {
    const source = sourceGeneration;
    const loaded = await ready;
    if (!loaded || source !== sourceGeneration || !sourceMatches()) return;
    if (!video.paused && !video.ended) { video.pause(); return; }
    if (video.ended || state.frame === total - 1) await seekFrame(0);
    if (source !== sourceGeneration) return;
    try { await video.play(); } catch (error) { if (source === sourceGeneration) setError(`影片無法播放：${error.message}`); }
  }
  function updateSourceUI() {
    $('sequence-select').value = activeSequence;
    $('sequence-label').textContent = meta.sequence || activeSequence;
    $('baseline-label').textContent = String(meta.baseline || '3cbaa518f0dc').slice(0, 12);
    $('source-format').textContent = `${meta.width} × ${meta.height} · 來源 ${fps} FPS`;
    $('frame-total').textContent = String(total);
    $('frame-slider').max = String(total - 1);
    $('frame-input').max = String(total);
    $('duration-label').textContent = formatTime(total / fps);
    const bookmarkFrames = [Math.min(60, total - 1), Math.floor(total / 2), Math.max(0, total - 54)];
    document.querySelectorAll('[data-bookmark]').forEach((button, index) => {
      button.dataset.bookmark = String(bookmarkFrames[index]);
    });
    $('source-download').href = meta.sourceVideo || 'assets/source.mp4';
    $('backup-download').href = meta.backupVideo || (activeSequence === 'MOT17-02-SDP' ? 'assets/tracking-demo-02.mp4' : 'assets/tracking-demo.mp4');
    $('metric-offline-label').textContent = `完整 ${total} 幀離線完成`;
    const metrics = data.metrics;
    const measured = metrics && Number.isFinite(metrics.p95Ms);
    document.querySelector('.measurement-section').classList.toggle('unmeasured', !measured);
    $('measurement-title').textContent = `${activeSequence}｜${measured ? '獨立效能量測' : '尚未量測'}`;
    $('measurement-scope').textContent = measured ? '播放速度依來源時間軸；以下數字取自此序列的獨立量測。' : '此序列尚無對應的獨立效能量測；不沿用其他序列數值。';
    const value = (field, digits) => metrics && Number.isFinite(metrics[field]) ? metrics[field].toFixed(digits) : '—';
    $('metric-single').textContent = value('p95Ms', 3);
    $('metric-queued').textContent = value('queuedP95Ms', 3);
    $('metric-throughput').textContent = value('throughputFps', 1);
    $('metric-offline').textContent = value('offlineTotalSeconds', 3);
    const notes = {
      'metric-single-note': 'JPEG 預載 · 480 筆暖機後樣本',
      'metric-queued-note': '含在途等待與 caller 接收政策',
      'metric-throughput-note': '處理間隔換算；不是逐幀延遲',
      'metric-offline-note': '含啟動、讀檔、定稿與最終寫檔',
    };
    for (const [id, note] of Object.entries(notes)) $(id).textContent = measured ? note : `${activeSequence} 尚未量測`;
  }
  async function switchSequence(key) {
    const selected = catalog[key];
    if (!selected || !selected.meta || !Array.isArray(selected.frames) || !selected.frames.length) throw new Error(`Unknown sequence: ${key}`);
    const generation = ++sourceGeneration;
    ++seekGeneration;
    resolveReady(false);
    for (const finish of [...pendingSeeks]) finish();
    if (videoFrameCallbackId !== null && typeof video.cancelVideoFrameCallback === 'function') video.cancelVideoFrameCallback(videoFrameCallbackId);
    videoFrameCallbackId = null;
    const preferredRate = Number($('playback-rate').value) || video.playbackRate || 1;
    video.pause();
    data = selected;
    if (!gapIndexes.has(data)) {
      const index = window.SaccadeGapFill.build(data.frames);
      const frames = index.byFrame.map((_, frame) => new Map(window.SaccadeGapFill.sample(index, frame).map(gap => [gap.outputId, gap])));
      gapIndexes.set(data, {index, frames});
    }
    const gaps = gapIndexes.get(data);
    gapIndex = gaps.index;
    gapFrameRows = gaps.frames;
    meta = data.meta;
    fps = Number(meta.fps);
    total = data.frames.length;
    activeSequence = key;
    window.SACCADE_DEMO = data;
    state.loaded = false;
    state.error = null;
    state.frame = 0;
    state.selectedId = null;
    lastSidebarFrame = -1;
    lastSidebarSelection = undefined;
    lastSidebarLoaded = undefined;
    lastSidebarGapFill = undefined;
    $('id-list').removeAttribute('data-signature');
    lastTrailFrame = -1;
    trailCache = new Map();
    videoPaintKey = '';
    lastDecodedFrame = null;
    displayedRows = [];
    displayedGaps = [];
    panels = [];
    waitingForVideo = true;
    resetDisplayClock();
    loadedSourceGeneration = -1;
    ready = new Promise((resolve) => { resolveReady = resolve; });
    const sourceReady = ready;
    expectedSourceUrl = new URL(meta.sourceVideo || 'assets/source.mp4', document.baseURI).href;
    $('loading-state').hidden = false;
    $('error-state').hidden = true;
    $('frame-input').value = '1';
    for (const id of ['frame-slider', 'play-button', 'restart-button', 'previous-button', 'next-button']) $(id).disabled = true;
    updateSourceUI();
    video.poster = meta.poster || (key === 'MOT17-02-SDP' ? 'assets/poster-02.jpg' : 'assets/poster.jpg');
    video.defaultPlaybackRate = preferredRate;
    video.src = expectedSourceUrl;
    video.load();
    video.playbackRate = preferredRate;
    render(0, 0);
    updatePlaybackStatus();
    scheduleVideoFrame(generation);
    const loaded = await sourceReady;
    if (!loaded || generation !== sourceGeneration) return getState();
    await seekFrame(Math.min(total - 1, 220));
    return getState();
  }
  function selectAtPoint(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    const panel = panels.find((candidate) => candidate.tracks && x >= candidate.x && x <= candidate.x + candidate.w && y >= candidate.y && y <= candidate.y + candidate.h);
    if (!panel || !state.boxes) return;
    const sourceX = (x - panel.x) * Number(meta.width) / panel.w;
    const sourceY = (y - panel.y) * Number(meta.height) / panel.h;
    const candidates = [...displayedRows.map(row => ({id:row[0],box:row.slice(2,6),filled:false})), ...displayedGaps.map(gap => ({id:gap.outputId,box:gap.xyxy,filled:true}))];
    const hits = candidates.filter(({box}) => sourceX >= box[0] && sourceX <= box[2] && sourceY >= box[1] && sourceY <= box[3]);
    hits.sort((a, b) => Number(a.filled) - Number(b.filled) || (a.box[2]-a.box[0])*(a.box[3]-a.box[1]) - (b.box[2]-b.box[0])*(b.box[3]-b.box[1]));
    setSelected(hits.length ? hits[0].id : null);
  }
  for (const option of $('sequence-select').options) option.disabled = !catalog[option.value];
  $('sequence-select').addEventListener('change', (event) => switchSequence(event.target.value));
  $('play-button').addEventListener('click', togglePlayback);
  $('restart-button').addEventListener('click', () => seekFrame(0));
  $('previous-button').addEventListener('click', () => seekFrame(state.frame - 1));
  $('next-button').addEventListener('click', () => seekFrame(state.frame + 1));
  $('frame-slider').addEventListener('input', (event) => seekFrame(Number(event.target.value)));
  $('frame-input').addEventListener('change', (event) => { const requested = clampFrame(Number(event.target.value) - 1); event.target.value = String(requested + 1); seekFrame(requested); });
  $('playback-rate').addEventListener('change', (event) => { video.playbackRate = Number(event.target.value); updatePlaybackStatus(); });
  $('show-boxes').addEventListener('change', (event) => { state.boxes = event.target.checked; render(); });
  $('show-trails').addEventListener('change', (event) => { state.trails = event.target.checked; render(); });
  $('show-interpolation').addEventListener('change', (event) => setInterpolation(event.target.checked));
  $('show-gap-fill').addEventListener('change', (event) => setGapFill(event.target.checked));
  $('clear-selection').addEventListener('click', () => setSelected(null));
  $('previous-appearance').addEventListener('click', () => { const frame = findAppearance(-1); if (frame !== null) seekFrame(frame); });
  $('next-appearance').addEventListener('click', () => { const frame = findAppearance(1); if (frame !== null) seekFrame(frame); });
  for (const button of document.querySelectorAll('[data-view]')) button.addEventListener('click', () => setView(button.dataset.view));
  for (const button of document.querySelectorAll('[data-bookmark]')) button.addEventListener('click', () => seekFrame(Math.min(total - 1, Number(button.dataset.bookmark))));
  $('fullscreen-button').addEventListener('click', async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await stage.requestFullscreen(); } catch { $('player-status').textContent = '此瀏覽器未開放全螢幕；可使用瀏覽器 F11。'; }
  });
  canvas.addEventListener('click', selectAtPoint);
  window.addEventListener('keydown', (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.target.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) return;
    if (event.code === 'Space' && event.target.tagName !== 'BUTTON') { event.preventDefault(); togglePlayback(); }
    else if (event.code === 'ArrowLeft') { event.preventDefault(); seekFrame(state.frame - 1); }
    else if (event.code === 'ArrowRight') { event.preventDefault(); seekFrame(state.frame + 1); }
    else if (event.code === 'Escape') setSelected(null);
  });
  for (const name of ['play', 'pause', 'ratechange', 'ended']) video.addEventListener(name, updatePlaybackStatus);
  for (const name of ['pause', 'waiting', 'stalled', 'seeking', 'ratechange']) video.addEventListener(name, () => {
    if (!state.loaded || !sourceMatches()) return;
    if (name === 'waiting' || name === 'stalled' || name === 'seeking') waitingForVideo = true;
    resetDisplayClock();
    render(state.frame, 0);
  });
  video.addEventListener('playing', () => { if (state.loaded && sourceMatches()) { waitingForVideo = false; reanchorPlayback(); } });
  video.addEventListener('ended', () => { if (state.loaded && sourceMatches() && video.ended) { waitingForVideo = true; resetDisplayClock(); render(total - 1, 0); } });
  video.addEventListener('seeked', () => {
    if (!state.loaded || !sourceMatches() || video.seeking || video.readyState < 2) return;
    waitingForVideo = false;
    const frame = clampFrame(Math.floor(video.currentTime * fps + 0.00001));
    lastDecodedFrame = { frame, mediaTime: frame / fps };
    resetDisplayClock();
    videoPaintKey = '';
    render(frame, 0);
    if (!video.paused) reanchorPlayback();
  });
  video.addEventListener('error', () => {
    if (!sourceMatches()) return;
    resolveReady(false);
    state.loaded = false;
    setError(`無法讀取 ${activeSequence} 的本機影片。請保持整份展示包的相對位置，並使用支援 H.264 的瀏覽器。`);
  });
  video.addEventListener('loadedmetadata', () => {
    if (!sourceMatches() || video.readyState < 1 || loadedSourceGeneration === sourceGeneration) return;
    loadedSourceGeneration = sourceGeneration;
    state.loaded = true;
    for (const id of ['frame-slider', 'play-button', 'restart-button', 'next-button']) $(id).disabled = false;
    video.playbackRate = Number($('playback-rate').value) || 1;
    resolveReady(true);
    updatePlaybackStatus();
  });
  video.addEventListener('loadeddata', () => { if (state.loaded && sourceMatches()) { $('loading-state').hidden = true; render(state.frame, 0); } });
  function scheduleVideoFrame(generation) {
    if (!hasVideoFrameClock || generation !== sourceGeneration) return;
    videoFrameCallbackId = video.requestVideoFrameCallback((now, metadata) => {
      if (generation !== sourceGeneration) return;
      videoFrameCallbackId = null;
      if (!state.loaded || !sourceMatches()) { scheduleVideoFrame(generation); return; }
      const frame = clampFrame(Math.floor(metadata.mediaTime * fps + 0.0001));
      lastDecodedFrame = { frame, mediaTime: metadata.mediaTime };
      if (!video.seeking) {
        if (video.paused || video.ended) { resetDisplayClock(); render(frame, 0); }
        else {
          waitingForVideo = false;
          displayAnchor = { frame, mediaTime: metadata.mediaTime, expectedDisplayTime: Number.isFinite(metadata.expectedDisplayTime) ? metadata.expectedDisplayTime : now, playbackRate: video.playbackRate };
          // Commit the packet with the native video's newly presented frame.
          // rAF below paints only the intermediate overlay positions.
          render(frame, 0);
        }
      }
      scheduleVideoFrame(generation);
    });
  }
  // Only this loop schedules animation frames. rVFC is a decoded-frame clock,
  // so overlays update at display cadence without synthesizing any image frames.
  const onAnimation = (now) => {
    if (state.loaded && sourceMatches() && !video.paused && !video.ended && !video.seeking) {
      const frame = hasVideoFrameClock ? state.frame : clampFrame(Math.floor(video.currentTime * fps + 0.00001));
      const fraction = phaseAt(now, frame);
      if ((!hasVideoFrameClock && frame !== state.frame) || (fraction > 0 && fraction !== state.interpolationPhase)) {
        render(frame, fraction);
      }
    }
    requestAnimationFrame(onAnimation);
  };
  requestAnimationFrame(onAnimation);
  const resize = () => { stageRect = null; videoPaintKey = ''; render(); };
  if ('ResizeObserver' in window) new ResizeObserver(resize).observe(stage);
  window.addEventListener('resize', resize);
  window.demoAPI = { seekFrame, setView, setSelected, setInterpolation, setGapFill, sampleDisplayRows, sampleGapRows, switchSequence, allCatalog: catalog, getState };
  window.demoState = state;
  drawPlaceholder(); updateFrameUI();
  switchSequence(activeSequence);
})();
