import fs from 'node:fs';
import path from 'node:path';

const capturePid = Number(process.argv[2]);
const logPath = path.resolve(process.argv[3]);
const thresholdKiB = 8 * 1024 * 1024;
const intervalMs = 5000;
const log = fs.openSync(logPath, 'a');
const emit = text => fs.writeSync(log, `${new Date().toISOString()}\t${text}\n`);
function readProc(pid) {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
    const tail = stat.slice(stat.lastIndexOf(')') + 2).trim().split(/\s+/);
    const status = fs.readFileSync(`/proc/${pid}/status`, 'utf8');
    const field = name => status.match(new RegExp(`^${name}:\\s*(.+)$`, 'm'))?.[1] ?? 'NA';
    const cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').filter(Boolean);
    return {pid, ppid: Number(field('PPid')), comm: field('Name').replace(/[\t\r\n]/g,' '),
      rss: field('VmRSS'), hwm: field('VmHWM'), swap: field('VmSwap'), start: tail[19],
      executable: cmdline[0] ? path.basename(cmdline[0]) : '', cmdline};
  } catch { return null; }
}
const initial = readProc(capturePid);
if (!initial || initial.cmdline.join(' ') !== 'bash evidence/s04/2026-10-30/capture.sh build-s04-evidence2') {
  emit(`ABORT\tcapture identity mismatch\tpid=${capturePid}`);
  throw new Error('capture identity mismatch');
}
const captureStart = initial.start;
function descendants() {
  const found = new Map();
  const pending = [capturePid];
  while (pending.length) {
    const pid = pending.shift();
    if (found.has(pid)) continue;
    const record = readProc(pid);
    if (!record) continue;
    found.set(pid, record);
    try {
      for (const tid of fs.readdirSync(`/proc/${pid}/task`)) {
        try {
          const children = fs.readFileSync(`/proc/${pid}/task/${tid}/children`, 'utf8').trim();
          if (children) pending.push(...children.split(/\s+/).map(Number));
        } catch {}
      }
    } catch {}
  }
  return found;
}
const sent = new Set();
let stopping = false;
let timer;
function close(reason) {
  if (stopping) return;
  stopping = true;
  if (timer) clearInterval(timer);
  emit(`MONITOR_END\t${reason}`);
  fs.fsyncSync(log);
  fs.closeSync(log);
}
function sample() {
  const root = readProc(capturePid);
  if (!root || root.start !== captureStart) {
    close('capture exited or PID identity changed');
    return;
  }
  const memory = fs.readFileSync('/proc/meminfo', 'utf8');
  const available = Number(memory.match(/^MemAvailable:\s*(\d+) kB$/m)?.[1] ?? NaN);
  const swapFree = Number(memory.match(/^SwapFree:\s*(\d+) kB$/m)?.[1] ?? NaN);
  emit(`MEM\tMemAvailable_KiB=${available}\tSwapFree_KiB=${swapFree}\tthreshold_KiB=${thresholdKiB}`);
  const tree = descendants();
  for (const p of tree.values()) emit(`PROC\tPID=${p.pid}\tPPID=${p.ppid}\tcomm=${p.comm}\tVmRSS=${p.rss}\tVmHWM=${p.hwm}\tVmSwap=${p.swap}\tstart_ticks=${p.start}`);
  if (Number.isFinite(available) && available < thresholdKiB) {
    const targets = [...tree.values()].filter(p => ['compute-sanitizer','head_trt_tests','scan_trt_tests'].includes(p.executable));
    emit(`LOW_MEMORY_STOP\tMemAvailable_KiB=${available}\ttargets=${targets.map(p=>p.pid).join(',') || 'none'}`);
    for (const p of targets) {
      const key = `${p.pid}:${p.start}`;
      if (sent.has(key)) continue;
      const current = readProc(p.pid);
      if (!current || current.start !== p.start || current.ppid !== p.ppid) continue;
      try {
        process.kill(p.pid, 'SIGTERM');
        sent.add(key);
        emit(`TERM_SENT\tPID=${p.pid}\tPPID=${p.ppid}\texecutable=${p.executable}\tMemAvailable_KiB=${available}`);
      } catch (error) { emit(`TERM_ERROR\tPID=${p.pid}\terror=${error.code}`); }
    }
  }
  fs.fsyncSync(log);
}
emit(`MONITOR_BEGIN\tmonitor_pid=${process.pid}\tcapture_pid=${capturePid}\tcapture_start_ticks=${captureStart}\tinterval_ms=${intervalMs}\tthreshold_KiB=${thresholdKiB}\taction=SIGTERM capture descendant compute-sanitizer/head_trt_tests/scan_trt_tests only`);
process.on('SIGINT', () => close('monitor SIGINT'));
process.on('SIGTERM', () => close('monitor SIGTERM'));
sample();
if (!stopping) timer = setInterval(sample, intervalMs);
