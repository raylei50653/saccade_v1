// Independent CPU-only fixtures for capture.sh's actual run_default_memcheck.
// A mock sanitizer emits immutable/sample logs; no GPU command is executed.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync, execFileSync} from 'node:child_process';
import os from 'node:os';

// Run: node SCRIPT [evidence-input-root] [repo-root]. JSON goes to stdout.
const repoRoot = path.resolve(process.argv[3] ?? execFileSync('git', ['rev-parse', '--show-toplevel'], {encoding: 'utf8'}).trim());
const inputRoot = path.resolve(process.argv[2] ?? path.join(repoRoot, 'evidence/s04/2026-10-30'));
const relative = file => path.relative(repoRoot, file) || '.';
const scriptPath = path.join(repoRoot, 'evidence/s04/2026-10-30/capture.sh');
const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), 's04-memcheck-fixtures-'));
process.on('exit', () => fs.rmSync(outputDir, {recursive: true, force: true}));
const script = fs.readFileSync(scriptPath, 'utf8');
const helper = script.match(/^run_default_memcheck\(\) \{\n[\s\S]*?^\}/m)?.[0];
if (!helper) throw new Error('run_default_memcheck helper missing');
fs.mkdirSync(outputDir, {recursive: true});
const mock = path.join(outputDir, 'mock-sanitizer.sh');
fs.writeFileSync(mock, '#!/usr/bin/env bash\nset -euo pipefail\ncat -- "$FIXTURE_INPUT"\nexit "$FIXTURE_EXIT"\n', {mode: 0o755});
const success = suite => `^${suite}_trt_tests: [1-9][0-9]* checks passed$`;
const sources = [
  {name: 'head-old-known-note', suite: 'head', code: 99,
    file: path.join(inputRoot, 'capture/failed-attempt/sanitizer-head-memcheck.log')},
  {name: 'scan3-known-note', suite: 'scan', code: 99,
    file: path.join(inputRoot, 'capture/failed-attempt3/sanitizer-scan-trt-memcheck.log')},
  {name: 'head-clean-output', suite: 'head', code: 0,
    file: path.join(inputRoot, 'capture/failed-attempt2/sanitizer-head-memcheck.log')},
  {name: 'scan-clean-output', suite: 'scan', code: 0,
    file: path.join(inputRoot, 'capture/failed-attempt2/sanitizer-scan-trt-memcheck.log')},
].map(source => ({...source, log: fs.readFileSync(source.file, 'utf8')}));
const summary = /^========= ERROR SUMMARY: \d+ errors?\n/gm;
const known = /^========= CUDA API Warning: Selective device code recompilation in progress\.\n/gm;
const successfulLine = /^(head|scan)_trt_tests: [1-9][0-9]* checks passed\n/gm;
const fixtures = [];
const add = (name, suite, code, log, expected, provenance) => fixtures.push({name, suite, code, log, expected, provenance});
for (const source of sources) {
  add(source.name, source.suite, source.code, source.log, 0, relative(source.file));
  for (const [suffix, prefix] of [
    ['extra-api-error', '========= Program hit cudaErrorInvalidValue (error 1) due to invalid argument on CUDA API call to cudaMemcpy.\n'],
    ['extra-memory-error', '========= Invalid __global__ write of size 4 bytes\n'],
    ['extra-api-warning', '========= CUDA API Warning: unrelated driver warning.\n'],
  ]) add(`${source.name}-${suffix}`, source.suite, source.code, prefix + source.log, 1, 'Synthetic additional sanitizer header; same original summary isolates header validation.');
  add(`${source.name}-missing-success`, source.suite, source.code, source.log.replace(successfulLine, ''), 1, 'Synthetic missing success line.');
  const passLine = source.log.match(successfulLine)?.[0];
  if (!passLine) throw new Error(`${source.name}: source log has no expected success line`);
  add(`${source.name}-duplicate-success`, source.suite, source.code, source.log + passLine, 1, 'Synthetic duplicate success line.');
  add(`${source.name}-zero-checks`, source.suite, source.code,
    source.log.replace(successfulLine, `${source.suite}_trt_tests: 0 checks passed\n`), 1, 'Synthetic zero-check suite success.');
  add(`${source.name}-wrong-suite-success`, source.suite === 'head' ? 'scan' : 'head', source.code, source.log, 1, 'Use another suite success regex against the unchanged raw log.');
  add(`${source.name}-missing-summary`, source.suite, source.code, source.log.replace(summary, ''), 1, 'Synthetic missing summary.');
  add(`${source.name}-duplicate-summary`, source.suite, source.code,
    source.log + (source.log.match(summary)?.[0] ?? ''), 1, 'Synthetic duplicate summary.');
  add(`${source.name}-summary-mismatch`, source.suite, source.code,
    source.log.replace(summary, '========= ERROR SUMMARY: 2 errors\n'), 1, 'Synthetic summary count differs from known-note count.');
  for (const code of [2, 99, 124, 137, 139]) {
    if (code === source.code) continue;
    add(`${source.name}-wrong-exit-${code}`, source.suite, code, source.log, 1, 'Synthetic exit perturbation, including timeout and signal/crash exits.');
  }
}
for (const source of sources.filter(source => source.code === 99)) {
  add(`${source.name}-warning-with-zero-summary-and-exit`, source.suite, 0,
    source.log.replace(summary, '========= ERROR SUMMARY: 0 errors\n'), 1, 'Known warning present while summary and exit falsely claim no errors.');
  add(`${source.name}-plain-application-text-is-not-tool-note`, source.suite, 99,
    source.log.replace(known, '') + 'application: CUDA API Warning: Selective device code recompilation in progress\n',
    1, 'Unanchored application text must not be counted as a known sanitizer header.');
  add(`${source.name}-two-known-notes`, source.suite, 99,
    source.log.replace(summary, '========= CUDA API Warning: Selective device code recompilation in progress.\n========= ERROR SUMMARY: 2 errors\n'),
    0, 'Synthetic two known headers, both accounted for by the summary.');
}

const driver = `set -euo pipefail\nrun() { local name=$1; shift; "$@"; }\n${helper}\nsanitizer=$1\nrun_default_memcheck fixture "$2" "$3" ignored-suite-command\n`;
const results = fixtures.map((fixture, index) => {
  const stem = `${String(index + 1).padStart(3, '0')}-${fixture.name}`;
  const input = path.join(outputDir, `${stem}.input.log`);
  const raw = path.join(outputDir, `${stem}.raw.log`);
  fs.writeFileSync(input, fixture.log);
  const result = spawnSync('bash', ['-c', driver, 'fixture', mock, raw, success(fixture.suite)], {
    encoding: 'utf8', env: {...process.env, FIXTURE_INPUT: input, FIXTURE_EXIT: String(fixture.code)},
  });
  fs.writeFileSync(path.join(outputDir, `${stem}.classification.log`), result.stdout + result.stderr);
  const rawPreserved = fs.existsSync(raw) && fs.readFileSync(raw).equals(Buffer.from(fixture.log));
  return {name: fixture.name, suite: fixture.suite, rawToolExit: fixture.code, expectedClassifierExit: fixture.expected,
    classifierExit: result.status, signal: result.signal, rawPreserved,
    ok: result.status === fixture.expected && rawPreserved, provenance: fixture.provenance,
    classificationTail: (result.stdout + result.stderr).trimEnd().split('\n').slice(-3)};
});
const report = {captureScript: relative(scriptPath), inputRoot: relative(inputRoot), captureScriptSha256: crypto.createHash('sha256').update(script).digest('hex'),
  helperSha256: crypto.createHash('sha256').update(helper).digest('hex'),
  inputLogs: sources.map(source => ({file: relative(source.file), sha256: crypto.createHash('sha256').update(source.log).digest('hex')})),
  note: 'CPU-only classification fixtures; raw capture outcomes remain unchanged and no GPU acceptance is asserted.',
  fixtureCount: results.length, allPassed: results.every(result => result.ok), failures: results.filter(result => !result.ok), results};
console.log(JSON.stringify(report, null, 2));
if (!report.allPassed) process.exitCode = 1;
