// CPU-only classifier regression fixtures: no builds or GPU execution.
// Uses only rejection_pattern() from the committed shell script; never runs
// mutation .command files or modifies the immutable capture.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync, execFileSync} from 'node:child_process';

// Run: node SCRIPT [evidence-input-root] [repo-root]. Writes JSON to stdout only.
const repoRoot = path.resolve(process.argv[3] ?? execFileSync('git', ['rev-parse', '--show-toplevel'], {encoding: 'utf8'}).trim());
const inputRoot = path.resolve(process.argv[2] ?? path.join(repoRoot, 'evidence/s04/2026-10-30'));
const relative = file => path.relative(repoRoot, file) || '.';
const classificationScript = path.join(repoRoot, 'evidence/s04/2026-10-30/mutation-check.sh');
const mutations = path.join(inputRoot, 'capture/failed-attempt2/mutations');
const script = fs.readFileSync(classificationScript, 'utf8');
const functionSource = script.match(/^rejection_pattern\(\) \{\n[\s\S]*?^\}/m)?.[0];
if (!functionSource) throw new Error('rejection_pattern function not found');
const names = [...script.matchAll(/^  '([a-z0-9_]+)\|/gm)].map(match => match[1]);
const patterns = new Map(names.map(name => {
  const result = spawnSync('bash', ['-c', `${functionSource}\nrejection_pattern "$1"`, 'fixture', name], {encoding: 'utf8'});
  if (result.status !== 0 || !result.stdout) throw new Error(`${name}: no current rejection pattern`);
  return [name, result.stdout];
}));
function classify(name, code, log) {
  if (code === 0) return 'INSENSITIVE';
  if (code !== 1) return 'INVALID_RUNTIME';
  const result = spawnSync('grep', ['-Eq', '--', patterns.get(name)], {input: log, encoding: 'utf8'});
  if (result.status !== 0 && result.status !== 1) throw new Error(`${name}: grep failed: ${result.stderr}`);
  return result.status === 0 ? 'DETECTED' : 'INVALID_RUNTIME';
}
const resultText = fs.readFileSync(path.join(mutations, 'results.tsv'), 'utf8');
const captureRows = resultText.split('\n').slice(1).filter(line => line.split('\t').length === 4).map(line => {
  const [name, check, exit, capturedResult] = line.split('\t');
  const logPath = path.join(mutations, `${name}.log`);
  const log = fs.existsSync(logPath) ? fs.readFileSync(logPath, 'utf8') : '';
  return {name, check, exit, capturedResult, logFile: relative(logPath),
    logSha256: crypto.createHash('sha256').update(log).digest('hex'),
    currentClassification: classify(name, Number(exit), log),
    expectedDiagnostic: patterns.get(name),
    finalDiagnostic: log.trimEnd().split('\n').at(-1) ?? '',
    note: 'Captured result is preserved; currentClassification is a classifier regression check only.'};
});

const tests = [];
function fixture(label, name, code, log, expected, provenance) {
  const actual = classify(name, code, log);
  tests.push({label, name, exit: code, expected, actual, ok: actual === expected, provenance});
}
// Source-derived positive examples cover the two new CPU mutants even before
// their actual logs arrive; they are explicitly not GPU/runtime evidence.
const sourceExamples = [
  ['ref_sigma_zero_for_dot_products', 'head_reference_tests: FP32 CPU head inside the policy: cls_p3\n'],
  ['converter_level_swap', 'head_weights_tests: bundle level of p4.input_proj.weight is the checkpoint index of input_proj.2.weight\n'],
];
for (const [name, log] of sourceExamples) fixture('source-derived-positive', name, 1, log, 'DETECTED', 'Literal current assertion and catch prefix; synthetic fixture, not runtime evidence.');
const tf32Log = fs.readFileSync(path.join(mutations, 'tf32_enabled.log'), 'utf8');
fixture('actual-tf32-full-build-prefix', 'tf32_enabled', 1, tf32Log, 'DETECTED', 'Immutable capture2 raw log; original capture classification remains INVALID_RUNTIME.');
fixture('unrelated-tactic-guard', 'tf32_enabled', 1,
  'head_trt: head engine build: tactics outside the FP32 direct-summation model: sm80_xmma_gemm_f16f16_f16f32\n',
  'INVALID_RUNTIME', 'Synthetic non-TF32 guard rejection.');
fixture('unwrapped-tf32-message', 'tf32_enabled', 1,
  'head_trt: tactics outside the FP32 direct-summation model: sm80_xmma_tf32f32\n',
  'INVALID_RUNTIME', 'Current source always adds head engine build prefix; do not accept guessed alternate wrapping.');
for (const row of captureRows) {
  const logPath = path.join(mutations, `${row.name}.log`);
  if (!fs.existsSync(logPath)) continue;
  const log = fs.readFileSync(logPath, 'utf8');
  if (row.exit === '1') fixture('captured-target-rejection', row.name, 1, log, 'DETECTED', 'Saved failed-attempt2 raw diagnostic; original attempt verdict is unchanged.');
  for (const code of [0, 2, 99, 124, 137, 139]) fixture('wrong-exit-even-with-matching-diagnostic', row.name, code, log,
    code === 0 ? 'INSENSITIVE' : 'INVALID_RUNTIME', 'Synthetic exit perturbation; no GPU execution.');
}
for (const name of names) {
  for (const log of ['head_trt: cannot open input file\n', 'head_trt_tests: CUDA error: unknown error\n',
    'head_reference_tests: cannot open input file\n', 'head_weights_tests: cannot write standard output\n',
    'timeout: failed to run command\n']) fixture('unrelated-runtime-error', name, 1, log, 'INVALID_RUNTIME', 'Synthetic I/O/CUDA/tool failure.');
}
const summary = {
  classificationScript: relative(classificationScript),
  inputRoot: relative(inputRoot),
  classificationScriptSha256: crypto.createHash('sha256').update(script).digest('hex'),
  mutationDirectory: relative(mutations),
  captureCoverage: {expected: names.length, completed: captureRows.length,
    absent: names.filter(name => !captureRows.some(row => row.name === name)),
    allCompletedDiagnosticsRecognized: captureRows.every(row => row.exit === '1' && row.currentClassification === 'DETECTED'),
    note: 'All 26 saved failed-attempt2 rows are required. This checks the current classifier and does not change the original failed capture verdict.'},
  captureRows,
  fixtures: {count: tests.length, failures: tests.filter(test => !test.ok), allPassed: tests.every(test => test.ok), tests},
};
console.log(JSON.stringify(summary, null, 2));
if (!summary.fixtures.allPassed || !summary.captureCoverage.allCompletedDiagnosticsRecognized || captureRows.length !== names.length || names.length !== 26) process.exitCode = 1;
