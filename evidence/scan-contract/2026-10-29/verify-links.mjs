// Check local file targets in the delivered Markdown; no network or models.
import fs from 'node:fs';
import path from 'node:path';
const root = path.dirname(new URL(import.meta.url).pathname);
const repo = path.resolve(root, '../../..');
const files = [
  'README.md', 'docs/SCAN_CONTRACT.md', 'docs/MODEL_FEASIBILITY.md',
  'docs/MODEL_ARTIFACT_READINESS.md',
  ...['REPORT.md', 'design/FINDINGS.md', 'upstream/FINDINGS.md',
      'mambapsa/FINDINGS.md', 'repository/README.md'].map(f => path.relative(repo, path.join(root, f)))
];
let count = 0;
for (const file of files) {
  const absolute = path.join(repo, file);
  const text = fs.readFileSync(absolute, 'utf8');
  for (const match of text.matchAll(/\[[^\]\n]*\]\(([^)\n]+)\)/g)) {
    const target = match[1].replace(/^<|>$/g, '');
    if (/^[a-z]+:/i.test(target)) continue;
    const local = decodeURIComponent(target.split('#')[0]);
    if (!local) continue;
    const resolved = path.resolve(path.dirname(absolute), local);
    if (!fs.existsSync(resolved)) throw new Error(`${file}: missing ${target}`);
    count++;
  }
}
console.log(`${files.length} Markdown documents: ${count} local file targets exist (anchors/network excluded)`);
