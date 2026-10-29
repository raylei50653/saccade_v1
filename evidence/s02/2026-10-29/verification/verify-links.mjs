// Check local Markdown link targets, including #anchors, for the documents this
// round delivers. Anchors use GitHub's heading slug rule (lowercase; keep
// letters, marks, numbers, '-', '_' and spaces; spaces become '-'). No network.
import fs from 'node:fs';
import path from 'node:path';
const repo = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../../..');
const files = [
  'README.md', 'docs/SCAN_CUDA.md', 'docs/SCAN_CONTRACT.md', 'docs/MODEL_FEASIBILITY.md',
  'docs/CUDA_SYNTHETIC.md', 'docs/ASYNC_CONTROL.md',
  'evidence/s02/2026-10-29/REPORT.md',
];
const slug = (heading) => heading.trim().toLowerCase().replace(/[^\p{L}\p{M}\p{N}\-_ ]/gu, '').replace(/ /g, '-');
const anchors = (file) => new Set(fs.readFileSync(file, 'utf8').split('\n')
  .filter((line) => /^#{1,6} /.test(line)).map((line) => slug(line.replace(/^#+ /, ''))));
let links = 0, withAnchor = 0;
for (const file of files) {
  const absolute = path.join(repo, file);
  for (const match of fs.readFileSync(absolute, 'utf8').matchAll(/\]\(([^)\s]+)\)/g)) {
    const target = match[1];
    if (/^[a-z]+:/i.test(target)) continue;
    const [local, anchor] = target.split('#');
    const resolved = local ? path.resolve(path.dirname(absolute), decodeURIComponent(local)) : absolute;
    if (!fs.existsSync(resolved)) throw new Error(`${file}: missing ${target}`);
    links++;
    if (anchor !== undefined && resolved.endsWith('.md')) {
      if (!anchors(resolved).has(decodeURIComponent(anchor))) throw new Error(`${file}: missing anchor ${target}`);
      withAnchor++;
    }
  }
}
console.log(`${files.length} Markdown documents: ${links} local targets exist, ${withAnchor} Markdown anchors resolve`);
