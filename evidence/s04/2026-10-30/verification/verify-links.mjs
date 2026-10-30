// Check local Markdown targets and anchors in the S04 delivery documents.
// Reuses S02's heading-slug rule: lowercase; retain letters, marks, numbers,
// '-', '_' and spaces; replace spaces with '-'. No network requests.
import fs from 'node:fs';
import path from 'node:path';
const repo = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../../..');
const files = [
  'README.md', 'docs/HEAD_TRT.md', 'docs/HEAD_POLICY.md', 'docs/MODEL_MANIFEST.md', 'docs/SCAN_TRT.md',
  'docs/MODEL_ARTIFACT_READINESS.md', 'docs/MODEL_FEASIBILITY.md', 'docs/SCAN_CONTRACT.md', 'fixtures/model-manifest/README.md',
  'evidence/s04/2026-10-30/REPORT.md',
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
