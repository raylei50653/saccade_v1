#!/usr/bin/env python3
"""Lossless small-artifact archive; binaries stay in ignored capture directories."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists():
        raise RuntimeError('refusing to overwrite archive')
    files = []
    for item in sorted(source.rglob('*')):
        if not item.is_file() or item.name == 'annotation':
            continue
        relative = item.relative_to(source)
        # Tool logs are losslessly compressed; all commands/exits and outputs
        # retain their original bytes. Do not execute imported binaries.
        data = item.read_bytes()
        compressed = item.name == 'tool.log'
        target_relative = Path(str(relative) + '.gz') if compressed else relative
        target = out / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(data, mtime=0) if compressed else data
        target.write_bytes(encoded)
        files.append({'source_relative': str(relative),
                      'archive_relative': str(target_relative),
                      'compression': 'gzip' if compressed else None,
                      'original_bytes': len(data), 'original_sha256': digest(data),
                      'archive_sha256': digest(encoded)})
    record = {'source': str(source), 'files': files,
              'excluded_binary': 'annotation',
              'note': 'Exact byte archive; tool.log gzip round-trip checked by verifier.'}
    (out / 'ARCHIVE.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'archived {len(files)} files from {source} to {out}')


if __name__ == '__main__':
    main()
