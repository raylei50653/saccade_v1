#!/usr/bin/env python3
"""Verify archived bounded detection-session evidence, without a CUDA runtime."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root):
    cap = root / 'capture'
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert sha((root / name).read_bytes()) == digest, name
    manifest = (cap / 'source-files.sha256').read_bytes()
    source = dict(line.split('=', 1) for line in (cap / 'source.txt').read_text().splitlines())
    assert source['allow_dirty'] == 'true'
    assert sha(manifest) == source['source_digest']
    expected = dict((line[66:], line[:64]) for line in manifest.decode().splitlines())
    with tarfile.open(cap / 'source-build-inputs.tar.gz', 'r:gz') as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        assert set(members) == set(expected)
        for name, digest in expected.items():
            assert sha(archive.extractfile(members[name]).read()) == digest, name
    for before, after in [('binaries.sha256', 'binaries-after.sha256'),
                          ('artifacts.sha256', 'artifacts-after.sha256')]:
        assert (cap / before).read_bytes() == (cap / after).read_bytes(), before
    controls = dict(memcheck='Invalid __global__ write', initcheck='Uninitialized __global__ memory read',
                    racecheck='Error: Race reported between', synccheck='Barrier error detected')
    exits = {}
    for file in cap.glob('*.exit'):
        code = int(file.read_text())
        exits[file.stem] = code
        if file.stem.startswith('control-'):
            assert code == 99, file.name
        elif file.stem == 'initcheck-diagnostic':
            assert code in (0, 99), file.name
        else:
            assert code == 0, file.name
    assert 'exit=0\n' in (cap / 'capture-status.txt').read_text()
    for tool, diagnostic in controls.items():
        text = (cap / f'kernels-{tool}.log').read_text()
        assert 'PASS checks=78 ' in text, tool
        summary = ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck'
                   else '========= ERROR SUMMARY: 0 errors')
        assert text.splitlines().count(summary) == 1, tool
        assert diagnostic in (cap / f'control-{tool}.log').read_text(), tool
        assert exits[f'control-{tool}'] == 99
    normal = (cap / 'session.log').read_text()
    assert 'detection_session=PASS checks=4293 mode=normal' in normal
    assert 'callback_gates=PASS' in normal
    assert len(re.findall(r'^independent_input=', normal, re.M)) == 7
    memcheck = (cap / 'memcheck.log').read_text()
    assert re.search(r'^detection_session=PASS checks=\d+ mode=sanitizer$', memcheck, re.M)
    assert memcheck.splitlines().count('========= ERROR SUMMARY: 0 errors') == 1
    assert 'callback_gates=UNEXECUTED_SANITIZER' in memcheck
    if 'initcheck-diagnostic' not in exits:
        assert (cap / 'initcheck-classification.txt').read_text().startswith('UNEXECUTED')
        initcheck = 'UNEXECUTED in final capture; prior instrumented attempt FAIL/unresolved'
    else:
        diagnostic = (cap / 'initcheck-diagnostic.log').read_text()
        assert re.search(r'^detection_session=PASS checks=\d+ mode=sanitizer$', diagnostic, re.M)
        if exits['initcheck-diagnostic'] == 99:
            sites = re.findall(r'^=========     at (.+)$', diagnostic, re.M)
            reads = re.findall(r'^========= Uninitialized __global__ memory read of size (\d+) bytes$', diagnostic, re.M)
            assert sites and len(sites) == len(reads)
            assert all(site.startswith('void nvjpeg::rgba2rgb<') for site in sites), sites
            assert all(size == '3' for size in reads), reads
            initcheck = 'FAIL diagnostic: printed sites nvjpeg::rgba2rgb, size 3; unprinted sites not attributed'
        else:
            assert diagnostic.splitlines().count('========= ERROR SUMMARY: 0 errors') == 1
            initcheck = 'PASS in bounded run'
    failed = root / 'failed-initcheck-attempt/capture'
    assert 'exit=1\n' in (failed / 'capture-status.txt').read_text()
    prior = (failed / 'initcheck-diagnostic.log').read_text()
    assert 'detection_session=FAIL checks=67' in prior
    assert 'the launch timed out and was terminated' in prior
    assert 'ERROR SUMMARY: 12704 errors' in prior
    assert 'ERROR SUMMARY: 12604 errors were not printed' in prior
    prior_sites = re.findall(r'^=========     at (.+)$', prior, re.M)
    assert len(prior_sites) == 100
    assert all(site.startswith('void nvjpeg::rgba2rgb<') for site in prior_sites)
    assert not re.search(r'^independent_input=', prior, re.M)
    prior_manifest = dict((line[66:], line[:64]) for line in (failed / 'source-files.sha256').read_text().splitlines())
    assert set(expected) == set(prior_manifest)
    changed = [name for name in expected if expected[name] != prior_manifest.get(name)]
    assert sorted(changed) == ['docs/DETECTION_SESSION.md', 'scripts/run_detection_session_acceptance.sh']
    assert re.search(r'100% tests passed(?:, 0 tests failed)? out of 13', (cap / 'cpu-ctest.log').read_text())
    assert 'PASS configurations=18 mutations=6' in (cap / 'LastTest.log').read_text()
    return dict(archive_checksums='PASS', source_files=len(expected), source_digest=source['source_digest'],
                source_archive='PASS', source_dirty=True, binary_hashes='unchanged', artifact_hashes='unchanged',
                command_exits=exits, normal_checks=4293, session_kernel_checks=78,
                initcheck=initcheck, prior_attempt='FAIL retained; root cause unresolved', changed_since_prior=changed, scope='bounded detection session; no full tracking/P01 approval')


if __name__ == '__main__':
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent), indent=2))
