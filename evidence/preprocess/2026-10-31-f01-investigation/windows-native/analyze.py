#!/usr/bin/env python3
"""Offline Windows/WSL comparison. Does not execute Windows or GPU tools."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CAP = ROOT / 'capture'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(line):
    return dict(re.findall(r'([A-Za-z_]\w*)=([^\s]+)', line))


def analyze(name):
    command = json.loads((CAP/(name+'.command.json')).read_text(encoding='utf-8-sig'))
    status = json.loads((CAP/(name+'.status.json')).read_text(encoding='utf-8-sig'))
    code = int((CAP/(name+'.exit')).read_text().strip())
    stdout = (CAP/(name+'.stdout.log')).read_text()
    stderr = (CAP/(name+'.stderr.log')).read_text()
    text = stdout + '\n' + stderr
    output = ROOT/(name+'.bin')
    assert code == status['exit'] and not status['timed_out'], name
    summaries = [int(n) for n in re.findall(r'ERROR SUMMARY: (\d+) errors?\s*$', text, re.M)]
    api_failures = [line for line in text.splitlines() if re.match(r'(api|cleanup)=',line) and re.search(r'(nvjpeg|cuda)_status=[1-9]\d*',line)]
    assert not api_failures, (name,api_failures)
    identity = [fields(line) for line in text.splitlines() if line.startswith('nvjpeg=')]
    result = [fields(line) for line in text.splitlines() if line.startswith('result=')]
    assert len(identity) == 1 and len(result) == 1 and result[0]['result'] == 'complete', name
    expected = {
        'windows-rgb-normal': (0, []),
        'windows-rgb-memcheck': (0,[0]),
        'windows-rgb-initcheck': (99,[1890]),
        'windows-skip-uninit-memcheck': (0,[0]),
        'windows-skip-uninit-initcheck': (99,[2835]),
        'windows-skip-zero-initcheck': (0,[0]),
        'windows-rgb-no-preallocate-initcheck': (99,[1890]),
        'windows-rgb-no-pipeline-initcheck': (99,[1890]),
        'windows-rgbi-initcheck': (99,[1890]),
        'windows-y-initcheck': (99,[108]),
        'windows-hybrid-initcheck': (0,[0]),
    }
    assert (code,summaries) == expected[name], (name,code,summaries)
    assert result[0]['repeat_exact'] == '1', name
    if identity[0]['backend'] == 'hw' and identity[0]['format'] in ('rgb','rgbi'):
        assert result[0]['rgb_equivalence'] == 'exact'
        assert output.read_bytes() == (ROOT/'baseline.rgb').read_bytes(), name
    return {'name':name,'exit':code,'errors':summaries,'command':command,'status':status,
            'identity':identity[0],'result':result[0],'api_failures':api_failures,
            'printed_kernels':sorted(set(re.findall(r'^=========\s+at (.+)$',text,re.M))),
            'output_bytes':output.stat().st_size,'output_sha256':sha(output),
            'stdout_sha256':sha(CAP/(name+'.stdout.log')),
            'stderr_sha256':sha(CAP/(name+'.stderr.log'))}


def main():
    assert sha(ROOT/'repro.cu') == sha(ROOT.parent/'repro.cu')
    assert sha(ROOT/'baseline.rgb') == sha(ROOT.parent/'baseline.rgb')
    before = json.loads((CAP/'artifact-identity.json').read_text(encoding='utf-8-sig'))
    after = json.loads((CAP/'post-artifact-identity.json').read_text(encoding='utf-8-sig'))
    before_hashes = {item['path'].rsplit('\\',1)[-1]:item['sha256'] for item in before}
    for item in after:
        name = item['path'].rsplit('\\',1)[-1]
        if name in before_hashes:
            assert item['sha256'] == before_hashes[name], name
        if (ROOT/name).exists():
            assert sha(ROOT/name) == item['sha256'], name
    for command in ('compile','nvcc-version','sanitizer-version','gpu-identity','pe-dependencies'):
        assert (CAP/(command+'.exit')).read_text().strip() == '0', command
    assert '2026.3.0.0' in (CAP/'sanitizer-version.stdout.log').read_text()
    assert 'V13.4.92' in (CAP/'nvcc-version.stdout.log').read_text()
    cases = [analyze(path.name.removesuffix('.command.json')) for path in sorted(CAP.glob('windows-*.command.json'))]
    assert len(cases) == 11
    wsl = json.loads((ROOT.parent/'summary.json').read_text())
    wsl_baseline = next(item for item in wsl['commands'] if item['name']=='new-rgb-initcheck')
    windows_baseline = next(item for item in cases if item['name']=='windows-rgb-initcheck')
    assert windows_baseline['exit'] == wsl_baseline['exit'] == 99
    assert windows_baseline['errors'] == wsl_baseline['summary_errors'] == [1890]
    assert windows_baseline['output_sha256'] == wsl_baseline['output_sha256'] == sha(ROOT/'baseline.rgb')
    assert any('rgba2rgb' in k and '+0x170' in k for k in windows_baseline['printed_kernels'])
    summary = {'schema':'f01-native-windows-comparison/1',
               'scope':'Standalone native Windows diagnostics; unsuppressed initcheck failure preserved; WSL is not necessary to reproduce, no native Linux verdict.',
               'source_sha256':sha(ROOT/'repro.cu'),
               'exe_sha256':before_hashes['repro.exe'],
               'source_matches_prior_linux_harness':True,
               'before_after_artifacts_unchanged':True,
               'cases':cases,'wsl_baseline_comparison':{
                   'windows_exit':windows_baseline['exit'],'wsl_exit':wsl_baseline['exit'],
                   'windows_errors':windows_baseline['errors'],'wsl_errors':wsl_baseline['summary_errors'],
                   'rgb_output_sha256':windows_baseline['output_sha256'],
                   'same_error_kernel_offset_observed':True}}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'comparison':summary['wsl_baseline_comparison']},indent=2))


if __name__ == '__main__':
    main()
