#!/usr/bin/env python3
"""Read-only receipt/packet/data verification; no model or GT access."""
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

EVIDENCE=Path(__file__).resolve().parent
ROOT=EVIDENCE.parents[2]
ASSETS=ROOT/'docs/demos/report1/assets'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    checks=0
    for line in (EVIDENCE/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        assert sha(EVIDENCE/name)==expected,name
        checks+=1
    summary=json.loads((EVIDENCE/'summary.json').read_text())
    assert summary['status']=='PASS_COMPLETE_SOURCE_CAPTURE' and summary['metrics'] is None
    assert summary['source_inputs_binaries_unchanged']
    for kind in ['source','inputs','binaries']:
        before=json.loads((EVIDENCE/(kind+'-before.json')).read_text())
        after=json.loads((EVIDENCE/(kind+'-after.json')).read_text())
        assert before==after,kind
    raw=EVIDENCE/'raw/run1'
    payloads={}
    for name,expected in summary['outputs'].items():
        payload=gzip.decompress((raw/(name+'.gz')).read_bytes()) if name in ['packets.bin','motion.scmo'] else (raw/name).read_bytes()
        assert len(payload)==expected['bytes'] and hashlib.sha256(payload).hexdigest()==expected['sha256'],name
        payloads[name]=payload
    log=(EVIDENCE/'run1.log').read_text()
    assert 'source_commit='+summary['contract']['runtime_base_commit']+'\n' in log
    assert 'source_dirty=false\n' in log and 'source_dirty=true\n' not in log
    spec=importlib.util.spec_from_file_location('packet_export',ROOT/'docs/demos/report1/methods/build_data.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    frames,stats=common.packets(payloads['packets.bin'],1050)
    common.motions(payloads['motion.scmo'],frames)
    assert stats['rows']==stats['flags_1']==43003
    for f in frames:
        assert len({r[0] for r in f['rows']})==len(f['rows'])
        assert len({r[1] for r in f['rows']})==len(f['rows'])
        assert all(0<r[0]<2**53 and 0<r[1]<2**53 for r in f['rows'])
    exported=json.loads((ASSETS/'data-04.json').read_text())
    assert exported['frames']==frames and exported['metrics'] is None
    assert exported['meta']['sequence']=='MOT17-04-SDP' and exported['meta']['frameCount']==1050
    js=(ASSETS/'data-04.js').read_text()
    marker='"MOT17-04-SDP":'
    value,consumed=json.JSONDecoder().raw_decode(js.split(marker,1)[1])
    assert value==exported
    assert js.split(marker,1)[1][consumed:].strip()=='});'
    provenance=json.loads((ASSETS/'provenance-04.json').read_text())
    for name,expected in provenance['artifacts'].items():
        p=ASSETS/name
        assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'],name
    assert provenance['source_inputs']['capture_summary_sha256']==sha(EVIDENCE/'summary.json')
    stream,=provenance['ffprobe']['streams']
    assert int(stream['nb_read_frames'])==1050 and float(provenance['ffprobe']['format']['duration'])==35
    assert provenance['ffprobe']['faststart_verified']
    rows=list(csv.reader(payloads['mot.txt'].decode().splitlines()))
    assert len(rows)==45056 and sum(float(r[6])==-1 for r in rows)==2053
    print(f'PASS: {checks} evidence checksums; 1050 canonical packets/motion contexts; 43003 exact JSON rows; full outputs/custody; 1050-frame source video; no benchmark metrics')


if __name__=='__main__':main()
