#!/usr/bin/env python3
"""Offline verification of actual converter arguments and submitted dependencies."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXPECTED_RGB='761809bdcad84f79c1b01f0d2369bb4cf52f280cb2dff92be87c3ef9f11c7162'
EXPECTED_LIBRARY='c8fb7a95fa58f20eeb58c21760ad4017845fd051c6a70672ff25736696404338'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_jsonl(path):
    text=gzip.decompress(path.read_bytes()).decode() if path.suffix=='.gz' else path.read_text()
    return [json.loads(s) for s in text.splitlines()]

def analyze(root):
    identity=json.loads((root/'identity.json').read_text())
    for path,before in identity['inputs_before'].items():
        assert identity['inputs_after'][path]==before,('source/library drift',path)
    assert EXPECTED_LIBRARY in identity['loaded_modules'].values()
    results=[]
    for name in ('default','no-pipeline','no-preallocate'):
        trace=root/f'{name}.jsonl'
        if not trace.exists():trace=trace.with_suffix('.jsonl.gz')
        rows=read_jsonl(trace)
        assert [r['seq'] for r in rows]==list(range(1,len(rows)+1))
        assert rows[0]['kind']=='tracer_ready' and rows[-1]['kind']=='tracer_end'
        assert len({r['tid'] for r in rows})==1,'this artifact assumes one observed callback thread'
        assert int((root/'capture'/f'{name}.exit').read_text())==0
        log=(root/'capture'/f'{name}.stdout').read_text()
        assert 'result=complete repeat_exact=1 rgb_equivalence=exact' in log
        assert 'rgb_differences_across_repeats=0 rgb_max_difference=0' in log
        assert (root/f'{name}.rgb').stat().st_size==2835
        assert sha(root/f'{name}.rgb')==EXPECTED_RGB
        calls=[r for r in rows if r.get('kind')=='api' and r.get('domain')=='driver']
        assert all(r.get('status',0)==0 for r in calls if r.get('site')=='exit')
        entries=[r for r in calls if r['site']=='enter']
        exits={(r['api'],r['correlation']):r for r in calls if r['site']=='exit'}
        for r in entries:
            exit=exits[(r['api'],r['correlation'])]
            assert exit['seq']>r['seq'] and exit['status']==0
        decodes=[r for r in rows if r['kind']=='decode_enter']
        assert len(decodes)==2
        summaries=[]
        for d in decodes:
            end=next(r for r in rows if r['kind']=='decode_exit' and r['decode']==d['decode'])
            subset=[r for r in entries if d['seq']<r['seq']<end['seq']]
            kernels=[r for r in subset if r.get('converter_launch')]
            assert len(kernels)==1 and end['converter_snapshots']==1 and end['status']==0
            k=kernels[0]
            assert k['api']=='cuLaunchKernel' and k['kernel_params_present'] and not k['extra_present']
            assert k['width']==35 and k['height']==27
            assert k['source']['pitches']==[512,0,0,0]
            assert k['destination']['pitches']==[35,35,35,0]
            src=int(k['source']['channels'][0],16)
            dst=[int(s,16) for s in k['destination']['channels'][:3]]
            assert dst==[dst[0],dst[0]+945,dst[0]+1890]
            assert all(not(src<=a<src+16384) for a in dst)
            surface=next(r for r in rows if r['kind']=='surface_identity' and r['decode']==d['decode'])
            assert surface['range_status']==surface['id_status']==0
            assert surface['source']==k['source'] and surface['converter_stream']==k['stream']
            base=int(surface['base'],16)
            assert base<=src and src+26*512+34*4+4<=base+surface['allocation_bytes']
            records=[r for r in subset if r['api']=='cuEventRecord' and r.get('stream')==k['stream'] and r['seq']>k['seq']]
            assert len(records)==1
            record=records[0]
            waits=[r for r in subset if r['api']=='cuStreamWaitEvent' and r.get('event')==record['event'] and r.get('stream')==d['caller_stream'] and r['seq']>record['seq']]
            assert len(waits)==1 and k['stream']!=d['caller_stream']
            wait=waits[0]
            assert exits[(k['api'],k['correlation'])]['seq']<record['seq']
            assert exits[(record['api'],record['correlation'])]['seq']<wait['seq']
            assert exits[(wait['api'],wait['correlation'])]['seq']<end['seq']
            before_query=[r for r in entries if end['seq']<r['seq']<surface['seq']]
            assert [r['api'] for r in before_query]==['cuMemGetAddressRange_v2','cuPointerGetAttribute']
            summaries.append({'decode':d['decode'],'tid':d['tid'],'source':k['source'],'destination':k['destination'],'surface_base':surface['base'],'allocation_bytes':surface['allocation_bytes'],'buffer_id':surface['buffer_id'],'caller_stream':d['caller_stream'],'converter_stream':k['stream'],'launch_seq':k['seq'],'event_record_seq':record['seq'],'event':record['event'],'caller_wait_seq':wait['seq'],'decode_exit_seq':end['seq'],'snapshot_seq':surface['seq'],'nvjpeg_stack_offsets':[f['offset'] for f in k['frames'] if 'libnvjpeg' in f['module']]})
        assert summaries[0]['buffer_id']==summaries[1]['buffer_id']
        results.append({'name':name,'records':len(rows),'rgb_sha256':sha(root/f'{name}.rgb'),'decodes':summaries,'driver_api_entries':dict(sorted(Counter(r['api'] for r in entries).items()))})
    return {'verified':True,'classification':'normal standalone diagnostic; no sanitizer; host callback order and submitted event dependency, not hardware fence proof','captures':results,'unproven':['hardware producer identity and completion fence','initcheck shadow implementation and metadata consumption','uninstrumented timing and all schedules','original application sanitizer acceptance']}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path,nargs='?',default=HERE/'capture-success');ap.add_argument('--output',type=Path)
    args=ap.parse_args();result=analyze(args.capture)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified':result['verified'],'captures':len(result['captures']),'decodes':sum(len(c['decodes']) for c in result['captures'])}))
