#!/usr/bin/env python3
"""Verify saved evidence only. Does not execute a GPU program."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]

def manifests():
    checked={}
    for name in ('static','host-static','dynamic','contracts','annotation'):
        manifest=HERE/name/'SHA256SUMS'
        assert manifest.is_file(),manifest
        count=0
        for line in manifest.read_text().splitlines():
            digest,name=line.split('  ',1)
            path=manifest.parent/name
            assert path.is_file(),path
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,path
            count+=1
        checked[manifest.parent.name]=count
    return checked

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    records=[]
    commands=[
        [sys.executable,str(HERE/'static/verify.py'),str(HERE/'static/capture')],
        [sys.executable,str(HERE/'host-static/analyze_gdb.py')],
        [sys.executable,str(HERE/'host-static/analyze_inner.py')],
        [sys.executable,str(HERE/'host-static/analyze_driver_edge.py')],
        [sys.executable,str(HERE/'dynamic/analyze.py')],
        [sys.executable,str(HERE/'contracts/verify.py')],
        [sys.executable,str(HERE/'annotation/verify.py')],
    ]
    for argv in commands:
        p=subprocess.run(argv,cwd=REPO,capture_output=True,text=True,timeout=30)
        records.append({'argv':argv,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        assert p.returncode==0,(argv,p.stdout,p.stderr)
    result={'saved_evidence_verified':True,'gpu_executed':False,'subtree_manifest_files':manifests(),'commands':records,'classification':'artifact consistency and bounded reconstruction checks; original F01 initcheck FAIL unchanged'}
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='commands'},indent=2))

if __name__=='__main__':main()
