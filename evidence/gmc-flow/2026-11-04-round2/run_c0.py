from pathlib import Path
import subprocess, hashlib,shlex
root=Path.cwd(); out=root/'build/gmc-flow2'; old=root/'build/gmc-flow'
with (out/'logs/C0.log').open('w',buffering=1) as log:
 for arm in ('G0','G1','L2'):
  (out/'mot17'/arm).mkdir(parents=True,exist_ok=True)
  (out/'diagnostics/mot17'/arm).mkdir(parents=True,exist_ok=True)
  for n in ('02','04','05','09','10','11','13'):
   s=f'MOT17-{n}-SDP'; mot=out/'mot17'/arm/(s+'.txt')
   cmd=['timeout','600',str(out/'gmc_flow_offline'),'track',str(root.parent/'m4-formal/build-m4-evaluation/scdr'/(s+'.scdr')),str(root.parent/'m1-data'/(s+'.scmo')),str(old/'flow'/(s+'.scfl')),arm,str(mot),str(out/'diagnostics/mot17'/arm/(s+'.csv'))]
   log.write('$ '+shlex.join(cmd)+'\n');subprocess.run(cmd,stdout=log,stderr=log,check=True)
   reference=old/'arms'/arm/(s+'.txt')
   if mot.read_bytes()!=reference.read_bytes():
    log.write(f'FAIL {arm} {s}\n');raise SystemExit(1)
   log.write(f'PASS {arm} {s} sha256={hashlib.sha256(mot.read_bytes()).hexdigest()}\n')
 log.write('C0 PASS 21/21 byte-identical MOT files\n')
print('C0 PASS 21/21 byte-identical MOT files')
