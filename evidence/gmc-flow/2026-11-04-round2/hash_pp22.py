from pathlib import Path
import hashlib
root=Path.cwd();b=root/'build/gmc-flow2';p=b/'pp22';e=root/'evidence/gmc-flow/2026-11-04-round2'
with (e/'inputs.sha256').open('a') as dst:
 for kind,suffix in (('scdr','scdr'),('scmo','scmo'),('flow','scfl')):
  for path in sorted((p/kind).glob('*.'+suffix)):
   h=hashlib.sha256()
   with path.open('rb') as src:
    for chunk in iter(lambda:src.read(8*1024*1024),b''):h.update(chunk)
   dst.write(f'{h.hexdigest()}  {path}\n')
 path=root/'scripts/eval_pp22.sh';dst.write(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n')
print('Hashed PP22 inputs')
