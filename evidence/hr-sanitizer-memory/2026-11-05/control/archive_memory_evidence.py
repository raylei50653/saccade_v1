from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
root=Path.cwd();base=root/'build/racecheck-memory-20261105';out=root/'evidence/hr-sanitizer-memory/2026-11-05';out.mkdir(parents=True);(out/'raw').mkdir();(out/'control').mkdir();(out/'analysis').mkdir()
cases=sorted(p for p in base.iterdir() if p.is_dir() and (p/'execution.json').exists());gpu=[p for p in cases if not p.name.startswith('runner-cpu')]
subprocess.run(['python3',str(base/'summarize_process_memory.py'),'--output',str(base/'process-memory-all25.json'),*map(str,gpu)],check=True)
with tarfile.open(out/'raw/probes.tar.gz','w:gz') as a:
 for d in cases:
  for p in sorted(d.rglob('*')):
   if p.is_file() and not p.is_symlink():a.add(p,arcname=p.relative_to(base))
with tarfile.open(out/'raw/allocator-traces.tar.gz','w:gz') as a:
 for d in ['traces','selftest-final','selftest-growth']:
  p=base/d
  if p.exists():a.add(p,arcname=d)
with tarfile.open(out/'raw/observer-and-probe-binaries.tar.gz','w:gz') as a:
 for name in ['alloc_trace.so','alloc_trace_growth.so','constructor_probe','block_memory_probe','scan_memory_probe','alloc_trace_selftest','alloc_trace_growth_selftest']:
  a.add(base/name,arcname=name)
for p in sorted(base.iterdir()):
 if not p.is_file():continue
 if p.suffix in ('.py','.c','.cpp','.cu','.sass','.log'):
  shutil.copy2(p,out/'control'/p.name)
 elif p.suffix=='.json':shutil.copy2(p,out/'analysis'/p.name)
for p in cases:
 d=out/'analysis/captures'/p.name;d.mkdir(parents=True)
 for name in ['execution.json','manifest.json','command.json']:
  if (p/name).exists():shutil.copy2(p/name,d/name)
checks=[]
with tarfile.open(out/'raw/probes.tar.gz','r:gz') as a:
 for m in a.getmembers():
  if m.isfile():assert hashlib.sha256(a.extractfile(m).read()).digest()==hashlib.sha256((base/m.name).read_bytes()).digest();checks.append(m.name)
(out/'analysis/archive-verification.json').write_text(json.dumps({'raw_probe_files_exact':len(checks),'gpu_probe_cases':len(gpu),'cpu_runner_cases':len(cases)-len(gpu),'all_originals_match':True},indent=2)+'\n')
print(json.dumps({'out':str(out),'gpu_cases':len(gpu),'raw_files_verified':len(checks)}))
