from pathlib import Path
import hashlib,json,tarfile,re,urllib.parse
root=Path.cwd();out=root/'evidence/hr-sanitizer-memory/2026-11-05';base=root/'build/racecheck-memory-20261105'
def index():
 rows=[]
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name!='ARTIFACTS.json':rows.append({'path':str(p.relative_to(out)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
 (out/'ARTIFACTS.json').write_text(json.dumps({'schema':'racecheck-memory-artifacts/1','artifacts':rows},indent=2)+'\n');return len(rows)
index();checks=[]
for name in ['probes.tar.gz','allocator-traces.tar.gz','observer-and-probe-binaries.tar.gz']:
 count=0
 with tarfile.open(out/'raw'/name,'r:gz') as a:
  for m in a.getmembers():
   if m.isfile():assert hashlib.sha256(a.extractfile(m).read()).digest()==hashlib.sha256((base/m.name).read_bytes()).digest(),m.name;count+=1
 checks.append({'check':'archive_members_match_originals','archive':name,'files':count,'pass':True})
links=0
for p in [out/'REPORT.md',root/'docs/STATUS.md',root/'README.md']:
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if re.match(r'(https?://|mailto:|#)',target):continue
  t=urllib.parse.unquote(target.split('#')[0]);assert (p.parent/t).exists(),(p,target);links+=1
checks.append({'check':'local_markdown_links_exist','links':links,'pass':True})
c=json.loads((out/'analysis/source-contract.json').read_text())
for path,info in c['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==info['sha256'],path
checks.append({'check':'runtime_probe_observer_engine_library_hashes_stable','files':len(c['files']),'pass':True})
summary={'scope':'Diagnostic allocation-source investigation only','runtime_revision':c['runtime_revision'],'allocation_owner':'libsanitizer-collection.so+0x5b9ef9','owner_context':'sanitizer internal worker threads','pointwise_tracked_large_heap_share':0.9699,'actual_hr_custom_one_frame_share':0.9809,'quantities':'intercepted requestedbytes>=64KiB;notRSS;observerlimitsapply','native_pointwise':'OOMat6GiB','forced_blocking_pointwise':'constructorcomplete0hazards;peak1,162,190,848B','global_probe_increment_per_extra_256thread_block':'512KiB cumulative toolrequestedbytes','shared_probe_increment_per_extra_256thread_block':'704KiB cumulative toolrequestedbytes','leak_classification':'notestablished;completedcontrolsreturntocommonfinalwatermark','private_payload_type':'unknown','full_native_hr_racecheck':'OPEN/INCOMPLETE, prioracceptancenotchanged','gpu_probe_cases':25}
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'analysis/final-verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps({'checks':len(checks),'artifacts':index(),'raw_members':sum(x.get('files',0) for x in checks if 'archive' in x),'links':links},indent=2))
