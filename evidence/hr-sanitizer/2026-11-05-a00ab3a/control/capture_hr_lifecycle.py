#!/usr/bin/env python3
import argparse, fcntl, importlib.util, json, re, sys
from pathlib import Path
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('hr_validation',root/'run_hr_sanitizer_validation_nodump.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tools',nargs='+',choices=m.TOOLS,default=list(m.TOOLS));p.add_argument('--timeout',type=int,default=900);a=p.parse_args()
if a.timeout<1 or len(set(a.tools))!=len(a.tools):p.error('positive timeout and unique tools required')
a.source=root.parent/'hr-sanitizer-20261105-source';a.build=root/'cuda';a.memory_high='24G';a.memory_max='24G';out=a.output.resolve();out.mkdir()
c=m.Capture(a,out);engine=Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine');seq=Path('/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP');records=Path('/home/ray/developer/ai/saccade_v1-worktrees/hr-records-v2');binary=root/'harness/tracking_session_hr_tests';source=root/'harness/tracking_session_tests.cpp'
command=[str(binary),str(engine),str(seq.parent),str(records),'MOT17-10-SDP','--profile','hr-1280x768','--lifecycle']
def positive_summary(tool,text):
 if tool=='racecheck':
  rows=re.findall(r'^========= RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)$',text,re.M)
  return len(rows)==1 and int(rows[0][0])>0 and int(rows[0][1])+int(rows[0][2])>0
 rows=re.findall(r'^========= ERROR SUMMARY: (\d+) errors?$',text,re.M)
 return len(rows)==1 and int(rows[0])>0

def custody():
 return {'runtime':m.source_snapshot(a.source,a.build,engine,seq,30),'harness':{str(x):m.digest(x) for x in [binary,source,root/'harness/commands.json',records/'MOT17-10-SDP.scdr',records/'MOT17-10-SDP.scmo']},'runner':m.digest(__file__),'capture_runner':m.digest(root/'run_hr_sanitizer_validation_nodump.py'),'sanitizer':m.digest('/opt/cuda/bin/compute-sanitizer'),'target_wrapper':m.digest(c.wrapper)}
def complete(text):
 return len(re.findall(r'^tracking_session_tests: 180 checks passed$',text,re.M))==1 and 'tracking_session_tests profile=hr-1280x768 mode=lifecycle\n' in text
results=[];manifest={'formal_acceptance':False,'scope':'Clean a00ab3a runtime libraries, separately compiled HR test harness extension. Full listed lifecycle, all CUDA kernels, selected tools. No kernel/launch filters or forced synchronization. OFA hardware itself is outside CUDA instrumentation.','tools':a.tools,'command':command,'started':m.now()}
m.save(out/'manifest.json',manifest)
with (Path.home()/'.cache/saccade-v1/gpu.lock').open('a') as lock:
 print('Waiting for GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
 before=custody();m.save(out/'custody-before.json',before)
 if before['runtime']['revision']!='a00ab3a6abd7cc2c1f823fd3fbac66a345f3ce48':raise RuntimeError('unexpected runtime revision')
 if next(iter(before['runtime']['engine'].values()))!='8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876':raise RuntimeError('unexpected deployed engine')
 if f'CMAKE_HOME_DIRECTORY:INTERNAL={a.source}\n' not in (a.build/'CMakeCache.txt').read_text():raise RuntimeError('wrong CMake source')
 try:
  r=c.run('baseline-lifecycle',command);text=(out/'baseline-lifecycle.log').read_text(errors='replace');ok=r['returncode']==0 and r.get('target',{}).get('returncode')==0 and complete(text) and not m.incomplete(r,text)
  results.append({'name':'baseline-lifecycle','status':'PASS' if ok else 'FAIL','execution':r});m.save(out/'results.json',results)
  if not ok:raise RuntimeError('HR lifecycle baseline failed')
  for tool in a.tools:
   options=[str(Path('/opt/cuda/bin/compute-sanitizer')),'--tool',tool,'--error-exitcode','99','--print-limit','1000' if tool=='initcheck' else '0']
   if tool=='racecheck':options+=['--racecheck-num-workers','1']
   if tool=='memcheck':options+=['--report-api-errors','explicit']
   name='control-'+tool;r=c.run(name,[*options,str(a.build/'postprocess_sanitizer_control'),tool]);text=(out/(name+'.log')).read_text(errors='replace');control=r['returncode']==99 and r.get('target',{}).get('returncode')==99 and m.DIAGNOSTIC[tool] in text and ('control='+tool+' launch=') in text and not m.incomplete(r,text) and positive_summary(tool,text) and '========= COMPUTE-SANITIZER' in text  and not m.INTERNAL.search(text.replace('Target application returned an error',''))
   results.append({'name':name,'status':'PASS' if control else 'FAIL','execution':r});m.save(out/'results.json',results)
   name=tool+'-lifecycle';r=c.run(name,[*options,*command]);text=(out/(name+'.log')).read_text(errors='replace');done=complete(text);zero=m.clean_sanitizer_log(tool,text);incomplete=m.incomplete(r,text);ok=r['returncode']==0 and r.get('target',{}).get('returncode')==0 and done and zero and control and not incomplete
   row={'name':name,'status':'PASS' if ok else 'INCOMPLETE' if incomplete else 'OPEN','target_completed':done,'zero_summary':m.summary_ok(tool,text),'clean_sanitizer_log':zero,'execution':r};results.append(row);m.save(out/'results.json',results);print(name+': '+row['status'],flush=True)
 except BaseException as error:
  manifest['error']=repr(error)
  raise
 finally:
  c.stop();after=None
  try:
   after=custody();m.save(out/'custody-after.json',after)
  except Exception as error:manifest['custody_error']=str(error)
  manifest.update(finished=m.now(),custody_unchanged=before==after,status='PASS' if before==after and not manifest.get('error') and len(results)==1+2*len(a.tools) and all(x['status']=='PASS' for x in results) else 'OPEN');m.save(out/'manifest.json',manifest)
sys.exit(0 if manifest['status']=='PASS' else 1)
