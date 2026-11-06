#!/usr/bin/env python3
"""Summarize run_probe samples without equating cgroup charge, RSS, or PSS."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('captures',type=Path,nargs='+');a=p.parse_args()
rows=[]
for capture in a.captures:
 manifest=json.loads((capture/'manifest.json').read_text()) if (capture/'manifest.json').exists() else {}
 execution=json.loads((capture/'execution.json').read_text()) if (capture/'execution.json').exists() else {}
 command=json.loads((capture/'command.json').read_text()) if (capture/'command.json').exists() else {}
 samples=[]
 if (capture/'telemetry.jsonl').exists():
  for line in (capture/'telemetry.jsonl').read_text().splitlines():
   try:samples.append(json.loads(line))
   except json.JSONDecodeError:pass
 processes={};cgroup_peak=0;stat_peaks={};at_peak=None
 for sample in samples:
  current=sample.get('memory.current',0)
  cgroup_peak=max(cgroup_peak,sample.get('memory.peak',0))
  if at_peak is None or current>at_peak['memory.current']:
   at_peak={'time':sample['time'],'memory.current':current,'memory.stat':sample.get('memory.stat',{}),'memory.events':sample.get('memory.events',{}),'pids':[x['pid'] for x in sample.get('processes',[])]}
  for key,value in sample.get('memory.stat',{}).items():stat_peaks[key]=max(stat_peaks.get(key,0),value)
  for proc in sample.get('processes',[]):
   record=processes.setdefault(proc['pid'],{'pid':proc['pid'],'comm':proc['comm'],'cmdline':proc['cmdline'],'first_sample':sample['time'],'last_sample':sample['time'],'samples':0,'status_peaks':{},'smaps_peaks':{},'maps_snapshots':[]})
   record['last_sample']=sample['time'];record['samples']+=1
   for label,field in [('status_peaks','status_numeric'),('smaps_peaks','smaps_numeric')]:
    for key,value in proc.get(field,{}).items():record[label][key]=max(record[label].get(key,0),value)
   if proc.get('maps_snapshot'):record['maps_snapshots'].append(proc['maps_snapshot'])
 ordered=sorted(processes.values(),key=lambda x:-x['status_peaks'].get('VmRSS_bytes',0))
 for proc in ordered:
  files=set()
  for rel in proc['maps_snapshots']:
   path=capture/rel
   if path.exists():
    for line in path.read_text().splitlines():
     fields=line.split(None,5)
     if len(fields)==6 and fields[5].startswith('/'):files.add(fields[5])
  proc['mapped_files']=sorted(files)
 runlog=(capture/'run.log').read_text(errors='replace') if (capture/'run.log').exists() else ''
 reported_result=re.findall(r'^\s*Finished with result: ([^\n]+)',runlog,re.M)
 reported_peak=re.findall(r'^\s*Memory peak: ([^\n]+)',runlog,re.M)
 limit=manifest.get('memory_max','')
 lm=re.fullmatch(r'(\d+)([KMGT]?)',str(limit));limit_bytes=int(lm[1])*(1024**('KMGT'.find(lm[2])+1)) if lm and lm[2] else int(lm[1]) if lm else None
 resource_evidence={'service_result_text':reported_result[-1] if reported_result else None,'service_memory_peak_text':reported_peak[-1] if reported_peak else None,'service_reported_oom':bool(execution.get('service_reported_oom') or any('oom-kill' in r for r in reported_result)),'configured_memory_cap_bytes':limit_bytes,'sampled_peak_is_not_proof_of_true_requirement':True,'target_final_record_exists':(capture/'target-final.json').exists()}
 rows.append({'resource_evidence':resource_evidence,'service_started':manifest.get('service_started'),'capture':str(capture.resolve()),'status':manifest.get('status','UNKNOWN'),'command':command.get('target'),'environment':command.get('target_environment'),'samples':len(samples),'first_sample':samples[0]['time'] if samples else None,'last_sample':samples[-1]['time'] if samples else None,'sampled_cgroup_peak_bytes':cgroup_peak,'recorded_cgroup_peak_bytes':execution.get('sampled_memory_peak_bytes'),'sampled_cgroup_stat_peaks':stat_peaks,'cgroup_stat_at_max_current':at_peak,'processes':ordered,'execution':execution,'limits':{'memory':manifest.get('memory_max'),'swap':manifest.get('memory_swap_max'),'timeout':manifest.get('timeout_seconds')}})
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'schema':'racecheck-process-memory-summary/1','note':'Per-field maxima may occur at different times. RSS and PSS are process observations; cgroup charges include accounted anon/file/kernel and are not their sum. No allocation-owner or leak conclusion is inferred.','captures':rows},indent=2)+'\n')
for row in rows:
 print(row['capture'],row['status'],'peak',row['recorded_cgroup_peak_bytes'])
 for proc in row['processes'][:3]:print(' ',proc['pid'],proc['comm'],'RSS',proc['status_peaks'].get('VmRSS_bytes'),'anon',proc['status_peaks'].get('RssAnon_bytes'),'PSS',proc['smaps_peaks'].get('Pss_bytes'))
