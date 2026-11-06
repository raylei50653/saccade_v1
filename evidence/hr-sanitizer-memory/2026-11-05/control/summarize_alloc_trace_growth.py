#!/usr/bin/env python3
"""Resolve bounded preload trace addresses against captured per-PID mappings.
All sizes are intercepted requested bytes, not RSS. mmap and heap are separate.
"""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();processes=[]
elf_symbols={}
def defined_functions(module):
 if module in elf_symbols:return elf_symbols[module]
 symbols=[]
 try:
  r=subprocess.run(['readelf','-WsW',module],capture_output=True,text=True,timeout=10)
  for line in r.stdout.splitlines():
   cols=line.split()
   if len(cols)>=8 and cols[0].endswith(':') and cols[3] in ('FUNC','IFUNC') and cols[6]!='UND':
    try:value=int(cols[1],16);size=int(cols[2],16 if cols[2].startswith('0x') else 10)
    except ValueError:continue
    if size:symbols.append((value,size,cols[7]))
 except (OSError,subprocess.SubprocessError):pass
 elf_symbols[module]=sorted(set(symbols));return elf_symbols[module]

for file in sorted(a.directory.glob('alloc-*.jsonl')):
 rows=[];bad=0
 for line in file.read_text(errors='replace').splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:bad+=1
 if not rows:continue
 pid=next((r['pid'] for r in rows if r.get('type')=='header'),None);maps=a.directory/f'maps-{pid}.txt';segments=[]
 if maps.exists():
  for line in maps.read_text(errors='replace').splitlines():
   cols=line.split(None,5)
   if len(cols)<6:continue
   start,end=[int(v,16) for v in cols[0].split('-')];segments.append((start,end,int(cols[2],16),cols[5]))
 cache={}
 def resolve(address):
  if address in cache:return cache[address]
  value=int(address,16);out={'address':address,'module':None}
  for start,end,offset,module in segments:
   if start<=value<end:
    elf_offset=value-start+offset;out.update(module=module,module_file_offset=hex(elf_offset));path=Path(module)
    if path.is_file() and not module.endswith(' (deleted)'):
     try:
      with path.open('rb') as binary:header=binary.read(18)
      v=value if header[:4]==b'\x7fELF' and int.from_bytes(header[16:18],'little')==2 else elf_offset
      r=subprocess.run(['addr2line','-C','-f','-e',module,hex(v)],capture_output=True,text=True,timeout=5)
      out['symbol']=r.stdout.strip().splitlines()
      out['symbol_note']='addr2line may report nearest symbol; containing ELF function range is authoritative when available'
      out['elf_lookup_address']=hex(v)
      out['containing_elf_function_ranges']=[{'name':name,'start':hex(start),'end_exclusive':hex(start+size),'size':size} for start,size,name in defined_functions(module) if start<=v<start+size]
     except (OSError,subprocess.SubprocessError):pass
    break
  cache[address]=out;return out
 sites={str(r['id']):dict(r,caller_resolution=resolve(r['caller']),stack_resolution=[resolve(x) for x in r.get('stack',[])]) for r in rows if r.get('type')=='site'}
 growth_stacks=[dict(r,caller_resolution=resolve(r['caller']),stack_resolution=[resolve(x) for x in r.get('stack',[])]) for r in rows if r.get('type')=='growth_stack']
 snapshots=[];current=None
 for r in rows:
  if r.get('type')=='snapshot':current=dict(r,top_owners=[]);snapshots.append(current)
  elif r.get('type')=='owner' and current is not None:current['top_owners'].append(dict(r,site=sites.get(str(r['id']))))
 processes.append({'pid':pid,'trace':str(file),'maps':str(maps),'bad_or_truncated_lines':bad,'header':next((r for r in rows if r.get('type')=='header'),{}),'snapshots':snapshots,'sites':sites,'growth_stacks':growth_stacks})
a.output.write_text(json.dumps({'schema':'bounded-allocator-trace-growth/2','limitations':['Requests below configured threshold are not tracked.','Requested-byte counters are not RSS or allocation usable sizes.','Heap and mmap API figures may overlap and must not be summed as physical memory.','mmap tracking follows exact-base full/leading unmaps only; unmatched/partial flags qualify precision.','Direct syscalls, private allocator entrypoints, and fork-only children before exec are outside coverage.','Trace table virtual size12MiB plus bounded owner data; observer overhead can affect timing.','Site backtraces are first-witness only; growth_stacks capture current allocating thread at each128MiB heap growth trigger.'], 'processes':processes},indent=2)+'\n')
print(json.dumps({'processes':len(processes),'output':str(a.output),'snapshots':[len(x['snapshots']) for x in processes]}))
