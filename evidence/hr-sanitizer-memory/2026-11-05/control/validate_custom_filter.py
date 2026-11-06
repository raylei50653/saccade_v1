#!/usr/bin/env python3
import argparse,collections,json,re,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--regex',default='_ZN7saccade|_ZN3cub');p.add_argument('--expected-launches',type=int,default=303);p.add_argument('--expected-selected',type=int,default=28);a=p.parse_args();lines=a.log.read_text(errors='replace').splitlines();rows=[];current=None;headers=collections.Counter()
for line in lines:
 m=re.fullmatch(r'========= Launch #(\d+)(.*)',line)
 if m:
  label=m[2].strip();flag=False if not label or re.fullmatch(r'\(filtered: (no|false|0)\)',label) else True if 'not in include list' in label or 'in exclude list' in label else None
  current={'launch':int(m[1]),'label':label,'filtered':flag};rows.append(current)
 elif current and (m:=re.fullmatch(r'=========\s+Kernel:\s+(.*)',line)):current['kernel']=m[1]
 elif current and (m:=re.fullmatch(r'=========\s+(Grid|Block|Cluster):\s+(.*)',line)):current[m[1].lower()]=m[2]
 if line.startswith('========= ') and len(line)>10 and not line[10].isspace() and not line.startswith('========= Launch #'):headers[line[10:]]+=1
selected=[r for r in rows if r['filtered'] is False];excluded=[r for r in rows if r['filtered'] is True];unknown=[r for r in rows if r['filtered'] is None];wrong_selected=[r for r in selected if not re.search(a.regex,r.get('kernel',''))];wrong_excluded=[r for r in excluded if re.search(a.regex,r.get('kernel',''))];ids=[r['launch'] for r in rows]
okay=(len(rows)==a.expected_launches and len(selected)==a.expected_selected and not unknown and not wrong_selected and not wrong_excluded and ids==list(range(1,a.expected_launches+1)) and any(line.startswith('constructor_probe complete repeats=') for line in lines))
result={'source':str(a.log.resolve()),'log_sha256':hashlib.sha256(a.log.read_bytes()).hexdigest(),'predicate':a.regex,'total_launches':len(rows),'expected_launches':a.expected_launches,'selected_launches':len(selected),'expected_selected':a.expected_selected,'filtered_launches':len(excluded),'selected_kernel_counts':dict(collections.Counter(r.get('kernel') for r in selected)),'selected_rows':selected,'excluded_kernel_names':sorted({r.get('kernel','') for r in excluded}),'selection_contains_nonmatching_names':wrong_selected,'filtered_names_matching_predicate':wrong_excluded,'unknown_filter_labels':unknown,'diagnostic_base_headers':dict(headers),'inventory_validation_status':'PASS' if okay else 'INCOMPLETE_OR_MISMATCH','sanitizer_acceptance':'INVENTORY_ONLY_NOT_ACCEPTANCE'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['inventory_validation_status'],len(rows),'launches',len(selected),'selected',len(excluded),'filtered');raise SystemExit(0 if okay else 1)
