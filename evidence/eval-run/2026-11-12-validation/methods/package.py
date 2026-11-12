"""Archive both validation attempts and render the run-2 report; no GPU work."""
import csv, datetime, gzip, hashlib, json, shutil, subprocess
from pathlib import Path
ROOT=Path.cwd()
WORK=ROOT/'build/eval-run-work2'
EVIDENCE=ROOT/'evidence/eval-run/2026-11-12-validation'
BENCHMARKS=['MOT17','MOT20','MOT15','DanceTrack','PP22']
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(p,q):
 q.parent.mkdir(parents=True,exist_ok=True)
 if p.name.endswith('.log') or p.name=='mot.txt':
  q=q.with_name(q.name+'.gz')
  q.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
 else:shutil.copyfile(p,q)
parity=json.loads((WORK/'parity.json').read_text())
assert parity['status']=='PASS'
assert len(parity['G1'])==len(parity['G2'])==len(parity['G4'])==7
assert len(parity['G3'])==2
assert all(x['pass'] for k in ['G1','G2','G3','G4'] for x in parity[k])
assert parity['DanceTrack_link_gate']['pass']
# Preserve every run-1 byte, including its original checksum file and stop record.
assert not (EVIDENCE/'attempt-1').exists()
old=list(EVIDENCE.iterdir())
(EVIDENCE/'attempt-1').mkdir()
for p in old:p.rename(EVIDENCE/'attempt-1'/p.name)
for benchmark in BENCHMARKS:
 source=ROOT/('build/eval-run-val2-'+benchmark)
 target=EVIDENCE/'runs'/benchmark
 for p in sorted(source.rglob('*')):
  if not p.is_file() or p.is_symlink() or p.suffix=='.bin':continue
  relative=p.relative_to(source)
  if relative.parts[:2] in [('scoring','gt'),('scoring','trackers')]:continue
  copy(p,target/relative)
# Receipts include build commands, preflight, CPU tests, GPU run commands and logs.
for p in sorted(WORK.iterdir()):
 if p.is_file() and p.suffix in ['.json','.log','.tsv'] and p.name!='parity.json':
  copy(p,EVIDENCE/'receipts'/p.name)
copy(WORK/'validate.py',EVIDENCE/'methods/validate.py')
for benchmark in ['MOT17','PP22']:
 source=WORK/('parity-'+benchmark)/'trackeval'
 for p in source.rglob('*'):
  if not p.is_file() or 'trackers' in p.relative_to(source).parts:continue
  copy(p,EVIDENCE/'controls'/benchmark/p.relative_to(source))
# Validate that every run used the same binary and engine bytes with a clean identity.
environments={b:json.loads((EVIDENCE/'runs'/b/'environment.json').read_text()) for b in BENCHMARKS}
files=[v['files'] for v in environments.values()]
assert all(v==files[0] for v in files)
assert all(v['git_head']==parity['code_commit'] and not v['git_status_porcelain'] and v['status']=='PASS' for v in environments.values())
identities=[item for v in environments.values() for item in list(v['captures'].values())+[s for a in v['speed'].values() for s in a['sessions']]]
assert len(identities)==63
assert all(v==parity['code_commit'] for item in identities for v in item['source_commit'])
assert all(v=='false' for item in identities for v in item['source_dirty'])
assert all(a['tracker_config_check']=='PASS' and len(a['sessions'])==8 and a['graph']['latency_samples']==480 for v in environments.values() for a in v['speed'].values())
parity['build_identity']={'code_commit':parity['code_commit'],'runs':5,'captures':7,'speed_sessions':56,'identity_records':63,'source_dirty':False,'clean_git_porcelain_runs':5,'warning_count':0,'same_binary_and_engine_hashes':True,'files':files[0]}
# Record the only local orchestration failure; it occurred before any GPU work.
parity['own_defects_fixed']=[{'scope':'local CPU-test command orchestration','issue':'Parallel command redirection raced with creation of build/eval-run-work2; the evaluation-tools test did not start on that invocation.','fix':'Created the log directory first and reran the command; both CPU suites subsequently passed, including clean-HEAD G5.','runner_worker_test_code_failure':False,'gpu_reruns':0}]
parity['own_defects_fixed'].append({'scope':'evidence wording','issue':'Notes initially mentioned CRLF only for detailed CSV, while native summary TXT also uses CRLF.','fix':'Corrected the report Notes and preserved all native bytes.','gpu_reruns':0})
dump(EVIDENCE/'parity.json',parity)
telemetry={}
for b in BENCHMARKS:
 with (EVIDENCE/'runs'/b/'telemetry.csv').open() as f: rows=list(csv.DictReader(f,skipinitialspace=True))
 def values(key):return [float(r[key].split()[0]) for r in rows]
 util=values('utilization.gpu [%]');mem=values('memory.used [MiB]');temp=values('temperature.gpu');clock=values('clocks.current.sm [MHz]')
 telemetry[b]={'samples':len(rows),'gpu_utilization_mean_percent':sum(util)/len(util),'gpu_utilization_max_percent':max(util),'gpu_memory_max_mib':max(mem),'temperature_max_c':max(temp),'sm_clock_min_mhz':min(clock),'sm_clock_max_mhz':max(clock)}
dump(EVIDENCE/'telemetry-summary.json',telemetry)
# Remove only the explicitly retired run-1 build output directories, after all gates pass.
removed=[]
for name in ['eval-run-val-MOT17','eval-run-val-MOT20','eval-run-val-MOT15','eval-run-val-DanceTrack','eval-run-val-PP22','eval-run-work']:
 p=ROOT/'build'/name
 if p.exists():
  assert p.is_dir() and not p.is_symlink()
  removed.append(str(p.relative_to(ROOT)))
  shutil.rmtree(p)
dump(EVIDENCE/'receipts/retired-run1-outputs.json',{'after_all_gates_pass':True,'removed':removed,'new_outputs':[f'build/eval-run-val2-{b}' for b in BENCHMARKS]})
report=f'''# 單指令品質與速度 runner 驗證（2026-11-12，run 2）

**結果：PASS；五個 benchmark、七個序列完成 G1–G5 與 DanceTrack link gate。** 本報告的結果、gates、速度與 telemetry 均只使用 run 2；程式提交為 `{parity['code_commit']}`，五組資料皆由同一組重建 binaries 產生。

## 範圍與身分

- GPU 驗證（UTC）：`{parity['started_utc']}` 至完整驗證結束 `{parity['completed_utc']}`。
- 前置輸入檢查 19/19 通過：engine hash、driver 621.14、七組 GT／seqinfo 共十四個 hashes、frozen per-sequence CSV hash，以及 eval interpreter 的 NumPy／SciPy imports。詳見 `receipts/preflight.json`。
- 以 run 1 的完整 configure 參數重新 configure／build `build/eval-run`，Release、SM 120、`--parallel 2`，只建 `track_sequence`、`tracking_benchmark`。相對 `9b0d14f8` 的 C++／CUDA sources 及指定凍結目錄差異為空（`receipts/source-control.json`）。
- 五次 runner 的 git porcelain 均為空；七個 capture、五十六個 speed sessions，共 63 份 native identity records，全部 `source_commit` 等於程式提交且 `source_dirty=false`。五份 RESULTS 沒有 dirty／mismatch warning；五組 binary／engine hashes 完全相同，見 `parity.json` 的 `build_identity`。
- GPU 為 NVIDIA GeForce RTX 5070 Ti Laptop GPU，driver 621.14；engine SHA-256 為 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`，未重建 engine。
- 固定 `hr-1280x768`、NSA／measurement box OFF；capture 不輸出 packets／motion。五組 GPU jobs 循序執行，GPU phase 持有 gpu.lock、每個 native 呼叫經 `timeout -k 30s 3600s`、每秒 telemetry；評分在釋放 lock 後執行。

## Gates（僅 run 2）

| Gate | 結果 | 數量與規則 |
| --- | --- | --- |
| G1 MOT bytes | PASS | 7/7 MOT SHA-256 等於 primary156 main |
| G2 逐序列分數 | PASS | 7/7 序列、42/42 指標完全相同；HOTA／DetA／AssA／IDF1／MOTA parsed floats exact equality，IDSW integer equality |
| G3 既有 scorer parity | PASS | 2/2 組、12/12 指標；eval_mot17.sh 評分三份 MOT17 複本，eval_pp22.sh 評分 PP22 複本；COMBINED 依既有輸出精度（品質五位有效數字、IDSW 整數）完全相同 |
| G4 速度 | PASS | 7/7 summary status PASS；56/56 fresh sessions、56/56 tracker config checks PASS；每序列 480 single samples，合計 3360；沒有 short-sequence skip |
| G5 CPU 測試 | PASS | eval_run_tests.py 12/12、evaluation_tools_tests.py 12/12；stub GPU 工具＋真實 repository TrackEval worker，沒有 GPU 推論 |
| DanceTrack link gate | PASS | 1203/1203 symlinks，以雙方逐 frame 的 readlink -f 結果比對，全部指向相同實體影像 |

DanceTrack G1 重現 `503f5596d5b17ea335bfae78027e87e1a14cea76de14916aaf08e49578c94c59`。原始八位檔名影像透過 `build/eval-run-val2-DanceTrack/inputs/dancetrack0004` 的六位 symlink view 供兩個 native 工具讀取，`seqinfo.ini` 複製原檔；GT、評分 seqinfo 與 shared-parent check 仍使用原始序列。`environment.json` 記錄 original path、view path、source_digits 8、link_count 1203。逐 frame 雙方連結與解析結果保留於 `receipts/dancetrack-links.tsv`；primary156 view 僅讀取、未修改。

所有 COMBINED 均直接來自該組 TrackEval detailed CSV，沒有平均各序列品質。G3 控制工具透過 `SACCADE_DATASET_ROOT=/home/ray/developer/ai/dataset` 指定資料位置，既有 scripts byte-unchanged；控制輸出見 `controls/` 與 `receipts/parity-*.log.gz`。

## 五組 RESULTS

品質為百分比（IDSW 除外）；CSV／JSON 保留完整精度，以下為 runner 的三位小數表格。
'''
for b in BENCHMARKS:
 table='\n'.join(line for line in (EVIDENCE/'runs'/b/'RESULTS.md').read_text().splitlines() if line.startswith('|'))
 report+=f'\n### {b}\n\n[RESULTS](runs/{b}/RESULTS.md) · [CSV](runs/{b}/results.csv) · [JSON](runs/{b}/results.json)\n\n{table}\n'
report+='''
## MOT17-10 速度對照

百分比為 `(run 2 / 基準 − 1) × 100`；此項為資訊對照，不作效能 gate，也沒有為追逐速度差異重跑。

| 指標 | g1-live | attempt 1 | run 2 | 相對 g1-live | 相對 attempt 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
for k,v in parity['MOT17-10_speed_comparison'].items():
 a=parity['MOT17-10_attempt1_comparison'][k]
 report+=f"| {k} | {v['g1_live']:.6f} | {a['attempt_1']:.6f} | {v['runner']:.6f} | {v['difference_percent']:+.3f}% | {a['difference_percent']:+.3f}% |\n"
report+='''
本輪與 attempt 1 均取 native benchmark summary；g1-live 的指定 single 基準採 host timestamp instrumentation 的 enqueue→receive 返回時間，與 summary 有少量計時邊界差異。clocks 未鎖定，此對照不代表受控加速。

## Telemetry

一秒取樣包含初始化、warmup、capture、處理與清理，不能當作穩態 kernel／Tensor Core／OFA utilization。

| benchmark | samples | GPU utilization mean / max (%) | GPU memory max (MiB) | temperature max (°C) | SM clock min / max (MHz) |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
for b,v in telemetry.items():
 report+=f"| {b} | {v['samples']} | {v['gpu_utilization_mean_percent']:.1f} / {v['gpu_utilization_max_percent']:.0f} | {v['gpu_memory_max_mib']:.0f} | {v['temperature_max_c']:.0f} | {v['sm_clock_min_mhz']:.0f} / {v['sm_clock_max_mhz']:.0f} |\n"
report+='''
## Attempt 1 歷史與規格原因

Run 1 在指定 DanceTrack 原始目錄只有八位檔名、而 native 工具要求六位檔名時依規則停止；當時規格沒有授權 image view，這是規格缺口，不是 runner 缺陷。當時只完成五個序列的 G1／G2／G4，G3 與 PP22 未執行。本輪依補充規格新增與 primary156 相同的 symlink view，沒有變更 native sources。Run 1 的全部 run records、失敗／停止收據、報告、parity 與原始 SHA256SUMS 均原樣移到 `attempt-1/`；不併入本輪 gates 或結果。

## 自有缺陷與修正

本輪 runner、worker、測試沒有觸發需修正的程式失敗，沒有 GPU 重跑或放寬比較。準備階段一次本機測試命令將 log 目錄建立與另一個測試輸出平行啟動，造成 shell redirection 先於目錄建立而失敗，該次測試尚未啟動；改為先建立目錄，再執行測試後通過，乾淨 HEAD 的正式 G5 也再次通過。此為本機命令編排錯誤，不影響 GPU 結果。封裝檢查另發現 Notes 原先只列 detailed CSV，但原生 summary TXT 也保留 CRLF；已補正文字，未改動工具輸出。Run 1 已修正的證據文字錯誤保留於其原始報告。

## 保管與限制

- `runs/` 的五組結果皆來自 `build/eval-run-val2-<benchmark>`；保留完整精度結果、identity、commands、telemetry、MOT gzip、resolved configs、benchmark summary／runs.csv／single CSV／session identities、scoring detailed CSV 與 gzip logs。影像連結以逐 frame TSV 和 environment 記錄保管，沒有把外部影像或 symlinks 複製入 git。
- 全部 gates 通過後，已刪除 run 1 的 `build/eval-run-val-*` 及其暫存 `build/eval-run-work`；清單見 `receipts/retired-run1-outputs.json`。run 2 build／輸出留在 worktree 的 build/，沒有提交 build trees 或 packet .bin。
- 速度只量每序列前 140 frames，20 warmup 後量 frames 21–140；single 為 480 pooled samples，queued FPS 是預載 JPEG 的完成吞吐，排除磁碟 I/O、constructor、warmup 與輸出檔案；它不是逐 frame latency、live-stream 或 camera FPS。clocks 未鎖定。
- 只驗證指定七個序列，沒有重跑 primary156 全集，也沒有進行 sanitizer、nsys、GPU CTest、training 或其他 GPU 工作。短序列 skip 另由 CPU fixture 驗證。
- evidence commit 只更動證據；binary 身分對應程式提交，量測時 HEAD 與 binary 相同。未 push、未 merge；沒有修改 main checkout 或資料集。
- `SHA256SUMS` 涵蓋本目錄除自身以外的所有檔案，包含 attempt-1 的原始 checksum；在 attempt-1/ 內另可驗證原有清單。

## Notes

1. 三個 MOT17 序列的 primary156 `command.json` 未包含於原封存 checkout；七個 capture hashes 均存在且本輪 G1 全部相同，沒有觸發 mismatch argv diff。
2. TrackEval 原生 detailed CSV 與 summary TXT 保留 CRLF bytes；一般 git diff --check 會列出行尾空白提示。程式與 Markdown whitespace check 通過，沒有為消除提示改寫原始 CSV。
'''
(EVIDENCE/'REPORT.md').write_text(report)
dump(EVIDENCE/'source-after.json',{'code_commit':parity['code_commit'],'head_before_evidence_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cpp_diff_from_9b0d14f8':subprocess.check_output(['git','diff','9b0d14f8','--','*.cpp','*.cc','*.cxx','*.cu','*.h','*.hpp','*.cuh'],text=True),'no_push':True,'no_merge':True})
# Include this packaging method before hashing the complete evidence bundle.
copy(WORK/'package.py',EVIDENCE/'methods/package.py')
paths=sorted(p for p in EVIDENCE.rglob('*') if p.is_file() and p!=EVIDENCE/'SHA256SUMS')
(EVIDENCE/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(EVIDENCE)}\n' for p in paths))
print('packaged',len(paths),'files; all five run-2 gates PASS')
