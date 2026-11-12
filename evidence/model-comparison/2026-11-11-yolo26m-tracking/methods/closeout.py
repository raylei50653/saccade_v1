"""Preserve the completed comparison, inputs, methods, and replay outputs."""
from pathlib import Path
import gzip, hashlib, json, shutil, subprocess
from run_main import ROOT, REPO, EVIDENCE, DATA, RECORDS, SEQS, inventory, save, sha

ARMS=['official','finetuned','main']
LABELS={'official':'官方 YOLO26m 權重','finetuned':'微調 YOLO26m UA2 e4','main':'目前主方案 HR Mamba FP16'}

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)

def zipped(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with src.open('rb') as f,dst.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as z:
        shutil.copyfileobj(f,z)

if __name__=='__main__':
    scores=json.loads((EVIDENCE/'scores.json').read_text())
    after=inventory()
    save(EVIDENCE/'inputs-after.json',after)
    assert after==json.loads((EVIDENCE/'inputs-before.json').read_text()),'inputs changed'
    for p in [ROOT/'run_main.py',ROOT/'score_results.py',ROOT/'closeout.py']:
        copy(p,EVIDENCE/'methods'/p.name)
    for sub in ['detector-tools','replay-tools']:
        for p in (ROOT/sub).iterdir():
            if p.is_file() and p.suffix in ('.py','.cpp','.json','.log'):
                copy(p,EVIDENCE/'methods'/sub/p.name)
    for p in [ROOT/'main-driver.log', ROOT/'gpu-telemetry.csv']:
        copy(p,EVIDENCE/p.name)
    for a in ARMS:
        for s in SEQS:
            dest=EVIDENCE/'outputs'/a/s
            zipped(ROOT/a/(s+'.scdr'),dest/'detections.scdr.gz')
            copy(ROOT/a/'mot'/(s+'.txt'),dest/'mot.txt')
            copy(ROOT/a/'mot'/(s+'.config'),dest/'resolved.config')
            zipped(ROOT/a/'mot'/(s+'.packets.bin'),dest/'packets.bin.gz')
            copy(ROOT/a/'mot'/(s+'.packet-validation.log'),dest/'packet-validation.log')
            copy(ROOT/a/(s+'.ap50.json'),dest/'ap50.json')
            if a!='main':
                capture=json.loads((ROOT/a/s/'inference.json').read_text())
                assert capture['status']=='PASS'
                for p,h in capture['input_hashes'].items(): assert sha(Path(p))==h,(p,'input changed')
                assert capture['weights_state_before']==capture['weights_state_after']
                for p,h in capture['source_hashes'].items(): assert sha(Path(p))==h,(p,'source changed')
                copy(ROOT/a/s/'inference.json',dest/'inference.json')
            else:
                for suffix in ['.command.json','.log']:
                    copy(ROOT/a/(s+suffix),dest/('capture'+suffix))
        if a!='main':copy(ROOT/a/'replay-receipt.json',EVIDENCE/'outputs'/a/'replay-receipt.json')
        else:copy(ROOT/a/'capture.json',EVIDENCE/'outputs'/a/'capture.json')
        resultdir=ROOT/a/'mot/trackeval'
        for name in ['pedestrian_summary.txt','pedestrian_detailed.csv']:
            copy(resultdir/'results/v1'/name,EVIDENCE/'scoring'/a/name)
        copy(resultdir/'trackeval.log',EVIDENCE/'scoring'/a/'trackeval.log')
        copy(resultdir/'run.py',EVIDENCE/'scoring'/a/'run.py')
    save(EVIDENCE/'validation.json',{
        'result':'PASS_FOR_THIS_COMPARISON',
        'all_three_arms_complete':True,'sequences':SEQS,'frames_per_arm':sum(SEQS.values()),
        'input_hashes_unchanged':True,'main_fresh_scdr_matches_frozen':True,
        'main_replay_mot_and_packets_match_formal_G1':True,
        'shared_tracker_and_MOT_formatter':True,'SCDR_invariants_all_pass':True,
        'canonical_packets_all_pass':True,'same_GT_coverage_all_arms':True,
        'formal_acceptance':False,'performance_comparison':False,'deployment_changed':False,
    })
    text=['# YOLO26m 三方固定追蹤比較（2026-11-11）','',
          '本機實際執行三個偵測器，各涵蓋 MOT17-10 654 幀與 MOT17-02 600 幀，共 1,254 幀／方案、3,762 幀。固定 1280×768、同一 tracker、G1 motion、MOT formatter 與 TrackEval。這是開發研究比較，未調整門檻或部署預設，不是新的正式驗收。','',
          '官方臂指官方預訓練權重在專案 one2many＋F04 路徑的結果；微調臂為 UA2 seed 1 e4 primary EMA，保留 YOLO head、關閉 gate；主方案為現行固定 HR Mamba TensorRT FP16 engine。官方／微調臂使用 PyTorch strict FP32，主方案 scan plugins 與 input／output 為 FP32。','',
          '## 追蹤結果','',
          '所有百分比分數以 0–100 表示。FP／FN 是 TrackEval CLEAR 的計數，並非偵測 AP50 的固定 0.25 門檻計數。COMBINED 為 TrackEval 正式聚合，不是兩序列的分數平均。','',
          '| 序列 | 方案 | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | MOTA ↑ | IDSW ↓ | Frag ↓ | FP ↓ | FN ↓ |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in [*SEQS,'COMBINED']:
        for a in ARMS:
            r=scores[a][s]
            text.append(f"| {s} | {LABELS[a]} | {r['HOTA']:.4f} | {r['DetA']:.4f} | {r['AssA']:.4f} | {r['IDF1']:.4f} | {r['MOTA']:.4f} | {r['IDSW']} | {r['Frag']} | {r['CLR_FP']} | {r['CLR_FN']} |")
    text+=['','## 主方案相對微調 YOLO 的差值','',
           '| 序列 | ΔHOTA（pp） | ΔIDF1（pp） | ΔMOTA（pp） | ΔIDSW | ΔFP | ΔFN |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in [*SEQS,'COMBINED']:
        m,t=scores['main'][s],scores['finetuned'][s]
        text.append(f"| {s} | {m['HOTA']-t['HOTA']:+.4f} | {m['IDF1']-t['IDF1']:+.4f} | {m['MOTA']-t['MOTA']:+.4f} | {m['IDSW']-t['IDSW']:+d} | {m['CLR_FP']-t['CLR_FP']:+d} | {m['CLR_FN']-t['CLR_FN']:+d} |")
    text+=['','## 同次偵測 AP50','', '| 序列 | 官方權重 | 微調 YOLO | 主方案 |', '| --- | ---: | ---: | ---: |']
    for s in SEQS:text.append('| '+s+' | '+' | '.join(f"{scores[a][s]['AP50']:.6f}%" for a in ARMS)+' |')
    text+=['','## 驗證與解讀邊界','',
        '- 三方均保留 NMS kept 與 suppressed 的完整候選、原始 anchor candidate ID，經同一原生 CPU tracker 與 EOF 插值輸出。NSA／measurement box 皆關閉。官方與微調臂由 native scalar F04 產生 SCDR；主方案使用原生 CUDA F04。',
        '- 新跑主方案兩序列 SCDR 與固定部署紀錄逐 byte 相同；本次來源 CPU replay 的 MOT 與完整 packets 均重現正式 G1 參考。每個新臂亦通過完整 SCDR 與 canonical packet 驗證，沒有排除失敗幀。',
        '- 三方共用已凍結的 G1 SCMO。它的 person mask 源自主方案偵測器；本次控制固定 motion，比較的是偵測器接到相同追蹤器的結果，沒有各自重算 detector-conditioned GMC。',
        '- MOT17-10 已用於選模；MOT17-02 是 M4 之後重用資料的比較，不再視為新的獨立 eval。微調 Teacher 是既有兩 seed 中選定的一個；這不是跨 seed 的估計。',
        '- PyTorch FP32 與 TensorRT FP16 的精度／backend、模型訓練配方不同，因此不能將差距全部歸因於 Mamba 架構。官方臂不是 Ultralytics 預設 one2one/end2end 的官方原生 baseline。',
        '- 原歷史 AP50 使用 PyTorch decode；本次統一採原生 F04 的候選產生與 tie／rounding 規則，小幅數值差異需以本次 AP50 表為準。',
        '- 本次沒有量測可比較的三方 FPS／latency，也未執行新的 CUDA session 或 sanitizer 驗收；既有 OPEN 不變。',
        '', '完整原始精度見 [scores.csv](scores.csv)／[scores.json](scores.json)，事前設定見 [PLAN.json](PLAN.json)，驗證見 [validation.json](validation.json)。方法、逐序列收據、MOT、壓縮 SCDR／packets、GT／像素與權重 hashes 均保留於本目錄。']
    (EVIDENCE/'REPORT.md').write_text('\n'.join(text)+'\n')
    save(EVIDENCE/'source-after.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
         'status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True)})
    entries=[]
    for p in sorted(EVIDENCE.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS': entries.append(f'{sha(p)}  {p.relative_to(EVIDENCE)}')
    (EVIDENCE/'SHA256SUMS').write_text('\n'.join(entries)+'\n')
    print(EVIDENCE/'REPORT.md')
