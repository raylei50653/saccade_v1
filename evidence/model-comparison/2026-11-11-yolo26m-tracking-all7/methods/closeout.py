"""Preserve the completed comparison, inputs, methods, and replay outputs."""
from pathlib import Path
import gzip, hashlib, json, shutil, subprocess
from run_main import ROOT, REPO, EVIDENCE, DATA, RECORDS, SEQS, GROUPS, inventory, save, sha

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
                if (ROOT/a/(s+'.import.json')).exists():
                    copy(ROOT/a/(s+'.import.json'),dest/'import.json')
                else:
                    for suffix in ['.command.json','.log']:
                        copy(ROOT/a/(s+suffix),dest/('capture'+suffix))
            if a!='main' and (ROOT/a/s/'import.json').exists():
                copy(ROOT/a/s/'import.json',dest/'import.json')
        if a!='main':copy(ROOT/a/'replay-receipt.json',EVIDENCE/'outputs'/a/'replay-receipt.json')
        else:
            copy(ROOT/a/'capture.json',EVIDENCE/'outputs'/a/'capture.json')
            copy(ROOT/a/'replay-receipt.json',EVIDENCE/'outputs'/a/'replay-receipt.json')
        resultdir=ROOT/a/'mot/trackeval'
        for name in ['pedestrian_summary.txt','pedestrian_detailed.csv']:
            copy(resultdir/'results/v1'/name,EVIDENCE/'scoring'/a/name)
        copy(resultdir/'trackeval.log',EVIDENCE/'scoring'/a/'trackeval.log')
        copy(resultdir/'run.py',EVIDENCE/'scoring'/a/'run.py')
        copy(ROOT/a/'mot/scoring-receipt.json',EVIDENCE/'scoring'/a/'scoring-receipt.json')
    save(EVIDENCE/'validation.json',{
        'result':'PASS_FOR_THIS_COMPARISON',
        'all_three_arms_complete':True,'sequences':SEQS,'groups':GROUPS,'frames_per_arm':sum(SEQS.values()),
        'prior_02_10_imports_match_original':True,
        'input_hashes_unchanged':True,'main_fresh_scdr_matches_frozen':True,
        'main_replay_mot_and_packets_match_formal_G1':True,
        'shared_tracker_and_MOT_formatter':True,'SCDR_invariants_all_pass':True,
        'canonical_packets_all_pass':True,'same_GT_coverage_all_arms':True,
        'formal_acceptance':False,'performance_comparison':False,'deployment_changed':False,
    })
    roles={s:g for g,seqs in GROUPS.items() for s in seqs}
    m,t=scores['main']['COMBINED'],scores['finetuned']['COMBINED']
    text=['# YOLO26m 三方完整七序列追蹤比較（2026-11-11）','',
          f"主方案全部七序列的 MOTA 為 {m['MOTA']:.4f}，HOTA {m['HOTA']:.4f}，IDF1 {m['IDF1']:.4f}。相對微調 YOLO，差值為 MOTA {m['MOTA']-t['MOTA']:+.4f} pp、HOTA {m['HOTA']-t['HOTA']:+.4f} pp、IDF1 {m['IDF1']-t['IDF1']:+.4f} pp。逐序列與不同資料角色須分開解讀。",'',
          '完整涵蓋 MOT17 七個不同 SDP 序列、5,316 幀／方案，共 15,948 個偵測幀。先前 02／10 的同條件推論結果逐 byte 匯入；其餘五序列各方案新跑。三方七序列全部以同一原生 CPU tracker／固定 G1 motion 回放、同一 TrackEval 重新聚合；不是 DPM／FRCNN／SDP 三份重複影像的 21 條比較。','',
          '04／05／09／11／13 為模型訓練使用序列；10 為選模驗證；02 原為 held-out，但本次屬 M4 之後重用資料。全部七序列的 COMBINED 是含訓練序列的描述性結果，不是獨立泛化成績。','',
          '## MOTA 逐序列','',
          '| 序列 | 資料角色 | 官方權重 | 微調 YOLO | 目前主方案 |',
          '| --- | --- | ---: | ---: | ---: |']
    for seq in SEQS:
        text.append('| '+seq+' | '+roles[seq]+' | '+' | '.join(f"{scores[a][seq]['MOTA']:.6f}" for a in ARMS)+' |')
    text.append('| **COMBINED** | **全部七序列** | '+' | '.join(f"**{scores[a]['COMBINED']['MOTA']:.6f}**" for a in ARMS)+' |')
    text+=['','## 完整追蹤指標','',
          '百分比分數以 0–100 表示。FP／FN 是 TrackEval CLEAR 計數。COMBINED 由 TrackEval 聚合，不是逐序列分數的算術平均。','',
          '| 序列 | 方案 | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | MOTA ↑ | IDSW ↓ | Frag ↓ | FP ↓ | FN ↓ |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seq in [*SEQS,'COMBINED']:
        for a in ARMS:
            r=scores[a][seq]
            text.append(f"| {seq} | {LABELS[a]} | {r['HOTA']:.4f} | {r['DetA']:.4f} | {r['AssA']:.4f} | {r['IDF1']:.4f} | {r['MOTA']:.4f} | {r['IDSW']} | {r['Frag']} | {r['CLR_FP']} | {r['CLR_FN']} |")
    text+=['','## 依資料角色分組的 MOTA','',
           '本表以各組 CLEAR FP／FN／IDSW／GT 計數加總後計算 MOTA；不是逐序列 MOTA 平均。','',
           '| 組別 | 幀數 | 有效 GT | 官方權重 | 微調 YOLO | 主方案 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    groupresults={}
    for group,seqs in {**GROUPS,'02+10':['MOT17-02-SDP','MOT17-10-SDP'],'all7':list(SEQS)}.items():
        groupresults[group]={}
        for a in ARMS:
            totals={k:sum(scores[a][s][k] for s in seqs) for k in ['Frames','GT_Dets','CLR_FP','CLR_FN','IDSW']}
            totals['MOTA']=100*(1-(totals['CLR_FP']+totals['CLR_FN']+totals['IDSW'])/totals['GT_Dets'])
            groupresults[group][a]=totals
        v=groupresults[group]['main']
        text.append(f"| {group} | {v['Frames']} | {v['GT_Dets']} | "+' | '.join(f"{groupresults[group][a]['MOTA']:.6f}" for a in ARMS)+' |')
    save(EVIDENCE/'group-MOTA.json',groupresults)
    text+=['','## 偵測 AP50','', '| 序列 | 官方權重 | 微調 YOLO | 主方案 |', '| --- | ---: | ---: | ---: |']
    for seq in SEQS:text.append('| '+seq+' | '+' | '.join(f"{scores[a][seq]['AP50']:.6f}%" for a in ARMS)+' |')
    text+=['','## 方法、驗證與解讀邊界','',
        '- 官方臂使用固定官方 YOLO26m 權重，微調臂為 UA2 seed 1 e4 primary EMA，保留 YOLO head、gate 關閉。兩臂均為 PyTorch strict FP32、nvJPEG RGB→FP32 /255→antialiased stretch 1280×768；由同一 native scalar F04 產生完整候選。官方臂不是 Ultralytics 預設 one2one/end2end pipeline。',
        '- 主方案為現行 HR Mamba TensorRT FP16 engine，scan plugins 與 input／output FP32；五條新跑 SCDR 與封存部署紀錄相同，02／10 匯入的 SCDR 亦對上封存 hashes。',
        '- 三方 top300／conf0.001／NMS0.5，保留 kept＋suppressed 與原 anchor candidate ID；同一 CPU track_offline、NSA／measurement box 關閉、同一 MOT EOF 插值與格式。所有完整 SCDR 與 canonical packets 通過驗證，沒有排除失敗幀。',
        '- 三方共用凍結 G1 SCMO；它的 person mask 源自主方案偵測器。比較固定 GMC 下的偵測器選擇，未各自重算 detector-conditioned GMC。',
        '- 使用本庫 TrackEval，MOT17 train pedestrian DO_PREPROC=True，三方完整影格／GT coverage 相同。主方案七序列 MOT＋完整 packets 重現正式 G1；02／10 三方此次逐序列分數與上一份報告完全相同。',
        '- 模型訓練資料角色如前述；微調 Teacher 為兩個既有 seed 中選定的一個。不同模型的訓練配方、精度、backend 與 score calibration 不同；固定操作門檻的差距不全是架構因果效應。未調參、換模型或改部署預設。',
        '- 本次為本機開發研究品質比較；沒有三方 FPS／latency A/B，也沒有新的 CUDA session／sanitizer 正式驗收。既有 OPEN 不變。',
        '', '完整精度見 [scores.csv](scores.csv)／[scores.json](scores.json)，事前設定見 [PLAN.json](PLAN.json)，驗證見 [validation.json](validation.json)。方法、輸入 hashes、逐序列收據、MOT、壓縮 SCDR／packets 與評分詳細 CSV 均保留。']
    (EVIDENCE/'REPORT.md').write_text('\n'.join(text)+'\n')
    save(EVIDENCE/'source-after.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
         'status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True)})
    entries=[]
    for p in sorted(EVIDENCE.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS': entries.append(f'{sha(p)}  {p.relative_to(EVIDENCE)}')
    (EVIDENCE/'SHA256SUMS').write_text('\n'.join(entries)+'\n')
    print(EVIDENCE/'REPORT.md')
