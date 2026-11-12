"""Preserve TRT comparison and produce a source-bounded engineering report."""
from pathlib import Path
import csv,gzip,hashlib,json,shutil,subprocess,datetime
from run_quality import ROOT,REPO,EVIDENCE,SEQ,ENGINES,sha,save

ARMS=['official','finetuned','main']
LABEL={'official':'官方 YOLO26m raw6','finetuned':'微調 YOLO26m raw6','main':'目前 HR Mamba'}

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def zipped(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with src.open('rb') as f,dst.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)

def main():
    quality=json.loads((EVIDENCE/'quality-scores.json').read_text())
    modelroot=ROOT/'benchmark-tools/model-runs';piperoot=ROOT/'benchmark-tools/pipeline-runs'
    model=json.loads((modelroot/'summary.json').read_text());pipe=json.loads((piperoot/'summary.json').read_text())
    assert model['status']==pipe['status']=='PASS'
    assert len(json.loads((modelroot/'runs.json').read_text()))==20
    assert len(json.loads((piperoot/'runs.json').read_text()))==96
    subgroup_path=ROOT/'pipeline-tools/quality-subgroups.json'
    subgroups=json.loads(subgroup_path.read_text()) if subgroup_path.exists() else {}
    if subgroups:copy(subgroup_path,EVIDENCE/'quality-subgroups.json')
    def mg(a,comparison='raw6',mode='graph'):
        return next(r for r in model['groups'] if r['comparison']==comparison and r['arm']==a and r['mode']==mode)
    def pg(a,s='MOT17-10-SDP',path='tracking',mode='single'):
        return next(r for r in pipe['groups'] if (r['arm'],r['sequence'],r['path'],r['mode'])==(a,s,path,mode))
    # No performance assertion is a quality/architecture qualification gate.
    comparisons={}
    for a in ['official','finetuned']:
        comparisons[a]={'model_graph_mean_extra_percent':100*(mg('main')['gpu_event_single_ms']['mean']/mg(a)['gpu_event_single_ms']['mean']-1),
            'all7_quality_delta':{k:quality['mot']['main']['COMBINED'][k]-quality['mot'][a]['COMBINED'][k] for k in ['MOTA','HOTA','IDF1']},'prefixes':{}}
        for s in ['MOT17-10-SDP','MOT17-02-SDP']:
            comparisons[a]['prefixes'][s]={'single_mean_extra_percent':100*(pg('main',s)['delivery_ms']['mean']/pg(a,s)['delivery_ms']['mean']-1),
                'single_p95_extra_percent':100*(pg('main',s)['delivery_ms']['p95']/pg(a,s)['delivery_ms']['p95']-1),
                'queued_FPS_change_percent':100*(pg('main',s,mode='queued')['throughput_fps']/pg(a,s,mode='queued')['throughput_fps']-1)}
    save(EVIDENCE/'comparison.json',comparisons)
    # Engine, dataset and scorer custody; main binary/weights have not changed.
    before=json.loads((EVIDENCE/'inputs-before.json').read_text())
    after={p:{'bytes':Path(p).stat().st_size,'sha256':sha(Path(p))} for p in before}
    assert before==after
    save(EVIDENCE/'inputs-after.json',after)
    artifacts={}
    for a,p in {**ENGINES,'official-default':ROOT/'artifacts/official-default/model.engine',
                'official-fp32-diagnostic':ROOT/'artifacts/official-fp32-diagnostic/model.engine'}.items():
        artifacts[a]={'path':str(p),'bytes':p.stat().st_size,'MiB':p.stat().st_size/(1<<20),'sha256':sha(p)}
    save(EVIDENCE/'ARTIFACTS.json',artifacts)
    # Methods, compile/verification history, numerical FAIL receipts, actual layer formats.
    for sub in ['export-tools','pipeline-tools','benchmark-tools']:
        for p in (ROOT/sub).iterdir():
            if p.is_file() and p.suffix in ['.py','.cpp','.cu','.hpp','.json','.log','.txt']:
                copy(p,EVIDENCE/'methods'/sub/p.name)
    for name in ['run_quality.py','closeout.py','quality-driver.log','gpu-telemetry.csv']:
        copy(ROOT/name,EVIDENCE/'methods'/name)
    for a in ['official','finetuned','official-default','official-fp32-diagnostic']:
        folder=ROOT/'artifacts'/a
        if folder.exists():
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix in ['.json','.csv','.log','.txt','.status']:
                    copy(p,EVIDENCE/'exports'/a/p.relative_to(folder))
    for a in ARMS:
        for s in SEQ:
            folder=ROOT/'quality'/a/s;out=EVIDENCE/'quality'/a/s
            for p in folder.iterdir():
                if p.suffix in ['.json','.log','.config'] or p.name=='mot.txt':copy(p,out/p.name)
            for name in ['detections.scdr','packets.bin','motion.scmo']:zipped(folder/name,out/(name+'.gz'))
            fixed=ROOT/'quality'/a/'fixed-motion'
            copy(fixed/(s+'.txt'),out/'fixed-main-motion.txt')
            copy(fixed/(s+'.config'),out/'fixed-main-motion.config')
        for mode in ['mot','fixed-motion']:
            folder=ROOT/'quality'/a/mode/'trackeval'
            for name in ['pedestrian_summary.txt','pedestrian_detailed.csv']:
                copy(folder/'results/v1'/name,EVIDENCE/'quality-scoring'/mode/a/name)
            for name in ['trackeval.log','run.py']:copy(folder/name,EVIDENCE/'quality-scoring'/mode/a/name)
    for scope,folder in [('model',modelroot),('pipeline',piperoot)]:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ['.json','.csv','.log','.txt','.status']:
                copy(p,EVIDENCE/'performance'/scope/p.relative_to(folder))
    # Preserve one identical raw model output set per arm; repeats have verified hashes.
    first={}
    for r in json.loads((modelroot/'runs.json').read_text()):first.setdefault(r['arm'],Path(r['output']))
    for a,folder in first.items():
        for p in folder.glob('input-*.f32'):zipped(p,EVIDENCE/'model-output-references'/a/(p.name+'.gz'))
    memory={}
    for a,folder in first.items():memory[a]=json.loads((folder/'memory.json').read_text())
    save(EVIDENCE/'model-memory.json',memory)
    # Report all scopes distinctly: model time, detection path, full tracking, queue delivery.
    q=quality['mot'];d=comparisons['finetuned'];m=q['main']['COMBINED'];f=q['finetuned']['COMBINED']
    text=['# TensorRT 三方性能與品質比較（2026-11-11）','',
        f"同條件 TensorRT 下，主方案相對微調 YOLO 的整體 MOTA {m['MOTA']-f['MOTA']:+.4f} pp、HOTA {m['HOTA']-f['HOTA']:+.4f} pp、IDF1 {m['IDF1']-f['IDF1']:+.4f} pp；模型 Graph GPU 平均耗時增加 {d['model_graph_mean_extra_percent']:.2f}%。應依下面完整路徑與資料角色判讀，不能將此整體品質差距視為未見場景的普遍收益。",'',
        '## 給決策者的主要表格','',
        '以下完整追蹤計時使用 MOT17-10 前 140 幀、20 幀暖機、四次 fresh session／480 個量測樣本；CUDA Graph 開啟、G1、candidate／motion／result slots=2／2／4。JPEG 預載，排除檔案讀取、初始化、暖機與寫檔；single 是 enqueue→receive 交付延遲，queued FPS 是整段完成吞吐，兩者不互換。品質為三方實際 TRT 原生全七序列各自 G1 結果。','',
        '| 方案 | F03 Graph GPU mean ms ↓ | 完整 tracking single mean ms ↓ | single p95 ms ↓ | queued FPS ↑ | 七序列 MOTA ↑ | HOTA ↑ | IDF1 ↑ | engine MiB ↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in ARMS:
        text.append(f"| {LABEL[a]} | {mg(a)['gpu_event_single_ms']['mean']:.6f} | {pg(a)['delivery_ms']['mean']:.6f} | {pg(a)['delivery_ms']['p95']:.6f} | {pg(a,mode='queued')['throughput_fps']:.3f} | {q[a]['COMBINED']['MOTA']:.4f} | {q[a]['COMBINED']['HOTA']:.4f} | {q[a]['COMBINED']['IDF1']:.4f} | {artifacts[a]['MiB']:.3f} |")
    text+=['','## 完整路徑的兩個場景','',
        '| 序列 | 路徑 | 方案 | single mean ms | single p95 ms | queued interval ms/frame | queued FPS | queued delivery mean ms | queued delivery p95 ms |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in ['MOT17-10-SDP','MOT17-02-SDP']:
        for path in ['detect','tracking']:
            for a in ARMS:
                single,queued=pg(a,s,path),pg(a,s,path,'queued')
                text.append(f"| {s} | {path} | {LABEL[a]} | {single['delivery_ms']['mean']:.6f} | {single['delivery_ms']['p95']:.6f} | {queued['interval_ms_per_frame']:.6f} | {queued['throughput_fps']:.3f} | {queued['delivery_ms']['mean']:.6f} | {queued['delivery_ms']['p95']:.6f} |")
    text+=['','DetectionSession 目前只支援 ordinary enqueue，因此 detect 表是相同原生 ordinary detection 路徑；tracking 才開啟部署 D/M Graph。這兩條是分別直接量測的範圍，不能以兩個均值相減推算各 stage 的精確耗時。','',
        '## 模型層與原始 one2one 控制','',
        '模型計時包含同一 stream 的 TensorRT enqueue 或 Graph launch，GPU events 與 CPU wall 分列；四張相同真實輸入的 H2D、D2H、檢查、初始化、暖機及磁碟均排除。每組四次 fresh process、每 mode 1,280 筆單次樣本；另外 2,048 次 back-to-back replay 量吞吐。','',
        '| 比較組 | 方案 | mode | GPU mean ms | GPU p95 ms | CPU wall mean ms | queued GPU interval ms |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in model['groups']:
        text.append(f"| {r['comparison']} | {r['arm']} | {r['mode']} | {r['gpu_event_single_ms']['mean']:.6f} | {r['gpu_event_single_ms']['p95']:.6f} | {r['cpu_wall_single_ms']['mean']:.6f} | {r['queued_gpu_interval_ms']:.6f} |")
    text+=['','official-default 是官方原始 one2one/end2end、output0[1,300,6]，engine 內含 decode／TopK，與 raw6 表的輸出終點不同，另外與主方案做平衡對照。沒有把它冒稱成 raw6／同一 F04 的品質等價替代；本次其品質及完整追蹤未重評。','',
        '## TRT 品質與資料角色','',
        '| 序列 | 官方 MOTA | 微調 MOTA | 主方案 MOTA | 官方 HOTA | 微調 HOTA | 主方案 HOTA |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in [*SEQ,'COMBINED']:
        text.append('| '+s+' | '+' | '.join(f"{q[a][s][metric]:.6f}" for metric in ['MOTA','HOTA'] for a in ARMS)+' |')
    text+=['','04／05／09／11／13 用於模型訓練、10 用於選模、02 是 M4 後重用的原 held-out；訓練序列占整體有效 GT 約72%。`quality-subgroups.json` 保留真實 TrackEval 分組聚合；不得以七序列總分代替未見場景的判定。`quality-scores.json` 另保留相同 main-derived G1 的 fixed-motion CPU replay，用於分開看 detector 與 detector-conditioned GMC。','',
        '## Engine、精度與驗證','',
        '- 原始權重、微調 primary EMA 固定；每個 ONNX 的 542／542 initializer 逐 byte 對到來源權重，四張 FP32 PT／ONNX 數值核對通過。相同 SDK 10.16.1.11、batch1、1280×768、FP16 allowed、TF32／BF16／INT8 關閉、opt3、aux0、workspace256MiB；I/O FP32。實際 layers／formats 與 builder readback 已保存，沒有僅憑 FP16 flag 宣稱全部算子為 FP16。',
        '- 主方案維持固定部署 engine，scan plugins 與 I/O FP32。研究 adapter 只讓 YOLO raw6 支援零 engine-plugin-status，仍保留全 input／output 有限值與真實 producer status。四個正常／NaN 定點負控制通過，無合成 scan status 或關閉品質檢查。主方案七序列 SCDR／motion／MOT／packets 重現原部署，三方全部 native 對 CPU 自身 motion 的 MOT＋packets 逐 byte 相同。',
        '- FP16 原定 1% raw relative-L2 screen **FAIL 保留**：官方最大 2.406%、微調最大1.537%；實際 finite／repeat／Graph bitwise 檢查通過。相同官方 ONNX 的 FP32/no-TF32 控制最大6.497e-6，支持差異來自精度／tactic，而非錯載權重或接錯 graph。完整品質以本次 TRT 重新評分，不繼承上一輪 FP32 偵測分數。官方 decoded Top300 亦有集合／順序差異，未宣稱輸出等價。',
        '- `model-memory.json` 的 getDeviceMemorySizeV2 為 execution-context scratch bound，explicit I/O／status allocations 另列；cudaMemGetInfo 與 NVML 是整張 device 的觀察，不能解讀為本行程私有顯存或單幀能耗。Engine 檔案大小不是完整推論顯存。',
        '- 兩輪 raw M/O/F/F/O/M、另兩輪 M/default/default/M；模型共20 fresh processes，完整 detection／tracking 共96 fresh sessions。單一 GPU flock 橫跨兩個 campaign，量測時沒有建置、GPU 診斷或 profiler。未刪除長尾；以 nearest-rank 算分位數。沒有測試攝影機曝光到外部消費者的全系統延遲。',
        '- 本次是本機、固定 GPU／SDK／模型／輸入的研究比較；沒有切換部署、提交或發布，也不是新的 sanitizer／跨 GPU 正式驗收。原有 OPEN／FAIL 保留。','',
        'TRT 檔案實體位置與 hashes 見 [ARTIFACTS.json](ARTIFACTS.json)，數值／建置交付見 [delivery.json](methods/export-tools/delivery.json)，品質、模型與 pipeline 的所有原始精度樣本／配置／hash 收據均在本目錄；大型原始建置與全量 dumps 留在 `build/yolo26m-trt-comparison-20261111/`。']
    (EVIDENCE/'REPORT.md').write_text('\n'.join(text)+'\n')
    save(EVIDENCE/'source-after.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True)})
    entries=[]
    for p in sorted(EVIDENCE.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS':entries.append(f'{sha(p)}  {p.relative_to(EVIDENCE)}')
    (EVIDENCE/'SHA256SUMS').write_text('\n'.join(entries)+'\n')
    print(EVIDENCE/'REPORT.md')

if __name__=='__main__':main()
