"""Render only final, gated primary156 results; never invent missing results."""
ARMS = ['official', 'finetuned', 'main']
LABELS = {'official': '官方 YOLO26m', 'finetuned': '微調 YOLO26m', 'main': '目前 HR Mamba'}
METRICS = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA']


def number(value, signed=False):
    return format(value, '+.6f' if signed else '.6f')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def render(summary, audit, delivery, archive_count):
    if summary['result'] != 'PASS' or audit['result'] != 'PASS':
        raise RuntimeError('report requires final full156 PASS authorities')
    contexts = {(r['kind'], r['group']): r for r in audit['context_comparisons']}
    groups = summary['primary_groups']
    deltas = [g['metrics']['main']['HOTA'] - g['metrics']['finetuned']['HOTA'] for g in groups.values()]
    tol = audit['win_tie_loss_tolerance_pp']
    wins = sum(x > tol for x in deltas)
    ties = sum(abs(x) <= tol for x in deltas)
    losses = sum(x < -tol for x in deltas)
    overall = next(r for r in audit['context_comparisons'] if r['kind'] == 'pooled')
    wtl = overall['per_sequence_win_tie_loss']['HOTA']
    rows = []
    for group, entry in groups.items():
        m = entry['metrics']
        stat = contexts[('primary_group', group)]['per_sequence_win_tie_loss']['HOTA']
        rows.append([group, entry['sequence_count'], entry['frames'],
                     *[number(m[a]['HOTA']) for a in ARMS],
                     number(m['main']['HOTA'] - m['finetuned']['HOTA'], True),
                     f"{stat['wins']}/{stat['ties']}/{stat['losses']}"])
    text = [
        '# YOLO26m／HR Mamba：primary156 TensorRT 品質比較', '',
        f'完整 156 段、5 個資料集、7 個互斥 primary groups 已完成。以目前 HR Mamba 相對微調 YOLO26m 的 HOTA 比較，'
        f'7 組為 {wins} 勝／{ties} 平／{losses} 負；逐段為 {wtl["wins"]} 勝／{wtl["ties"]} 平／{wtl["losses"]} 負。'
        '這是指定已曝露資料池的研究比較，並非 blind test、正式部署驗收或新採用門檻。', '',
        '每個模型完整覆蓋 135,503 frames，合計 468 cells／406,509 frames；21 個 MOT17 cells 從前輪相同原生 TRT 品質實驗逐檔驗證匯入，'
        '其餘 447 cells 本輪重新推論。未訓練、未調參、未更換沿用的三個 TensorRT engines。', '',
        '所有數值以百分點尺度顯示；Δ 為 HR Mamba 減微調 YOLO26m。表格只為閱讀四捨五入，完整精度留在 JSON／CSV。', '',
        '## 七個 primary groups：HOTA', '',
        table(['群組', '段數', 'frames', '官方', '微調', 'HR Mamba', 'Δ HOTA pp', '逐段 HR/微調 勝/平/負'], rows), '',
        '主要解讀保留以上七組。跨資料集 pooled 結果僅為描述性補充，不能取代各組結果；所有合併均使用 frozen TrackEval 的完整 raw metric objects 與 combine_sequences，沒有平均各段分數。', '',
        '## 五資料集補充', '',
    ]
    rows = []
    for dataset, entry in summary['datasets'].items():
        for arm in ARMS:
            rows.append([dataset, entry['sequence_count'], entry['frames'], LABELS[arm],
                         *[number(entry['metrics'][arm][k]) for k in METRICS]])
    text += [table(['資料集', '段數', 'frames', '模型', *METRICS], rows), '',
             '## Descriptive pooled：全部 156 段', '',
             table(['模型', *METRICS], [[LABELS[a], *[number(summary['pooled_all156']['metrics'][a][k]) for k in METRICS]] for a in ARMS]), '',
             '各資料集的 GT 密度與標註語意不同，pooled 值受其支援量影響；不解讀為泛化的單一排名或正式採用結論。', '',
             '## 逐段勝平負與退步', '',
             f'勝平負容差為 {tol:g} pp，僅處理浮點捨入，不代表有意義的品質等價帶。以下均是 HR Mamba 相對微調 YOLO26m。', '']
    rows = []
    for row in audit['context_comparisons']:
        if row['kind'] not in ('primary_group', 'pooled'):
            continue
        values = row['per_sequence_win_tie_loss']
        rows.append([row['group'], *[f"{values[k]['wins']}/{values[k]['ties']}/{values[k]['losses']}" for k in METRICS]])
    text += [table(['範圍', *METRICS], rows), '',
             f"HOTA 退步共 {audit['HOTA_regression_sequence_count']} 段；以下列出最大十個退步（不足十段則全部）。"]
    regressions = audit['worst_ten_HOTA_regressions']
    if regressions:
        rows = [[r['sequence'], r['primary_group'], r['frames'], r['GT_Dets'],
                 *[number(r['delta_pp'][k], True) for k in METRICS]] for r in regressions]
        text += ['', table(['序列', '群組', 'frames', 'GT detections', *['Δ ' + k + ' pp' for k in METRICS]], rows)]
    else:
        text += ['', '本資料池在上述浮點容差下沒有 HOTA 退步段。']
    text += ['', '完整逐段三模型結果見 [per-sequence.csv](metrics/per-sequence.csv)；'
             '完整 main／fine Δ 見 [scoring-audit.per-sequence.csv](audits/scoring-audit.per-sequence.csv)。'
             'DetA／AssA 的差值是描述性分解，並非因果歸因。', '',
             '## 固定方法與資料邊界', '',
             '- 三模型使用同一原生 F01／F02 前處理、F04 後處理與 G1 tracker 路徑，各自產生自己的 detector-conditioned motion；本輪沒有固定 main motion 的額外控制。'
             'tracker.kalman.nsa_enabled=false、tracker.rows.measurement_box=false；CUDA Graph 開啟，candidate／motion／result slots 為 2／2／4。',
             '- 官方與微調 YOLO26m 使用 raw6 one2many head 接共用 F04；不將它宣稱為 Ultralytics 預設 one2one 端到端流程。'
             'main 保持原 producer；研究 adapter 的 YOLO finite/status 合約與既有 main parity 證據見 methods／prior-TRT-characterization。',
             '- MOT15 使用 MOT15 adapter；MOT20 使用 MOT20 class6 distractor 規則；DanceTrack 保留 GT 前七欄並設 class=1、visibility=1；'
             'PP22 僅在有 GT 的原 frame IDs 計分，保留完整 seqLength，不重編 frame。固定 pedestrian、DO_PREPROC=True。',
             '- 全部 156 段已在先前研究曝露，本輪不是新的 blind holdout。依既有使用者政策，SportsMOT 45 段排除在 primary inference／scoring／gate／參數選擇之外。',
             '- 微調模型為 UA2 teacher e4 primary EMA（checkpoint 08be5aba…）；HR Mamba 為 Full25 mixed student e10 primary EMA（53bccffe…），沿用 UA2 teacher。'
             '兩份原始訓練 manifest 的 MOT17 分工均為 04／05／09／11／13 train、10 選模，且各有 MOTSynth 訓練資料；Full25 選定 epoch 的 frame slots 為 MOT17 1536／MOTSynth 512。'
             '這項分工只適用這兩條本地權重 lineage，不用來描述官方預訓練 corpus；本輪未全面稽核官方預訓練資料。詳見 [training-provenance.json](training-provenance.json) 的原始 manifest／selection hashes 與摘錄。',
             '- 原始訓練 manifest 當時將 MOT17-02 記為未開啟 held-out；前輪比較已把它列為 post-M4 重用評估，本輪亦再次使用。'
             '訓練當時的 holdout 宣告不等於目前仍為盲測。不同配方、訓練來源與分數校準也使本比較不能單獨歸因於 Mamba 架構。',
             '- HOTA／DetA／AssA 使用完整 19-alpha raw arrays；IDF1 與 MOTA 使用各自正式 combine_sequences。'
             'MOTA = 100 × [1 − (FP + FN + IDSW)／GT_Dets]；FP／FN／IDSW／GT／frames 原始計數保留於完整 metrics。', '',
             '## FP16 數值 screen 與驗收限制', '']
    for item in delivery['models']:
        if item['arm'] in ('official', 'finetuned'):
            text += [f"- {LABELS[item['arm']]}：既有 1% raw relative-L2 screen 狀態 **{item['trt_vs_pt_diagnostic_status']}**，"
                     f"最大值 {100 * item['trt_vs_pt_max_raw_rel_l2']:.6f}%。本輪保留原 engine 並直接量測其 native TRT 品質，沒有把任務分數當作數值 screen 通過。"]
    text += ['- 前輪 finite／repeat／Graph 檢查與 FP32 ONNX 診斷另見沿用 metadata；數值 FAIL 邊界維持，未升格為正式 session／sanitizer／部署驗收。',
             '- 本輪不測速。capture wall time 含 lock wait，不能當受控 latency／throughput；前輪 TRT 性能研究是分開的實驗。'
             '此次沒有更改預設方案或部署，沒有新增 GPU benchmark、commit 或 push。', '',
             '## 完整性與交付', '',
             'capture-complete、全 JPEG／GT／seqinfo／engine／binary 的 inputs-after、final capture audit 與獨立 scoring audit 均須完整 PASS 後才允許產生本報告。'
             '七組 scoring configs、GT 轉換／keyframe filter、原始 raw components、468 cell receipts、逐段驗證與 logs 均附於本包。', '',
             f'{archive_count} 個 gzip（每 cell 各 mot.txt、packets.bin、motion.scmo）均逐檔解壓後核對原始 SHA256 與 byte count；'
             '來源原檔保留於 build study，不移除或修改。JPEG、weights、ONNX、TensorRT engines 不複製到本包，僅保留其 hash／位置／方法。', '',
             '[OUTPUT-ARCHIVES.json](OUTPUT-ARCHIVES.json) 記錄逐檔壓縮與解壓驗證；[SHA256SUMS](SHA256SUMS) 包含整包所有檔案（manifest 自身除外），'
             '發布前已檢查每項 SHA 與完整檔案集合。打包程式自檢 receipt 留在 build study 的 closeout-tools；此項自檢不稱作獨立封裝審核。', '',
             '最終完整原始分數：[summary.json](metrics/summary.json)；完整 metric objects：[raw-components.json](metrics/raw-components.json)；'
             '獨立審核：[scoring-audit.json](audits/scoring-audit.json)。', '']
    return '\n'.join(text)
