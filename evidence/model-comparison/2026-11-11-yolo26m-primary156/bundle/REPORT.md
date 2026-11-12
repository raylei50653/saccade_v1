# YOLO26m／HR Mamba：primary156 TensorRT 品質比較

完整 156 段、5 個資料集、7 個互斥 primary groups 已完成。以目前 HR Mamba 相對微調 YOLO26m 的 HOTA 比較，7 組為 3 勝／0 平／4 負；逐段為 55 勝／0 平／101 負。這是指定已曝露資料池的研究比較，並非 blind test、正式部署驗收或新採用門檻。

每個模型完整覆蓋 135,503 frames，合計 468 cells／406,509 frames；21 個 MOT17 cells 從前輪相同原生 TRT 品質實驗逐檔驗證匯入，其餘 447 cells 本輪重新推論。未訓練、未調參、未更換沿用的三個 TensorRT engines。

所有數值以百分點尺度顯示；Δ 為 HR Mamba 減微調 YOLO26m。表格只為閱讀四捨五入，完整精度留在 JSON／CSV。

## 七個 primary groups：HOTA

| 群組 | 段數 | frames | 官方 | 微調 | HR Mamba | Δ HOTA pp | 逐段 HR/微調 勝/平/負 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DanceTrack/all | 65 | 67304 | 35.869190 | 36.245102 | 35.701698 | -0.543404 | 29/0/36 |
| MOT15/moving | 5 | 3185 | 40.244344 | 44.184075 | 43.334649 | -0.849426 | 1/0/4 |
| MOT15/other | 6 | 2315 | 47.116369 | 45.783215 | 47.275060 | +1.491844 | 1/0/5 |
| MOT17/all | 7 | 5316 | 46.010671 | 61.030214 | 63.082927 | +2.052713 | 4/0/3 |
| MOT20/all | 4 | 8931 | 16.891659 | 29.647488 | 37.389975 | +7.742487 | 4/0/0 |
| PP22/moving | 9 | 4004 | 45.152151 | 44.693599 | 42.606932 | -2.086667 | 2/0/7 |
| PP22/static | 60 | 44448 | 43.527065 | 43.546008 | 41.581049 | -1.964960 | 14/0/46 |

主要解讀保留以上七組。跨資料集 pooled 結果僅為描述性補充，不能取代各組結果；所有合併均使用 frozen TrackEval 的完整 raw metric objects 與 combine_sequences，沒有平均各段分數。

## 五資料集補充

| 資料集 | 段數 | frames | 模型 | HOTA | DetA | AssA | IDF1 | MOTA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DanceTrack | 65 | 67304 | 官方 YOLO26m | 35.869190 | 60.478294 | 21.514681 | 36.674792 | 74.087668 |
| DanceTrack | 65 | 67304 | 微調 YOLO26m | 36.245102 | 61.007502 | 21.774742 | 37.519811 | 76.593947 |
| DanceTrack | 65 | 67304 | 目前 HR Mamba | 35.701698 | 59.366768 | 21.701642 | 37.430759 | 73.995868 |
| MOT15 | 11 | 5500 | 官方 YOLO26m | 43.277301 | 40.579182 | 46.501729 | 55.483402 | 22.676356 |
| MOT15 | 11 | 5500 | 微調 YOLO26m | 44.901250 | 41.253039 | 49.103392 | 56.521654 | 19.776970 |
| MOT15 | 11 | 5500 | 目前 HR Mamba | 45.053982 | 41.217971 | 49.556598 | 56.815762 | 19.178048 |
| MOT17 | 7 | 5316 | 官方 YOLO26m | 46.010671 | 42.637510 | 50.107045 | 55.596889 | 45.275475 |
| MOT17 | 7 | 5316 | 微調 YOLO26m | 61.030214 | 63.036248 | 59.669964 | 71.328692 | 73.291361 |
| MOT17 | 7 | 5316 | 目前 HR Mamba | 63.082927 | 66.295312 | 60.522542 | 71.527824 | 76.215749 |
| MOT20 | 4 | 8931 | 官方 YOLO26m | 16.891659 | 8.196064 | 34.995269 | 15.976078 | 9.598154 |
| MOT20 | 4 | 8931 | 微調 YOLO26m | 29.647488 | 27.799596 | 32.076483 | 41.115832 | 36.965435 |
| MOT20 | 4 | 8931 | 目前 HR Mamba | 37.389975 | 41.121448 | 34.505313 | 51.204179 | 54.961423 |
| PP22 | 69 | 48452 | 官方 YOLO26m | 43.637621 | 45.211036 | 42.521164 | 48.216047 | 47.942301 |
| PP22 | 69 | 48452 | 微調 YOLO26m | 43.622624 | 45.662920 | 42.239533 | 50.268885 | 43.333179 |
| PP22 | 69 | 48452 | 目前 HR Mamba | 41.648654 | 44.103234 | 39.914077 | 47.947344 | 39.137815 |

## Descriptive pooled：全部 156 段

| 模型 | HOTA | DetA | AssA | IDF1 | MOTA |
| --- | --- | --- | --- | --- | --- |
| 官方 YOLO26m | 29.913422 | 30.240969 | 29.908210 | 30.985066 | 33.437191 |
| 微調 YOLO26m | 36.230272 | 41.655000 | 32.233126 | 43.146249 | 50.289601 |
| 目前 HR Mamba | 39.722313 | 48.033722 | 34.095971 | 47.804945 | 59.253468 |

各資料集的 GT 密度與標註語意不同，pooled 值受其支援量影響；不解讀為泛化的單一排名或正式採用結論。

## 逐段勝平負與退步

勝平負容差為 1e-09 pp，僅處理浮點捨入，不代表有意義的品質等價帶。以下均是 HR Mamba 相對微調 YOLO26m。

| 範圍 | HOTA | DetA | AssA | IDF1 | MOTA |
| --- | --- | --- | --- | --- | --- |
| DanceTrack/all | 29/0/36 | 18/0/47 | 30/0/35 | 33/0/32 | 17/0/48 |
| MOT15/moving | 1/0/4 | 1/0/4 | 2/0/3 | 0/0/5 | 2/0/3 |
| MOT15/other | 1/0/5 | 3/0/3 | 2/0/4 | 2/0/4 | 4/0/2 |
| MOT17/all | 4/0/3 | 5/0/2 | 4/0/3 | 2/0/5 | 4/0/3 |
| MOT20/all | 4/0/0 | 4/0/0 | 3/0/1 | 3/0/1 | 3/0/1 |
| PP22/moving | 2/0/7 | 3/0/6 | 3/0/6 | 2/0/7 | 2/0/7 |
| PP22/static | 14/0/46 | 15/0/45 | 17/0/43 | 14/0/46 | 13/1/46 |
| all156 | 55/0/101 | 49/0/107 | 61/0/95 | 56/0/100 | 45/1/110 |

HOTA 退步共 101 段；以下列出最大十個退步（不足十段則全部）。

| 序列 | 群組 | frames | GT detections | Δ HOTA pp | Δ DetA pp | Δ AssA pp | Δ IDF1 pp | Δ MOTA pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0024 | DanceTrack/all | 763 | 4512 | -10.715380 | +2.308540 | -14.910729 | -18.784156 | -2.437943 |
| uid_vid_00087.mp4 | PP22/static | 240 | 203 | -9.897948 | -7.330571 | -12.831131 | -9.412226 | -7.389163 |
| dancetrack0007 | DanceTrack/all | 1203 | 9471 | -8.571400 | +1.007308 | -10.338606 | -7.381593 | -0.221729 |
| TUD-Campus | MOT15/other | 71 | 359 | -8.493830 | -2.913492 | -14.716193 | -15.115093 | -13.370474 |
| dancetrack0032 | DanceTrack/all | 604 | 3360 | -7.444665 | -0.744636 | -9.232512 | -12.809628 | -2.619048 |
| uid_vid_00018.mp4 | PP22/static | 409 | 1398 | -7.434699 | -4.724306 | -11.020459 | -16.004308 | -20.028612 |
| uid_vid_00019.mp4 | PP22/static | 353 | 7141 | -6.963412 | -8.216168 | -5.260380 | -8.260844 | -11.286935 |
| dancetrack0082 | DanceTrack/all | 603 | 12370 | -5.792482 | -1.896552 | -7.234067 | -6.419711 | -2.166532 |
| uid_vid_00079.mp4 | PP22/static | 240 | 704 | -5.718048 | -3.124641 | -8.733884 | -6.243274 | -3.835227 |
| dancetrack0043 | DanceTrack/all | 183 | 1675 | -5.705831 | +1.009529 | -11.271968 | -9.192707 | +1.552239 |

完整逐段三模型結果見 [per-sequence.csv](metrics/per-sequence.csv)；完整 main／fine Δ 見 [scoring-audit.per-sequence.csv](audits/scoring-audit.per-sequence.csv)。DetA／AssA 的差值是描述性分解，並非因果歸因。

## 固定方法與資料邊界

- 三模型使用同一原生 F01／F02 前處理、F04 後處理與 G1 tracker 路徑，各自產生自己的 detector-conditioned motion；本輪沒有固定 main motion 的額外控制。tracker.kalman.nsa_enabled=false、tracker.rows.measurement_box=false；CUDA Graph 開啟，candidate／motion／result slots 為 2／2／4。
- 官方與微調 YOLO26m 使用 raw6 one2many head 接共用 F04；不將它宣稱為 Ultralytics 預設 one2one 端到端流程。main 保持原 producer；研究 adapter 的 YOLO finite/status 合約與既有 main parity 證據見 methods／prior-TRT-characterization。
- MOT15 使用 MOT15 adapter；MOT20 使用 MOT20 class6 distractor 規則；DanceTrack 保留 GT 前七欄並設 class=1、visibility=1；PP22 僅在有 GT 的原 frame IDs 計分，保留完整 seqLength，不重編 frame。固定 pedestrian、DO_PREPROC=True。
- 全部 156 段已在先前研究曝露，本輪不是新的 blind holdout。依既有使用者政策，SportsMOT 45 段排除在 primary inference／scoring／gate／參數選擇之外。
- 微調模型為 UA2 teacher e4 primary EMA（checkpoint 08be5aba…）；HR Mamba 為 Full25 mixed student e10 primary EMA（53bccffe…），沿用 UA2 teacher。兩份原始訓練 manifest 的 MOT17 分工均為 04／05／09／11／13 train、10 選模，且各有 MOTSynth 訓練資料；Full25 選定 epoch 的 frame slots 為 MOT17 1536／MOTSynth 512。這項分工只適用這兩條本地權重 lineage，不用來描述官方預訓練 corpus；本輪未全面稽核官方預訓練資料。詳見 [training-provenance.json](training-provenance.json) 的原始 manifest／selection hashes 與摘錄。
- 原始訓練 manifest 當時將 MOT17-02 記為未開啟 held-out；前輪比較已把它列為 post-M4 重用評估，本輪亦再次使用。訓練當時的 holdout 宣告不等於目前仍為盲測。不同配方、訓練來源與分數校準也使本比較不能單獨歸因於 Mamba 架構。
- HOTA／DetA／AssA 使用完整 19-alpha raw arrays；IDF1 與 MOTA 使用各自正式 combine_sequences。MOTA = 100 × [1 − (FP + FN + IDSW)／GT_Dets]；FP／FN／IDSW／GT／frames 原始計數保留於完整 metrics。

## FP16 數值 screen 與驗收限制

- 官方 YOLO26m：既有 1% raw relative-L2 screen 狀態 **FAIL**，最大值 2.405992%。本輪保留原 engine 並直接量測其 native TRT 品質，沒有把任務分數當作數值 screen 通過。
- 微調 YOLO26m：既有 1% raw relative-L2 screen 狀態 **FAIL**，最大值 1.537103%。本輪保留原 engine 並直接量測其 native TRT 品質，沒有把任務分數當作數值 screen 通過。
- 前輪 finite／repeat／Graph 檢查與 FP32 ONNX 診斷另見沿用 metadata；數值 FAIL 邊界維持，未升格為正式 session／sanitizer／部署驗收。
- 本輪不測速。capture wall time 含 lock wait，不能當受控 latency／throughput；前輪 TRT 性能研究是分開的實驗。此次沒有更改預設方案或部署，沒有新增 GPU benchmark、commit 或 push。

## 完整性與交付

capture-complete、全 JPEG／GT／seqinfo／engine／binary 的 inputs-after、final capture audit 與獨立 scoring audit 均須完整 PASS 後才允許產生本報告。七組 scoring configs、GT 轉換／keyframe filter、原始 raw components、468 cell receipts、逐段驗證與 logs 均附於本包。

1404 個 gzip（每 cell 各 mot.txt、packets.bin、motion.scmo）均逐檔解壓後核對原始 SHA256 與 byte count；來源原檔保留於 build study，不移除或修改。JPEG、weights、ONNX、TensorRT engines 不複製到本包，僅保留其 hash／位置／方法。

[OUTPUT-ARCHIVES.json](OUTPUT-ARCHIVES.json) 記錄逐檔壓縮與解壓驗證；[SHA256SUMS](SHA256SUMS) 包含整包所有檔案（manifest 自身除外），發布前已檢查每項 SHA 與完整檔案集合。打包程式自檢 receipt 留在 build study 的 closeout-tools；此項自檢不稱作獨立封裝審核。

最終完整原始分數：[summary.json](metrics/summary.json)；完整 metric objects：[raw-components.json](metrics/raw-components.json)；獨立審核：[scoring-audit.json](audits/scoring-audit.json)。
