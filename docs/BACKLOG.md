# Saccade v1 後續工作

更新日期：2026-11-05。tracking M0–M4 已完成，部署模型為 HR 1280×768 FP16，**P01 CLOSED（有範圍）**。本頁先列現行結果與剩餘界線，再保留各階段原始契約及驗收紀錄；2026-10-31 的待辦已由後續證據接續。

| 項目 | 現行狀態與入口 |
| --- | --- |
| T01 實驗 | 已結案並封存；[closeout](history/2026-10-31-t01-closeout.md) |
| F01／F02、F03 producer | [前處理使用驗收](../evidence/preprocess/2026-10-31-usage/REPORT.md)／[producer 獨立正式驗收](../evidence/f03-producer/2026-10-31-formal-a9c3d9b/REPORT.md) PASS（受測範圍）；見 §1–§2 的來源與政策 |
| 完整 tracking session | M3 clean-source 正式驗收 PASS；[M3 證據](../evidence/tracking-session/2026-11-03-formal-e021bd1/REPORT.md) |
| HR FP16、M4／P01 | M4 正式驗收與計分完成，P01 CLOSED（有範圍）；[M4 證據](../evidence/m4/2026-11-03-formal-cad5298/REPORT.md) |
| F04 平行 finalize | clean-source 正式驗收 PASS；[第二輪證據](../evidence/f04/2026-11-04-parallel-finalize-e4d45de/REPORT.md) |
| CUDA Graph／tracking 排序 | clean-source 驗收 PASS：步驟 1–3 使用 `4329ed7`，步驟 4–5 使用 `d21a144`；HR 七個序列的 21 個輸出與 M4 逐位元相同；[正式證據](../evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md) |
| F02 tile fusion／F03 N16 與部署特化 | [clean-source 整合補驗](../evidence/f02-f03/2026-11-05-clean-23ade53/REPORT.md) PASS：乾淨 `23ade53`、`v1-motion/1`；CPU 16/16、CUDA 59/59、原 session gate、F02／scan 四 sanitizer、M4 21/21 輸出相同；原 A/B 測時保留開發來源身分 |
| 現行 G1／v1-motion/2 | [正式驗收](../evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2/REPORT.md)的完整 session memcheck 仍 OPEN／NOT A FULL PASS，其餘八步通過；F02／F03 的舊 profile 補驗不關閉此項 |
| 延遲與資源排程 | 開發來源量測已記錄；[資源報告](../evidence/resource-schedule/2026-11-04/REPORT.md)、[排程表](../evidence/resource-schedule/2026-11-04/SCHEDULE.md)。單幀延遲、queued 交付延遲與吞吐分列，不設速度門檻 |
| 即時影像串流（2026-11-05） | [FFmpeg／NVDEC／CUDA 前處理／track_stream](STREAMING.md) 已實作，CLI 預設 GPU 解碼，開發驗證獨立記錄。外部攝影機／RTSP 連線、多來源、重連與可變 dt 不由現有證書涵蓋 |

## 1. F04：先固定觀測政策，再實作

狀態：**clean-source 正式驗收 PASS（受測範圍），獨立結案**。2026-11-04 已接續平行 finalize 與 Graph request 更新；最新驗收見 [CUDA Graph 正式證據](../evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md)，F04 CTest 5/5、37,094 checks、四種 sanitizer 與 15/15 mutations 通過。

以下為 2026-10-31 乾淨 `995bca5` 版本的契約與原始驗收紀錄；當時的分級排序與測時不代表後續改版的效能。

輸入是 F03 的六個 raw cls／ltrb outputs 與 `ImageMapping`；輸出為保留穩定 candidate ID、原始 score／class、原圖 xyxy、kept／suppressed、device count／capacity／status 的 `DetectionCandidates`。依 [DATAFLOW](DATAFLOW.md#4-跨階段資料契約)，suppressed 候選須留給 F04b，suppression 只執行一次。

已依版本化 [`f04-postprocess-fp32/1`](POSTPROCESS_POLICY.md) 實作 [CPU scalar reference 與 CUDA API](POSTPROCESS.md)，沿用 T01 已固定的語義：P3→P5 row-major、stride 8／16／32、anchor `(x+0.5,y+0.5)`、raw ltrb、80 類 sigmoid 後 max／argmax、conf≥0.001、**class 篩選前 top-300**、只留 class 0、greedy NMS IoU>0.5，再映射原圖。保留完整 kept／suppressed pool，不能將此契約縮成 person-only top-k 或只交付 kept boxes。

政策已固定 class／top-k／NMS ties、FP32 計算與閾值邊界、不 clipping、退化／非有限框、capacity／padding／overflow，以及被 NMS suppressed 與低分／非 person／top-k 淘汰的區別。T01 evaluator 的 invalid-topk 只有歷史統計；新增 native 處置不回寫其實驗結果。候選 ID 由篩選前模型序號建立並在 compact 後保留。

完成條件：依 GPU 前固定的 reference／cases／政策，驗證完整輸出與穩定順序、重跑與資源重用、同分與閾值邊界、空候選及明確失敗；保留 F04b 可用的 suppressed pool。正常 runtime 的中途 count／status 留 device，不增加 per-frame postprocess D2H。

當時實作為 **decode 融合壓實＋device 分級 Top-300**：M 是 class／退化篩選前的合格位置數，M≤100 用 compact shared-memory rank，101–512 用單 block radix，M>512 用 tiles 與全域合併；唯一 key 保留原始 ID 決勝。[正式驗收](../evidence/f04/2026-10-31-formal-995bca5/REPORT.md) 的 49 cases、10-phase 重用、完整輸出 bytes、repeat／graph、六份 replay 與 F04 CTest 4/4 通過，正常 suite 共 34,915 checks。四種 sanitizer 各 34,610 checks、0 errors／hazards，四個正控制及 13/13 語義變體均以預期診斷與退出碼偵測；sanitizer callback gates 為 UNEXECUTED，正常 gates 已驗。

先前 [adaptive dirty development 報告](../evidence/f04/2026-10-31-adaptive/REPORT.md) 保留原身分與 attempt 紀錄；三份真實輸入的完整 F04 API 測時為 1.77–1.97× 加速，排除 H2D／D2H 與 CPU reference。正式驗收未重測效能，也未驗完整模型或 tracking 端到端速度。

[原始 rank 版本的 F04 證據](../evidence/f04/2026-10-31/REPORT.md) 保留 31 個固定 cases、正確解析與獨立 historical kept-count 核對的六份 F03 dumps、正常 pending gates、四種 sanitizer、正控制與 10 個 mutations 的歷史結果；當時 `source_dirty=true`、sanitizer callback gates 為 UNEXECUTED，clean-source 正式驗收亦未執行，不能將該 PASS 套用到本次 adaptive 改版。F03 plugin／finite statuses 的 [producer 聚合器](MODEL_PRODUCER.md) 已完成有界 engine→F04 乾淨 source 正式驗收；native 前處理／模型接續已另行實作於 [detection session](DETECTION_SESSION.md)，依其獨立證據判定。

## 2. F03 producer 與 bounded detection session

F03 producer 與獨立 bounded detection session 的 2026-10-31 證據如下。完整 tracking session 已在 2026-11-03 完成 M3，後續 Graph 版本亦已正式驗收；兩者的 source 與驗收身分分開記錄。

[`f03-producer/1`](MODEL_PRODUCER.md) 以三個同 stream kernels 完成全部模型 input／raw finite checks、九個 plugin status 的角色檢查、來源退休及 F04 aggregate 發布；**乾淨 `a9c3d9b` 正式驗收 PASS（受測範圍）**，見 [正式證據](../evidence/f03-producer/2026-10-31-formal-a9c3d9b/REPORT.md)。原 [開發證據](../evidence/f03-producer/2026-10-31/REPORT.md) 保留。Fresh slot 初始化、ready／consumed、identity 與 session 首次失敗屬 caller 責任，不由 primitive 的 PASS 推導。

[F01–F04 bounded async detection session](DETECTION_SESSION.md) 已實作：D 串接原 nvJPEG／F02、完整 F03、producer 與 F04；M 管理候選 consumer、device first-error latch 及固定容量 detection result。CS ready／consumed ring 防止尚未消費即覆寫，pinned HP ring 在 host receive 後重用；支援 frame／epoch identity、overflow／failure／後續 skip、EOF／cancel、D-only tail 的雙 stream drain、checked close 及新 instance 隔離。[開發來源驗證 PASS（受測範圍）](../evidence/detection-session/2026-10-31/REPORT.md)；不能記為 clean-source session 正式結案，也不納入 `a9c3d9b` 的既有 producer 驗收。

本版沿用單 JPEG staging，尚未實作雙 host staging overlap。正常 detection 資料面每幀交付一次完整固定容量 `DetectionSessionResult` D2H，沒有中間 count／status 讀回；這不是 tracking `ResultPacket`，M consumer 也不是 F04b。

F05–F09 的數值與決策規格、參數、容量及已完成的 M0–M4 里程碑見 [TRACKING](TRACKING.md)。

完整 tracking session 已接通 F05、main 的 F06→F08a→F04b→F07／F08b 及最後 F09，依 [ASYNC_CONTROL](ASYNC_CONTROL.md)、[DATAFLOW](DATAFLOW.md#6-gpu-所有權與同步依賴) 與 [ResultPacket](RESULT_PACKET.md) 管理 MotionImage／prior／tracker state 的 owner／ready／last-use／consumed。候選須活到 F04b 最後消費，F04b 使用本幀 GMC＋predict prior，continuation-only 候選維持 `can_birth=false`。

真實影像 serial／queued、跨幀／跨 session 隔離、pending consumer 下的重用、frame／epoch／版本一致性、錯誤／overflow／缺幀／drain／checked close，以及 tracking state／MOT bytes 的獨立對照，已有 [M3 正式證據](../evidence/tracking-session/2026-11-03-formal-e021bd1/REPORT.md)。Graph 改版的 CPU／ordinary serial／graph pipelined packet bytes、HR 路徑、sanitizer 範圍與七序列輸出重驗見 [2026-11-04 正式證據](../evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md)；不能只由前處理 usage 或 bounded detection 驗證推導。

## 3. P01：依原 readiness 缺件整體判定

**P01 CLOSED（有範圍），2026-11-03 完成。** 原本 2026-10-31 的 OPEN 狀態已由 M3／M4 的獨立證據接續。完整 tracking session 在 M3 完成（[TRACKING §14](TRACKING.md)），部署模型改為 HR FP16（§15–§16）。M4 以部署中的 HR FP16 模型依 [TRACKING §17.4](TRACKING.md) 判定，[正式證據](../evidence/m4/2026-11-03-formal-cad5298/REPORT.md)：

- 身分鏈：checkpoint、ONNX、bundle、engine 與 runtime 一致。
- engine 精度與 shape binding。
- TensorRT 與 cuFFT 的 stream 行為。
- MOT17 七個序列的偵測與追蹤指標（只記錄）。

Head／F03（TensorRT 內部 kernel）的 racecheck／synccheck 仍為 UNEXECUTED，與其他不涵蓋項目一起明列在 §17.4。

2026-11-04 的 Graph／排序改版保留相同部署模型與 tracker 設定，HR 七個序列的 SCDR、MOT 與 offline MOT 共 21 個輸出均與 M4 相同；此次沒有重新計分或讀取 `gt/`。MOT17-02 已於 M4 開啟，後續更改模型、tracker 或參數的結果須標為「M4 之後」，不再算獨立 eval。

## 4. 尚未完成與適用界線

| 項目 | 現行界線與後續依據 |
| --- | --- |
| TensorRT 內部 racecheck／synccheck、完整 session initcheck | 仍 UNEXECUTED；完整 session memcheck 與 v1 kernel 的 scoped racecheck／synccheck 已有正式證據，不能擴成函式庫內部全通過。nvJPEG initcheck 依使用政策獨立診斷 |
| 完整 HR 自訂路徑 racecheck | [資源報告 §8](../evidence/resource-schedule/2026-11-04/REPORT.md#8-無競態的條件與本次驗證) 的開發嘗試 OOM，另一次有界 T01 session 嘗試 timeout；後續正式驗收完成的是 tracking／motion／session-control 範圍，未補成完整 HR 自訂路徑驗收 |
| 雙 host staging 與重疊配置 | 尚未實作雙 STAGE。depth 1／2／4 的交付延遲觀測在 [資源報告](../evidence/resource-schedule/2026-11-04/REPORT.md)，depth 2 未改為預設；新增重疊或變更 buffers／consumed 時須重驗所有權與完整 bytes |
| P07 支援與恢復 | 跨 GPU、driver、TensorRT 版本的支援矩陣及 GPU fatal 後恢復未由目前單機證據涵蓋；依 [DATAFLOW §9](DATAFLOW.md#9-重抽象差異與待補規格) 與 M4 不涵蓋項目接續 |
| P08 設定與行為驗證工具 | 現有 tracking lifecycle、packet／MOT 比較與有限協定模型已驗；完整 resolved schema／覆寫優先序、設定 hash 與 bridge 事件模型等其餘工具仍依 [DATAFLOW §4.4、§8.1、§9.2](DATAFLOW.md) 判定，不由 M4 一併關閉 |
| 輸出耐久性 | checked close 與一般檔案輸出已有各自測試；額外 fsync／durability 及異常後部分檔案處置不由現有速度或 bytes 比對證明，仍依 [DATAFLOW P07](DATAFLOW.md#92-真正待補齊的工程規格) 判定 |
| 效能與品質 | 資源／延遲報告是指定來源、SDK、輸入與設定的量測；不推導其他環境的速度保證。M4 品質只記錄，後續調整仍只用 val |

## nvJPEG／NVJPG 診斷：目前不主動續查

原 initcheck FAIL、Bug 7021871、逆向／trace／annotation／poison 與所有失敗紀錄保留；依 [使用政策](PREPROCESS_USAGE_POLICY.md)，正常使用輸出等價才是目前驗收目標，不要求 NVIDIA 分類或消除既知報告。本診斷不作上述使用結案的前置工作。

只有新增輸出不符、重跑／重用漂移、API／同步失敗、新環境無法通過輸出驗證，或 NVIDIA 提供可行動的新資訊時，才按具體訊號建立新的診斷項。既有 annotation 不進正常 runtime，也不把被排除的檢查改稱原路徑 PASS。
