# Saccade v1 目前狀態與文件狀態

更新日期：2026-11-04。依目前 checkout 的文件、正式報告與驗收腳本整理；最新 F02／F03 加速的完整 MOT 開發重測與原 clean-source 正式證書分列。完整文件分組見 [README](../README.md#文件分組)，接續工作見 [BACKLOG](BACKLOG.md)。

## 目前部署與最新判定

目前路徑為 **JPEG sequence → F01／F02 → F03／F04 → F05–F09 tracking → MOT**；部署模型為 **HR 1280×768 FP16**，scan plugins 與模型 input／raw outputs 維持 FP32。部署 engine 以 SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876` 固定；來源、精度、shape binding 與支援邊界見 [TRACKING §15–§17](TRACKING.md#15-d1hr-偵測器部署2026-11-03-使用者核准)。

**M0–M4 完成，P01 CLOSED（有範圍）**。P01 的正式判定由 [M4 報告](../evidence/m4/2026-11-03-formal-cad5298/REPORT.md) 與 [TRACKING §17.4](TRACKING.md#174-p01-判定標準) 定義，不表示任意模型、GPU 或 SDK 的一般支援。

CUDA Graph 預設啟用，D graph 包含 F03→producer→F04，M graph 包含候選檢查→F04b→關聯更新→commit→packet；前處理、motion／GMC、跨 stream events 與結果 D2H 仍由 session 排程。[最新 clean-source 正式驗收](../evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md) 全部 PASS：

| 項目 | 受測來源與結果 |
| --- | --- |
| CPU、F04、語義變體 | 乾淨 `4329ed7`；CPU 16/16、F04 CTest 5/5、mutations 15/15；F04 四種 sanitizer 與正控制通過 |
| TrackingSession | 乾淨 `d21a144`；CPU 16/16、CUDA 59/59，七序列 CPU／ordinary serial／graph pipelined packet bytes 相同，含 HR 路徑及報告所列 sanitizer 範圍 |
| HR 七序列回歸 | 乾淨 `d21a144`；身分鏈、offline／sequence 比對及 stream 守門通過；7 個 SCDR、7 個 MOT、7 個 offline MOT 與 M4 **21/21 相同**；未讀 GT、未重新計分 |
| Stream 位置 | node／graph 兩次 capture 配對 CUDA Graph launch；8 個負控制均被偵測，規則見 [TRACKING §17.3](TRACKING.md#173-正式執行) |

各步的受測 source 分開記錄；後續開發來源重測不改寫這些正式證書。

後續 [HR F02 tile fusion](../evidence/f02-fusion/2026-11-04/REPORT.md) 為開發來源改版，僅分派 1920×1080→1280×768；128 組舊／新 bits、4 張 HR Python references、60/60 CTest 與新 kernel 的有範圍 sanitizer 通過。這份工作樹尚未執行新的 clean-source 正式 gate，不改寫上表的證書來源。

再後續的 [F03 N16 特化](../evidence/f03-scan/2026-11-04/REPORT.md) 保留既有 Graph；scan 精確比較、四張完整模型輸出、16 sessions 的 140 幀完整 packets、此 build 的 49/49 CTest 與 scan 四種 sanitizer 通過。該輪為開發來源，未執行新的 clean-source 正式 gate 或品質重評。

上述 F02 fusion、F03 N16 與部署 shape／128-thread 特化已一併重新建置，完成[完整 MOT 開發測試](../evidence/mot/2026-11-04-full-development/REPORT.md)：CPU 16/16、CUDA 59/59、報告所列 17 次 sanitizer 與四個正控制、七序列 offline／native MOT 及 node／graph stream 守門均通過。HR 七個序列共 **5,316 幀**，7 個 SCDR、7 個 MOT、7 個 offline MOT 與 M4 **21/21 逐位元相同**；重新讀 GT 計分後品質指標不變。吞吐為 **195.85 FPS**（加權 5.10608 ms／幀，包含 JPEG 讀檔及 pipeline，排除 engine 初始化與輸出寫檔）。本次保留 `source_dirty=true`、`formal_acceptance=false`，MOT17-02 記為 **M4 之後**的重跑，不作新的獨立 eval 或 clean-source 正式證書。

## 各階段狀態

| 階段 | 狀態與依據 | 範圍與接續 |
| --- | --- | --- |
| 協定／合成 CUDA | [協定模型](PROTOCOL_MODEL.md)、[CUDA 骨架](CUDA_SYNTHETIC.md) 已有有限驗證 | 合成 profile 與真實 tracking 的證據分開 |
| S01–S04 scan／head | [分階段判定](MODEL_FEASIBILITY.md#3-s01s04-的接續條件)、[S04 正式證據](../evidence/s04/2026-10-30/REPORT.md) PASS（各自受測範圍） | 原 FP32／640 候選；不可直接作 HR FP16 的品質判定 |
| T01 實驗 | [2026-10-31 結案與封存](history/2026-10-31-t01-closeout.md)，固定 tag `t01-experiment-final-2026-10-31`／`a47fb16` | 原 checkpoint、選模與封存身分保留；目前部署已換 HR 模型 |
| T01 backbone／F03 | [正式證據](../evidence/t01/2026-10-30/f03/REPORT.md) PASS，乾淨 `4b52b2f` | 六個固定輸入的 FP32 驗收；後續 FP16 模式見 [MODEL_TRT](MODEL_TRT.md) |
| F01／F02 | [使用驗收](../evidence/preprocess/2026-10-31-usage/REPORT.md) PASS（固定輸入與使用範圍） | 原 7 張 JPEG、4 sessions／208 幀；nvJPEG initcheck 診斷獨立保留 |
| F03 producer | [正式驗收](../evidence/f03-producer/2026-10-31-formal-a9c3d9b/REPORT.md) PASS，乾淨 `a9c3d9b` | 原 primitive 驗收與後續 HR／session 的證據分列 |
| F04 | [原 adaptive 正式驗收](../evidence/f04/2026-10-31-formal-995bca5/REPORT.md)、[平行 finalize 驗收](../evidence/f04/2026-11-04-parallel-finalize-e4d45de/REPORT.md) 及最新 Graph gate PASS | 保留版本化政策、kept／suppressed、穩定 ID 與完整輸出 ABI；各輪結果見 [POSTPROCESS §5](POSTPROCESS.md#5-平行-finalize2026-11-04-實作前固定) |
| DetectionSession | [2026-10-31 開發來源驗證](../evidence/detection-session/2026-10-31/REPORT.md) PASS（受測範圍） | 獨立 F01–F04 API；不將後續 TrackingSession 正式驗收移作它的 clean-source 證書 |
| TrackingSession／M3 | [乾淨 `e021bd1` 正式驗收](../evidence/tracking-session/2026-11-03-formal-e021bd1/REPORT.md) PASS | 完整 F05–F09、CPU／GPU packets、MOT、生命週期與有範圍 sanitizer；Graph／排序改版另由最新 gate 驗證 |
| HR／M4／P01 | [乾淨 `cad5298` 正式評估](../evidence/m4/2026-11-03-formal-cad5298/REPORT.md) 完成，P01 CLOSED（有範圍） | MOT17-02 HOTA 40.6、AP50 72.91%；只記錄品質，不設品質門檻；最新 gate 只比對輸出 |

## 效能報告的狀態

| 報告 | 已觀察結果 | 判讀限制 |
| --- | --- | --- |
| [CUDA Graph 開發量測](../evidence/cuda-graph/2026-11-04/REPORT.md) | 單幀平均 8.59→7.06 ms，p95 9.08→7.44 ms | 開發來源的 ABBA 量測；首幀 setup 與 steady-state 分列 |
| [Tracking 排序量測](../evidence/tracking-sort/2026-11-04/REPORT.md) | 固定 F04b replay 143.57→102.81 µs；完整 tracking 7.037→6.978 ms | 吞吐幾乎不變，不宣稱穩定端到端加速 |
| [硬體資源／排程](../evidence/resource-schedule/2026-11-04/REPORT.md) | Graph 單幀平均 6.988 ms；queued wall time/frame 5.412 ms，約 184.8 FPS | 後者是吞吐換算；預設 4 格 ring 的另次交付延遲平均 21.217 ms。2 格 ring 的單次觀測為 10.740 ms，不能混作同一量測 |
| [HR F02 tile fusion](../evidence/f02-fusion/2026-11-04/REPORT.md) | 隔離 F02 interval 149.06→89.76 µs（−39.8%）；queued 5.367→5.312 ms/frame | 同起點兩輪 ABBA、32 sessions／140 幀完整 packets 相同；single p50 7.010→6.905 ms，mean 受一筆 102.958 ms baseline 長尾影響。workspace 配置仍保留 |
| [F03 N16 kernel 特化](../evidence/f03-scan/2026-11-04/REPORT.md) | scan API Graph 1083.52→770.103 µs（−28.9%）；完整 F03 Graph 5.051→4.762 ms；tracking 單幀平均 7.013→6.647 ms，queued 5.361→5.068 ms/frame | 同起點 ABBA，原部署 engine；四張模型輸出與 16 sessions／140 幀完整 packets 相同。先測的 broadcast 較慢未採用，正式 gate／品質重評未執行 |
| [llama.cpp 啟發的 F03 部署特化](../evidence/f03-llama/2026-11-04/REPORT.md) | 在既有N16之上，shape＋block128使完整F03 Graph兩輪改善1.15%／1.64%；四張輸出與statuses相同 | 開發來源；scan suite與四種sanitizer通過，未跑新正式gate／tracking重評。raw load＋shuffle與B／C shared copy使完整模型變慢，保留負控制及NCU counters |
| [F03 backbone／head 瓶頸分析](../evidence/f03-backbone-head/2026-11-04/REPORT.md) | 20次Graph／6040 kernel calls完整對映：backbone／neck占kernel duration加總66.12%，conv／GEMM占63.73%，scan core占15.63% | 本輪沒有可採用的新加速。Cache收益確認輪反轉；causal直接對現行engine只差0.014%，重建FP16 engines均非bitwise替代品；開發來源，未跑新正式gate／品質重評 |
| [加速後完整 MOT 開發測試](../evidence/mot/2026-11-04-full-development/REPORT.md) | 七序列共 5,316 幀，吞吐 195.85 FPS；M4 輸出 21/21 相同，品質指標不變 | 每序列單次完整執行；吞吐不當作單幀交付延遲，與歷史測時未做 ABBA。未提交來源，MOT17-02 為 M4 之後重跑；sanitizer 範圍依報告 |

這些報告保留各自的開發來源身分、樣本與失敗嘗試。先前 Graph／排序的 clean-source gate 補上對應來源的正確性與七序列回歸，不改寫原測時身分或取代其量測條件；後續完整 MOT 開發重測補上 F02／F03 加速的全序列回歸與計分，不改寫各輪原始量測或正式證書。

## 未涵蓋與未完成邊界

- **完整函式庫插樁**：TensorRT 內部 kernels 的 racecheck／synccheck 仍未執行；完整 session initcheck 受 nvJPEG 既知診斷影響。已通過的 session racecheck／synccheck 限 T01 前 10 幀的 tracking／motion／session-control kernels，不等於完整 HR／函式庫路徑驗收。
- **完整 HR racecheck**：[資源報告](../evidence/resource-schedule/2026-11-04/REPORT.md#8-無競態的條件與本次驗證) 的 OOM／timeout 嘗試仍為未完成；較後的有範圍 session PASS 不覆蓋這些失敗嘗試。
- **nvJPEG**：F01 initcheck FAIL／原因 OPEN 保留；[使用政策](PREPROCESS_USAGE_POLICY.md) 與 [診斷紀錄](PREPROCESS.md#6-逆向與初始化診斷的證明邊界) 分開判定，不將介入後零報告當作原 runtime 修復。
- **部署與支援**：HR FP16 engine 的 build provenance、HR deploy manifest 與 native schemas 1–3 的邊界見 [TRACKING §17.4](TRACKING.md#174-p01-判定標準)。跨 GPU／SDK 支援、GPU fatal 後恢復、額外 durability 與完整 P08 工具仍依 [BACKLOG](BACKLOG.md) 及 [DATAFLOW §9.2](DATAFLOW.md#92-真正待補齊的工程規格) 各自判定。
- **排程擴充**：雙 host staging／完整 host decode overlap 尚未實作；多 caller、新 streams、shape／buffer 或 slot 回收邊改動，需更新 owner／last-use 及對應驗證。當前安全使用條件見 [ownership](../evidence/resource-schedule/2026-11-04/ownership.csv) 與 [安全審查](../evidence/resource-schedule/2026-11-04/safety-review.md)。

## 文件狀態與維護方式

| 類別 | 文件入口 | 維護規則 |
| --- | --- | --- |
| 現行導覽與狀態 | [README](../README.md)、本頁、[BACKLOG](BACKLOG.md) | 隨新增證據更新目前結論、日期與來源；驗收範圍保留 |
| 架構／API／操作 | README 的架構、runtime、模型分組 | 保留契約與重跑入口；早期缺件標明原日期，再連目前部署與對應正式報告 |
| 固定政策 | [HEAD](HEAD_POLICY.md)、[BACKBONE](BACKBONE_POLICY.md)、[PREPROCESS](PREPROCESS_POLICY.md)、[USAGE](PREPROCESS_USAGE_POLICY.md)、[POSTPROCESS](POSTPROCESS_POLICY.md) | 原版本、方法與門檻不因狀態整理變更 |
| 歷史規格與實驗 | [舊訓練流程](TRAINING_WORKFLOW.md)、[T01](T01_TRAINING.md)、[結案報告](history/2026-10-31-t01-closeout.md) | 原選模、checkpoint、實驗與封存結果保留；不當作目前 HR 狀態 |
| Evidence／archive | [README 證據分組](../README.md#驗收與效能證據)、`evidence/`、`docs/history/` | 原路徑、報告、FAIL／UNEXECUTED、hash 與 custody 保留；後續結果另建並由入口連接 |

狀態衝突時，先核對對應階段的最新正式證據、受測 source 與 profile；早期文件的「當時 OPEN」是歷史紀錄，不是目前待辦。文件檢查通過只表示索引與連結一致，不代表重跑 GPU 或品質驗收。
