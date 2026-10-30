# P01 模型 artifact 與 native scan 可行性

P01 保持 OPEN。P01-A 將 manifest 工具、native SDK 路徑、真實 artifact readiness 分別驗收，不能互相替代。P01-A 調查日期：2026-10-28；head 契約查核與 S01：2026-10-29。

2026-10-29 使用者決策將 v1 模型基線改為 YOLO26m backbone／neck＋原版 m T3→T1 Mamba-FPN head、單幀 T=1。[SCAN_CONTRACT](SCAN_CONTRACT.md) 已由該 head 的實作、checkpoint 容器與部署 ONNX 核實 scan 契約；**S01 PASS，範圍限該文件 §3–§5 的 `SelectiveScan` 算子與 Cross-Scan**（§3 表後說明）。**S02 PASS，範圍限 [SCAN_CUDA](SCAN_CUDA.md) 的 CUDA FP32 forward**（§3 表後說明，[S02 evidence](../evidence/s02/2026-10-29/REPORT.md)）。證據見 [S01 evidence](../evidence/s01/2026-10-29/REPORT.md)；同日稍早的 s／D1 查核（[scan-contract evidence](../evidence/scan-contract/2026-10-29/REPORT.md)）保留為歷史。

[DATAFLOW §1.1 B03](DATAFLOW.md#11-由舊架構收斂的-v1-設計基線) 現為 YOLO26m＋原版 m head、640×640、P3／P4／P5、單幀推論、C++／CUDA。下面 §1–§2 的 P01-A 於 2026-10-28 以當時的 s 目標完成（從驗收基線 `437c99edbae1a34854aa6966dc210b5053069929` fetch 後接續），其結論保留原適用範圍，不改寫成 m 驗收；m 的 artifact 查核見 [readiness §7](MODEL_ARTIFACT_READINESS.md#7-2026-10-29-原版-m-head-的-artifact-對應)。沒有搬用舊程式、權重或實驗結論。

## 1. P01-A 的交付與驗收邊界

| 項目 | 實作與查核入口 | 驗收含義 |
| --- | --- | --- |
| Manifest 工具 | [schema 與 CLI](MODEL_MANIFEST.md)，`model_manifest`、CPU 測試及明標用途的 fixtures | 驗證宣告與檔案完整性；不自行保證模型內容符合宣告 |
| Native SDK 路徑 | [SDK pin、環境、命令與比較契約](NATIVE_TENSORRT.md)，可選 `tensorrt_smoke` | 僅限指定環境的固定 affine 網路 build → serialize → checked save/read → deserialize → enqueue；不代表正式模型可匯入 |
| 真實 artifact readiness | [來源查核、native ONNX inspection、缺件清單](MODEL_ARTIFACT_READINESS.md) | 一般 YOLO26s 的來源／檔案／可核對結構與整個目標模型的完成度分開（P01-A，s 時期）；m 目標的可載入權重與 engine 身分未交付時保持 OPEN |

實際執行結果、source commit、命令／exit、完整輸出與 hashes 集中在 [P01-A 證據](../evidence/p01-a/2026-10-28/REPORT.md)。既有合成 CUDA 骨架證據維持原封不動：[原始驗收](../evidence/cuda-synthetic/2026-10-28/REPORT.md)、[checked I/O／cleanup closeout](../evidence/cuda-synthetic/2026-10-28-closeout/REPORT.md)。

## 2. 版本化模型輸入契約

第一版選定 **JSON 的嚴格 ASCII profile、schema=1**，由純 C++ parser／validator 實作；完整欄位、型別、限制、canonical bytes 與相對路徑基準見 [MODEL_MANIFEST](MODEL_MANIFEST.md)。這是 P01 的模型輸入契約，完整 P08 設定治理仍 OPEN。

| 契約 | 必須提供的內容 |
| --- | --- |
| 新 artifact 來源 | 角色、來源、revision、授權、產生來源、檔案大小與 SHA-256。Checkpoint 另需不使用 Python 的 native 轉換／權重交付途徑。 |
| 組件與 tensor 邊界 | 合併／分開／缺件狀態，backbone／neck／head 到 artifact 的映射；tensor 名稱、順序、shape、dtype、layout、語義，以及 cls／reg／decode 的責任。 |
| 固定輸入 profile | batch 1、RGB CHW、float32 0–1、640×640；其他 tensor 的尺寸依新 artifact 核對。 |
| Scan 身分與完整語義 | domain／op type／opset、plugin 身分、全部 attributes、每個輸入的 shape／stride／broadcast／角色、序列方向、展平順序、initial state、輸出、精確公式的來源／revision／定位及每幀重設規則。未知資料明列 unverified，不用通用 Mamba 公式填補。 |
| 精度與比較 | 計算／累加精度、TF32／FP16／INT8 與 build 選項、對照來源、metric 與事先固定的門檻。宣告與模型內容的查核結果分開。 |

**輸入 manifest 與 build evidence 分離。** Engine hash、實際 plugin binary hash、tactic／timing cache、SDK／CUDA／driver／GPU／compiler／builder 身分與實際設定在建置後紀錄；不成為首次 manifest 驗證的依賴。Manifest parser 拒絕未列入 schema 的 build evidence 欄位。

檔案相對於 manifest 所在目錄解析，使用既有 checked I/O；穩定編碼與 hash 不依賴 JSON 欄位排列、縮排或執行時工作目錄。共用 SHA-256 只是移出原函式，synthetic config 的編碼與 hash 契約保持原樣。

查核層級始終分成：①宣告合法且內部一致；②實際檔案可讀、大小與 SHA-256 相符；③實際 tensor／圖／scan attributes 與規格交叉確認。①②成功不推導③，test fixture 也不能取得 real-model readiness。

## 3. S01–S04 的接續條件

| 階段 | 必要驗證 | 成功後可聲稱的範圍 |
| --- | --- | --- |
| S01 語義／容量 | 先固定新 head 對應的 source/config/module map 與已查核 scan 契約，再實作獨立 C++ scalar reference 及數值 fixtures：零／單 token、非零 initial state（若支援）、長短序列、不同通道／state size、極端有限值、非法 shape／非有限值拒絕。 | 該明定 scan 契約的可執行參考 |
| S02 CUDA forward | 同一契約的獨立 CUDA forward 對照 reference；顯式 D stream、固定 workspace、無幀中途 D2H、無 default-stream 依賴；initcheck／memcheck。 | 該 scan 的實作及數值結果 |
| S03 TensorRT 接入 | 新 pin SDK 的 plugin build／deserialize、身分及參數 round-trip、shape／dtype 拒絕、顯式 stream、session workspace 隔離、固定輸出生命週期。 | 該 plugin 在指定環境可執行 |
| S04 完整 head／F03 | 同一新 artifact、固定輸入與完整輸出對照；重跑決定性、graph replay、重疊與資源需求；cls／reg／候選邊界。 | 有證據的完整推論範圍 |

**S01 判定：PASS（2026-10-29，受測 source `97f9b8c43f4127feb4bd1056893648f9b6cfc6c5`）。** 條件逐項：原版 m head 的 source／config／module map 已固定（[SCAN_CONTRACT §1–§2](SCAN_CONTRACT.md#1-目標與來源)）；scan 數學、tensor 邊界、狀態生命週期與比較政策已核實並先行固定（§3–§5）；[`selective_scan_reference`](../include/saccade/selective_scan.hpp) 為獨立 binary64 C++ 實作。[fixtures](../fixtures/selective-scan/README.md) 覆蓋零／單 token、長短序列（含 m 的 L=25／100／400、B=4、Dc=256、N=16）、N=1／2／4／8／16／32 與多種 channel 數、shared／per-channel A、rank-1／full C、D 有無、四方向展開與合併、極端有限值，以及非法 shape、容量溢位、非有限值與 FP32 範圍溢出的拒絕。非零 initial state 不受支援，ABI 中沒有該欄位。預設 CPU CTest 與 ASan／UBSan 均通過，expected 由 bc 重產逐 byte 相同，8 個語義 mutation 全部被測出。

PASS 不涵蓋：MambaBlock 的投影、conv1d 與 gate（於 S04 以 native TensorRT network API 建圖對照）、FP32／FP16 CUDA kernel 的數值容差（S02）、plugin（S03）、完整 head 與真實權重（S04／P01）。可供 v1 native 載入的 m 權重、engine 身分與 cls／reg／decode 容量缺件見 [SCAN_CONTRACT §7](SCAN_CONTRACT.md#7-來源差異衝突與缺件) 與 [readiness §7](MODEL_ARTIFACT_READINESS.md#7-2026-10-29-原版-m-head-的-artifact-對應)。S02 已接續完成（下段）。

**S02 判定：PASS（2026-10-29，受測 source `7f7bfa73d53d5ee5d41b416e8abf0b20b1d8e434`）。** [SCAN_CUDA](SCAN_CUDA.md) 在第一次 GPU 執行前固定呼叫契約、FP32 算法、支援範圍與逐元素 running error bound 政策。`selective_scan_cuda` 與 Cross-Scan 展開／合併對 S01 reference 使用同一份 FP32 輸入 bytes：70 個案例、1,294,286 個元素全部在 2×E 內，最大 誤差／門檻 0.295，含 S01 全部 fixtures、m 的 L=25／100／400 形狀、N=1…32 與各種 broadcast、softplus 掃描（舊式 `logf(1+expf(x))` 在 18 點超出門檻）、中間量溢位／下溢與相消案例。失敗以 device status 回報、整個輸出改為 NaN；每次呼叫重設；工作只在顯式 stream、可被 global-mode capture 成 1 memset＋3 kernels，沒有配置、同步或 D2H；workspace 為 0。memcheck／racecheck／synccheck 完整通過，initcheck 只略過 host callback 擋住 stream 的一項，四個工具都有正控制組；17 個 CUDA mutation 全部被測出。開發中發現 lazy loading 會讓第一次呼叫隱式等待已在執行的工作，已列入呼叫契約（`selective_scan_cuda_load_kernels`）。

S02 PASS 不涵蓋：FLT_MAX 附近幾個 ulp 的範圍判定灰區（SCAN_CUDA §4）、FP16 I/O、plugin 與其 status 通道（S03）、MambaBlock 其他算子、完整 head 與真實權重（S04／P01）、效能與跨 GPU 一致性。後續 S03 已以 v1 自建 plugin 接上同一個 kernel，結果如下。

**S03 判定：PASS（2026-10-29，受測 source `e468044f0ab003fe5f76d6d2ef7260b618121516`）**，範圍與證據見 [SCAN_TRT](SCAN_TRT.md) 與 [S03 證據](../evidence/s03/2026-10-29/REPORT.md)。

**S04 判定：PASS（2026-10-30，受測 source `3857e243e8fc32663aa688f073343002ad12cb30`）。** 指定的 T1 候選 checkpoint 已 native 轉成權重 bundle；完整 FP32 head 以 TensorRT network API 自建；數值政策 [`s04-head-fp32/1`](HEAD_POLICY.md) 在第一次 GPU 比較前提交。正式 clean-source capture 的三組輸入 × 六輸出逐元素對照、frame 原子性、status 完整性、所有權與生命週期結果見 [S04 證據](../evidence/s04/2026-10-30/REPORT.md)。head memcheck／initcheck、S03 四種 sanitizer 與正控制已執行；head 與 S03 採 explicit API 錯誤報告，並各自另跑預設模式 memcheck 分類 driver 記錄。**完整 head 的 racecheck／synccheck 未執行**。三次失敗 attempt（`222af61`：head sanitizer API 記錄分類及 dirty build；`0f5829d`：TF32 mutation 診斷 matcher；`198ad63`：API 記錄分類範圍漏掉 S03）、開發 racecheck 期間原因未判定的 WSL 當機與數值觀察保留於 [HEAD_TRT §9](HEAD_TRT.md#9-執行紀錄與待完成事項) 與正式報告。S04 不涵蓋 backbone／neck、前處理、decode 與模型品質；P01 仍 OPEN。

**F03（T01 候選）判定：PASS，有範圍（2026-10-30，受測 source `4b52b2fbbf54344a1c7dcb7a68812fb2a9e63bd9`）。** T01 的 matching backbone ONNX 與 S04 head 建成一個 FP32 engine；政策 [`t01-backbone-fp32/1`／`t01-f03-fp32/1`](BACKBONE_POLICY.md) 在第一次 GPU 比較前提交。六個固定輸入上 backbone 與完整模型的逐元素對照、執行與失敗傳遞見 [MODEL_TRT](MODEL_TRT.md) 與 [F03 證據](../evidence/t01/2026-10-30/f03/REPORT.md)。不涵蓋 native 前處理、decode 與模型品質；racecheck／synccheck 未執行；P01 仍 OPEN。

`enqueue` 不使用 host count readback 決定正式工作量；固定 shape／容量在 session 前配置。未來 plugin 只在 D 及其明定資源執行；新增 stream 須先補 ASYNC_CONTROL 的交付／回收邊。失敗交由 F04 slot header／M 控制流程鎖定首次錯誤，既有 TensorRT 工作仍須依 stream／event 收尾。此次 affine 工具的 D2H 屬獨立驗證流程。
