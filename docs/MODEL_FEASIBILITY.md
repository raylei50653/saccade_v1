# P01 模型 artifact 與 native scan 可行性

P01 保持 OPEN。P01-A 將 manifest 工具、native SDK 路徑、真實 artifact readiness 分別驗收，不能互相替代。調查日期：2026-10-28。

[DATAFLOW §1.1 B03](DATAFLOW.md#11-由舊架構收斂的-v1-設計基線) 維持 YOLO26s backbone／neck＋Mamba-FPN head、640×640、P3／P4／P5、幀內推論、C++／CUDA。從驗收基線 `437c99edbae1a34854aa6966dc210b5053069929` fetch 後確認 `origin/main` 相同，再接續實作；沒有搬用舊程式、權重或實驗結論。

## 1. P01-A 的交付與驗收邊界

| 項目 | 實作與查核入口 | 驗收含義 |
| --- | --- | --- |
| Manifest 工具 | [schema 與 CLI](MODEL_MANIFEST.md)，`model_manifest`、CPU 測試及明標用途的 fixtures | 驗證宣告與檔案完整性；不自行保證模型內容符合宣告 |
| Native SDK 路徑 | [SDK pin、環境、命令與比較契約](NATIVE_TENSORRT.md)，可選 `tensorrt_smoke` | 僅限指定環境的固定 affine 網路 build → serialize → checked save/read → deserialize → enqueue；不代表正式模型可匯入 |
| 真實 artifact readiness | [來源查核、native ONNX inspection、缺件清單](MODEL_ARTIFACT_READINESS.md) | 一般 YOLO26s 的來源／檔案／可核對結構與整個目標模型的完成度分開；Mamba-FPN head 與 scan 語義未取得時保持 OPEN |

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
| S01 語義／容量 | 先取得新 head 與已查核 scan 契約，再實作獨立 C++ scalar reference 及數值 fixtures：零／單 token、非零 initial state（若支援）、長短序列、不同通道／state size、極端有限值、非法 shape／非有限值拒絕。 | 該明定 scan 契約的可執行參考 |
| S02 CUDA forward | 同一契約的獨立 CUDA forward 對照 reference；顯式 D stream、固定 workspace、無幀中途 D2H、無 default-stream 依賴；initcheck／memcheck。 | 該 scan 的實作及數值結果 |
| S03 TensorRT 接入 | 新 pin SDK 的 plugin build／deserialize、身分及參數 round-trip、shape／dtype 拒絕、顯式 stream、session workspace 隔離、固定輸出生命週期。 | 該 plugin 在指定環境可執行 |
| S04 完整 head／F03 | 同一新 artifact、固定輸入與完整輸出對照；重跑決定性、graph replay、重疊與資源需求；cls／reg／候選邊界。 | 有證據的完整推論範圍 |

S01 依賴的 head artifact／scan 精確公式目前尚未交付，具體所需輸入見 [缺件紀錄](MODEL_ARTIFACT_READINESS.md)。本次不自行替未知模型選一個公式，因此尚未產生可宣稱為目標 scan 的 scalar reference 或數值 fixtures。取得該契約後，下一個工作直接從 S01 接續。

`enqueue` 不使用 host count readback 決定正式工作量；固定 shape／容量在 session 前配置。未來 plugin 只在 D 及其明定資源執行；新增 stream 須先補 ASYNC_CONTROL 的交付／回收邊。失敗交由 F04 slot header／M 控制流程鎖定首次錯誤，既有 TensorRT 工作仍須依 stream／event 收尾。此次 affine 工具的 D2H 屬獨立驗證流程。
