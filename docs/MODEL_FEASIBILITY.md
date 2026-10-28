# P01 模型 artifact 與 native scan 可行性

狀態：介面調查；P01 保持 OPEN。檢查日期：2026-10-28。

[DATAFLOW §1.1 B03](DATAFLOW.md#11-由舊架構收斂的-v1-設計基線) 已選定 YOLO26s backbone／neck＋Mamba-FPN head、640×640、P3／P4／P5、幀內推論。實作方向維持 C++ TensorRT＋native CUDA scan。本次只核對文件、檔案存在與工具資訊，沒有載入舊模型、下載權重、建立 engine 或執行推論。

## 1. 已核對的條件

| 項目 | 本次觀察 | 保證範圍 |
| --- | --- | --- |
| v1 artifact | 基準 `e08a1da` 只有 README 與兩份規格，沒有模型、lineage、TensorRT SDK pin、plugin 或 engine。 | 尚無可交給 builder 的 v1 模型輸入；不表示舊目錄的模型可直接採用。 |
| GPU | `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader` 回報 NVIDIA GeForce RTX 5070 Ti Laptop GPU、driver 617.42、12227 MiB。 | 工具能列出 GPU；未驗證新 engine、記憶體需求或效能。 |
| CUDA compiler | `/opt/cuda/bin/nvcc --version` 回報 release 13.4、V13.4.92。 | compiler 存在；不是 v1 已核定的 runtime／SDK 組合。 |
| TensorRT | `trtexec` 不在 PATH；標準系統 header 路徑未找到 `NvInferVersion.h`。相鄰舊專案 `.venv/lib/python3.12/site-packages/` 有 TensorRT 10.16.1.11 的 package metadata。 | 舊環境的套件存在不代表 v1 已可編譯／連結 native builder，也不作為 v1 的版本 pin。 |
| 舊模型邊界 | [舊 math model §4.1](../../saccade/docs/reference/math_model.md#41-detector-內部結構fpn--mamba-head--decode) 描述三尺度 head 與 scan 架構；相鄰 `models/yolo/` 可找到 ONNX／engine／lineage 檔案。 | 僅支援架構來源的理解；不遷移檔案、權重 pin、舊實驗數據或 parity 結論。 |

NVIDIA 提供 C++ `IPluginV3`／creator 註冊與 ONNX parser 的 plugin 接入路徑；runtime `enqueue` 接收呼叫端的 CUDA stream。這支持以 native plugin 包裝 scan 的介面可行性，尚未證明本模型可匯入或數值相符。[NVIDIA C++ plugin 文件](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/inference-library/plugins-cpp.html)

## 2. 開始模型 build 前的輸入契約

第一份 v1 模型 manifest 應明列下表，不從舊目錄的檔名或 runtime 自動推定。manifest 的穩定編碼／hash 應與 P08 設定規格一併收斂；下表目前是必要欄位清單，尚不是已實作的 schema。

| 契約 | 必須提供的內容 |
| --- | --- |
| 新 artifact 來源 | 可取得的來源、版本／revision、檔案大小與 SHA-256、模型家族、訓練或產生來源、使用授權；若只有 checkpoint，須另定不使用 Python 的轉換／權重交付途徑，不能把 pickle 載入當 native 入口。 |
| 完整模型邊界 | backbone／neck 與 head 分開或合併；所有 tensor 的名稱、順序、shape、dtype、layout；cls／reg 的語義與 decode 所屬邊界。 |
| 固定輸入 profile | batch 1、RGB CHW、float32 0–1、640×640；P3／P4／P5 的通道／尺寸由新 artifact 核對，不能只沿用舊實現常數。 |
| scan 算子身分 | ONNX domain／op type／opset（若用 ONNX）、plugin name／namespace／version、所有 attributes、序列方向與展平順序、是否輸出 final state。 |
| scan 精確語義 | 每個輸入的 shape／stride／broadcast、A／B／C／D 的角色、delta bias／softplus 的先後、離散化公式、可選 gate、initial state；每次 F03 的狀態初始化與幀間無記憶規則。 |
| 精度與對照 | scan 累加精度、TensorRT 網路／各層精度、TF32 等 build 選項與讀回結果；先定新模型的對照來源、誤差量測與接受門檻，再執行測試。不得以舊實驗結果替代。 |
| build／runtime 身分 | TensorRT、CUDA、driver、GPU／SM、compiler、plugin ABI、builder commit、設定 hash、tactic／timing cache 身分、engine／plugin hash；重建條件。 |

## 3. 第一個 native scan 實作的界線

從經 manifest 確認的 scan 語義開始，先寫獨立 C++ 參考 recurrence 與 CUDA forward；再接 TensorRT plugin。參考與 CUDA 必須各自表達同一個已確認公式，不能只比較同一 kernel 的兩個包裝。跨 token 的順序、initial state 與輸出 gate 都需要明確測例；目前不預先選定一個「通用 Mamba」公式當成新 artifact 的實際語義。

| 階段 | 必要驗證 | 成功後可聲稱的範圍 |
| --- | --- | --- |
| S01 語義／容量 | C++ scalar reference：零／單 token、非零 initial state（若介面支援）、長短序列、不同通道／state size、極端但有限輸入、無效 shape／非有限值拒絕。 | 該明定 scan 契約的可執行參考。 |
| S02 CUDA forward | 相同 fixtures 對照 reference；獨立 D stream、固定 workspace、無幀中途 D2H、無 default-stream 依賴；initcheck／memcheck。 | 該 scan 的實作與數值結果；不是完整 head 的判定。 |
| S03 TensorRT 接入 | 新 pin SDK 的 C++ build／deserialize、plugin 身分及參數 round-trip、shape／dtype 拒絕、顯式 stream、session workspace 隔離、固定輸出生命週期。 | 該 plugin 可在指定環境執行。 |
| S04 完整 head／F03 | 相同新 artifact、固定輸入、完整輸出對照；重跑決定性、graph replay、重疊與資源需求；核對 cls／reg／候選邊界。 | 有證據的完整推論範圍；scan 單元通過不能替代。 |

`enqueue` 不使用 host count readback 決定工作量；支援的固定 shape／容量於 session 前配置。plugin 只在 D 與其明定資源上執行，若未來使用額外 stream，須先補足 ASYNC_CONTROL 的對外交付與回收邊。失敗狀態應交給 F04 slot header，由 M 的控制流程鎖定首次錯誤；TensorRT 已排入的工作完成與否，仍依 stream／event 收尾，不能假定自訂旗標會取消函式庫工作。

## 4. 目前的 OPEN 邊界

P01 的下一個具體輸入是「新的 artifact 來源與 manifest」，以及選定、可供 native C++ build 的 SDK。這兩項尚未存在於 v1，因此本次不宣稱 ONNX importer、scan plugin、完整 head 或效能已通過。S01–S04 尚未實作；本文件把其依賴與接受範圍拆開，供後續實作依序落地。
