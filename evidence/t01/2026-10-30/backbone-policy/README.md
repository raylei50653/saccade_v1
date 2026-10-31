# T01 backbone／F03 政策的 GPU 前證據（2026-10-30）

政策見 [BACKBONE_POLICY](../../../../docs/BACKBONE_POLICY.md)。本目錄只含 CPU 結果，在任何 native GPU backbone 比較之前產生並與政策同一 commit 提交。大型檔案（`f03-inputs.f32` 31,457,280 bytes、`backbone-reference.f64` 383,385,600 bytes）留在 `~/.cache/saccade-v1/t01/export/`，hash 記在下列 JSON。

| 檔案 | 內容 |
| --- | --- |
| `f03-inputs.json` | 六個固定輸入的標籤、整檔與逐輸入 SHA-256、產生方式與 `fp32-strict` 讀回 |
| `backbone-reference.json` | binary64 reference 的檔案 hash、常數、每個輸入的自我核對（對 Ultralytics float64 modules）、σ ≤ E、σ 統計、運算次數與 ONNX 節點對照 |
| `backbone-precheck.json` | PyTorch FP32 CPU 與 ONNX Runtime FP32 CPU 對 reference 的逐元素判定、ONNX Runtime Sigmoid 量測、12 個語義變體 |
| `sdk-files-before-parser.txt`、`sdk-files-after-parser.txt` | 追加 ONNX parser 前後的 SDK 檔案清單與 SHA-256（原有 24 項不變，新增 9 項） |

重跑（export venv，`training/t01/` 下）：`t01_backbone_ref.py inputs`、`reference`、`precheck`。
