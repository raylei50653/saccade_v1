# T01 正式 run 證據（2026-10-30）

摘要與完成狀態見 [T01_TRAINING §8](../../../../docs/T01_TRAINING.md#8-執行紀錄正式-run2026-10-30)。大型檔案（checkpoint、cache、ONNX）留在 `~/.cache/saccade-v1/t01/`；`artifacts.json` 記錄它們的路徑、大小與 SHA-256，提交前全部重新計算過 hash。

| 檔案 | 內容 |
| --- | --- |
| `pipeline.log` | runner 的輸出：每個 attempt 的結果、每個 epoch 的 val 結果、各 stage 的選定 |
| `attempts.json` | 每個 attempt 的計算精度（policy、環境變數、程序內開始與結束讀回、monitor 取樣、偏離）與儲存（cache manifest、每個 epoch checkpoint 的 tensor dtype 與存檔時精度）；另含 122 個評估結果的精度讀回 |
| `val-tables.json` | 五個 stage 每個 epoch 的 val 結果：AP50、小目標 recall、recall 與 FP（score ≥ 0.25）、非有限值、training loss、checkpoint hash、評估與存檔時的精度 |
| `selections/` | 各 stage 的選定紀錄（teacher 為固定 e12；學生 stage 含完整選模表） |
| `opened.json`、`final_eval.json` | eval split 打開的紀錄（含 T1 選定紀錄的 hash），以及 final eval：T1 在 eval 與 val 各評兩次、lineage 與 teacher 診斷 |
| `artifacts.json` | 選定 checkpoint、三個 cache manifest、backbone ONNX 的 SHA-256 |
| `backbone-export.json`、`backbone-verify.json` | teacher e12 backbone／neck ONNX 的匯出紀錄（440 個 initializer 對應到 checkpoint）；以及 ONNX Runtime vs PyTorch vs cache 的比較，未訓練權重為負向對照 |
| `feature-precision.json` | `t01_verify_features.py` 對正式 cache 的驗收：**PASS** |
