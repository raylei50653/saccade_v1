# T01 backbone engine 與 F03 完整模型驗證報告（2026-10-30）

**結論：backbone PASS（`t01-backbone-fp32/1`）；F03 PASS（`t01-f03-fp32/1`）。**

- YOLO26m backbone／neck 的 TensorRT engine 在六個固定輸入上，p3／p4／p5 的每個元素都在 binary64 reference 的 16σ 內；T01 事先固定的兩個次要容差與 12 個語義變體也全部通過。
- F03 完整模型 engine（`images` → 六個 raw outputs）在同樣六個輸入上，六個輸出的每個元素都在以 backbone reference 為輸入的 head reference 的 16σ 內。執行、失敗傳遞與 graph 重播檢查全部通過；memcheck 與 initcheck 都是 0 errors。

範圍：模型輸入 tensor 到六個 raw outputs，限 §身分 的 engine、六個固定輸入與本機環境。engine 輸入是訓練側（PyTorch `fp32-strict`）依 `t01-input/1` 產生的 FP32 tensor，native 前處理（F01／F02）、decode／NMS（F04）、tracking 與模型品質不在範圍內。**P01 未關閉。** 規格見 [BACKBONE_POLICY](../../../../docs/BACKBONE_POLICY.md) 與 [MODEL_TRT](../../../../docs/MODEL_TRT.md)。

## 身分

| 項目 | 值 |
| --- | --- |
| 受測 source | `4b52b2fbbf54344a1c7dcb7a68812fb2a9e63bd9`（乾淨工作樹；archive SHA-256 `19ae8bc8…13d57d55`）；capture 期間每一步前後核對 HEAD 與工作樹，沒有變動；全部報告 `build_dirty=false` |
| 政策 | `fb10377`（先於任何 GPU backbone 比較提交）；capture 核對 BACKBONE_POLICY 與其 GPU 前證據自 `fb10377` 起未改、HEAD_POLICY 自 `4d38bcc` 起未改 |
| 未改動的程式 | GPU kernels、plugins、frame session、權重轉換與 head 比較器自 S04 受測 source `3857e243` 起未改（`git diff --exit-code`）。既有 head 程式碼自 T01 head 紀錄 `f9077c3` 起只多了 `head_reference_from_refs` 與 `HeadBuildOptions::features`（完整 diff 在 `capture/head-code-changes-since-t01-head.log`）；S01–S04 與 T01 既有證據未改 |
| teacher checkpoint | e12，251,887,299 bytes，SHA-256 `14e86231…2899e1b`，與選定紀錄相符 |
| backbone ONNX | 76,186,163 bytes，SHA-256 `007c3260…a0c236f`；匯出紀錄的 teacher hash 相符，float initializer 無未對應項 |
| T1 checkpoint／bundle | `070580f9…6e6962b`；capture 中重新轉換得到 bundle `4fccfda6…098f046a3`（69 active＋39 excluded） |
| 輸入／reference | 在獨立的 `SACCADE_T01_ROOT` 重新產生：`f03-inputs.f32` `14dda686…5faf60aa0`、`backbone-reference.f64` `ccb6f9da…2419584`，皆與政策 commit 的紀錄逐 byte 相同 |
| backbone engine | 76,879,796 bytes，SHA-256 `7c256a1339f92178da187d3007e3240f2ed874a13c01bba6c73f3798946ad52c`；同一 capture 重建得到相同 plan bytes |
| F03 engine | 118,173,420 bytes，SHA-256 `c6cff1936194e679ce11c3736cab552e4a73a2b1a9ae5508bc3e5d1b3fbab9c5`；同一 capture 重建的 plan bytes 不同（`7716f741…`），但 layer／tactic JSON 完全相同。依政策只作紀錄 |
| head engine | ctest 建出與重建都是 `5b6d30fa…5276024a`，與 HEAD_TRT §10 相同 |
| 環境 | RTX 5070 Ti Laptop（sm_120，driver 617.42，CUDA runtime／driver API 13.4）、nvcc 13.4.92、TensorRT 10.16.1.11＋同版本 nvonnxparser、GCC 16.2.1、WSL2；全程約 11 min（23:27:35–23:38:40），可用記憶體最低 38.1 GiB（低於 4 GiB 會停止 capture） |

## 結果

| 項目 | 結果 |
| --- | --- |
| CPU CTest | 13/13（含 T01 bundle 轉換與 hash 檢查、`head_reference_candidate`、manifest schema 1／2／3） |
| head reference（T01 bundle） | 241 checks；13 個 head 語義變體 `missed=none`；`from_refs` 對 FP32 入口逐位元相同 |
| CUDA＋TRT CTest | 31/31（S01–S03 全部、head engine 建置／執行／未註冊、`head_trt` suite 824 checks、api_trace） |
| CPU 預檢（重跑） | PyTorch FP32 CPU 最大 誤差／門檻 0.024，PASS；12 個變體全部偵測到；ONNX Runtime 的 341 個 SiLU 尾端超界照政策紀錄 |
| **Backbone 逐元素** | 6 輸入 × 2,662,400 元素：0 超界、0 非有限值；最大 誤差／門檻 **0.0302**（MOT17-13/400 的 p4）；各輸入最大值見下表 |
| Backbone 次要容差 | rel L2(TensorRT, PyTorch `fp32-strict`) 最大 8.92×10⁻⁶ ≤ 10⁻⁴；rel L2(訓練 cache, TensorRT) 最大 2.13×10⁻⁴ ≤ 2⁻¹¹＋10⁻⁴ |
| Backbone 語義變體 | 12/12 對 reference 與對 TensorRT 都偵測到；每個變體在每個受測 level 都有 ≥ 20.7% 的元素超出 TensorRT 輸出的判定範圍 |
| Backbone 執行 | 6 輸入重跑逐位元相同、無非有限值；186 節點的 CUDA graph 重播與 enqueueV3 逐位元相同；tactic 檢查無違規（Myelin 的 FP32 implicit-GEMM conv、FFMA gemm、融合的 SiLU／加法與明確的 max／exp／sum／div softmax） |
| **F03 逐元素** | 6 輸入 × 6 輸出（705,600 元素／輸入）：36 個判定全部 PASS、0 超界；最大 誤差／門檻 **0.0766**（MOT17-13/400 的 `reg_p3`） |
| F03 執行 | 每個輸入都被接受（9 個 plugin statuses 與兩個有限值檢查 OK）；重跑逐位元相同；失敗序列 OK／FAILED(8)／SKIPPED，committed 不變；graph 重播與 enqueueV3 逐位元相同；未註冊 plugin 時 context 建立被拒 |
| F03 sanitizers | memcheck 0 errors、initcheck 0 errors（`--inputs 1`：MOT17-02/1、重跑、失敗序列、graph 重播；explicit API 報告） |
| 兩段式（backbone engine → head engine） | S04 政策下 54 個判定全部 PASS（固定輸入 3 組＋backbone engine 的 6 組 features），features 最大 誤差／門檻 0.012；失敗序列與 graph 重播 PASS |
| Manifest | [schema 3 manifest](../../../../manifests/t01/yolo26m-t1.json) validate 與 verify（六個 artifact 的大小與 SHA-256）通過；canonical SHA-256 `6bc885e9…2cbccd2`；`real_model_ready=false`（工具不據此判定） |

| 輸入 | backbone p3／p4／p5 最大 誤差／門檻 |
| --- | --- |
| MOT17-02-SDP/1 | 0.0096／0.0201／0.0044 |
| MOT17-10-SDP/1 | 0.0134／0.0270／0.0102 |
| MOT17-13-SDP/400 | 0.0136／0.0302／0.0064 |
| zeros | 0.0070／0.0104／0.0030 |
| ones | 0.0162／0.0275／0.0243 |
| random/1 | 0.0079／0.0095／0.0027 |

**診斷（不是驗收）**：
- F03 輸出與兩段式結果在六個輸入上**逐位元相同**；
- 對訓練側 PyTorch（backbone `fp32-strict`＋T1 head）的相對 L2 ≤ 2.1×10⁻⁵；
- 經 T01 decode（conf 0.001、top-300、NMS 0.5）後，偵測數相同（32／28／39；合成輸入 2／3／1），次序相同；框最大差 0.0013 px，score 最大差 8.7×10⁻⁶。
- 開發建置（`fb10377`＋未提交的工具）與本 capture 的 engine hash 不同，但兩者的 backbone 與 F03 輸出在六個輸入上逐位元相同。

## 未執行或不在範圍

- F03 與 head 的 racecheck／synccheck 未執行，原因同 S04；
- head suite 與 S03 suite 的 sanitizer 不在此重跑：code 自 S04 未改，T01 head 的 memcheck／initcheck 見 [native-head](../native-head/REPORT.md)；
- native 前處理（F01／F02）、decode／NMS 與容量（F04）、tracking；
- 其他 GPU、driver 或 TensorRT 版本。

## 檔案

`capture/` 是 [capture.sh](capture.sh) 每一步的 `.command`／`.log`／`.exit`，以及 `memory.log` 與報告。其中：
- `backbone-compare.json`：backbone 逐元素、次要容差與變體；
- `model-run-report.txt`：F03 的逐輸出判定與執行檢查；
- `chain-head-run-report.txt`：兩段式；
- `f03-diagnostics.json`：PyTorch 與兩段式診斷；
- `*-build-report.txt`：builder 設定、身分與每層 tactic；
- `sanitizer-skips.txt`：省略項目。

大型檔案（engines、bundle、inputs、reference、dumps）留在本機 `build-t01-f03-evidence/` 與 `~/.cache/saccade-v1/t01/`，hash 見 `capture/binary-hashes.log`。本目錄全部檔案的 SHA-256 見 [SHA256SUMS](SHA256SUMS)。
