# T01：YOLO26m backbone／neck engine 與 F03 完整模型推論（native FP32 TensorRT）

把 T01 選定的模型接成完整的 F03：**模型輸入 `images` → YOLO26m backbone／neck（teacher e12，ONNX）→ P3／P4／P5 → S04 head（T1 epoch 1）→ 六個 raw outputs**，FP32，batch 1，T=1。數值政策固定在 [BACKBONE_POLICY](BACKBONE_POLICY.md)（`t01-backbone-fp32/1`、`t01-f03-fp32/1`，於 `fb10377` 先於任何 GPU 比較提交）；head 的比較器與規則仍是 [HEAD_POLICY](HEAD_POLICY.md)（`s04-head-fp32/1`）。不含 native 前處理（F01／F02：engine 輸入是訓練側產生的 FP32 tensor）、decode／NMS（F04）、tracker 與效能。

結果見 [§5](#5-結果2026-10-30)。

## 1. Artifacts

| 項目 | 值 |
| --- | --- |
| backbone ONNX | `yolo26m-backbone-t01.onnx`，76,186,163 bytes，SHA-256 `007c3260…a0c236f`；IR 9、opset 20、BatchNorm 未摺疊；440 個 initializer 逐 byte 對應 teacher e12 checkpoint（`14e86231…`），見 [T01_TRAINING §8](T01_TRAINING.md#8-執行紀錄正式-run2026-10-30) |
| head bundle | T1 epoch 1 的 SCWB 1 bundle，`4fccfda6…`（[HEAD_TRT §10](HEAD_TRT.md#10-t01-候選的延續驗證2026-10-30)） |
| ONNX parser | 同一 SDK root 內追加的 TensorRT 10.16.1.11 `nvonnxparser`（[NATIVE_TENSORRT](NATIVE_TENSORRT.md#onnx-parsert01-backbone)） |
| manifest | [`manifests/t01/yolo26m-t1.json`](../manifests/t01/yolo26m-t1.json)（[schema 3](MODEL_MANIFEST.md#schema-3t01-完整模型)）：teacher checkpoint、backbone ONNX、T1 checkpoint、head bundle 與兩份政策文件 |

## 2. 建圖

[`model_trt_network.cpp`](../src/model_trt_network.cpp)：

- **backbone engine**（`build_backbone_engine`）：`nvonnxparser` 把 ONNX 解析進 strongly typed network；I/O 必須恰為 `images [1,3,640,640]` 與依序的 `p3_feat`、`p4_feat`、`p5_feat`（FP32，head 輸入形狀），否則拒絕並列出 parser 錯誤。builder 設定同 head：TF32 off、optimization level 0、workspace 256 MiB、auxiliary streams 0、不用 timing cache、detailed profiling；建成後以 `head_tactic_violations` 檢查每個 tactic。
- **F03 model engine**（`build_model_engine`）：`HeadBuildOptions::features`（[head_trt_network.hpp](../include/saccade/head_trt_network.hpp)）在 head 的 network 內先解析同一份 ONNX，取消其三個輸出，再由 S04 的建圖程式在這三個 tensor 上建 head；status、tactic 檢查與 S04 相同（`build_head_engine`）。parser 持有 ONNX 權重直到 plan 序列化。I/O 為 `model_io_spec()`：`head_io_spec()` 的輸入換成 `images`。
- **S04 head 不變**：不設 `features` 時，head 的建圖路徑與 S04 相同（同一程式建出的 T01 head engine 必須仍是 `5b6d30fa…`）。

## 3. 工具與驗收

| 工具 | 內容 |
| --- | --- |
| [`backbone_trt build`](../tools/backbone_trt.cpp) | 以 SHA-256 讀 ONNX，建 backbone engine，checked 寫出後讀回逐 byte 比對、重新反序列化；build report 含 builder 設定、parser／TensorRT／CUDA／GPU 身分、source commit／dirty、I/O 與每層 tactic |
| `backbone_trt run` | 新程序：反序列化、一個 execution context、non-default stream；每個輸入先把輸出填成全 1 bits（NaN），enqueueV3 後 D2H；重跑必須逐位元相同；只含 enqueueV3 的 CUDA graph 重播必須與 enqueueV3 逐位元相同；`input-<i>.f32` 依序存 p3／p4／p5 |
| [`t01_backbone_ref.py compare`](../training/t01/t01_backbone_ref.py) | BACKBONE_POLICY §5：逐元素判定、T01 事先固定的次要容差、12 個語義變體 |
| [`model_trt build`](../tools/model_trt.cpp) | 以 SHA-256 讀 ONNX 與 bundle，建 F03 engine，記錄同 `head_trt build` |
| `model_trt run` | HEAD_TRT §5 的 frame session（`model_io_spec()`）：每個輸入接受、`head_reference_from_refs` 以 backbone reference 為輸入的 head policy 判定、重跑逐位元相同、失敗序列（`images` 內一個 NaN）、CUDA graph 重播；`--inputs N` 只取前 N 個（sanitizer） |
| `head_trt run-unregistered` | F03 engine 在未註冊 plugin 時不得建立 execution context |
| `head_trt run --features` | 診斷兼 S04 政策檢查：backbone engine 的輸出接 head engine（兩段式） |
| `t01_backbone_ref.py diagnose` | 診斷：F03 輸出對 PyTorch（backbone `fp32-strict`＋T1 head）與對兩段式結果 |

CMake：`-DSACCADE_ENABLE_MODEL_TRT=ON`（需 `SACCADE_ENABLE_CUDA` 與 `SACCADE_ENABLE_TENSORRT`），只在 `SACCADE_TENSORRT_ROOT` 內找 `libnvonnxparser.so.10.16.1` 與 `NvOnnxParser.h`，並以 probe 核對 header 與 library 的 API 版本；未開啟時既有 build 不變，CI 的 CPU build 不受影響。

## 4. 重跑

完整 capture（乾淨的已提交 source；需 `~/.cache/saccade-v1/t01` 的 checkpoints、ONNX、export venv 與 T01 cache，pinned SDK 含 parser，CUDA、compute-sanitizer 與 GPU）：

```sh
bash evidence/t01/2026-10-30/f03/capture.sh build-NEW-OUTPUT
```

capture 在獨立的 `SACCADE_T01_ROOT` 重新產生六個輸入與 binary64 reference，並要求與政策 commit 記錄的 SHA-256 相同，再建三個 engine（head 重建、backbone、F03）、執行上表的全部檢查、F03 的 memcheck／initcheck（第一個真實輸入＋失敗序列＋graph 重播，可用記憶體低於 4 GiB 即停）、manifest validate／verify 與 source identity。

## 5. 結果（2026-10-30）

**Backbone PASS、F03 PASS**：clean-source capture（受測 `4b52b2f`），完整報告見 [F03 證據](../evidence/t01/2026-10-30/f03/REPORT.md)。

| 項目 | 結果 |
| --- | --- |
| backbone engine | 76,879,796 bytes，`7c256a13…46ad52c`；同一 capture 重建相同；tactic 檢查無違規 |
| backbone 逐元素 | 6 輸入 × 2,662,400 元素全部在 16σ 內，最大 誤差／門檻 0.030；無非有限值 |
| backbone 次要容差 | 對 PyTorch `fp32-strict` 8.9×10⁻⁶ ≤ 10⁻⁴；對訓練 cache 2.1×10⁻⁴ ≤ 2⁻¹¹＋10⁻⁴ |
| 語義變體 | 12/12 對 reference 與 TensorRT 都偵測到 |
| F03 engine | 118,173,420 bytes，`c6cff193…3fbab9c5`；重建的 plan bytes 不同但 layer／tactic JSON 相同（只作紀錄） |
| F03 逐元素 | 6 輸入 × 6 輸出全部在 16σ 內，最大 0.077（`reg_p3`）；36 個判定 PASS |
| F03 執行 | 重跑與 graph 重播逐位元相同；失敗序列 OK／FAILED(8)／SKIPPED；未註冊 plugin 被拒 |
| sanitizers | F03 memcheck、initcheck 0 errors；racecheck／synccheck 未執行 |
| head 不變 | 同一程式建出的 T01 head engine 仍是 `5b6d30fa…` |
| 診斷 | F03 輸出與兩段式逐位元相同；對 PyTorch 相對 L2 ≤ 2.1×10⁻⁵，decode 後偵測相同 |

開發中的觀察：ONNX Runtime CPU 的 Sigmoid 在 x < −8 相對誤差很大（政策 §5 的預檢）；TensorRT 的 FP32 SiLU 沒有這個問題（backbone 最大比值 0.030）。開發建置與 capture 的 engine hash 不同，輸出卻逐位元相同。

**本 F03 capture 未涵蓋**：native 前處理（F01／F02）、decode／NMS（F04）、racecheck／synccheck、其他 GPU／版本。前處理與 F04 的獨立驗收現況分別見 [PREPROCESS](PREPROCESS.md)、[POSTPROCESS](POSTPROCESS.md)；[F03 producer 聚合器](MODEL_PRODUCER.md) 已接續實作並有 [有界開發證據](../evidence/f03-producer/2026-10-31/REPORT.md)；完整 native 接續與 P01 仍 OPEN。

## Native detection session 接續

完整 engine 已由 [`DetectionSession`](DETECTION_SESSION.md) 串接 native F01／F02、producer 聚合器及 F04。session 驗證完整 engine I/O，使用 zero auxiliary streams、同 detect stream 的模型 scratch 與跨 stream 候選 leases，並另行管理首次失敗與 checked drain；新增 session 的證據不回寫本文件原有 F03 數值驗收或模型品質。完整 tracking session／P01 仍 OPEN。
