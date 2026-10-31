# T01 backbone／neck 與 F03 的 FP32 數值比較政策（`t01-backbone-fp32/1`、`t01-f03-fp32/1`）

固定日期：2026-10-30。**本政策在第一次 native GPU backbone 比較之前提交**，與其 reference [`t01_backbone_ref.py`](../training/t01/t01_backbone_ref.py)、[`head_reference_from_refs`](../include/saccade/head_reference.hpp) 及 §5 的 CPU 預檢結果同一 commit。之後若 GPU 驗收失敗，只能修正實作，或以新版本號明列修訂與理由（§9）；不依觀測到的 GPU 差異放寬門檻。

適用對象：

- **backbone（`t01-backbone-fp32/1`）**：T01 選定 teacher e12（`14e86231…`）的 YOLO26m layers 0–22，即 [T01_TRAINING §8](T01_TRAINING.md#8-執行紀錄正式-run2026-10-30) 的 backbone ONNX（76,186,163 bytes，SHA-256 `007c32600832b1be94b077c513210346f01ff280d95cd92b4200d8ba7a0c236f`，440 個 initializer 逐 byte 對應 checkpoint）。輸入 `images [1,3,640,640]` FP32，輸出依序 `p3_feat [1,256,80,80]`、`p4_feat [1,512,40,40]`、`p5_feat [1,512,20,20]`。
- **F03（`t01-f03-fp32/1`）**：同一 backbone 接 T01 T1 head（bundle `4fccfda6…`，[HEAD_TRT §10](HEAD_TRT.md#10-t01-候選的延續驗證2026-10-30)）的完整模型 engine：輸入 `images`，輸出六個 raw outputs。

範圍不含 native 前處理（F01／F02）、decode／NMS（F04）與模型品質。

## 1. Reference

| 項目 | 內容 |
| --- | --- |
| 實作 | `t01_backbone_ref.py reference`：CPU 上 PyTorch float64 的函數式 forward，依 Ultralytics 8.4.37 的 module 定義（Conv、C3k2、C3k、Bottleneck、SPPF、C2PSA、PSABlock、Attention、Upsample、Concat）逐一重寫，路由用模型自身的 `f`／`save`。不使用 ONNX、TensorRT 或 CUDA。這是訓練側的驗證工具；v1 runtime 不含 Python。 |
| 參數 | `t01_export.load_teacher_yolo`：全新 YOLO26m 載入 teacher 的 `yolo_model.*` tensors（與 ONNX 匯出同一份 checkpoint；ONNX initializer 對應見 `backbone-export.json`）。BN eps 與 attention scale 取其 FP32 值（等於部署圖中的 attribute／constant）。 |
| 精度 | 輸入 FP32 bytes 精確升為 binary64；所有中間量與輸出都是 binary64。binary64 的捨入比下列 FP32 尺度小約 2⁻²⁹，忽略。 |
| 自我核對 | 同一輸入以 Ultralytics modules 本身（`.double()`，同樣的 FP32 常數）計算，每個 level 的 max \|ref − module\| ≤ 10⁻¹² × max \|module\|；所有 value 與 σ 有限；σ ≤ E。另將 reference 的張量運算次數與 ONNX 節點逐類對照（Conv＝BatchNormalization＝88、Sigmoid 79、MatMul 4、Softmax 2、MaxPool 3、Resize 2），必須相同。 |
| 重現 | 16 threads；同一輸入重算逐 byte 相同（`backbone-reference.f64` SHA-256 `ccb6f9da…`）。 |

## 2. TensorRT 實作的運算模型

TensorRT 10.16.1.11 strongly typed FP32 network，TF32 關閉，ONNX 由同版本 `nvonnxparser` 解析（[NATIVE_TENSORRT](NATIVE_TENSORRT.md#onnx-parsert01-backbone)）。u＝2⁻²⁴。沿用 [HEAD_POLICY §2](HEAD_POLICY.md#2-gpu-實作的運算模型) 的 Conv 直接求和、ε_t＝2⁻¹⁷ 的 exp／sigmoid／SiLU 假設與純搬移運算；另加：

| 運算 | 模型 |
| --- | --- |
| Conv＋BatchNorm（eval） | TensorRT 可能把 BN 在 FP32 摺進權重（scale s＝γ/√(var+ε) 至多 3 次捨入，摺入的 w′＝w·s、b′＝β−μs 各再 1–2 次），或另跑 scale 層。兩者的參數誤差都 ≤ 5u·M_fold，M_fold＝Σ\|w′x\|＋\|β\|＋\|μs\|，對同一輸出通道同號，線性加入。點積本身同 HEAD_POLICY 的 K＋1 項直接求和。 |
| SiLU | ONNX 的 Sigmoid＋Mul；TensorRT 可能融合。同 HEAD_POLICY 的 SiLU 規則（含 2⁻¹²⁵·\|x\| 的下溢項）。 |
| MatMul（attention 的 qᵀk 與 v·pᵀ） | 兩個運算元都帶誤差；K 個乘積的直接求和。 |
| 乘以 scale 常數 | 1 次捨入。 |
| Softmax | 可能被融合成 online softmax：分子與分母各至多 4 次 exp（共 8ε_t），z−m 的捨入（相對 u\|z−m\|），N 項求和與除法；結果下溢時另計 2⁻¹²⁶ 絕對誤差。 |
| MaxPool 5×5（SPPF）、nearest Resize、Concat、Split、Slice、Reshape、Transpose | 精確（取最大值或搬移）。 |

轉換型 conv（Winograd、FFT）與低精度 tactic 不在模型內，建置時以 HEAD_TRT 的 tactic 檢查拒絕（§5）。

## 3. 每個元素的尺度 σ 與界 E

σ 的意義與組合方式同 HEAD_POLICY §3（獨立、零均值、有界的捨入；可能共享來源者線性相加）。

| 步驟 | σ 規則 |
| --- | --- |
| 輸入 images | 0 |
| Conv＋BN | σ_y ＝ √(Σ w′²σ_x² ＋ 2·terms·u²·M²) ＋ 5u·M_fold，terms＝C_in/g·k_h·k_w＋1，M＝Σ\|w′x\|＋\|b′\| |
| SiLU、加法 | 同 HEAD_POLICY（head_ref::silu、head_ref::add） |
| 乘以常數 c | \|c\|σ_x ＋ u\|cx\| |
| MatMul y＝Σ_k a_k b_k | σ_y ＝ √(Σ_k (\|b_k\|σ_{a_k} ＋ \|a_k\|σ_{b_k})² ＋ 2K·u²·M²)，M＝Σ\|a b\| |
| Softmax（長度 N） | σ_{y_i} ＝ y_i(σ_{z_i} ＋ Σ_j y_j σ_{z_j}) ＋ (8ε_t ＋ u(\|z_i−m\| ＋ Σ_j y_j\|z_j−m\|) ＋ (√N＋1)u)·y_i ＋ 2⁻¹²⁶ |
| MaxPool | 視窗內 σ 的最大值（\|max x̂ − max x\| ≤ max \|x̂ − x\|） |
| Resize、Concat、Split 等 | 複製 |

E 是同結構的 first-order 最壞界（HEAD_POLICY §3）。經 88 層 conv 後，E 在 P3 ≥ 10¹⁷⁶、在 P5 溢出 binary64（+inf）：**只作報告**，判定不使用。

## 4. 固定輸入

`t01_backbone_ref.py inputs` 產生 `f03-inputs.f32`（6 × [1,3,640,640] FP32 little-endian NCHW，SHA-256 `14dda6867e23ed8a2d83086224d71a445a7fdcb606c7507a3f39c0e5faf60aa0`）：

| 標籤 | 內容 | SHA-256 |
| --- | --- | --- |
| `MOT17-02-SDP/1` | T01 輸入契約 `t01-input/1`（nvJPEG 硬體 decode、/255、antialiased bilinear stretch），PyTorch `fp32-strict`；與 HEAD_TRT §10 的真實 features 同三幀。02 屬 eval split（已開啟），此處只作數值對照，不用於選模 | `a99482f8…` |
| `MOT17-10-SDP/1` | 同上（val split） | `ca9388d1…` |
| `MOT17-13-SDP/400` | 同上（train split） | `4a30d942…` |
| `zeros` | 全 0 | `f824c1a8…` |
| `ones` | 全 1 | `830eadd8…` |
| `random/1` | numpy `PCG64(1)` 的 `Generator.random(float32)`，[0,1) 均勻 | `a30e585e…` |

## 5. Backbone 判定規則（`t01-backbone-fp32/1`）

**數值（主判定）**：六個輸入、p3／p4／p5 的**全部元素**（每輸入 2,662,400 個）：\|trt − ref\| ≤ λσ ＋ 2⁻¹¹⁰，**λ＝16**（理由同 HEAD_POLICY §4），且全部有限。報告欄位同 HEAD_POLICY §4。

**次要（沿用 T01 事先固定的容差**，[`t01_verify_features.TOLERANCES`](../training/t01/t01_verify_features.py)**）**，三個真實幀、每個 level：
- rel L2(TensorRT, PyTorch `fp32-strict` GPU) ≤ 10⁻⁴（`fp32_implementation_rel_l2`）；
- rel L2(訓練 cache 的 FP16 features, TensorRT) ≤ 2⁻¹¹ ＋ 10⁻⁴（`train_vs_deploy_rel_l2`；02／10／13 分屬 cache_eval／cache_val／cache_train）。

**Engine**：strongly typed FP32、TF32 off、builder optimization level 0、workspace 256 MiB、auxiliary streams 0、不匯入或匯出 timing cache；ONNX 以 SHA-256 讀入；輸入恰為 `images`、輸出恰為依序的 `p3_feat`、`p4_feat`、`p5_feat`，靜態 shape；`head_tactic_violations` 不得列出任何 tactic，否則建置失敗。重新建置的 plan 是否逐 byte 相同只作紀錄。

**執行**：明確的 non-default stream；同一 context、同一輸入重跑逐位元相同；CUDA graph 重播與 enqueueV3 逐位元相同。

**語義變體（必須全部被偵測）**：在 `MOT17-02-SDP/1` 與 `random/1` 上，下列 12 個變體的 reference 都必須有元素落在真實 reference 的 λσ 之外，且 TensorRT 輸出也都必須有元素落在變體 reference 的 λσ 之外：BGR 輸入、BN 無 eps、bilinear upsample、layer 12 concat 次序反轉、3×3 kernel 翻轉、SPPF 無 shortcut、SPPF kernel 3、attention 無 scale、softmax 取錯軸、attention 無 pe、C3k2 chunk 兩半互換、未訓練的 base 權重。

**GPU 執行前已完成的 CPU 預檢**（`t01_backbone_ref.py precheck`，`backbone-precheck.json`）：

| 檢查 | 結果 |
| --- | --- |
| PyTorch FP32 CPU（glibc 的 exp），六個輸入 | 全部元素在 λσ 內，最大 誤差／門檻 **0.024**（必要條件） |
| 12 個語義變體 vs reference | **全部偵測到**；每個變體在每個受測 level 都有 ≥ 91% 的元素落在 λσ 之外 |
| ONNX Runtime 1.24.4 FP32 CPU（部署的 ONNX），六個輸入 | 341 個元素超界（最大 102 倍），**全部是 \|ref\| ≤ 7.7×10⁻⁵ 的 SiLU 尾端**。直接量測其 Sigmoid：絕對誤差約 6×10⁻⁸，但 x < −8 的相對誤差 2×10⁻⁴ 至 100%（x ≤ −18 回傳 0），不符合 ε_t 的相對精度假設。只作紀錄：它顯示本規則也檢驗 pointwise 尾端的相對精度；若 TensorRT 的 FP32 sigmoid 有同樣性質，backbone 判定會失敗並依 §9 處理。 |

## 6. 失敗處理範圍

非有限輸入不進 backbone 數值比較；F03 的失敗路徑見 §7。

## 7. F03 判定規則（`t01-f03-fp32/1`）

**Engine**：同一個 strongly typed FP32 network 內先以 `nvonnxparser` 解析 backbone ONNX，再以 S04 的 network API 在其 `p3_feat`／`p4_feat`／`p5_feat` 上建 head（[HEAD_TRT §3](HEAD_TRT.md)；`HeadBuildOptions::features`），原本的 backbone 輸出改為內部 tensor。builder 設定同 §5；tactic 檢查涵蓋整個 engine；status 為 head 的九個 plugin statuses，加上 harness 對 `images` 與六個輸出的有限值檢查。head 的建圖程式碼與 S04 相同：同一程式建出的獨立 T01 head engine 必須與 HEAD_TRT §10 的 `5b6d30fa…` 逐 byte 相同。

**Reference**：`head_reference_from_refs`，輸入為 §1 backbone reference 的 value、E、σ；head 的 σ／E 規則即 HEAD_POLICY §3，只是輸入尺度改由 §3 提供而非 0。

**數值**：§4 的六個輸入、六個輸出的全部元素：\|gpu − ref\| ≤ 16σ ＋ 2⁻¹¹⁰（HEAD_POLICY §4 的比較器）。

**執行行為**（HEAD_TRT §5 的 frame session）：每個輸入的 frame 都被接受（九個 plugin statuses 與兩個有限值檢查都是 OK）；同一輸入重跑逐位元相同；CUDA graph 重播與 enqueueV3 逐位元相同；失敗序列 OK／FAILED／SKIPPED（`images` 內一個 NaN），committed 結果不變；未註冊 plugin 時無法建立 execution context；engine hash、build report 與 source identity 記錄。

**Sanitizer**：F03 run 的 memcheck 與 initcheck，限第一個真實輸入加失敗序列與 graph 重播，沿用記憶體監測（可用記憶體低於 4 GiB 即停）；racecheck／synccheck 不執行，明列為未驗證。

**診斷（不作驗收）**：F03 輸出對訓練側 PyTorch（backbone `fp32-strict`＋T1 head）的相對 L2 與 decode 後偵測；以及對「backbone engine 的輸出接 head engine」兩段式結果的相對 L2。

## 8. 支援範圍與限制

- 運算模型假設直接求和與相對精度的 exp／sigmoid（§5 的 ONNX Runtime 觀察）；Winograd／FFT／低精度 tactic 在建置時拒絕。
- 判定只涵蓋 §4 的六個輸入、本機 RTX 5070 Ti Laptop（sm_120）與 TensorRT 10.16.1.11；不聲稱跨 GPU 或跨版本一致。
- 輸入是訓練側產生的 FP32 tensor；native nvJPEG decode 與 resize（F01／F02）另行驗收。

## 9. 修訂規則

GPU 驗收失敗時：先定位首次偏差（backbone 可另建標出中間 tensor 的診斷 engine）；屬實作錯誤則修正實作重跑；屬模型假設不成立（pointwise 精度、tactic、BN 處理）則以 `t01-backbone-fp32/2`／`t01-f03-fp32/2` 提交修訂、理由與受影響結果，原版本與失敗結果保留在 evidence。
