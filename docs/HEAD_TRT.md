# S04：完整 YOLO26m T=1 Mamba-FPN head 的 native FP32 TensorRT 整合

S04 把 [S01](SCAN_CONTRACT.md)／[S02](SCAN_CUDA.md)／[S03](SCAN_TRT.md) 接成完整的原版 m head：**固定權重身分 → native 權重 bundle → native TensorRT 建圖 → 獨立 binary64 reference 對照全部 raw cls／reg → 既有 frame 完成與失敗流程**。邊界是 **P3／P4／P5 features 輸入到六個 raw outputs**；FP32，batch 1，T=1，不保留跨 frame 狀態。不含新訓練、FP16、效能最佳化、decode／NMS、tracker、GMC、MOT writer，也不含 backbone／neck 推論。

本文件描述契約與重跑方式；數值政策固定在 [HEAD_POLICY](HEAD_POLICY.md)（`s04-head-fp32/1`，於 `4d38bcc` 先於任何 GPU 比較提交）。**S04 PASS**，正式結果來自 clean-source archive 的 [S04 證據](../evidence/s04/2026-10-30/REPORT.md)，範圍限指定候選權重、固定介面與受測 FP32 head 的數值、執行與失敗傳遞。head racecheck／synccheck 未執行；P01 仍 OPEN。開發紀錄、失敗 attempt 與未完成事項見 [§9](#9-執行紀錄與待完成事項)。

## 1. 固定範圍

| 項目 | 值 |
| --- | --- |
| 模型 | YOLO26m backbone／neck 輸出 P3／P4／P5（256／512／512 ch，80²／40²／20²）＋原版 m T3→T1 Mamba-FPN head |
| head 設定 | `d_model=128`、`d_inner=256`、`d_state=16`、每尺度 1 個 spatial block、spatial reduction 4、四方向 Cross-Scan、PixelShuffle 4、C rank 1、shared A、`reg_max=1`、80 類 |
| 精度 | FP32 資料與運算；TF32、FP16、INT8 關閉；TensorRT 10.16.1.11 strongly typed network |
| 時間 | T=1；temporal blocks 與 flow gate 不執行；每次呼叫 scan 零初始狀態 |

## 2. 權重候選與 native bundle

**候選**：使用者指定的 S04 工程驗證候選 `runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt`（舊庫，126,802,369 bytes，SHA-256 `a5a0e7091cd854ad68210ae4bbd43af73c7d3d098ea119f02ea65fab59c0676c`，epoch 15）。其 `selection.status=diagnostic_train_loss` 只被記錄，**不是**品質驗收；不重訓。既有 `mamba_head_26m.onnx`（權重屬 `best_recall.ckpt`）與舊 engine 都不使用。

**轉換**（[`head_weights convert`](../tools/head_weights.cpp)，[`torch_checkpoint`](../src/torch_checkpoint.cpp)、[`head_weights`](../src/head_weights.cpp)）只讀資料：

1. 先核對整個檔案的 SHA-256，不符即拒絕。
2. zip：只接受 STORED entry；EOCD／zip64 記錄一致；local／central header 一致；data descriptor 的 CRC 與大小須等於 central directory；每個 entry 驗 CRC-32；拒絕重複名稱、重疊或越界資料。
3. pickle：只解讀 torch.save 用到的容器 opcodes；GLOBAL 只記名稱，REDUCE／NEWOBJ 只成為物件節點，**不解析、不呼叫、不匯入**任何類別；未知 opcode、堆疊／memo 錯用、STOP 後多餘 bytes 都拒絕。
4. 設定：`mamba_args` 的 in_channels、d_model、d_state、num_blocks、spatial_reduction、num_classes、cross-scan、PixelShuffle、per_channel_a、legacy_n1_scan、hybrid／temporal attention／detail 開關逐一核對；缺 `reg_max`／`head_depth`／`reduction_variant` 時由參數形狀推導（4 個 reg 輸出、每支一個 3×3 conv 無 BN、4×4 stride-4 conv）並在報告註明。
5. 參數：`student` 的 108 個 key 必須恰為 69 個 active＋39 個 excluded；缺少、未知或重複 key 拒絕。active tensor 須為 `torch._utils._rebuild_tensor_v2`、`FloatStorage`、形狀等於 schema、row-major 連續 stride、view 落在 storage 內、storage record 恰為 numel×4 bytes、全部有限值。

| 排除的 key | 數量 | 理由 |
| --- | ---: | --- |
| `temporal_blocks.{0,1,2}.0.*` | 27 | temporal Mamba 只在 T_frames > 1 執行；T=1 forward 繞過 |
| `flow_gate_conv.{0,1,2}.{0,2}.{weight,bias}` | 12 | temporal 路徑的 flow gate，需要 T>1 與光流 |

**映射**：bundle 名稱為 `<level>.<module>`（level ＝ p3／p4／p5 ＝ checkpoint index 0／1／2），例如 `mamba_blocks.1.0.x_proj.weight → p4.mamba.x_proj.weight`。**不轉置、不 alias、不去重**：權重 bytes 與 PyTorch 布局（Conv OIHW、Linear (out,in)、Conv1d (C,1,K)）原樣保存；Linear 的轉置由 TensorRT matrix-multiply 的 `kTRANSPOSE` 表達，Conv1d kernel 在建圖時以 (256,1,1,4) depthwise 2-D kernel 讀同一份 bytes。三層的 `A_log`（ln 1…16）與 `D`（全 1）bytes 相同，仍各自保留並在報告列為 `identical_bytes`。逐 key 的 storage key／offset／stride／shape／SHA-256 與映射見 evidence 的 `t1-head-mapping.tsv`；108 個 tensor 的 SHA-256 與 S01 以 `unzip` 取得的稽核完全相同。

**Bundle 格式 `SCWB 1`**：ASCII header（`SCWB 1`、`schema yolo26m-mamba-fpn-head-t1/1`、`source_sha256`、`source_archive`、`source_epoch`、`tensors 69`，再依 schema 次序每個 tensor 一行 `tensor <name> f32 <d0>x<d1>… <offset> <bytes> <sha256>`，`end`），零填充到 64-byte 邊界；之後是 little-endian FP32 資料，每個 tensor 起點 64-byte 對齊、零填充。解碼要求名稱次序、dtype、形狀、範圍、逐 tensor SHA-256、有限值，並**重新編碼必須與輸入逐 byte 相同**（拒絕非標準 offset、padding 或尾端資料）。候選 bundle：40,711,552 bytes，SHA-256 `e1a8660fb9ad510f20ccc3d56a4feac61c6a2e5f771aab531caa741e7677f310`，存於 `~/.cache/saccade-v1/s04/`（不入 Git）。

## 3. 計算圖與 TensorRT 建圖

以 TensorRT network API 直接建圖（[`head_trt_network.cpp`](../src/head_trt_network.cpp)），沒有 ONNX 匯入或圖修改；Cross-Scan 與 SelectiveScan 為 S03 plugins（`saccade::CrossScanExpand` v1、`saccade::SelectiveScan` v2 `has_d=1`、`saccade::CrossScanMerge` v1 height=width=20／10／5）。每層：

| 步驟（SCAN_CONTRACT §2） | TensorRT 實作 | 形狀（P3） |
| --- | --- | --- |
| 1 input_proj | Conv 1×1＋bias | (1,128,80,80) |
| 2 downsample | Conv 4×4 stride 4＋bias | (1,128,20,20) |
| 3 Cross-Scan 展開 | CrossScanExpand plugin | (4,400,128) |
| 4a in_proj | MatMul(tokens, W(1,512,128)ᵀ)；slice x＝[:256]、z＝[256:] | (4,400,512) |
| 4b causal Conv1d＋SiLU | transpose→(4,256,1,L)；depthwise Conv (1,4)，padding (0,3)，groups 256＋bias；**取前 L**；轉回 (4,L,256)；sigmoid×x | (4,400,256) |
| 4c x_proj | MatMul→(4,L,33)；dt＝[0:16]、B＝[16:32]、C＝[32:33] | — |
| 4d dt_proj | MatMul＋bias（delta bias 在 softplus 之前，softplus 在 plugin 內） | (4,400,256) |
| 4e A | constant A_log → EXP → NEG（圖內） | (1,16) |
| 4f scan | SelectiveScan v2 (u, delta, A, B, C, D) | (4,400,256) |
| 4g gate／out_proj／殘差 | y·SiLU(z)；MatMul (128,256)ᵀ；＋tokens | (4,400,128) |
| 5 合併 | CrossScanMerge plugin | (1,128,20,20) |
| 6 upsample＋PixelShuffle | Conv 3×3 pad 1＋bias → 2048 ch；reshape (1,128,4,4,20,20)，transpose (0,1,4,2,5,3)，reshape (1,128,80,80)＝**CRD** | (1,128,80,80) |
| 7 concat、cls／reg | concat **[input_proj, upsampled]**；Conv 3×3＋SiLU＋Conv 1×1 → 80／4 | (1,80,80,80)／(1,4,80,80) |

**I/O**：輸入 `p3 [1,256,80,80]`、`p4 [1,512,40,40]`、`p5 [1,512,20,20]`；輸出依序 `cls_p3, cls_p4, cls_p5, reg_p3, reg_p4, reg_p5`（raw logits／raw ltrb，無 sigmoid、DFL、decode 或 NMS）；另有 9 個 INT32[2] plugin status outputs：`p{3,4,5}.{expand,scan,merge}_status`，由建圖時實際建立的 plugin 呼叫決定，`build_head_engine` 要求它們等於 [`head_io_spec()`](../include/saccade/head_trt_network.hpp)。

**Builder**：strongly typed FP32、清除 TF32、workspace 256 MiB、auxiliary streams 0、無 timing cache、profiling verbosity detailed、**builder optimization level 0**。開發中以 level 3 建置時，P3 upsample 與三層融合的 cls／reg 3×3 conv 選到 Winograd tactic（`…_fprop_wngd_…_wngd2x2`），不在 HEAD_POLICY §2 的直接求和模型內；level 0 只選 FP32 implicit-GEMM／GEMM／direct conv（FFMA）。建置後以 IEngineInspector 讀出每層 tactic，`head_tactic_violations` 拒絕 Winograd、FFT、FP16／BF16／TF32／HMMA／INT8 標記（以名稱 token 判斷，略過 `0x…` kernel hash）。layer 資訊與 tactic 名稱存入 build report。同一 source 兩次建置得到相同 plan bytes（evidence 記錄）；不同 build 之間的比較仍依 HEAD_POLICY，不要求逐位元相同。

## 4. Reference 與數值政策

[`head_reference`](../src/head_reference.cpp) 從同一份 FP32 feature bytes 與 bundle bytes 以 binary64 計算全部六個輸出，並逐元素計算兩種誤差量：機率尺度 σ（驗收）與 first-order 最壞界 E（只報告）。精度、捨入位置、各 primitive 的 σ 規則、λ＝16 的依據、近零與相消處理、報告欄位與 GPU 前已完成的 CPU 檢查見 [HEAD_POLICY](HEAD_POLICY.md)。要點：

- reference 不呼叫 TensorRT、CUDA 或 S02，只與 GPU 路徑共用唯讀的 bundle 解碼與 feature 產生器；GPU 中間值從不回灌。
- scan 步驟本身的誤差界逐位元等於 S02 的 SCAN_CUDA §5 bound（GPU 測試要求）；只有在此契約相符處沿用 S02。
- 每個 primitive 有手算／bc 推導的小案例；scan 與 S01 reference 逐位元相同；FP32 host 實作在 bound 內。
- 驗收輸入：`asymmetric` seed 1（座標／通道不對稱）、`random` seed 1、2（SiLU(1.5·N(0,1))）；全部 705,600 個輸出元素逐一判定。
- 定位：`HeadBuildOptions::debug_intermediates` 另建 debug engine，把 reference 的 17 個中間 tensor（每層）標為輸出，各自以同一規則比較，用來找首次偏差；debug engine 不是被驗收的 engine。

## 5. 六個輸出共同提交與 status 判定

[`ScanFrameSession`](../include/saccade/scan_trt_frame.hpp) 新增明確的 `FrameIoSpec`；沒有 spec 時維持 S03 的推定行為（一個 payload、1..4 status）。有 spec 時：

- **初始化核對**：engine 的 I/O 名稱集合必須恰等於 spec（不多、不少、不重複），mode、dtype、shape 逐一相符，status 必為 INT32[2]；否則建構失敗，不進入 frame。
- **slot 資源**：每個 slot 有自己的 execution context、context memory（16,386,048 bytes）、input arena、六個輸出所在的 **output arena**、11 個 status、pinned staging 與 `staged`／`ready`／`consumed` events。arena 內每個 tensor 在 256-byte 對齊的位置：

| 輸出 | arena offset（元素） | 元素數 |
| --- | ---: | ---: |
| cls_p3 | 0 | 512,000 |
| cls_p4 | 512,000 | 128,000 |
| cls_p5 | 640,000 | 32,000 |
| reg_p3 | 672,000 | 25,600 |
| reg_p4 | 697,600 | 6,400 |
| reg_p5 | 704,000 | 1,600 |

- **status（gate 次序）**：9 個 plugin status（`p3.expand`、`p3.scan`、`p3.merge`、`p4.…`、`p5.…`，生產者為對應 plugin 呼叫），加上 harness 的 `frame_input_check`（每個輸入元素的有限值）與 `frame_output_check`（arena 內每個輸出元素的有限值）。兩個 check 在 D stream、engine（或 graph）之後、`ready` 之前執行（memset → 掃描 kernel → `done`），旗標為 `kFrameNonFiniteInput`／`kFrameNonFiniteOutput`；gate 認得這兩個 bit。status 數量來自 spec（即實際圖），gate 上限 16。
- **判定**：gate 讀全部 11 個 status：`flags≠0` → NUMERIC；`done≠kScanDone` 或未知 bit → PROTOCOL；讀後 retire 為 {0,0}。guarded publish 以單一 kernel 把**整個 output arena** 複製到 working result version，所以六個輸出只能一起發布或都不發布；commit 只在無錯誤且已 publish 時前進。首次失敗鎖定 fail_frame／fail_code，後續 frame SKIPPED；ResultPacket v1 ABI 不變，host 只接受該 epoch／frame 的 OK packet。
- **所有權**：`consumed[slot]` 在 guarded publish（對 arena 與 status 的最後一次 M 讀取）之後記錄；D 重用 slot 前等待上一代 `consumed`。frame 流量的 D2H 只有 ResultPacket；raw tensors 只在兩個 terminal drain 之後讀為診斷。
- **非有限值**：權重在載入時檢查；frame 的輸入與最終輸出由上述 GPU check 判定，不因 scan statuses 成功就略過 scan 之後算子的輸出檢查。

## 6. 初始化、生命週期與命令

初始化沿用 S03：明確註冊 plugins → 綁定 device／context → deserialize → 每 slot `createExecutionContext(kUSER_MANAGED)`（plugins 在此 preload kernels）→ 配置全部資源、載入 frame kernels（含 check kernels）→ 每 slot warmup → retire status → 需要時以固定 context／shape／addresses capture graph（TensorRT 的 graph 含 programmatic dependent launch 的 typed edges，introspection 需讀 edge data）→ device synchronize。結束時 drain D、M，依所有權釋放（graph、contexts、engine、runtime，再釋放 context memory、buffers、pinned memory、events、streams），`close()` 檢查 release 錯誤。

```sh
# CPU（不需 GPU；給 checkpoint 才會跑候選轉換與真實權重的 reference 檢查）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_HEAD_CHECKPOINT=/home/ray/developer/ai/saccade/runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt
cmake --build build --parallel 8 && ctest --test-dir build --output-on-failure
build/head_weights convert CHECKPOINT a5a0e709…59c0676c t1-head.swb t1-head-mapping.tsv
build/head_weights verify t1-head.swb e1a8660f…7677f310

# CUDA + TensorRT
cmake -S . -B build-s04 -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_TENSORRT=ON \
  -DSACCADE_TENSORRT_ROOT=/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DSACCADE_HEAD_CHECKPOINT=CHECKPOINT
cmake --build build-s04 --parallel 8 && ctest --test-dir build-s04 --output-on-failure
build-s04/head_trt build BUNDLE BUNDLE_SHA head.engine build-report.txt          # checked save + readback + layers
CUDA_MODULE_LOADING=LAZY build-s04/head_trt run head.engine ENGINE_SHA BUNDLE BUNDLE_SHA run-report.txt
build-s04/head_trt run-unregistered head.engine
build-s04/head_trt_tests head.engine ENGINE_SHA BUNDLE BUNDLE_SHA report.tsv [--group NAME] [--skip NAME] [--no-held-stream]
```

`head_trt run` 是新程序：註冊、反序列化、初始化 session、三組輸入各自比較六個輸出、同一 engine 重跑逐位元相同、A／NaN-P4／A 失敗序列、CUDA graph 重播與 enqueueV3 逐位元相同、checked close 後正常退出；失敗時仍寫出已量測的部分。`run-unregistered` 確認未註冊 creators 時無法形成 execution context（TensorRT 10.16 將 plugin 實體化延後到 context 建立，同 SCAN_TRT §3）。CTest 未給 checkpoint 時以 `head_weights synthetic 3` 的合成權重建圖，只驗結構與失敗路徑，不代表任何模型。

## 7. 驗證矩陣

| 項目 | 測試 |
| --- | --- |
| 權重交付 | `head_weights_tests`：以 C++ 逐 byte 寫出的 torch.save 形式 archive 測 16 種拒絕（缺／未知／重複 key、形狀、dtype、stride、storage 範圍與大小、非有限、設定、壓縮 entry）、CRC／截斷／尾端資料、未支援 opcode、archive 目錄與 byteorder；bundle 的 SHA、截斷、padding、magic、次序、形狀、dtype。`head_weights_candidate`：真實 checkpoint 的兩個 hash 與錯誤 hash 拒絕。 |
| 完整數值 | `head_trt run`／`head_trt_tests numeric`：三組輸入 × 六輸出全部元素依 HEAD_POLICY；`intermediates`：51 個中間 tensor。 |
| 語義邊界 | `head_reference_tests`：13 個語義變體（候選權重下全部超出 λσ）；`io_contract`：feature／output 次序互換、status 與 output 角色互換、缺漏／多餘 status 被拒；mutation（見 S04 evidence）。 |
| Frame 原子性 | `atomicity`：11 個 status × {有限 payload＋失敗 flag、缺 done、未知 bit}、六個輸出各一個 NaN、三個輸入各一個 Inf：失敗 frame 與其後 SKIPPED 都不發布，committed 仍是前一 frame 的六輸出 bits。 |
| 所有權 | `ownership`（slots 1／2，gate 延遲 4,000,000 cycles，逐 frame digest）、`isolation`（兩個 session 交錯、失敗後 reset）。 |
| 生命週期 | `graph`（只含 kernel／memset，重播逐位元相同，graph 模式失敗序列）、`streams`（held stream 下 enqueueV3 與 graph launch 先返回）、`api_trace`（CUPTI：無同步、配置、module loading、D2H，stream 正確）、`head_trt run`／`run-unregistered`。 |
| 回歸 | 既有 CPU 與 CUDA／TRT CTest（S01–S03 全部）。 |

## 8. 支援範圍與未完成事項

- S04 head PASS 的意思是：**指定權重、固定介面與受測範圍內**，完整 FP32 head 的數值、執行與失敗傳遞已驗證。它不代表 detector、模型品質或 P01 通過。head suite 的 memcheck／initcheck、native ownership／graph／streams 與 S03 四種 sanitizer 有正式證據；**完整 head 的 racecheck／synccheck 未執行**，不能列為 sanitizer PASS。
- **P01 仍 OPEN**：matching YOLO26m backbone／neck 只完成舊 `yolo26m_backbone_640.onnx` 的 ordered I/O 結構查核（`images [b,3,640,640]` → `p3_feat`／`p4_feat`／`p5_feat`，形狀符合 head 輸入）；其權重歸屬、native engine、與 T1 head 的配對與推論未驗；影像前處理（RGB、/255、640 stretch、decode 後端）未實作；decode／NMS 與候選容量（F04）未定；候選 checkpoint 以 training loss 選取、無 holdout，品質未驗收。
- 不支援：FP16／INT8、batch≠1、動態 shape、T>1 temporal 路徑、Winograd／FFT tactics（建置時拒絕）。S02 的範圍灰區與次正常數限制照常適用。
- 舊 `mamba_head_26m.engine`／`.onnx` 未作執行基線；本輪 engine 由 v1 source 自建，身分記於 build evidence。

## 9. 執行紀錄與待完成事項

狀態（2026-10-30）：**S04 PASS，受測 source `3857e243e8fc32663aa688f073343002ad12cb30`。** 正式判定採 [capture.sh](../evidence/s04/2026-10-30/capture.sh) 從 clean HEAD 的 archive 重建後的結果，完整驗證表、執行身分、原始 logs 與限制見 [S04 報告](../evidence/s04/2026-10-30/REPORT.md)。下列開發數字只記錄進度與定位，與正式證據分開。

| commit | 內容 |
| --- | --- |
| `4d38bcc` | 權重交付、binary64 reference、[HEAD_POLICY](HEAD_POLICY.md)；提交時沒有任何完整 head 的 GPU 結果 |
| `222af61` | 完整 head、harness 擴充、`head_trt`／`head_trt_tests`、manifest schema 2、capture 與 mutation 腳本（第一次正式 capture 的受測 source） |
| `dc016ed` | capture：head sanitizer 改報 explicit API 錯誤，另以一次預設模式 memcheck 分類所有回報項目 |
| `0f5829d` | capture 強化，第二次正式 capture 的受測 source（該 attempt 不通過）：明列 head racecheck／synccheck 未執行；保留 memcheck／initcheck、S03 四工具與正控制；新增全程記憶體取樣、逐步 clean-source 守門、engine bytes 比對與 mutation 拒絕診斷檢查 |
| `198ad63` | 僅修正 TF32 mutation 的拒絕診斷 matcher，補上實際的 `head engine build:` 前綴並要求 TF32 tactic token；第三次正式 capture 的受測 source（該 attempt 不通過） |
| `3857e24` | 正式受測 source：S03 四種 sanitizer 也報 explicit API 錯誤，head 與 S03 各以獨立預設模式 memcheck 分類 driver 記錄；完整重跑第四次 capture |

**開發結果。** 三組輸入 × 六輸出全部元素在 λσ 內（最差 誤差／門檻 0.0063，`asymmetric/1` 的 `cls_p4`）；51 個中間 tensor 最差 0.013；同一 engine 重跑與 graph 重播逐位元相同；兩次建置 plan bytes 相同（SHA-256 `ca74a50d04980b7b15ffd66a968e4851459c4fb8c3fa2db0724d9d55f69509c1`，41,471,380 bytes）；`head_trt_tests` 824 checks、api_trace 通過；CUDA＋TRT CTest 31/31、CPU 13/13、ASan／UBSan 13/13；g++-13 與 clang 的 CPU 建置與測試通過；head suite 的 memcheck（explicit API 報告）與 initcheck（預設報告）0 錯誤。mutation 開發 pass 23/25 被測出，未測出的兩項已處理（見下）。

**第一次正式 capture（受測 `222af61`）：不通過。** 完整建置目錄 `build-s04-evidence-attempt1/` 不入 Git；其已產生的 logs 存於 `capture/failed-attempt/`。 CPU、ASan／UBSan、CUDA＋TRT CTest、權重轉換、`head_trt run`（數字與開發結果相同）、manifest verify、backbone ordered I/O 都 exit 0；在 head suite 的 memcheck 停止：suite 640 checks 全部通過、沒有記憶體錯誤，但預設 `--report-api-errors extended` 把 driver 記錄「Selective device code recompilation in progress」（工具重新插樁 TensorRT 的 kernel library）計為 1 個錯誤，exit 99。另外此次 attempt 的建置身分為 `build_dirty=true`，因為 capture 進行中修改了工作樹文件；下次 capture 期間必須保持工作樹不變。

**第二次正式 capture（受測 `0f5829d`，本機 `build-s04-evidence2/`）：不通過。** 26 個 mutation 中 25 個記為 `DETECTED`；TF32 mutation 的 tactic guard 已正確拒絕 TF32 engine，但新增的診斷 matcher 漏掉實際訊息的 `head engine build:` 前綴，將該次拒絕記為 `INVALID_RUNTIME`，capture 因而失敗。這是驗證腳本的診斷分類錯誤，並非 GPU 數值比較失敗；仍保留為失敗 attempt，不納入 PASS 證據。`198ad63` 只修正這一行 matcher；第三次完整重跑另遇下述 S03 sanitizer 記錄分類問題，各 attempt 的來源與結果分開記錄。

**第三次正式 capture（受測 `198ad63`，本機 `build-s04-evidence3/`）：不通過。** head memcheck 完成 640 checks、0 errors；S03 memcheck 的 842 checks 全部通過，受測 build 為 `build_dirty=false`，但預設 `--report-api-errors extended` 把同一句 driver 記錄「Selective device code recompilation in progress」（`cuModuleLoadFatBinary`，TensorRT builder kernel library）計為 1 error，工具 exit 99，capture 因而停止。先前分類處理只適用 head、漏掉 S03 suite，這次是 sanitizer 記錄分類適用範圍不完整；並非 GPU 數值失敗，也沒有回報記憶體存取錯誤。仍保留為失敗 attempt，不納入 PASS 證據。第四次受測 source `3857e24` 將 S03 四工具也改為 explicit API 報告，另跑獨立的預設模式 S03 memcheck 分類；歷史 S03 證據不改寫。

**WSL VM 當機與程序記憶體觀察。** 在開發機試跑 head suite 的 racecheck（先全範圍，後改 `--group ownership`）期間，WSL VM 於約 11:41 當機重開（`last -x` 記為 crash）。**使用者補充的確定觀察：被 Compute Sanitizer 檢查的 `head_trt_tests` 程序吃滿記憶體；目前仍無法區分工具的追蹤資料開銷與測試程式本身洩漏。** 這項程序用量觀察與 VM 當機根因分開記錄，**當機原因仍未判定**。NVIDIA 的 [Compute Sanitizer memory footprint 文件](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memory-footprint) 說明追蹤資料可能很大，並可能造成 host allocation 失敗或 crash；這提供待驗證的解釋，不能證實本例根因。主機沒有 low-memory 事件；主機在 10:28、10:53、11:06 有 NVIDIA TDR（`LiveKernelEvent 0x141`, `nvlddmkm.sys`），時間對應含 S03 `scan_trt_async_fault`（故意觸發 GPU fault）的 CTest，當機時間沒有主機事件。全範圍 racecheck 曾以約 19 個核心執行 16 分鐘以上（host 端分析 TensorRT kernels 的 shared-memory 存取）。

**開發中的數值觀察（詳見正式報告）。** (1) 合成隨機權重下 scan 分支耦合弱，三個 scan 內部語義變體不超出 λσ，所以合成權重的變體檢查只作資訊；候選權重下 13 個全部超出。(2) σ 由 SiLU／exp 的 2⁻¹⁷ 假設主導：只拿掉內積的局部捨入項不改變任何判定，政策偏保守。(3) TF32 主要由建置 guard 擋下；暫時關掉 guard 時，數值政策只在 6 個輸出中的 5 個判超界（最大 2.22 倍、`reg_p4` 0.82 倍），不是 HEAD_POLICY §7 所寫的「遠超」。

**正式 capture 與 sanitizer 範圍。** 新受測 source 將 head racecheck／synccheck 明列為未執行：TensorRT conv／GEMM 插樁成本極高，且開發試跑期間曾發生上述原因未判定的 WSL 當機。S04 新增的 `check_finite`／`check_done` 不使用 shared memory 或 barrier；沿用的 packet kernel 有 barrier，仍由 S03 suite 的 synccheck 覆蓋。正式 capture 保留 head memcheck（完整預設 suite 但不重建 debug engine）、initcheck（略過 callback-held stream）、S03 四種 sanitizer 與四個正控制。head 與 S03 suite 都採 explicit API 錯誤報告，並**各自另跑一次預設模式 memcheck**，只允許已列明的 driver recompilation 記錄、逐項分類並保留 raw log；native head ownership／graph／streams 照常執行。完整 skip 清單與各 raw log 見 [正式報告](../evidence/s04/2026-10-30/REPORT.md)。

[capture.sh](../evidence/s04/2026-10-30/capture.sh) 在每一步前後拒絕 HEAD／工作樹變動，核對報告中的受測 SHA 與 `build_dirty=false`；同一 source 兩次 engine 建置必須逐 byte 相同。mutation 只接受 exit 1 與指定的 test／tactic guard 拒絕診斷，timeout、crash 或無關 I/O 失敗不算 detection。全程 `free -m` 取樣連同三次失敗 attempt 的 source／archive hash、exit codes、memcheck raw log 或 mutation 分類結果保留於 evidence；三次失敗分別存於 `capture/failed-attempt/`、`capture/failed-attempt2/`、`capture/failed-attempt3/`，第四次正式結果來自本機 `build-s04-evidence4/`。

**後續缺件。** [MODEL_FEASIBILITY](MODEL_FEASIBILITY.md) 記錄 S04 判定；本輪 source 與後續 evidence／docs commit 的推送、遠端 SHA 與 CPU CI 於交付回報分開記錄。本輪不關閉 P01：matching backbone／neck、影像前處理、decode／NMS 與模型品質仍需各自驗收。
