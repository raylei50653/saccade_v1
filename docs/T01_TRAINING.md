# T01：YOLO26m＋Mamba-FPN 重新訓練

T01 從官方 YOLO26m 權重出發，重跑 `Teacher → Feature Cache → Distill → GT1 → T3 → T1`，產生新的 v1 候選模型。這是**新的訓練實驗**：不宣稱逐位元重現舊 checkpoint，也不繼承舊模型的品質結論。架構與 scan 語義沿用 [SCAN_CONTRACT](SCAN_CONTRACT.md)，部署為 T=1。

本文件分兩部分：§1–§7 是**正式訓練前固定的規格**（與第一個正式 run 一同提交，提交時沒有任何正式訓練或驗證結果）；§8 起是執行紀錄。訓練、選模與評估使用 PyTorch／Python；v1 的推論仍是 C++／CUDA／TensorRT，不引入 Python runtime 依賴。

## 1. 起點與來源身分

| 項目 | 固定內容 | 核實方式 |
| --- | --- | --- |
| 工程起點 | saccade_v1 `0b63f6f`（S04 交付），分支 `t01-yolo26m-retrain` | `git log`；S04 計算語義、ResultPacket ABI、FP32 政策與既有證據不改 |
| 預訓練權重 | `yolo26m.pt`，44,255,705 bytes，SHA-256 `401cea9a…0745d0b7` | 與 [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) 的 asset 340096038 API digest 逐字相同；授權為上游宣告的 AGPL-3.0 |
| 訓練程式 | 舊庫 `raylei50653/saccade` 的 `e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2`（Apache-2.0），以 `git archive` 匯出 `src scripts pyproject.toml LICENSE NOTICE uv.lock` 後套用 [T01 patch](../training/t01/patches/t01-e48f3b39.patch) | [source.lock.json](../training/t01/source.lock.json)：archive、pristine tree、patch、patched tree 的 SHA-256；舊庫本身不修改 |
| 環境 | Python 3.12.13；torch 2.11.0+cu130、CUDA 13.0、cuDNN 9.19；torchvision 0.26.0（內附 nvJPEG 13.0.1，`libnvjpeg.36e11081.so.13` SHA-256 `1a359ba7…e38f1d74`）；ultralytics 8.4.37；numpy 2.4.4 | 專用 venv（[requirements.lock.txt](../training/t01/requirements.lock.txt)），不使用舊庫 `.venv`（其 `.pth` 會把舊 `build/` 編譯擴充放進 `sys.path`） |
| 硬體 | RTX 5070 Ti Laptop（sm_120，12 GiB）、Ryzen 9 8940HX 32 threads、47 GiB RAM、WSL2 | 每個 attempt 的 `attempt.json` 記錄 |

舊庫只作為程式來源與規格參考，不視為原始訓練環境的重現；舊庫歷史訓練當時的 revision 為 dirty `06e539ce`，與此處的 `e48f3b39` 不同。

### 1.1 Patch 內容（必要修改）

| 檔案 | 修改 | 理由與語義影響 |
| --- | --- | --- |
| `mamba_head.py` `_selective_scan` | 不再呼叫舊編譯擴充 `saccade_tracking_ext`（舊 `mamba_scan.cu`）；需要 autograd 時用原有 pure-PyTorch JIT scan，不需要時用同一遞迴的逐步形式（不展開 (B,L,D,N) 張量） | 不沿用舊 CUDA kernel。逐步形式與 JIT 在 CPU／GPU 上**逐位元相同**（L=400／25 的 spatial 形狀與 T3 temporal 形狀 C 寬 16 都驗過）。舊訓練用的 kernel 與 JIT 數學相同、FP32 捨入不同 |
| `mamba_head.py` `MambaBlock.forward` | `scan_stop_grad=true` 時在 `torch.no_grad()` 下算 scan | 原本輸出立即 `detach()`，所以數值與梯度拓撲都不變；只省掉不會用到的 autograd 圖 |
| `mamba_head.py` legacy N=1 | 非 autograd 路徑改為拒絕 | T01 配方不用 `legacy_n1_scan` |
| `train_mamba_head.py`、`train_mamba_gt.py` | 移除 `import saccade_tracking_ext` 與 `build/` 路徑 | 同上；另以 [t01_exec.py](../training/t01/t01_exec.py) 在每個訓練行程拒絕匯入任何舊編譯擴充，並要求 `saccade` 解析到匯出的 tree |
| 三個 trainer 的 `save_checkpoint` | 每個 checkpoint 加 `t01_identity`（source／patch／tree／v1 commit／recipe／split hash、stage、attempt）與 `t01_compute_precision`（存檔時在程序內讀回並檢查的計算精度，§2.2） | 只新增 root key；S04 轉換器只讀 `epoch`／`selection`／`mamba_args`／`student` |
| `train_gated_detector.py` `_git_state` | 改報 T01 identity | 匯出的 tree 不是 git checkout，原函式會把未知記成 `clean` |
| `dataset.py` `gpu_decode_clip_batch` | 先除 255 再 stretch，改為 antialiased bilinear，回傳 [0,1] | §2.1 的輸入契約；cache 與 GT1 的 GPU decode 路徑共用此函式 |
| `train_gated_detector.py` | 新增 `--gpu-decode`（資料載入回傳 JPEG bytes，迴圈內以上述函式解碼） | teacher 訓練與 cache／GT1 用同一前處理；拒絕與 `--adapt-bn`／`--augment` 併用 |
| `train_mamba_head.py` `write_frame`、`train_mamba_gt.py` | GPU 路徑不再除 255 | 配合函式回傳 [0,1] |
| `train_mamba_head.py` cache manifest；`train_mamba_gt.py` 讀 cache | 建 cache 時 manifest 加 `storage_dtype`、`compute_precision`，並在開始與完成前檢查程序精度；distill 與 T3／T1 讀 cache 時要求兩欄位為 `float16`／`fp32-strict` | §2.2；舊 TF32 cache 不再相容 |
| `dataset.py` CPU 載圖函式 | `_preload_images`、`_load_and_resize`、`_load_detail_view` 在 `SACCADE_T01_GPU_DECODE_ONLY=1` 時直接拋錯 | §2.1 的強制點：T01 的每個訓練、評估與匯出行程都設此變數，CPU decode 無法被誤用 |

Loss、optimizer、scheduler、assigner、GT 篩選、clip 取樣、cache 格式與模型模組都未修改。

**與舊訓練實際用的 scan kernel 的差距**（2026-10-30，舊庫原環境與其 build 唯讀執行，舊 teacher e12＋舊 T1 候選 `a5a0e709…`，4 張真實 frame）：舊 CUDA kernel 與 pure-PyTorch scan 的 scan 輸出最大差 2.4e-7（相對 L2 2.7e-8）；六個 head 輸出相對 L2 ≤ 4.4e-7、最大絕對差 9.2e-5（logit 量級約 58）。差距在 FP32 捨入層級，不影響訓練語義。

## 2. 資料切分

版本化 manifest：[split.json](../manifests/t01/split.json)（每個 sequence 的 seqinfo、frame 數、影格清單 hash、GT hash 與統計、variant 與洩漏檢查）與 [split-files.tsv](../manifests/t01/split-files.tsv)（每張影像、`gt.txt`、`seqinfo.ini` 的 SHA-256 與大小）。產生工具 [t01_split.py](../training/t01/t01_split.py) 只讀標註與像素，不執行模型。

| 角色 | Sequences（SDP 目錄） | Frames | 有效 GT boxes | 最短邊 4–8／8–16 px |
| --- | --- | ---: | ---: | --- |
| train | MOT17-04、05、09、11、13 | 4,062 | 73,762 | 3,448／14,463 |
| val（選模） | MOT17-10 | 654 | 11,117 | 1,884／4,527 |
| eval（最終獨立評估） | MOT17-02 | 600 | 10,993 | 2,259／4,874 |

最短邊是 640 stretch 後的 min(w·640/W, h·640/H)。

- **隔離層級：sequence。** val／eval sequence 的每一幀都不進入 teacher、cache、distill、GT1、T3、T1 的訓練；每個 stage 以 `--seqs` 明列 train，GT 系列另以 `--holdout-seqs` 記錄 val＋eval。
- **選擇理由：** 只有 02、10、13 含大量 4–8 px 目標；若以 09 作 val，選模完全看不到小目標。13 留在 train 作小目標來源。val（移動鏡頭、夜間）與 eval（固定鏡頭、白天）條件不同。
- **Detector variant：** DPM／FRCNN／SDP 三個目錄的 `img1` 與 `gt.txt` 逐 byte 相同（七個 sequence 都核對），所以只讀 SDP，隔離以 base sequence 為單位。
- **重複／鄰近影格：** 32×32 灰階縮圖的 normalized cross-correlation，val 對 train 最大 0.587、eval 對 train 0.551、eval 對 val 0.415，門檻 0.95 以上 0 幀；同一 sequence 相鄰影格的中位數為 0.963–0.9998。時間鄰近只存在於同一 sequence 內。
- **限制：** val 與 eval 各只有一個 sequence，與 train 同屬 MOT17 train release，是 held-out sequence，不是 test-server 成績；MOT17 test sequences 沒有公開 GT，不使用。本機 MOT17 沒有下載來源、封存 hash 或授權檔；MOTChallenge 公告的授權為 CC BY-NC-SA 3.0（未由本機 artifact 核實）。
- GT 篩選與訓練相同：`mark==1`、class 1（行人）、visibility ≥ 0.1；行人對應 detector class 0。

### 2.1 輸入前處理契約 `t01-input/1`

**使用者決策（2026-10-30）：訓練輸入對齊 v1 的 F01／F02**（[DATAFLOW](DATAFLOW.md) F01 nvJPEG、B04「float32 正規化至 0–1，再 stretch 到 640×640」）。正式訓練前比較了三種路徑，影響如下（預訓練 YOLO26m，train sequence 的 8 張 frame，FP32 不用 TF32；特徵差為整張 map 的相對 L2，取最差 frame）：

| 與原本的 CPU 路徑（libjpeg-turbo＋uint8 antialias resize＋取整）比較 | 像素差 | P3／P4／P5 特徵差 |
| --- | --- | --- |
| 只換 decoder（全解析度 nvJPEG vs libjpeg-turbo 預設設定）；差異全部來自 4:2:0 chroma 升採樣方式，見下方色彩轉換第 3 步 | 最大 16–31 LSB、平均 0.13–0.31 LSB | — |
| nvJPEG＋float antialias stretch（**採用**） | 最大 19.6、平均 0.28 LSB | 10.5%／17.2%／14.0% |
| libjpeg＋float antialias（只去掉 uint8 取整） | 最大 0.5、平均 0.22 LSB | 4.9%／9.8%／7.2% |
| nvJPEG＋float bilinear 無 antialias | 最大 61、平均 1.05 LSB | 14.8%／21.4%／20.1% |
| 對照：±0.5 LSB 均勻雜訊 | — | 3.3%／6.6%／4.1% |

backbone 特徵對次 LSB 的輸入差就有數個百分點的反應，所以訓練與部署必須使用同一前處理。固定內容：

1. **解碼與色彩轉換**：`torchvision.io.decode_jpeg(..., device="cuda")`（nvJPEG 13.0.1）→ RGB uint8。本機 GPU 有獨立的 JPEG 硬體引擎：`nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE)` 成功，批次解碼 1080p 時 `nvidia-smi dmon` 的 jpg 引擎 86–92%、SM 7%，約 1,320 frames/s。批次與單張解碼逐位元相同，重複解碼逐位元相同。色彩轉換逐步如下（見下表）。
2. **正規化**：float32 ÷ 255，先於 resize。
3. **Resize**：`F.interpolate(mode="bilinear", antialias=True, align_corners=False)` stretch 到 640×640，float32，**不取整、不 clamp**（antialias 權重的捨入可使值略超過 1，實測最大 1.0000001）。
4. **使用範圍**：teacher 訓練、GT1 live features、cache_train／cache_val／cache_eval、teacher 的 val 診斷、backbone 匯出驗證與 head 輸入 features，都經同一個 `gpu_decode_clip_batch`；distill／T3／T1 只讀 cache。
5. **強制**：每個 T01 行程設 `SACCADE_T01_GPU_DECODE_ONLY=1`，舊 CPU 載圖函式一被呼叫即失敗（負向測試：preload 與逐張載入都被拒絕）；runner 只接受 manifest 為 `status=complete`、`decode_backend=torchvision_nvjpeg` 的 cache。smoke 與已停止的正式 attempt 實際 log 中都沒有 CPU preload，三個 cache manifest 都是 `torchvision_nvjpeg`。唯一的 CPU 讀圖是 `t01_split.py` 的洩漏檢查縮圖，不進入模型。

**色彩轉換流程**（2026-10-30 以實際檔案與程式核實）：

| 步驟 | 內容 | 依據 |
| --- | --- | --- |
| 0. 來源格式 | 5,316 張 MOT17 frame 全部是 baseline（非 progressive）、8-bit、3 component YCbCr、**4:2:0**；無 ICC profile、無 Adobe marker、無 EXIF。02／05／09／10 有 JFIF APP0，04／11／13 沒有（3-component 依慣例解為 YCbCr）。 | Pillow 逐張讀 header |
| 1. 熵解碼、反量化、IDCT | 8×8 整數 IDCT | 與 libjpeg-turbo `islow` 結果相同（第 3 步的逐位元比對） |
| 2. 解碼路徑 | torchvision 0.26.0 以 `NVJPEG_BACKEND_HARDWARE` 建 handle；每張先 `nvjpegDecodeBatchedSupported`，支援者走硬體 `nvjpegDecodeBatched`，否則走 default backend；輸出格式 `NVJPEG_OUTPUT_RGB`（planar，即 CHW） | [decode_jpegs_cuda.cpp @ v0.26.0](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp) L103–114、L149–182、L440–509 |
| 3. Chroma 升採樣 4:2:0→4:4:4 | **複製（2×2 重複），不是 libjpeg 預設的 fancy／三角濾波**。nvJPEG 輸出與 libjpeg-turbo 3.2.0 `djpeg -nosmooth`（merged upsampler）在 **5,316／5,316 張**逐位元相同；與 `djpeg` 預設（fancy）或 torchvision CPU `read_image` 在 81–93% 像素相同，差異集中在 Cb／Cr | 本機全量比對 |
| 4. YCbCr→RGB | JFIF 全範圍 BT.601（R＝Y＋1.402 Cr′，G＝Y−0.34414 Cb′−0.71414 Cr′，B＝Y＋1.772 Cb′，Cb′／Cr′＝值−128），定點捨入並 clamp 到 0–255 | 同第 3 步的逐位元相同 |
| 5. 通道 | planar **R, G, B**（不做 BGR 交換）；與 CPU decode 的逐通道相關 ≥ 0.9997，確認次序相同 | 本機比對 |
| 6. 數值域 | uint8 → float32 ÷ 255；不做 mean／std 正規化、gamma、線性化或其他色彩空間轉換 | `gpu_decode_clip_batch` |
| 7. Resize | 每通道獨立 antialiased bilinear（第 3 項） | 同上 |

YOLO26m 預訓練（Ultralytics）以 RGB 輸入，舊原版 teacher／cache 也是 RGB；T01 只改變 chroma 升採樣與 resize，不改通道次序或數值域。

**對 v1 的要求（F01／F02／F03 驗收項，T01 不關閉）**：F02 須實作同一 antialiased kernel。F01 應使用 nvJPEG 硬體 backend、`NVJPEG_OUTPUT_RGB`；其輸出有一個獨立的 CPU 參考：libjpeg-turbo（`JDCT_ISLOW`、`do_fancy_upsampling=FALSE`、`JCS_RGB`，即 `djpeg -nosmooth`），可對 MOT17 frame 做**逐位元**比對。若改用 CUDA hybrid backend、其他 nvJPEG 版本或 GPU 世代，須重新做此比對後才能宣稱一致。

### 2.2 計算精度與儲存 `t01-precision/1`

**發現（2026-10-30，正式訓練前）**：PyTorch 2.11 預設 `cudnn.allow_tf32=True`，舊 trainer 從未關閉，所以舊流程的 cache 與 GT1 features 都以 TF32 卷積計算。以 smoke teacher、3 張 val frame 分解（[證據](../evidence/t01/2026-10-30/precision/README.md)）：對同一輸入 tensor，TF32 重算後轉 fp16 與 cache **逐位元相同**；嚴格 FP32（與 ONNX Runtime 相差 ≤1.8e-5）與 cache 差 0.81%／1.58%／0.95%（P3／P4／P5 相對 L2），其中 fp16 儲存只佔 2.0e-4，其餘全部來自 TF32。同一分解的輸入對照：BGR 交換 29–51%、未除 255 460–1190%、HWC 當 CHW 讀 80–134%、無 antialias 39–66%、CPU fancy decode 5.7–10.4%、先 resize 再除 255 ≤1.5e-5。RGB 次序、NCHW 布局與 ÷255 正規化在訓練、驗證與匯出參考三條路徑相同（同一函式，逐位元相同的輸入 tensor）；native F01／F02 與 backbone engine 尚未實作，不在此驗證範圍。

**使用者決策（2026-10-30）**：head 訓練與選模所用的 features 必須與部署一樣以嚴格 FP32 計算；不把約 1% 的差距列為已知限制。teacher 保留原定的 bf16 訓練政策。

| Policy | 設定 | 用於 |
| --- | --- | --- |
| `fp32-strict` | cuDNN 與 cuBLAS 的 TF32 關閉、float32 matmul precision `highest`、`NVIDIA_TF32_OVERRIDE=0`（函式庫層再強制一次） | cache_train／cache_val／cache_eval；GT1（整個程序：live teacher features 與 head 的訓練運算在同一程序）；選模與 final eval；backbone 匯出驗證與 head 輸入 features |
| `torch-default` | 原配方：把 PyTorch 2.11 預設值明確設定（cuDNN TF32 卷積、FP32 matmul） | teacher（trainer 內的 bf16 autocast 不變）、distill、T3、T1 |

Distill／T3／T1 的輸入是 `fp32-strict` cache；它們自身訓練運算的精度不改變部署時的輸入，維持原配方。

強制與記錄（[t01_precision.py](../training/t01/t01_precision.py)）：

1. [t01_exec.py](../training/t01/t01_exec.py) 在 trainer 程序內、腳本開始前套用 policy，讀回 flags（legacy 與 `fp32_precision` 兩套 API）及環境變數，並以 conv2d／matmul 對 float64 的探針量測實際運算：FP32 約 1.8e-6／1.5e-7，TF32 卷積約 2.9e-4，門檻 3e-5。不符即不啟動。執行中每 10 s 重讀 flags，一旦偏離就記錄並以 SIGTERM 停止程序（測試：腳本中途打開 cuDNN TF32，10 s 內被停止，exit 143）；結束時再讀回並探針一次。紀錄寫入 attempt 的 `precision.json`。
2. 每個 checkpoint 存檔時在同一程序內讀回並檢查，結果寫入 checkpoint 的 `t01_compute_precision`。
3. Runner 在 `attempt.json` 分開記錄計算精度（policy、宣告值、程序內紀錄）與儲存（cache 的 `storage_dtype`、各 epoch checkpoint 的 tensor dtype）。以下任一情況 attempt 即失敗：紀錄缺少、開始或結束不符、有偏離、任一 epoch checkpoint 的存檔精度不符。
4. Cache manifest 新增 `storage_dtype: float16` 與 `compute_precision`（`fp32-strict` 的固定設定）；建 cache 的程序在開始時與標記 complete 前各檢查一次。runner（`checked_cache`）與兩個讀 cache 的 trainer（distill；T3／T1）都要求這兩個欄位，所以沒有這兩個欄位的舊 TF32 cache 一律被拒絕。
5. 評估（`t01_eval`）與匯出驗證（`t01_export`、`t01_verify_features`）在各自程序內套用並讀回 `fp32-strict`，結果檔記錄開始與結束時的讀回。ONNX Runtime 用 CPU execution provider（安裝的 `onnxruntime` 1.24.4 是 CPU 版，不涉及 TF32）；改用 CUDA EP 時，`ort_session` 以 `use_tf32=0` 建立並讀回確認，否則拒絕。

**驗收容差**（2026-10-30 在第一次驗收執行前固定於 [t01_verify_features.py](../training/t01/t01_verify_features.py)，不因結果放寬）。每張 frame、每個尺度的相對 L2；frame 取每個 cache sequence 的第一、1/3、2/3 與最後一幀：

| 比較 | 容差 | 依據 |
| --- | --- | --- |
| FP32 實作差：ONNX Runtime（CPU）vs PyTorch `fp32-strict`，batch 1 | ≤ 1e-4 | 兩種 FP32 實作只差累加次序；先前量到 0.5–1.8e-5 |
| Batch 不變性：PyTorch batch 4（GT1 的 clip batch）與 batch 8（評估 batch）vs batch 1 | ≤ 1e-4 | 同上 |
| FP16 儲存：cache（batch 1 前向後存 fp16）vs PyTorch batch-1 FP32 參考 | ≤ 2⁻¹¹ ≈ 4.88e-4 | fp16 就近捨入的單位捨入誤差 |
| cache vs PyTorch batch 4／8 | ≤ 2⁻¹¹＋1e-4 | 前兩項相加 |
| 訓練 vs 部署參考：cache vs ONNX Runtime | ≤ 2⁻¹¹＋1e-4 ≈ 5.88e-4 | 同上 |

另需：批次與單張 nvJPEG 解碼逐位元相同；負向對照中，舊 TF32 smoke cache 對同一 teacher 的 FP32 重算**必須**超過 FP16 儲存容差，且 runner 與兩個 trainer 都必須拒絕舊 cache。

**Smoke 驗收結果（PASS；[證據](../evidence/t01/2026-10-30/precision/README.md)）**：8 張 frame（MOT17-09、MOT17-10 各 4 張）。

| 檢查 | 最差值 | 容差 |
| --- | ---: | ---: |
| ONNX Runtime vs PyTorch | 1.81e-5 | 1e-4 |
| batch 4 vs batch 1 | 1.18e-5 | 1e-4 |
| batch 8 vs batch 1 | 1.22e-5 | 1e-4 |
| FP16 儲存 | 2.11e-4 | 4.88e-4 |
| cache vs ONNX Runtime | 2.12e-4 | 5.88e-4 |

cache 與「FP32 參考轉 fp16」逐元素相同（比例 1.0），所以儲存誤差只來自捨入。舊 TF32 cache 對同一 teacher 的 FP32 重算，最小差 3.4e-3，超過容差。runner、distill（mismatch 只有 `storage_dtype` 與 `compute_precision`）與 T3／T1 都拒絕舊 cache。全部 attempt 的程序內紀錄都相符、沒有偏離。

第一次執行時「trainer 拒用」一項判為失敗。原因是腳本取了 traceback 的原始碼行，而不是例外訊息；trainer 本身的行為正確。修正解析後整份重跑，容差未變；第一次的報告一併保留。

## 3. 訓練配方

[recipe.json](../training/t01/recipe.json) 是唯一的參數來源；[t01_run.py](../training/t01/t01_run.py) 由它組出每個 stage 的命令，命令全文寫入各 attempt 的 `attempt.json`。seed `20260612`。

| Stage | 輸入／監督 | Epochs | T／起點 stride | Batch | LR／warmup | 其他 |
| --- | --- | ---: | --- | ---: | --- | --- |
| teacher | YOLO26m；GT detection＋gate（gt_ratio 0.5） | 30 schedule | 2／2 | 4 | gate 1e-3、YOLO 1e-5／0 | bf16 autocast（`torch-default`）；grad clip 10；YOLO BN train mode；§2.1 前處理（`--gpu-decode`，`--num-workers 0`） |
| cache_train／cache_val | teacher e12，ungated；P3/P4/P5＋one2many cls/reg | — | 逐幀 | — | — | §2.1 前處理；batch 1 前向，`fp32-strict`（無 TF32），存 float16，schema `mamba-teacher-cache-v2` |
| distill | cache；MSE(student cls/reg, teacher cls/reg) | 30 | 1／1 | 8 | 1e-3／5 | `--no-preload-cache`（逐步讀檔，靠 page cache）；`torch-default` |
| gt1 | distill 選定；live teacher features＋GT，TAL v8DetectionLoss | 30 | 4／8 | 4 | 1e-4／5 | gt_ratio 0.5（前幀 GT gate feedback）；§2.1 前處理（`--gpu-decode`，`--num-workers 0`）；`fp32-strict`（整個程序） |
| t3 | gt1 選定＋新 temporal blocks；cache＋GT | 15 | 3／6 | 4 | 1e-4／3 | gt_ratio 0；`torch-default` |
| t1 | t3 選定；cache＋GT | 15 | 1／2 | 4 | 1e-4／3 | 單幀；temporal blocks 繞過；`torch-default` |

各 student stage：AdamW wd 1e-4、linear warmup＋cosine（最低 0.01）、grad clip 1.0、`scan_stop_grad=true`、`reg_max=1`（無 DFL）、d_model 128、d_state 16、每尺度 1 個 block、spatial reduction 4、Cross-Scan、PixelShuffle、80 類。`--mamba-ckpt` 是只載入模型權重的 warm-start（新 optimizer／scheduler）。T3 的 `clip_stride 6` 是 clip 起點步長，clip 內三幀連續。

**與舊原版配方的差異（新實驗設定）：**

1. 資料：train 只有五個 sequence（舊配方七個、無 holdout）；每 epoch 的 batch 數因此較少。
2. Teacher 選取：仍固定 e12、30-epoch schedule；**完整跑完 30 epochs** 並記錄每個 epoch 的 val 診斷，但 val 結果不改變 e12。
3. 學生 stage 的 parent：舊配方用 training-loss `best.ckpt`；T01 用 §4 的 val 選定 checkpoint。`best.ckpt`（training loss）仍由 trainer 寫出，只作診斷。
4. 所有 GT stage `--save-every 1`（每個 epoch 都需 val 評估）。
5. 輸入前處理改為 §2.1（舊原版 teacher／GT1 用 CPU decode＋uint8 antialias resize；舊 GPU cache 用 nvJPEG＋無 antialias bilinear；T01 全鏈統一為 nvJPEG＋float antialias）。
6. Distill 改 `--no-preload-cache`（記憶體安全；數值不變）。
7. 訓練 scan 實作不同（§1.1）。
8. 不跑 plain GT2 對照分支（不在 T1 lineage 上）。
9. Cache 與 GT1 的 features 改以 `fp32-strict` 計算（§2.2；舊原版與 PyTorch 預設相同，用 TF32 卷積）。

## 4. 選模政策 `t01-selection/1`

| 項目 | 固定內容 |
| --- | --- |
| 資料 | 只用 val（MOT17-10 全部 654 幀）；student 的 features 來自 `cache_val`（同一 teacher e12，ungated） |
| 前向 | T=1、eval mode、FP32（`fp32-strict`，在評估程序內讀回；§2.2）、cuDNN deterministic、batch 8；temporal blocks 繞過 |
| Decode | anchor (x+0.5, y+0.5)，stride 8／16／32；raw ltrb×stride；score＝80 類 sigmoid 的最大值，class＝argmax，只留 class 0；conf ≥ 0.001、class 篩選前取 top-300；NMS IoU 0.5；座標換回原解析度 |
| 配對 | 每幀依 score 由高到低，與尚未配對且 IoU 最大的 GT 配對，需 IoU ≥ 0.5；無 ignore region |
| 主要指標 | **AP50**：所有 val 幀的偵測依 score 合併排序，101 點內插 precision |
| Tie-break | (1) score ≥ 0.25 時最短邊 4–16 px GT 的 recall；(2) 較晚的 epoch |
| 不合格 | 任一 val 幀的 raw cls／reg 含非有限值 |
| 頻率 | 每個 epoch 的 checkpoint 都評估；每個 epoch 都保存 |
| 傳遞 | 下一 stage 由上一 stage 的選定 checkpoint warm-start；T1 的選定 checkpoint 是最終候選 |
| Teacher | 固定 e12；每個 epoch 的 val 診斷（teacher one2many 輸出走同一 decode）只記錄 |
| Eval split | 只在 T1 的選定紀錄寫出後打開一次（`final/opened.json` 記錄選定紀錄的 hash）；不參與任何調參或選模。打開後同時評估選定 T1 兩次（一致性）與各 parent／teacher（只作 lineage 診斷） |

報告另列各 threshold（0.001／0.10／0.25）的 TP／FP／FN、空偵測幀數、各尺寸區間 recall、top-300 內不合法 box（x2≤x1 或 y2≤y1）數、class 分布、score 範圍與偵測結果 SHA-256。這些不是 tracking 指標；HOTA／IDF1／MOTA 不在本任務驗收範圍。

## 5. 中斷、resume 與異常

| 情況 | 處置 |
| --- | --- |
| 中斷（非零退出、低記憶體停止） | 該 attempt 標 `failed`；新 attempt 寫到新目錄（舊 `open_run` 拒絕非空目錄）。teacher 為**完整 resume**（需 `--num-workers 0`：persistent workers 會改變 data-loader generator 的抽取次序，smoke 實測 resume 後 loss 不同）（模型、optimizer、scheduler、grad scaler、Python／torch／CUDA RNG、data-loader generator）；distill／GT 為 **optimizer／scheduler resume**（RNG 與資料順序不恢復，不是逐位元延續）。兩者都與 `--mamba-ckpt` warm-start 分開記錄 |
| NaN／Inf loss | distill／GT trainer 直接失敗，stage 失敗並停止；teacher 依原碼略過該 batch 並計數，計數由 log 收錄報告 |
| 低記憶體 | runner 每 2 s 取樣；MemAvailable < 4 GiB 即停止訓練程序群組（SIGTERM，20 s 後 SIGKILL），attempt 記為失敗 |
| 資料缺失 | cache 須 `status=complete` 且逐 sequence frame 數等於影格數；cache manifest 的 base／teacher hash 與 sequences 由 trainer 核對 |
| 未完成 stage | `best.ckpt` 存在不代表成功；只有 `stage.json` 為 `complete` 且 teacher 有最後一個 epoch 才算完成 |
| 計算精度偏離 | trainer 程序內偵測到即記錄並 SIGTERM；attempt 失敗（§2.2）。cache 不符 `float16`／`fp32-strict` 時 runner 與 trainer 拒用 |
| Formal run 前提 | saccade_v1 工作樹乾淨（identity 的 v1 commit 才精確）、source tree 與 lock 相同 |

## 6. Smoke run 與預算

Smoke run（`t01_run.py stage NAME --smoke-epochs N`，寫到 `smoke/`，不屬正式結果）：teacher 2 epochs（MOT17-09）＋完整 resume、cache 525＋654 幀、distill 40 batches×2＋resume、GT1／T3／T1 各 30 batches×2、每個 epoch 的 val 評估、選模與 final eval 全流程都在本機跑過。觀察：

- teacher 完整 resume 後 epoch 2 loss 與不中斷時相同（5.7661）；`--num-workers 4` 時不同（5.7661 vs 5.7838），因此 teacher／GT1 固定 0 workers（解碼本來就在 GPU）。distill resume 的 epoch 2 loss 不同（17.8968 vs 17.8382），與「不恢復 RNG／資料順序」一致。
- 改用 §2.1 前處理後整條 smoke 鏈重跑（7 min），lineage 檢查會拒絕舊 teacher 訓練出的 distill checkpoint（hash 不符），所以 smoke 鏈自 teacher 起整段重建。
- 加入 §2.2 計算精度控制後，以 `t01_run.py smoke` 在新的 smoke 目錄重跑全鏈（8.3 min，含 teacher 與 distill resume、每 stage 的 val 評估與選模、final eval）。teacher 完整 resume 的 epoch 2 loss 仍為 5.76607591455633，與精度控制前相同；smoke teacher 的 backbone ONNX 與控制前逐位元相同（teacher 用 `torch-default`，沒有改變）。舊 smoke 移到 `smoke-tf32-20261030/`，作為負向對照。
- 重複評估：同一 checkpoint 在 val 與 eval 兩次的偵測 SHA-256 相同。
- eval 單元檢查：完美偵測 AP＝1；anchor decode 與手算相同；teacher one2many 輸出與 Detect `cv2`／`cv3` 對 features 的結果差 0。

| Stage | 量測 | 估計（正式） | 峰值 |
| --- | --- | --- | --- |
| teacher | 0.18 s/batch | 508 batches/epoch，約 1.5 min/epoch，30 epochs 約 46 min；val 診斷 30×15 s | GPU 5.1 GiB |
| cache | 28 fps（含 torch.save），6.4 MB/幀；`fp32-strict` 後約 22–24 fps | train 4,062 幀約 3–4 min、26 GB；val 4.1 GB；eval 3.8 GB | RSS 5.4 GiB |
| distill | 0.075 s/batch | 508 batches，約 40 s/epoch，30 epochs 約 20–30 min | RSS 約 2 GiB＋page cache |
| gt1 | 0.35 s/batch；`fp32-strict` 後 backbone 前向約慢 1.6 倍 | 127 batches，約 45–55 s/epoch，30 epochs 約 23–28 min | GPU 1.8 GiB（allocator） |
| t3 | 0.27 s/batch | 170 batches，約 46 s/epoch，15 epochs 約 12 min | GPU 6.5 GiB；features 預載 RAM 約 22 GiB |
| t1 | 0.10 s/batch | 508 batches，約 51 s/epoch，15 epochs 約 13 min | GPU 3.6 GiB；RAM 同上 |
| val 評估 | 11 s／student checkpoint | 90 個 checkpoint 約 17 min | — |

合計 GPU 約 2.5–3 h；磁碟約 55 GB（teacher 30×252 MB、student 90×127 MB、cache 34 GB），在 160 GB 可用空間內；預估在 47 GiB RAM 內。沒有縮短 epoch、減少資料或修改模型來配合預算。

## 7. 重跑

```sh
uv venv --python 3.12.13 ~/.cache/saccade-v1/t01/venv
uv pip install --python ~/.cache/saccade-v1/t01/venv/bin/python -r training/t01/requirements.lock.txt
cd training/t01
PY=~/.cache/saccade-v1/t01/venv/bin/python
$PY t01_split.py                  # 重建 split manifest（應與已提交者相同）
$PY t01_run.py prepare-source     # 匯出 e48f3b39＋patch，與 source.lock.json 比對
$PY t01_run.py env
$PY t01_run.py smoke              # 完整 smoke 鏈（寫到 smoke/；舊的 smoke/ 須先移走）
$PY t01_run.py pipeline           # teacher → cache → distill → gt1 → t3 → t1，逐 stage 評估、選定，最後 final eval

# 匯出與 feature 驗證另用一個 venv（多 onnx、onnxruntime：requirements-export.lock.txt）
uv venv --python 3.12.13 ~/.cache/saccade-v1/t01/venv-export
uv pip install --python ~/.cache/saccade-v1/t01/venv-export/bin/python -r requirements-export.lock.txt
PYX=~/.cache/saccade-v1/t01/venv-export/bin/python
$PYX t01_export.py backbone [--smoke]          # 選定 teacher 的 backbone／neck ONNX 與逐 initializer 對應
$PYX t01_export.py verify-backbone [--smoke]   # ONNX Runtime vs PyTorch vs cache，未訓練權重為負向對照
$PYX t01_verify_features.py [--smoke] --out REPORT.json [--negative-root OLD_SMOKE_ROOT]   # §2.2 驗收
```

單一步驟：`stage NAME [--resume CKPT]`、`eval NAME`、`select NAME`、`final-eval`。artifact 根目錄 `~/.cache/saccade-v1/t01/`（`SACCADE_T01_ROOT` 可覆寫）：`src/`、`runs/<stage>/attempt-NN/{attempt.json,train.log,memory.log,run/}`、`runs/<stage>/{stage.json,val/,selection.json}`、`runs/final/`。大型 checkpoint、cache 與資料集不進 Git。

## 8. 執行紀錄：正式 run（2026-10-30）

**結論**：全鏈一次跑完，沒有中斷或重試。依預先固定的政策，最終候選為 **T1 epoch 1**（SHA-256 `070580f9…6e6962b`）：val AP50 0.6758，eval（MOT17-02，只開一次）AP50 0.5440。證據見 [evidence/t01/2026-10-30/formal](../evidence/t01/2026-10-30/formal/)。

**身分**
- v1 為 `7794c75`（工作樹乾淨、已推送）；舊來源 `e48f3b39`＋patch，patched tree `12de1236…`。recipe 與 split 的 hash 記在每個 attempt。
- 全程 2 h 32 min（UTC 11:26–13:58），每個 stage 一個 attempt。
- 精度控制加入前、只跑到 teacher epoch 1 就停止的 attempt（`runs-stopped-20261030T0941/`）未被使用。

| Stage | 時間 | 計算精度（程序內讀回） | 選定 epoch | 選定的 val AP50 | training-loss 最佳 epoch | 選定 checkpoint |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| teacher | 46.0 min | `torch-default`＋bf16 autocast | 12（固定規則） | 0.6881（診斷） | — | `14e86231…` |
| cache_train | 2.2 min | `fp32-strict`，存 float16 | — | — | — | 4,062 幀、27.4 GB |
| cache_val | 0.5 min | `fp32-strict`，存 float16 | — | — | — | 654 幀、4.4 GB |
| distill | 24.9 min | `torch-default` | 26 | 0.6508 | 30 | `1b310878…` |
| gt1 | 26.7 min | `fp32-strict` | 6 | 0.6781 | 30 | `72093767…` |
| t3 | 12.1 min | `torch-default` | 2 | 0.6844 | 15 | `bdf887aa…` |
| t1 | 14.0 min | `torch-default` | 1 | 0.6758 | 15 | `070580f9…` |
| cache_eval | 0.6 min | `fp32-strict`，存 float16 | — | — | — | 600 幀（T1 選定後才建） |

**資源**：沒有低記憶體停止。最低可用記憶體 17.5 GiB（T3／T1 把 features 預載到 RAM，trainer RSS 約 23.4 GiB）；GPU 峰值 7.5 GiB；`runs/` 共 53 GB。

**計算精度**
- 8 個 attempt 的程序內紀錄都與 policy 相符，沒有偏離；monitor 取樣次數從 teacher 的 275 次到 cache 的 2 次不等。
- 122 個 val 評估紀錄的開始與結束都是 `fp32-strict`。
- 每個 epoch checkpoint 都通過存檔精度與身分檢查。

**正式 cache 的 feature 驗收**（§2.2 的容差，事先固定）：**PASS**。取 6 個 sequence 各 4 幀，共 24 幀：

| 檢查 | 最差值 | 容差 |
| --- | ---: | ---: |
| ONNX Runtime vs PyTorch | 1.37e-5 | 1e-4 |
| batch 4 vs batch 1 | 8.9e-6 | 1e-4 |
| batch 8 vs batch 1 | 9.8e-6 | 1e-4 |
| FP16 儲存 | 2.13e-4 | 4.88e-4 |
| cache vs ONNX Runtime | 2.13e-4 | 5.88e-4 |

負向對照全部通過。teacher e12 的 backbone／neck ONNX 為 76,186,163 bytes，SHA-256 `007c3260…`，440 個 initializer 全部逐位元對應到 checkpoint。

**Eval split（MOT17-02，600 幀、10,993 個 GT；T1 選定紀錄寫出後打開一次）**

| 項目 | T1 epoch 1 |
| --- | --- |
| AP50 | 0.5440（val MOT17-10：0.6758） |
| score ≥ 0.25 | recall 0.478：TP 5,250、FN 5,743、FP 1,580 |
| score ≥ 0.25，最短邊 4–8 px／8–16 px 的 recall | 0.014／0.363 |
| score ≥ 0.001，最短邊 4–8 px／8–16 px 的 recall | 0.223／0.626 |
| 檢查 | raw 輸出非有限值 0；top-300 中不合法框 0、全部為 class 0；score 範圍 0.001–0.998；沒有空偵測幀 |
| 重複評估 | eval 與 val 各評兩次，偵測結果 SHA-256 相同 |

Lineage 在 eval 上的 AP50（只作診斷，不參與選模）：distill 0.5155、gt1 0.5420、t3 0.5520；teacher one2many 輸出 0.5598。這些都不是 tracking 指標。

**觀察（記錄，不是結論）**
1. teacher 的 val AP50 隨訓練下降：e1 0.7545、e2 0.7582（最高）、e12 0.6881、e30 0.6585。依規則仍固定用 e12。
2. 學生 stage 的 val AP50 都在早期達到高點（gt1 e6、t3 e2、t1 e1），之後緩慢下降，而 training loss 持續降低。這符合「對 5 個訓練 sequence 過擬合」的解釋，但沒有另外驗證。依 val 選模取到的是早期高點。
3. T1 的最佳點是第 1 個 epoch，val 略低於 T3 選定點（0.6758 vs 0.6844）。
4. 在 MOT17-02，最短邊 4–8 px 的行人在 score ≥ 0.25 時幾乎都沒有被偵測到。

**完成狀態**

| 狀態 | 結果 |
| --- | --- |
| TRAINING COMPLETE | **是**：全部 stage complete；每個 epoch checkpoint 都存在，且通過存檔精度與身分檢查 |
| MODEL SELECTED | **是**：T1 epoch 1，依 `t01-selection/1` 選定；eval split 只在選定後打開一次 |
| NATIVE ARTIFACT READY | **是**（2026-10-30）：head 的 SCWB bundle `4fccfda6…` 與 engine `5b6d30fa…` 通過 [HEAD_TRT §10](HEAD_TRT.md#10-t01-候選的延續驗證2026-10-30)；matching backbone ONNX 以使用者核准追加的同版本 ONNX parser 建成 TensorRT engine，並通過 [BACKBONE_POLICY](BACKBONE_POLICY.md) 的 `t01-backbone-fp32/1`；完整模型 [schema 3 manifest](../manifests/t01/yolo26m-t1.json) 的宣告與六個 artifact 檔案皆 verified（[MODEL_TRT](MODEL_TRT.md)） |
| F03 VERIFIED | **是**，有範圍（2026-10-30，[F03 證據](../evidence/t01/2026-10-30/f03/REPORT.md)）：`t01-f03-fp32/1`，六個固定輸入（三個 MOT17 幀＋三個合成）上的完整模型推論數值、執行、失敗傳遞與 graph 重播；memcheck／initcheck 0 errors。輸入是訓練側的 FP32 tensor：native 前處理、decode／NMS 不在範圍；racecheck／synccheck 未執行 |
| P01 | **OPEN**：仍缺 native 前處理（F01／F02）與 decode／NMS（F04）等原有缺件的證據 |
