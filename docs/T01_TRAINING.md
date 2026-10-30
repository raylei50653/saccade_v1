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
| 三個 trainer 的 `save_checkpoint` | 每個 checkpoint 加 `t01_identity`（source／patch／tree／v1 commit／recipe／split hash、stage、attempt） | 只新增 root key；S04 轉換器只讀 `epoch`／`selection`／`mamba_args`／`student` |
| `train_gated_detector.py` `_git_state` | 改報 T01 identity | 匯出的 tree 不是 git checkout，原函式會把未知記成 `clean` |
| `dataset.py` `gpu_decode_clip_batch` | 先除 255 再 stretch，改為 antialiased bilinear，回傳 [0,1] | §2.1 的輸入契約；cache 與 GT1 的 GPU decode 路徑共用此函式 |
| `train_gated_detector.py` | 新增 `--gpu-decode`（資料載入回傳 JPEG bytes，迴圈內以上述函式解碼） | teacher 訓練與 cache／GT1 用同一前處理；拒絕與 `--adapt-bn`／`--augment` 併用 |
| `train_mamba_head.py` `write_frame`、`train_mamba_gt.py` | GPU 路徑不再除 255 | 配合函式回傳 [0,1] |
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
| 只換 decoder（全解析度 nvJPEG vs libjpeg-turbo） | 最大 16–21 LSB、平均 0.12–0.31 LSB | — |
| nvJPEG＋float antialias stretch（**採用**） | 最大 19.6、平均 0.28 LSB | 10.5%／17.2%／14.0% |
| libjpeg＋float antialias（只去掉 uint8 取整） | 最大 0.5、平均 0.22 LSB | 4.9%／9.8%／7.2% |
| nvJPEG＋float bilinear 無 antialias | 最大 61、平均 1.05 LSB | 14.8%／21.4%／20.1% |
| 對照：±0.5 LSB 均勻雜訊 | — | 3.3%／6.6%／4.1% |

backbone 特徵對次 LSB 的輸入差就有數個百分點的反應，所以訓練與部署必須使用同一前處理。固定內容：

1. **解碼**：`torchvision.io.decode_jpeg(..., device="cuda")`（nvJPEG 13.0.1）→ RGB uint8。本機 GPU 有獨立的 JPEG 硬體引擎：`nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE)` 成功，批次解碼 1080p 時 `nvidia-smi dmon` 的 jpg 引擎 86–92%、SM 7%，約 1,320 frames/s。批次與單張解碼逐位元相同，重複解碼逐位元相同。
2. **正規化**：float32 ÷ 255，先於 resize。
3. **Resize**：`F.interpolate(mode="bilinear", antialias=True, align_corners=False)` stretch 到 640×640，float32，**不取整、不 clamp**（antialias 權重的捨入可使值略超過 1，實測最大 1.0000001）。
4. **使用範圍**：teacher 訓練、GT1 live features、cache_train／cache_val／cache_eval、teacher 的 val 診斷、backbone 匯出驗證與 head 輸入 features，都經同一個 `gpu_decode_clip_batch`；distill／T3／T1 只讀 cache。
5. **強制**：每個 T01 行程設 `SACCADE_T01_GPU_DECODE_ONLY=1`，舊 CPU 載圖函式一被呼叫即失敗（負向測試：preload 與逐張載入都被拒絕）；runner 只接受 manifest 為 `status=complete`、`decode_backend=torchvision_nvjpeg` 的 cache。smoke 與已停止的正式 attempt 實際 log 中都沒有 CPU preload，三個 cache manifest 都是 `torchvision_nvjpeg`。唯一的 CPU 讀圖是 `t01_split.py` 的洩漏檢查縮圖，不進入模型。

**對 v1 的要求（F01／F02／F03 驗收項，T01 不關閉）**：F02 須實作同一 antialiased kernel；F01 應使用 nvJPEG 硬體 backend。硬體與 CUDA hybrid backend、不同 nvJPEG 版本或 GPU 世代可能有小的像素差，需以固定參考 frame 做像素級比對後才能宣稱一致。

## 3. 訓練配方

[recipe.json](../training/t01/recipe.json) 是唯一的參數來源；[t01_run.py](../training/t01/t01_run.py) 由它組出每個 stage 的命令，命令全文寫入各 attempt 的 `attempt.json`。seed `20260612`。

| Stage | 輸入／監督 | Epochs | T／起點 stride | Batch | LR／warmup | 其他 |
| --- | --- | ---: | --- | ---: | --- | --- |
| teacher | YOLO26m；GT detection＋gate（gt_ratio 0.5） | 30 schedule | 2／2 | 4 | gate 1e-3、YOLO 1e-5／0 | bf16 autocast；grad clip 10；YOLO BN train mode；§2.1 前處理（`--gpu-decode`，`--num-workers 0`） |
| cache_train／cache_val | teacher e12，ungated；P3/P4/P5＋one2many cls/reg | — | 逐幀 | — | — | §2.1 前處理；FP32 forward，存 float16，schema `mamba-teacher-cache-v2` |
| distill | cache；MSE(student cls/reg, teacher cls/reg) | 30 | 1／1 | 8 | 1e-3／5 | `--no-preload-cache`（逐步讀檔，靠 page cache） |
| gt1 | distill 選定；live teacher features＋GT，TAL v8DetectionLoss | 30 | 4／8 | 4 | 1e-4／5 | gt_ratio 0.5（前幀 GT gate feedback）；§2.1 前處理（`--gpu-decode`，`--num-workers 0`） |
| t3 | gt1 選定＋新 temporal blocks；cache＋GT | 15 | 3／6 | 4 | 1e-4／3 | gt_ratio 0 |
| t1 | t3 選定；cache＋GT | 15 | 1／2 | 4 | 1e-4／3 | 單幀；temporal blocks 繞過 |

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

## 4. 選模政策 `t01-selection/1`

| 項目 | 固定內容 |
| --- | --- |
| 資料 | 只用 val（MOT17-10 全部 654 幀）；student 的 features 來自 `cache_val`（同一 teacher e12，ungated） |
| 前向 | T=1、eval mode、FP32、TF32 關閉、cuDNN deterministic、batch 8；temporal blocks 繞過 |
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
| Formal run 前提 | saccade_v1 工作樹乾淨（identity 的 v1 commit 才精確）、source tree 與 lock 相同 |

## 6. Smoke run 與預算

Smoke run（`t01_run.py stage NAME --smoke-epochs N`，寫到 `smoke/`，不屬正式結果）：teacher 2 epochs（MOT17-09）＋完整 resume、cache 525＋654 幀、distill 40 batches×2＋resume、GT1／T3／T1 各 30 batches×2、每個 epoch 的 val 評估、選模與 final eval 全流程都在本機跑過。觀察：

- teacher 完整 resume 後 epoch 2 loss 與不中斷時相同（5.7661）；`--num-workers 4` 時不同（5.7661 vs 5.7838），因此 teacher／GT1 固定 0 workers（解碼本來就在 GPU）。distill resume 的 epoch 2 loss 不同（17.8968 vs 17.8382），與「不恢復 RNG／資料順序」一致。
- 改用 §2.1 前處理後整條 smoke 鏈重跑（7 min），lineage 檢查會拒絕舊 teacher 訓練出的 distill checkpoint（hash 不符），所以 smoke 鏈自 teacher 起整段重建。
- 重複評估：同一 checkpoint 在 val 與 eval 兩次的偵測 SHA-256 相同。
- eval 單元檢查：完美偵測 AP＝1；anchor decode 與手算相同；teacher one2many 輸出與 Detect `cv2`／`cv3` 對 features 的結果差 0。

| Stage | 量測 | 估計（正式） | 峰值 |
| --- | --- | --- | --- |
| teacher | 0.18 s/batch | 508 batches/epoch，約 1.5 min/epoch，30 epochs 約 46 min；val 診斷 30×15 s | GPU 5.1 GiB |
| cache | 28 fps（含 torch.save），6.4 MB/幀 | train 4,062 幀約 3 min、26 GB；val 4.1 GB；eval 3.8 GB | RSS 5.4 GiB |
| distill | 0.075 s/batch | 508 batches，約 40 s/epoch，30 epochs 約 20–30 min | RSS 約 2 GiB＋page cache |
| gt1 | 0.35 s/batch | 127 batches，約 45 s/epoch，30 epochs 約 23 min | GPU 1.8 GiB（allocator） |
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
$PY t01_run.py pipeline           # teacher → cache → distill → gt1 → t3 → t1，逐 stage 評估、選定，最後 final eval
```

單一步驟：`stage NAME [--resume CKPT]`、`eval NAME`、`select NAME`、`final-eval`。artifact 根目錄 `~/.cache/saccade-v1/t01/`（`SACCADE_T01_ROOT` 可覆寫）：`src/`、`runs/<stage>/attempt-NN/{attempt.json,train.log,memory.log,run/}`、`runs/<stage>/{stage.json,val/,selection.json}`、`runs/final/`。大型 checkpoint、cache 與資料集不進 Git。
