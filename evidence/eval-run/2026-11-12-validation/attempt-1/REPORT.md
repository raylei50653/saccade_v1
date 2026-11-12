# 單指令品質與速度 runner 驗證（2026-11-12）

**結果：STOP_INPUT_FAILURE；未完成全部 GPU 驗證。** 程式、測試及文件已提交為 `61819bf85a41ab4f4fe8c3ee6234e16717caf87c`。依指定參數從乾淨來源建置兩個 native binaries，前三次 runner（MOT17、MOT20、MOT15）完成；第四次 DanceTrack capture 遇到指定資料目錄缺少六位數命名 JPEG 而停止。PP22 及 G3 既有 scorer parity 未執行。

## 停止原因

指定 `/home/ray/developer/ai/dataset/DanceTrack/val/dancetrack0004/img1/` 的實際檔名從 `00000001.jpg` 開始；native capture 的固定輸入為 `000001.jpg`，該檔不存在。`seqinfo.ini`／GT 的 hashes 與 freeze 完全相同，影像目錄本身也存在，因此通過題目指定的 layout validation；native 工具開啟第一張影像時回傳 1：

```text
track_sequence: cannot open input: /home/ray/developer/ai/dataset/DanceTrack/val/dancetrack0004/img1/000001.jpg
```

Runner 的 argv 符合題目指定內容。這是外部輸入失敗，不是 runner／worker／測試缺陷；依停止條件沒有更名、建立影像 view、變更 argv、加入 fallback、重跑或調查 runtime。停止後確認 `ps` 沒有殘留 `track_sequence`／`tracking_benchmark`；沒有 kill GPU job。完整失敗身分、argv、log、telemetry 與 `receipts/stop.json` 均保留。

## 範圍與身分

- GPU 驗證起點（UTC）：`2026-11-12T15:18:48.336616+00:00`；停止記錄時間：`2026-11-12T15:21:31.092341+00:00`。
- 前置輸入 19/19 通過：engine SHA-256、driver 621.14、7 組 GT／seqinfo 共 14 hashes、frozen per-sequence CSV hash、eval interpreter imports。詳見 `receipts/preflight.json`。題目未授權額外的 JPEG layout 檢查或修復。
- Engine SHA-256：`8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`，未重建 engine。
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU；driver 621.14；Release、CUDA 13.4.92、TensorRT 10.16.1.11、SM 120。建置使用 `--parallel 2`，只建兩個指定 target。
- Profile 固定 `hr-1280x768`，NSA／measurement box OFF；capture 沒有 packets／motion 輸出。每次 GPU phase 全程持有 `~/.cache/saccade-v1/gpu.lock`，native 呼叫使用 `timeout -k 30s 3600s`，每秒 nvidia-smi；四次循序執行，前三組評分於解鎖後執行。
- 四次 runner 的 porcelain 均為空白；已產生的 native identity 均為上述 commit、`source_dirty=false`。兩個 binary／engine hashes、成功 capture 的 identity、40 個 benchmark sessions 的 tracker hash 比對結果見各組 `environment.json`；失敗 capture 的 identity 見其 gzip log。

## Gates

| Gate | 結果 | 數量與比對方式 |
| --- | --- | --- |
| G1 MOT bytes | 已完成部分 PASS | 5/5 產出 MOT SHA-256 與 primary156 main 相同；DanceTrack、PP22 未產出，總需求 7 |
| G2 逐序列分數 | 已完成部分 PASS | 5/5 序列、30/30 指標完全相同；五項 parsed floats exact equality，IDSW integer equality；總需求 42 項，另 12 項未執行 |
| G3 既有 scorer parity | UNEXECUTED | 0/2 組、0/12 指標；依外部輸入失敗停止，未執行新控制評分 |
| G4 速度 | 已完成部分 PASS | 5/5 summary status PASS；每序列 4 single＋4 queued，共 40 fresh sessions；每序列 480 pooled single samples，共 2400；40/40 tracker hashes 相符；總需求 7 序列 |
| G5 CPU 測試 | PASS（提交前） | eval_run_tests.py 10/10；evaluation_tools_tests.py 12/12；停止後未重跑 |

完整已執行數字及期望值見 `parity.json`。COMBINED 取自一次 TrackEval 評分的 detailed CSV，沒有平均逐序列指標。CPU 測試使用 stub native binaries 與 nvidia-smi，透過 eval venv 執行真實 repository TrackEval worker，沒有 GPU 推論。

## 五組結果狀態

品質為百分比（IDSW 除外），完整精度保留於 JSON／CSV；表內品質及毫秒三位小數。前三組 `runs/<benchmark>/RESULTS.md` 含完整 single mean／p50／p95／p99、queued ms/frame 與 FPS。

### MOT17

[完整結果表](runs/MOT17/RESULTS.md) · [CSV](runs/MOT17/results.csv) · [JSON](runs/MOT17/results.json)

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single mean ms | p95 ms | p99 ms | queued FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MOT17-10-SDP | 654 | 50.869 | 54.407 | 47.799 | 64.975 | 66.890 | 56 | 6.696 | 6.904 | 7.157 | 195.851 |
| MOT17-02-SDP | 600 | 40.565 | 40.896 | 40.566 | 49.154 | 46.784 | 102 | 6.715 | 6.956 | 7.137 | 195.073 |
| MOT17-04-SDP | 1050 | 74.802 | 83.169 | 67.376 | 80.578 | 92.464 | 71 | 6.833 | 7.047 | 7.296 | 195.100 |
| COMBINED | 2304 | 64.256 | 67.382 | 61.768 | 71.516 | 77.559 | 229 |  |  |  |  |

### MOT20

[完整結果表](runs/MOT20/RESULTS.md) · [CSV](runs/MOT20/results.csv) · [JSON](runs/MOT20/results.json)

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single mean ms | p95 ms | p99 ms | queued FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MOT20-01 | 429 | 50.586 | 56.904 | 45.413 | 63.577 | 70.891 | 147 | 6.899 | 7.087 | 7.362 | 194.240 |
| COMBINED | 429 | 50.586 | 56.904 | 45.413 | 63.577 | 70.891 | 147 |  |  |  |  |

### MOT15

[完整結果表](runs/MOT15/RESULTS.md) · [CSV](runs/MOT15/results.csv) · [JSON](runs/MOT15/results.json)

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single mean ms | p95 ms | p99 ms | queued FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADL-Rundle-6 | 525 | 49.640 | 60.377 | 40.907 | 61.868 | 71.252 | 21 | 6.522 | 6.854 | 7.154 | 197.445 |
| COMBINED | 525 | 49.640 | 60.377 | 40.907 | 61.868 | 71.252 | 21 |  |  |  |  |

### DanceTrack

[輸入失敗收據](runs/DanceTrack/STATUS.md)：capture exit 1；沒有 results 表。

### PP22

[未執行狀態](runs/PP22/STATUS.md)：依停止條件沒有執行；沒有 results 表。

## MOT17-10 與 g1-live 速度對照

此項僅報告差異，不作效能 gate、不追逐差異；百分比為 `(本次 / g1-live − 1) × 100`。

| 指標 | g1-live | 本次 | 差異 |
| --- | ---: | ---: | ---: |
| single_mean_ms | 6.712865 | 6.696337 | -0.246% |
| single_p95_ms | 6.981673 | 6.903525 | -1.119% |
| single_p99_ms | 7.228440 | 7.157335 | -0.984% |
| queued_fps | 194.620666 | 195.850553 | +0.632% |

本次單幀延遲採原始 benchmark summary（包含工具本身的 packet 檢查／記憶體複製）；g1-live 指定基準採 host timestamp instrumentation 的 enqueue→receive 返回時間，兩者有少量計時邊界差異。此對照不代表受控加速。

## Telemetry

一秒取樣包含初始化、warmup、capture、處理及清理，不能當作穩態 kernel／Tensor Core／OFA utilization。完整 CSV 與 `telemetry-summary.json` 均保留。

| benchmark | samples | GPU utilization mean / max (%) | GPU memory max (MiB) | temperature max (°C) | SM clock min / max (MHz) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MOT17 | 51 | 59.2 / 100 | 495 | 73 | 180 / 11895 |
| MOT20 | 16 | 53.9 / 100 | 495 | 74 | 1312 / 2565 |
| MOT15 | 15 | 50.5 / 100 | 495 | 73 | 1582 / 2542 |
| DanceTrack | 2 | 1.5 / 3 | 0 | 58 | 817 / 1642 |

## 自有缺陷與修正

Runner、worker、測試沒有觸發需要修正的缺陷，沒有因自有程式錯誤重跑或放寬比較。證據草稿曾把「三個 MOT17 缺少舊 command.json」誤寫為「7/7 皆缺少」；封裝核對時 assertion 抓到此錯誤，已逐一確認並修正為 3/7。這是證據文字錯誤，不影響 MOT、評分或速度數字，沒有重跑 GPU。CPU fixture 中的錯誤參數、缺失輸入與 tracker hash 不符均為預期拒絕案例。

## 保管與限制

- `runs/` 保留前三次完整結果、第四次失敗收據、第五次未執行狀態；已完成序列的 MOT gzip、resolved config、benchmark summary／runs.csv／single samples／session identities、評分 detailed CSV 及 gzip logs 全部保留。`receipts/` 保留前置檢查、configure／build、runner 的 command／result 與 gzip logs，以及 CPU 測試 logs；`methods/validate.py` 為本次驗證驅動，按停止條件沒有走到其 G3／G5 重跑段落。
- 未壓縮 build 與 run artifacts 只留在 worktree 的 `build/`；沒有提交 build trees、packet `.bin` 或 packet gzip。`SHA256SUMS` 涵蓋本目錄除自身外的所有檔案。
- 每序列速度只量前 140 frames，20 warmup 後量 frames 21–140；JPEG 預載，排除磁碟 I/O、constructor、warmup 與輸出檔案。queued FPS 是完成吞吐，不是逐 frame latency、live-stream 或 camera FPS；clocks 未鎖定。
- GPU 證據只完成 5 個序列，沒有宣稱五個 benchmark 全部通過、G3 parity 完成、156 序列重跑、模型泛化、外部攝影機端到端 FPS、sanitizer 或額外 GPU 驗收。短序列 skip、DanceTrack GT transform、PP22 prediction filter 僅在 CPU fixtures 通過。
- 後續 evidence commit 只新增證據，不重建 binary；binary 身分仍對應程式提交。未 push／merge，沒有寫入 main checkout 或資料集。

## Notes

1. 指定 DanceTrack 輸入是八位數 JPEG 名稱，native 工具要求六位數；屬本輪停止的外部輸入問題。需要使用者另行決定資料或輸入契約，未擅自修復。
2. 三個 MOT17 序列的 `bundle/outputs/main/<seq>/command.json` 不在此封存 checkout（3/7），其餘四個存在；七個 capture.json／MOT gzip 均存在。成功的 5 個 capture 使用題目固定 argv 且 G1 完全相符，沒有觸發 MOT mismatch 的舊 argv diff。
3. DanceTrack 的既有 command.json 指向 primary156 建立的 `image-views/dancetrack0004`，與題目指定的原始資料路徑不同；該封存 argv 複本見 `receipts/frozen-DanceTrack-command.json`。沒有存取或修改 main checkout 的該 view，也沒有建立替代 view。
4. 三份 TrackEval 原生 `pedestrian_detailed.csv` 保留工具產生的 CRLF；一般 `git diff --check` 會將其列為行尾空白。未改寫原始 CSV bytes，程式與文件的 whitespace check 通過。
