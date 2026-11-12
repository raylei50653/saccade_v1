# 固定第一輪基準的 G1 延遲補量（2026-11-11）

**結果：有界效能量測與完整 packet 一致性 PASS。** 乾淨 runtime
`3cbaa518f0dc37d691404e95ed85c0109c81130d`，固定 HR engine、G1／`v1-motion/2`、
Graph ON、NSA／量測框 OFF。新增 host timestamp benchmark harness，未改 runtime。
本次不是品質重評、sanitizer 或完整即時來源到消費者驗收。

## 結果與計時邊界

MOT17-10-SDP 前 140 幀，JPEG 全數預載；每個 fresh session 從 frame 0 建立相同歷史，
20 幀暖機後量測 120 幀。四個 single 與四個 queued sessions 交替執行，各保留 480 個逐幀樣本。
單位 ms，p50／p95／p99 以合併後逐幀樣本採 nearest-rank，未刪除長尾。

| 範圍 | mean | p50 | p95 | p99 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single enqueue→receive | 6.712865 | 6.680952 | 6.981673 | 7.228440 | 6.392425 | 9.288572 |
| Queued enqueue→caller receive | 20.254368 | 20.478764 | 21.181116 | 21.571266 | 12.934183 | 24.747265 |

Single 起點沒有前幀在途；queued 使用 result ring=4，滿時先接收最舊結果再 enqueue 下一幀。
兩者起點均在 `session.enqueue()` 前、終點在 `session.receive()` 返回後、packet 檢查／複製前，
取樣使用 `steady_clock`。Queued 包含排隊、資源等待及 caller 延後取回的時間，
並保留量測區段起始填充與結尾 drain 的樣本。

Queued 每次 wall interval 為 5.148452、5.140223、5.127788、5.136339 ms/frame，
平均 **5.138200 ms/frame，194.620666 FPS**。它是完成吞吐，不能當作約 5.14 ms 的逐幀交付延遲。
wall 邊界另包含 benchmark packet 檢查與記憶體內複製；排除磁碟讀取、constructor、暖機及檔案輸出。

| 冷啟動相關範圍（8 次） | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: |
| Session constructor | 808.991091 | 443.964228 | 435.329625 | 3371.000716 |
| 首幀 enqueue→receive | 17.728839 | 16.894595 | 16.445182 | 23.061802 |

第一個 constructor 3371.000716 ms 的長尾原樣保留。這些是八次 fresh session 的描述統計，
不代表重開機後或清除所有 driver／檔案 cache 的冷啟動分布。

## 身分與方法

- Runtime source：新 detached worktree `build/report1-live-20261111/source`，固定上述完整 commit。
  CMake 編譯定義與每個 session identity 均為 `source_dirty=false`，原始受版控檔案前後 hash 相同。
- Engine：`/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine`，
  SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`，本次未重建。
- Profile：`hr-1280x768`，1920×1080 原圖；pipelined session，candidate／motion／result slots=2／2／4；
  G1 SLOW、grid4、cell8、mask score≥0.45、minimum cells=8。
- 環境：RTX 5070 Ti Laptop、WSL2、**driver 621.14**、CUDA 13.4.92、TensorRT 10.16.1.11、
  GCC 16.2.1、nvJPEG 13.2.3。此 driver 與 2026-11-05 歷史量測的 617.42 不同，不能合併樣本或作受控跨報告加速比較。
- 狀態：`kalman.nsa_enabled=0`、`rows.measurement_box=0`，tracker config hash
  `d846e2a5b447ce2c92f7dfda77f7fd52e27ee950e4065f93d1d0b3f99c960d6f`。
- Harness：先以乾淨 source 建置原 `tracking_benchmark` 與 runtime static libraries，
  再複製單一 benchmark `.cpp`，加入 enqueue／receive host timestamps，沿用相同編譯參數與原 runtime libraries 連結。
  原 runtime 與 worktree 內的 benchmark 檔案均未修改；新增檔案及差異在 `methods/`。
- 順序：`single, queued` 重複四次；每次 fresh session。原 benchmark 的 `--only graph --rounds 2`
  只執行每輪 ordinal 2／3，因此總計 4＋4 sessions，而非 ordinary／Graph 的 ABBA 對照。
- 排他：整個 campaign 持有 `/home/ray/.cache/saccade-v1/gpu.lock`；未鎖 clocks。
  量測程序期間持續記錄一秒一次 `nvidia-smi` 遙測，無 profiler。

環境、configure／build、獨立 harness compile／link 與 measure 的 argv、log、退出碼及 wall time
均保留。CMake target 只建必要 runtime 與 benchmark，未跑額外測試／訓練／sanitizer。

## 輸出與保管

全部八份完整 140-frame ResultPackets 直接逐 byte 比較相同，包含暖機、unused rows 與 reserved bytes；
每份 1,731,520 B，SHA-256
`db77362c1c5fff9ebe858087b4c3726601549b8f57b0ca3506557b20171fb022`。
這是各 session／caller 風格之間的一致性，沒有宣稱與舊 driver 的歷史 packet 比較相同。

- `source-before.json`／`source-after.json`：完整受版控 source、symlink 身分，前後完全相同。
- `inputs-before.json`／`inputs-after.json`：engine、seqinfo 與 140 張 JPEG 的 bytes／hash，前後完全相同。
- `binaries-before.json`／`binaries-after.json`：原 benchmark、instrumented benchmark、harness object 與 runtime static libraries，前後完全相同。
- `raw/*delivery.csv`：各幀 enqueue／receive timestamps 與 duration，含暖機註記。
- `raw/runs.csv`、`raw/*single.csv`：原 harness 的數值原樣保存。原 single.csv 的時間包含 lambda 內 packet 檢查／複製；正式新增統計採 delivery.csv 的 receive 返回時點，故 mean 約相差 1.2 µs，不能混用。
- `raw/*.identity.txt`：每 session 完整 runtime／flow／tracker 設定與來源身分。
- `raw/*.bin.gz`：八份完整 packets 的 gzip 檔，解壓 bytes 與 build 內原檔相同。
- `summary.json`：正式新增統計與限制；`contract.json` 為量測前固定範圍。
- 大型本機建置與未壓縮輸出保留於 `build/report1-live-20261111/`。

## 判讀限制

本次補上固定報告基準 G1 的暖機後 single／queued 逐幀分布與初始化觀測，
但只涵蓋一個 140-frame prefix、每風格四個 sessions。新增 host timestamps 有少量量測成本；
single 固定先於 queued，沒有反轉順序，因此不將其差異全部歸因於純排程因素。

一秒遙測涵蓋 constructor、暖機、處理與清理；GPU utilization 包含閒置區段，
不應當作穩態單一 kernel、Tensor Core 或 OFA 利用率。遙測摘要另存於 `summary.json`。

仍未量測外部攝影機曝光／可信來源時間戳到下游收到結果的完整路徑；
本輪不讀 GT、不重新計分、不補 sanitizer gate、不改 tracker／模型預設。
完整 G1 memcheck、HR 原生 Graph racecheck 與 nvJPEG initcheck 的既有 OPEN／FAIL 判定照舊。

## 重算與重跑

`run.py` 包含量測、保管與統計方法，僅可在新目的地執行；現有 build 目的地存在時會拒絕覆寫。
先為新 campaign 修改 `WORK`／`EVIDENCE` 並保留此次來源身分。
單純重算現存 CSV 不需要執行 GPU。

執行 `python3 -B evidence/report1/2026-11-11-g1-live/verify.py` 可只讀核對全部收據 hashes、
八份完整壓縮 packet bytes、480＋480 個逐幀樣本與統計、前後保管及八個 clean runtime identities；本次通過。
