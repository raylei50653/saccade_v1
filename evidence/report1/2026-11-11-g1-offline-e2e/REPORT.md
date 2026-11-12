# 完整離線序列的程序啟動到最終輸出 E2E（2026-11-11）

**結果：有界完整程序量測與輸出一致性 PASS。** 固定乾淨 runtime
`3cbaa518f0dc37d691404e95ed85c0109c81130d`，原 HR engine、G1、Graph ON、pipelined、
NSA／量測框 OFF，MOT17-10-SDP 全 654 幀，三個 fresh processes。

## E2E 邊界與結果

主計時從外部 Python 呼叫 `Popen` 前，到 child 正常退出且 blocking `waitpid` 返回後。
使用 `perf_counter_ns`，另以 watchdog 控制 timeout，沒有 `wait(timeout)` 輪詢延遲。
這段包含程序啟動、engine 載入、session constructor、全部 JPEG 讀檔、首幀 lazy capture、
全部處理、drain、checked session close、序列尾插值／ID 定稿、MOT／packet／motion／resolved config 寫檔與
checked file close、stdout 寫入及程序退出。所有測量都輸出上述同一組檔案。

| Fresh process | 完整程序 wall（ms） | 程式內處理 loop wall（ms） | 程式內 ms/frame |
| --- | ---: | ---: | ---: |
| 1 | 5365.146328 | 3427.13 | 5.24025 |
| 2 | 4480.109873 | 3412.49 | 5.21787 |
| 3 | 4664.052763 | 3461.44 | 5.29273 |
| mean | **4836.436321** | **3433.686667** | **5.250283** |

完整程序 wall 中位數 **4664.052763 ms**，範圍 4480.109873–5365.146328 ms。
以每次 654 幀攤提，平均 **7.395163 ms/frame**；三次合計 1962 幀／總 wall 的吞吐為 **135.223532 FPS**。
這些是完整離線序列完成時間與攤提成本，**不是逐幀延遲或逐幀 p95**。最終 MOT 依 `v1-mot/1` 在 EOF 後才定稿。

程式內 loop wall 的起訖在 `tools/track_sequence.cpp`，含 JPEG 讀取、處理、`output.add`、finish/drain，
排除 constructor、checked close 與尾端輸出定稿／寫檔。三次完整程序減此 loop 的平均差為
**1402.749655 ms**，是上述額外範圍的合計，未細分，不能全部歸為 constructor 或輸出成本。
程式內欄位沿用 stdout 的有限小數精度；主程序 wall 另用外部 nanosecond clock 計算。

## 固定條件

- Source 為上述 commit 的既有 clean detached worktree `build/report1-live-20261111/source`；
  本次只建 `track_sequence`，沒有修改／插樁 runtime 或 CLI source。
- Engine SHA-256：`8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`；未重建。
- `--profile hr-1280x768`；G1／`v1-motion/2`、Graph ON、pipelined、result slots=4，
  顯式設定 `tracker.kalman.nsa_enabled=false`、`tracker.rows.measurement_box=false`。
- RTX 5070 Ti Laptop／WSL2、driver **621.14**、CUDA 13.4.92、TensorRT 10.16.1.11、GCC 16.2.1。
  不與歷史 driver 617.42 的樣本合併或宣稱受控跨版本差值。
- 全程持有共享 `gpu.lock`，無 profiler，未鎖 clocks；一秒遙測涵蓋三次完整程序及其中的短暫間隔。
  15 筆遙測 GPU utilization 平均 72.93%、最高 100%；最高 70°C、142.17 W、495 MiB。
  此平均含啟動／輸出／閒置，並非穩態 kernel 或 OFA 利用率。
- 不清除 OS 檔案 cache。量測前核對全部 654 張 JPEG 與 seqinfo 的 hashes，可能預熱 page cache；
  確實經過檔案讀取 API，但不宣稱冷磁碟讀取。

## 正確性與保管

三次各 654 幀正常完成、exit 0；全部最終 MOT、完整 packets、MotionRecords 與 resolved config 逐 byte 相同：

| 輸出 | 每份 bytes | SHA-256 |
| --- | ---: | --- |
| MOT | 594269 | `ccd1c2197e20d8a13a2695da8c07b2ee8e38ba46792c8f5f77d12e9882a8a35f` |
| Packets | 8088672 | `dbb226d5d4d7944cabec1fd8530a57c9c02788bf2db1ade0f790aca3e40531db` |
| MotionRecords | 28820 | `85dfd326b65a384c87fedd6aacfd9f107063543a1551163a3a773ba426c0cc15` |
| Resolved config | 6397 | `1394d49366796c922760c0fa2c207c42b4d2819fa73dcaf7f9538468dce6b097` |

完整 packets 的前 140 幀亦等於先前 G1 single／queued latency campaign 的結果。
Source、engine／seqinfo／654 張影像、CLI binary／runtime static libraries 的 hashes 均前後不變。
這證明本輪有限重跑與不同入口的輸出一致性，不是獨立 CPU／GT 正確性或品質驗收。

`raw/run1`–`run3` 保存 MOT／resolved config 原始 bytes，packets／motion 用 gzip 保存完整內容；
原始未壓縮輸出及 binaries 在 `build/report1-live-20261111/offline-e2e/` 與同 clean build。
每個程序的 command、log、exit、UTC 與 monotonic timestamp 收據均保留，統計在 `summary.json`、逐次值在 `runs.json`。

首次嘗試誤將 identity 欄位名稱當成 resolved config 名稱，CLI 在建 session 前回報 unknown field、exit 1；
保留於 `attempt-invalid-config/`。更正 `tracker.` 前綴後，使用新輸出目錄完成上述三次，
失敗嘗試不納入統計。

## 適用界線與只讀核對

這是完整**離線序列**的程序 E2E；不含相機曝光、網路／RTSP、live acquisition 或下游消費者。
檔案寫入檢查到 `ofstream.close()`，沒有新增 `fsync`，因此沒有永久儲存 durability 保證。
僅一條完整序列、三個程序；不讀 GT、不重評品質、不跑 sanitizer，不改既有 OPEN／FAIL 或部署預設。

執行 `python3 -B evidence/report1/2026-11-11-g1-offline-e2e/verify.py` 可只讀核對 hashes、
三次完整輸出 equality、外部 timing 收據與 summary，以及前後 source／input／binary 保管。
`run.py` 的輸出目錄必須不存在，重跑需改用新 campaign 路徑，不能覆寫此次證據。
