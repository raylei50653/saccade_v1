# HR 完整路徑 sanitizer 補驗（2026-11-05）

**結果：部分補齊，NOT A FULL PASS。** 現行 HR／G1 的完整 synccheck 已補上四種排程與 HR lifecycle；未過濾 racecheck 及全部自訂 kernels／CUB 的 racecheck 仍因資源上限未完成。Memcheck 重現 NVOF 診斷，initcheck 出現 nvJPEG 未初始化讀取後 timeout。沒有將過濾、未完成或零報告尾行改判為完整 PASS。

## 1. 受測身分與保管

| 項目 | 實際範圍 |
| --- | --- |
| Runtime source | 乾淨 detached `a00ab3a6abd7cc2c1f823fd3fbac66a345f3ce48`；新建 Release CUDA build，`source_dirty=false` |
| 模型／motion | HR 1280×768 FP16；`v1-motion/2` G1；engine SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876` |
| 裝置 | RTX 5070 Ti Laptop，UUID `GPU-825ae551-ba16-539a-e444-22726b59ee9a`；WSL2，driver 617.42 |
| 工具 | CUDA 13.4.92、Compute Sanitizer 2026.3.0.0、TensorRT SDK 10.16.1.11 |
| 輸入 | MOT17-10-SDP 前 **30 幀**；HR lifecycle 用同序列及既有 `hr-records-v2`；未讀 GT、未重新計分 |
| 排程 | Graph pipelined、ordinary pipelined、ordinary serial、Graph serial；原生 ring／Graph 設定，不插入工具強制同步、不降低 hazard cap、不設 launch-count／skip |
| Test harness | `tracking_session_tests.cpp` 新增 `--profile NAME`；另行編譯此測試檔，連結上述乾淨 runtime libraries。驗證 harness 為本輪未提交改版，與 runtime clean source 分開記錄 |

各 capture 的 source、全部受測 build binaries／libraries、engine、前 30 張 JPEG 與 metadata 均有前後 hashes；完成的收尾檢查皆為 `custody_unchanged=true`，包括 OOM／timeout 與中止的 capture。NVOF、nvJPEG、TensorRT shared libraries 的前後 hash 核對通過。執行檔、實際命令、source snapshots、tool logs、遙測、outputs 與 test harness source 已封存；入口為 [ARTIFACTS.json](ARTIFACTS.json)、[環境](control/environment.json)、[runtime／harness binaries](raw/binaries.tar.gz)。

這是指定乾淨 runtime 的補驗；`formal_acceptance=false`。未重跑原 §20.8 全套評估、MOT 七序列計分或 semantic mutations，不改寫既有正式證書。

## 2. 通過與仍未完成的範圍

| 項目 | 實際結果 | 判定 |
| --- | --- | --- |
| 新建 CUDA build 的完整 CTest | **68/68**，包含原七序列 session／lifecycle；642.77 秒 | PASS |
| 新 HR test harness | 30 幀 **602 checks**：CPU／ordinary serial／Graph pipelined packets 相同，含 1/1/8、3/3/5 rings 與兩個 perturb seeds | PASS |
| 新 harness 預設 T01 | 原前 10 幀 **222 checks**；預設 profile 及 bytes 比較保留 | PASS |
| HR lifecycle 普通執行 | **180 checks**，含 ordinary／Graph、serial／pipelined、sticky failure、完整 committed state 凍結、backpressure、cancel／pending drain、D-only failure、destructor／重複 close | PASS |
| **完整 HR synccheck，無 kernel filter** | 四種排程各 30 幀，exit 0、唯一 `ERROR SUMMARY: 0 errors`；MOT／完整 packets／SCMO 三者皆與 baseline 相同 | **PASS** |
| **完整 HR lifecycle synccheck，無 kernel filter** | HR profile，180 checks，exit 0、0 errors | **PASS** |
| **完整 HR memcheck，無 kernel filter** | 四種排程各 30 幀全部跑完，三種 outputs 相同；各 exit 99，**86 次 NVOF `cuArrayGetPlane`**，無其他 API／device 診斷 | **OPEN** |
| **完整 HR lifecycle memcheck，無 kernel filter** | 180 checks 跑完；exit 99，**1,008 次同一 NVOF 診斷**，無其他 API／device 診斷 | **OPEN** |
| 完整未過濾 HR Graph racecheck | 原環境與明確 `CUDA_MODULE_LOADING=LAZY` 各一次；兩者皆 24 GiB cgroup OOM，runner exit 1；未完成 30 幀，無可比對 outputs | **INCOMPLETE** |
| 全部觀察到的自訂 kernels＋CUB racecheck | regex `_ZN7saccade\|_ZN3cub` 對應 **36 kernel names／2,518 launches** 的 Graph inventory；Graph／ordinary pipelined 兩次都 24 GiB OOM；ordinary serial 啟動後中止，Graph serial 未執行 | **INCOMPLETE**，不是未過濾路徑 PASS |
| 完整未過濾 HR Graph initcheck 嘗試 | 已印出 **389 則** nvJPEG `rgba2rgb` 的 3-byte uninitialized global read；print-limit 1,000 亦計入 launch 資訊，之後輸出截限；**900 秒 timeout、exit 124**，無完整 summary／完成 outputs | **診斷 FAIL，執行 INCOMPLETE** |
| 四種工具正控制 | 不開 dump 的四種 controls 均 exit 99、命中各自植入缺陷與完整非零 summary | PASS（工具正控制） |

HR 普通四種排程、完整 synccheck 四種排程及完成的 memcheck 四種排程，均要求 target 完成 30 幀、packet 檔長恰為 `30 × 12368` bytes，並逐 byte 比對三種 outputs；exit 99 本身不能取代 target 完成證據。Lifecycle 的 bytes／狀態比較由 test harness 的 180 checks 判定，沒有宣稱它另輸出 MOT／packet 檔。

主判定見 [四種排程](captures/no-dump/results.json)、[lifecycle](captures/lifecycle/results.json)、[NVOF 86 次分類](control/memcheck-classification.json)、[1,008 次分類](control/lifecycle-memcheck-classification.json)。分類沿用受測 clean source 的既有 classifier，結果仍明示 `memcheck stays open, not passed`。原 Bug 7048273 的原因與外部確認狀態沿用原報告，未將其改稱無害。

## 3. Launch inventory 與 dump 開關的判讀

第一批使用 `--dump-kernel-launches` 保管實際 kernel inventory。此版工具會把 dump 的資訊報告也計入 `ERROR SUMMARY`／exit 99：synccheck 有 **11,551 launches、summary 11,551**，而非 11,551 個同步缺陷；memcheck 有同樣 11,551 launches，另有 86 則 NVOF 診斷，literal summary 為 **11,637**。兩份原始紀錄均完整跑完 30 幀且三種 outputs 相同，但只列 **INVENTORY_ONLY_NOT_ACCEPTANCE**。

未藉由扣掉 launch 數產生 PASS；正式補判另以不開 dump 的命令重跑。新增 runner 預設不開 dump，若明確開啟，該 capture 強制保持 OPEN。原始 inventory 與正控制、分類器修正前後紀錄均保留。

Graph 的兩份完整 inventory multiset 相同：**188 個 exact kernel names、233 個 name／grid／block／reported-cluster 組合、0 filtered launches**：

| Owner | Launches | 證據 |
| --- | ---: | --- |
| TensorRT | 8,800 | 550 有直接 library backtrace；8,250 Graph launches 由相同 exact name 的暖機 witness 推論，與直接證據分欄 |
| 自訂 CUDA | 1,858 | scan 864、session control 331、flow 240、tracking 153、F04 150、producer 90、resize 30 |
| CUB | 660 | 5 個 names，依 mangled namespace 推論 |
| NVOF CUDA transforms | 203 | 3 個 names；不代表 OFA 硬體內部受到 CUDA 插樁 |
| nvJPEG | 30 | `rgba2rgb` |

見 [owner coverage](inventory/owner-coverage.json)、[counts](inventory/counts.csv)、[比較](inventory/comparison.csv)。完整逐 launch JSON／CSV 在 [launch-inventory.tar.gz](raw/launch-inventory.tar.gz)；raw unknown owner 保留，不把推論覆寫成直接 library 證據。初次 racecheck inventory 只有 constructor warmup 的 **302 launches**，不能套用完整 11,551 次的 coverage。

## 4. 資源隔離、失敗與停止條件

所有 sanitizer 檢查持有同一 `gpu.lock`，在獨立 user service 下執行，`MemoryMax=24G`、`MemorySwapMax=0`、`OOMPolicy=kill`、`KillMode=control-group`，有 timeout／RuntimeMaxSec 與每 0.5 秒 fsync 的 cgroup memory／events／service 遙測。每次 OOM 的 peak 為 **25,769,803,776 bytes**，events 有 `oom_kill`／`oom_group_kill`；初次 global OOM 的歷史紀錄不被這些隔離重跑取代。

初次 racecheck 的軟限額由 16→22→24 GiB 調整，24 GiB 硬上限始終不變；兩次介入的命令／時間／前後值保存在 inventory raw archive 的 `racecheck-resource-intervention*.json`。後續 racecheck 由一開始即採 24／24 GiB，未降低 hazard cap、未關閉插樁、未加 forced blocking／synchronization。

未過濾原環境與 lazy loading、再到自訂／CUB 的兩種排程，合計 **四次實際 cgroup OOM**。自訂／CUB 第三種排程開始後由 runner SIGTERM 中止，保留 manifest 的 `KeyboardInterrupt: SIGTERM`；後續排程未執行。該中止不是 PASS，也沒有製造完成幀數。由於第一個有 inventory 的失敗尚停在暖機、其餘失敗沒有完成 outputs，這些結果均不能宣稱 racecheck 已涵蓋任何完整的 30 幀 session。

原始 archive：[初次未過濾／inventory／initcheck](raw/inventory.tar.gz)、[不開 dump 的完整檢查](raw/no-dump.tar.gz)、[HR lifecycle](raw/lifecycle.tar.gz)、[lazy racecheck](raw/lazy-racecheck.tar.gz)、[自訂／CUB racecheck](raw/custom-racecheck.tar.gz)、[initcheck 正控制](raw/init-control.tar.gz)。

## 5. 工具、重跑與剩餘邊界

新增 [run_hr_sanitizer_validation.py](../../../scripts/run_hr_sanitizer_validation.py)：對既有 clean build 檢查 source／binary／engine／input custody，以逐模式 baseline、正控制、完成標記、三類 outputs、唯一零 summary 與 cgroup 結果共同判定。OOM、timeout、internal tool error、未知診斷、未追蹤 launches、缺少 target final record 均無法判 PASS。更新後 **60 項**純分類檢查通過；HR test harness 的 **21 組**參數解析檢查通過。

本次未過濾 synccheck／memcheck 的重跑入口（output 必須全新）：

```sh
python3 -I scripts/run_hr_sanitizer_validation.py \
  --source build/hr-sanitizer-20261105-source \
  --build build/hr-sanitizer-20261105/cuda \
  --output build/hr-sanitizer-rerun \
  --frames 30 --tools synccheck memcheck --memory-high 24G
```

`--tools racecheck` 可重現未過濾路徑；本輪原生排程尚無 PASS。個別 command JSON、harness 編譯／連結命令及 scoped racecheck／lifecycle 的實際輔助程式都在 raw archives 與 [control](control/)。

**仍 OPEN**：完整 HR racecheck（含完整自訂路徑）、NVOF memcheck、完整 session initcheck。TensorRT／nvJPEG／NVOF CUDA kernels 的 synccheck 已由本輪受測範圍補上；OFA 硬體本體、跨 GPU／SDK、更多輸入／caller，以及所有跨 stream global-memory lease 的一般無競態證明均不由本輪推出。Racecheck 檢查 shared-memory hazards；owner／ready-consumed／last-use／close-drain 契約仍須維持。未套用 sanitizer suppression、未修改部署 runtime。
