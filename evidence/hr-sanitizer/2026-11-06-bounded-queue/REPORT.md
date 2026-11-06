# HR racecheck 有界排隊補驗（2026-11-06）

**已跑通未過濾的完整 HR 30 幀 racecheck：ordinary serial、8 CPU workers、每次 launch 工具同步，exit 0、0 errors／0 warnings，MOT／完整 packets／SCMO 與 baseline 相同。** 整個工具命令約 50 分 20 秒，cgroup peak **1.031 GiB**，結果為 `PASS_FORCED_SYNC`；此診斷入口明確關閉 Graph。

全部自訂 CUDA／CUB 路徑另以 sync limit 8 跑完四種排程各 30 幀及 HR lifecycle 180 checks，均為 `PASS_FILTERED_FORCED_SYNC`，最高 peak **6.270 GiB**。部署 runtime、engine 與 Graph 預設均維持原版。

這個結果明確包含 kernel filter 與工具同步。**原生排隊、未過濾的完整 HR racecheck 仍 OPEN**；本輪兩次未過濾 Graph 嘗試亦因 cgroup OOM 未完成，不能由自訂路徑或 lifecycle 的通過關閉。

## 1. 受測身分與保管

| 項目 | 實際範圍 |
| --- | --- |
| Runtime source | 乾淨 `a00ab3a6abd7cc2c1f823fd3fbac66a345f3ce48`，沿用前輪 clean build；主 checkout 的開發檔案不作本輪 runtime source |
| Engine | 原 HR 1280×768 FP16，SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876` |
| Motion | `v1-motion/2`，G1 保留 |
| GPU／環境 | RTX 5070 Ti Laptop，UUID `GPU-825ae551-ba16-539a-e444-22726b59ee9a`；WSL2、driver 617.42、Compute Sanitizer 2026.3.0.0、CUDA 13.4、TensorRT 10.16.1.11 |
| Session 輸入 | MOT17-10-SDP 前 30 幀；沒有讀 GT、重新計分或修改 tracker 參數 |
| Lifecycle | 同序列及既有 `hr-records-v2`；沿用前輪另行編譯、連結 clean runtime 的 HR 測試 harness |
| 正式性 | `formal_acceptance=false`；不重發原 §20.8 固定驗收或原 CTest／品質證書 |

本輪三個執行檔逐 byte 比對[前輪 frozen archive](../2026-11-05-a00ab3a/raw/binaries.tar.gz)完全相同。完整驗證見 [binary identity](control/binary-identity.json)，archive 本身的 SHA-256 與原 `ARTIFACTS.json` 核對見 [prior binary contract](analysis/PRIOR_BINARY_CONTRACT.json)。

| Binary | SHA-256 |
| --- | --- |
| `track_sequence` | `d92a4fe87c44d55cf7dd8aa68a563d4e50a287f8512fbc41ad71ea3edff05f4d` |
| `postprocess_sanitizer_control` | `0f82f04fdd73c07e28ac0882696b188df6f2e867e79c199602415852bb40aa47` |
| `tracking_session_hr_tests` | `cbba35fd548c781ab33e9c433078f3050f055990dcce4c3149ea3ea4ce8b1311` |

沒有修改部署 runtime、kernel、engine 或部署的 ring／Graph 預設；ordinary serial／Graph 等測試 mode 與工具同步設定另行明列。新增內容是驗證 runner、控制測試與證據；所有已完成 session／lifecycle captures 的 source、binary、engine、input 前後 custody 皆不變。環境與函式庫身分另見 [environment](control/environment.json)、[libraries before](control/libraries-before.json)、[libraries after](control/libraries-after.json)；各 capture 原始 custody 與 commands 保存在 raw archives。

## 2. 能跑完的自訂／CUB 範圍

Session 固定 filter 為 `_ZN7saccade|_ZN3cub`，不設 launch count／skip，不降低 hazard caps，不關閉 racecheck 的其他檢查。這個 filter 在前輪完整 Graph inventory 對應 **36 個 kernel names／2,518 launches**；這是受測 binary 的既有 scope reference，並非宣稱本輪每種排程另量到相同 launch 數。TensorRT、nvJPEG 與 NVOF library CUDA kernels 不在此通過範圍，OFA 硬體本體也不是 CUDA racecheck 的插樁對象。

四組均使用 `--racecheck-num-workers 8 --force-synchronization-limit 8`、8 GiB cgroup 上限、swap 0。下列時間為 target wrapper 包含工具執行的 elapsed time，排除等待 GPU lock；不是部署效能指標。RAM 採 periodic telemetry／execution record／最終 target `memory.peak` 的最大值，不只取最後一次輪詢。

| HR 排程 | 完成 | Target 時間 | RAM peak | 結果 |
| --- | ---: | ---: | ---: | --- |
| Graph pipelined | 30 幀 | 159.019 s | 6,457,942,016 B（6.014 GiB） | `PASS_FILTERED_FORCED_SYNC` |
| Ordinary pipelined | 30 幀 | 176.495 s | 3,315,884,032 B（3.088 GiB） | `PASS_FILTERED_FORCED_SYNC` |
| Ordinary serial | 30 幀 | 158.287 s | 3,277,430,784 B（3.052 GiB） | `PASS_FILTERED_FORCED_SYNC` |
| Graph serial | 30 幀 | 232.521 s | 6,732,693,504 B（6.270 GiB） | `PASS_FILTERED_FORCED_SYNC` |
| HR lifecycle | 180 checks | 372.783 s | 6,533,558,272 B（6.085 GiB） | `PASS_FILTERED_FORCED_SYNC` |

每個 session 均滿足 target 與外層 exit 0、完整 30 幀完成標記、唯一零 hazards summary、沒有未知／internal／untracked 診斷，以及 MOT／完整 packet／SCMO 三類輸出逐 byte 等於普通 baseline。Packet 檔長恰為 `30 × 12368 = 371040` bytes。四種普通 baseline 彼此也完全相同；不是只比較 sanitizer 的 exit code。

Lifecycle 同時要求唯一 `tracking_session_tests: 180 checks passed` 與 HR lifecycle 身分行，保留原測試的失敗凍結、drain／cancel、backpressure、destructor／重複 close 等檢查。它不另輸出 session 的三種檔案，沒有把 session 的 byte-parity 判定套到 lifecycle。

兩類 runner 都先以植入 shared race 的正控制驗證工具；控制案例 exit 99、完整非零 summary，之後才可接受受測 target 的零 summary。正控制使用同一同步界限與 workers，filter 額外只加入 test-only `_Z11shared_racePf`；這個額外名字不加入 session filter。根據測試 source 與 clean binary `nm` 確認正控制確實被選中。

原始證據：[四種 session 排程](raw/custom-sync8-workers8-four-modes.tar.gz)、[HR lifecycle](raw/lifecycle-custom-sync8-workers8.tar.gz)。精確 statuses、耗時、峰值、完成與 parity 欄位見 [SUMMARY](analysis/SUMMARY.json)。

## 3. 未過濾 Graph 仍未完成

| 未過濾設定 | 要求 | 實際結果 | 執行時間 | RAM 證據 |
| --- | --- | --- | ---: | --- |
| Graph pipelined、workers 8、sync limit 8、MemoryMax 8G | 30 幀 | cgroup OOM，`INCOMPLETE` | 249.443 s | 記錄 peak 8,589,934,592 B（8 GiB） |
| Graph pipelined、workers 8、sync limit 1、MemoryMax 24G | 30 幀 | cgroup OOM，`INCOMPLETE` | 200.718 s | 最後採樣 25,594,753,024 B（23.837 GiB）；systemd 終止紀錄為 24G peak／`oom-kill` |

兩組均無完成 30 幀、無可接受的三類 outputs 或唯一完整零 summary；沒有用正控制通過代替 session 完成。第二組的 observer 記錄到第一張 JPEG 的 open，但未提供幀完成證據。這些資料只支持「未過濾 Graph 路徑在上述設定仍會 OOM」；沒有把 OOM 精確命名為某個 Graph 建立 API、實例化階段或私有工具資料結構。

原始證據：[sync8／8G](raw/full-sync8-workers8-graph.tar.gz)、[sync1／24G](raw/full-sync1-workers8-graph-24g.tar.gz)。24G 案例的最後採樣值不是真實 peak 的精確量測；systemd 終止摘要與 cgroup 上限另行保留，沒有補造 final target record。

### 已完成：未過濾 ordinary serial（Graph 關閉、逐次同步）

`full-sync1-workers8-ordinary-serial` 完整跑完 MOT17-10-SDP 前 **30 幀**。使用 `--serial --no-cuda-graph`、8 workers、sync limit 1、8 GiB 上限、swap 0、3600 秒 timeout，**沒有 kernel filter、launch skip／count 或 hazard-cap override**。

| 項目 | 實際結果 |
| --- | --- |
| 狀態 | **PASS_FORCED_SYNC**；`native_schedule_acceptance=false` |
| Target／外層 exit | 0／0；無 OOM、timeout、internal／unknown／untracked 診斷 |
| Target 執行時間 | **3019.959 s（約 50 分 20 秒）**，包含工具／初始化，不是部署效能指標 |
| Cgroup RAM peak | **1,107,017,728 B（1.031 GiB）**；swap peak 0 |
| 完成與 summary | `track_sequence frames=30 mode=serial`；唯一 `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` |
| Outputs | MOT／完整 packets／SCMO 與 ordinary baseline 相同，亦逐 byte 等於既有 Graph baseline；packet 檔長 371040 bytes |
| Custody／正控制 | 前後 source／binary／engine／input 一致；同工具設定的植入 shared race 正控制通過 |

[最終獨立核對](control/FINAL_VERIFICATION.json)對 30 份 baseline／受測 output 檔案做實際 bytes equality，並核對三個 binaries 與前輪 archive 相同、17 個 SDK／driver／injection library hashes 前後一致。這項未過濾成功只適用於 **ordinary serial＋工具逐次同步**；原生排隊與未過濾 Graph 的驗收邊界保留。

[未過濾 ordinary serial 原始 capture](raw/full-sync1-workers8-ordinary-serial.tar.gz)包含完整命令、環境、outputs、零 summary、前後 custody 與 memory telemetry。輸入 open observer 沒有 queue overflow，但不作幀完成證據；上述完成判定來自 target log／exit、outputs 與 final record。

## 4. 為何使用這些工具選項，以及不能推出什麼

前輪已將主要大型 heap 成長定位至 target 內 Compute Sanitizer collection worker；其 allocator 與 grid／排隊存活量證據見 [RAM 來源追蹤](../../hr-sanitizer-memory/2026-11-05/REPORT.md)。本輪使用公開工具選項控制在途工作，沒有改 tracker 的排程程式。

NVIDIA 手冊說明大量並行 launches 可能令工具耗盡 host memory，並提供 `--force-synchronization-limit` 限制每個 stream 在強制同步前的 launch 數；值 1 等同 blocking launches。[NVIDIA Memory Footprint](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memory-footprint)

手冊亦說明 blocking 會限制錯誤報告，而同步選項會改變 CUDA API serialization；因此即使輸出相同、零 hazards，也不能將結果視為原生並行排程驗收。`racecheck-num-workers` 控制 CPU worker 數，0 為自動；本輪固定使用 8。[NVIDIA Nonblocking Mode](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#nonblocking-mode)、[Racecheck options](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#command-line-options)

Racecheck 的主要範圍是 shared-memory hazards。這一輪沒有證明所有跨 stream global-memory leases 一般無競態；ready／consumed、last-use 與 close／drain 的所有權契約仍須維持。也沒有把 NVOF memcheck 或 nvJPEG initcheck 的既有 OPEN／FAIL 關閉。[NVIDIA Compute Sanitizer tools](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#compute-sanitizer-tools)

## 5. 診斷 probes、中止與原始失敗保留

Constructor-only probes 與完整 session 分開判讀。8 workers／sync8 的完整 constructor inventory 約 80.347 秒完成，最後有 `constructor_probe complete repeats=1`，peak 2,687,922,176 B（2.503 GiB）。它開了 `--dump-kernel-launches`，literal summary 是 **306 hazards displayed（0 errors、0 warnings）**，exit 99。此版工具將 dump 資訊計入該總數；這是 inventory 診斷，沒有把 306 減去後製造零 summary PASS，也沒有任何 session frame coverage。

同一批亦保留原生排隊的 1／8 workers constructor OOM、1 worker 的 90／180 秒 timeout、初次遺漏 `REPEATS` 參數而在首個 instrumented API 前退出的無效 probe，以及先前中止的一 worker session／等待 GPU lock 的重啟。未執行 GPU 的 preflight 中止須同時有明確 interruption reason 且沒有 command／execution／results／target／telemetry 執行紀錄；它們列為 `ABORTED_OR_UNSTARTED_PREFLIGHT`，不作成功 capture。

全部原始 bytes 依目錄分別封存；[ARCHIVE_VERIFICATION](analysis/ARCHIVE_VERIFICATION.json)記錄 tar.gz 成員的大小／SHA-256 與 symlink target 核對，以及封存前後 source 不變。[ABORTED_OR_UNSTARTED](analysis/ABORTED_OR_UNSTARTED.json)另外列出中止與 preflight 原因。前輪原生 OOM／timeout 及本輪失敗均保留，不被後續 scoped 結果取代。

## 6. Runner 的拒絕條件與 CPU 檢查

所有 GPU captures 持有同一 `gpu.lock`，在獨立 user service 使用 MemoryHigh／MemoryMax、MemorySwapMax=0、OOMPolicy=kill、KillMode=control-group 與 timeout。每次命令、環境、target record、log、telemetry 與原始 outputs 都保留。任何 timeout／OOM、internal／unknown／untracked 診斷、摘要缺失／重複、正控制失敗、缺少 target final／完成記錄、bytes 不符或 custody 改變，都不能判 scoped PASS。

| CPU 檢查 | 通過數 | 證據 |
| --- | ---: | --- |
| Root runner scope／分類、workers 與最終 peak 記帳 | 175 | [runner-tests](control/runner-tests.json) |
| Custom runner 固定 filter／命令傳遞與拒絕條件 | 131 | [custom-runner-tests](control/custom-runner-tests.json) |
| Lifecycle runner 完成／scope／控制與 custody | 175 | [lifecycle-runner-tests](control/lifecycle-runner-tests.json) |
| Archive preflight／active guard、portable-source 衍生與編譯回歸 | 46 | [archive-tests](control/archive-tests.json) |
| 輸入 open observer 身分／overflow／停止條件 | 33 | [watch-inputs-tests](control/watch-inputs-tests.json) |

封存器首次產生 portable source 時，路徑樣式的全域替換誤改自身字串，衍生版本未能編譯；已改為只匹配完整 assignment 行，所有衍生來源編譯與回歸檢查通過。[封存過程紀錄](control/archive-build-attempts.json)保留第一次錯誤與修正，原 GPU captures bytes 未修改。

這些是分別驗證 runner／observer／封存判定的 CPU 測試；不當作 CUDA racecheck、完整 CTest 或 tracker 品質證據。輸入 open observer 沒有 PID 級事件歸屬，open 也不等於幀完成；其 overflow／觀察缺口欄位與 target 完成判定分開保管。

## 7. 重跑與 artifacts

所有 output 目錄必須全新，並在 repository root 執行；既有 clean source/build 與原 engine 必須仍可用。未過濾 ordinary serial 的入口為：

```sh
python3 -I scripts/run_hr_sanitizer_validation.py \
  --source build/hr-sanitizer-20261105-source \
  --build build/hr-sanitizer-20261105/cuda \
  --output build/hr-racecheck-unfiltered-ordinary-rerun \
  --frames 30 --tools racecheck --modes ordinary-serial \
  --racecheck-num-workers 8 --force-synchronization-limit 1 \
  --memory-high 8G --memory-max 8G --timeout 3600
```

自訂／CUB 四種排程與 HR lifecycle 的 portable evidence runners：

```sh
python3 -I evidence/hr-sanitizer/2026-11-06-bounded-queue/control/capture_custom.py \
  --output build/hr-racecheck-custom-rerun \
  --frames 30 --racecheck-num-workers 8 --force-synchronization-limit 8 \
  --memory 8G --timeout 900

python3 -I evidence/hr-sanitizer/2026-11-06-bounded-queue/control/capture_lifecycle.py \
  --output build/hr-racecheck-lifecycle-rerun \
  --custom-kernels-only --racecheck-num-workers 8 \
  --force-synchronization-limit 8 --memory 8G --timeout 900
```

`control/*.captured.py` 保存實際受測來源的原始 bytes；可執行的 evidence copies 另以 repository ancestor discovery 取代固定 parent depth／絕對 root，沒有覆寫原 capture source。衍生關係、修改與 hashes 見 [RERUN_DERIVATIONS](control/RERUN_DERIVATIONS.json)。重跑成功仍必須保留相同 scope tag；exit 0 僅表示所要求的受限 capture 完整通過。

[ARTIFACTS.json](ARTIFACTS.json)收錄本輪檔案 hashes；[SUMMARY](analysis/SUMMARY.json)保留原始 scope/status、每項耗時／RAM／完成／parity 與資料入口。沒有修改部署程式或 engine，也沒有將 `PASS_FILTERED_FORCED_SYNC`／`PASS_FORCED_SYNC` 改稱原生完整 HR racecheck PASS。

Analysis JSON 的 artifact 相對路徑均以本證據目錄為基準。所有 raw 成功、OOM、timeout 與中止紀錄保留；未更新部署 binaries、未重跑品質評分或重新發行原正式證書。
