# Racecheck 主機記憶體來源追蹤（2026-11-05）

**已定位到 Compute Sanitizer 注入目標程序的 `libsanitizer-collection.so` worker 所配置的大型暫存 heap。** 本機版本的直接配置呼叫點為 **`libsanitizer-collection.so + 0x5b9ef9`**。TensorRT pointwise 與實際 HR 自訂／CUB 路徑的 OOM 探針，在最後捕捉到的 ≥64 KiB heap requests 中，該呼叫點分別占約 **97%／98%** 的 live bytes。

記憶體高峰與 kernel grid 規模、非同步排隊及工具配置的存活時間有關。同一 TensorRT pointwise 暖機只增加工具的 `--force-blocking-launches`，便由 **6 GiB OOM** 變成完整跑完、cgroup peak **1.08 GiB**。此為定位用對照，**不改判原生排程的完整 HR racecheck PASS**；原補驗的 OOM／OPEN 保留。

## 1. 實驗身分與方法

- 原受測 runtime：乾淨 `a00ab3a6abd7cc2c1f823fd3fbac66a345f3ce48`；`track_sequence` bytes 與[上一輪封存 binary](../../hr-sanitizer/2026-11-05-a00ab3a/raw/binaries.tar.gz)相同。
- 同一 HR 1280×768 FP16 engine，SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`；RTX 5070 Ti Laptop、WSL2、driver 617.42、Compute Sanitizer 2026.3.0.0、CUDA 13.4、TensorRT 10.16.1.11。
- 新增的 constructor-only／CUDA 最小探針、malloc observer 均只在 `build/` 編譯，沒有修改部署 runtime。Constructor 探針使用原 HR config，完全不讀 JPEG、不提交 tracking frame，且在建立前／建立後／close 後 flush 標記。
- 各探針持有 GPU lock，使用獨立 user service、3–8 GiB 硬限額、swap 0、60／90／120 秒 timeout，並記錄 cgroup memory.stat、每 PID RSS／PSS／anonymous、maps 與 OOM 狀態。
- Allocator observer 攔截 malloc／calloc／realloc／free、mmap／munmap，計數 ≥64 KiB requests；固定 table 約 12 MiB。V2 每增加 128 MiB heap peak 捕捉當時 allocating thread 最多 32 frames 的 stack。CPU selftests 已驗證 allocate／free／realloc／mmap 與 growth stack。

來源、binary／library hashes 及 observer 限制見 [source-contract](analysis/source-contract.json)、[observer metadata](analysis/allocator-observer-metadata.json)。全部 **25 組 GPU probes** 的命令與程序記憶體彙總見 [process-memory-all25.json](analysis/process-memory-all25.json)。此為來源診斷，不是新正式驗收。

## 2. RAM 位於 target，不在 sanitizer 前端

TRT-only probe（排除自訂 kernels）在 8 GiB 限額 OOM 前，記錄到：

| 程序／計數 | 已觀察值 |
| --- | ---: |
| 被插樁 target RSS | 8,432,054,272 B |
| 同一 target anonymous | 7,851,876,352 B |
| Compute Sanitizer frontend RSS | 約 8.9 MB |

因此不能只看 executable 名稱，就把 target RSS 全算成 tracker 自身配置；工具 library 注入在 target 內。cgroup charge、各程序 RSS／PSS 與 tracked requested bytes 分別記錄，沒有直接相加。冷／熱 shared-file page charging 亦會使 cgroup peak 和 target RSS 不同。

## 3. 直接配置點與 worker stack

| V2 probe | 最後捕捉的 tracked heap live | 同一 collection caller live | 已追蹤大配置中的比例 |
| --- | ---: | ---: | ---: |
| TensorRT `__myl` pointwise，原生排隊，6 GiB OOM | 5,234,969,548 B | 5,077,437,368 B | 96.99% |
| 原 `track_sequence`，HR 自訂／CUB，要求 1 幀，6 GiB OOM | 5,234,767,499 B | 5,134,563,376 B | 98.09% |

兩個 OOM probes 各有 39 次 heap growth captures，其中 37 次為 collection worker paths，35 次的 triggering allocation 恰為 **524,288 B（512 KiB）**。主要 raw chain：

```text
libsanitizer-collection.so +0x5b9ef9  (allocation caller)
  <- +0x1987da <- +0x196e4f <- +0x1a0b6f <- +0x5fbf41
  <- libc start_thread <- __clone3
```

libc thread boundary 使用相符 `.gnu_debuglink`／CRC 的本機 debuginfo 驗證，支持這是工具內部 worker 配置。Sanitizer private symbols 已 stripped，只能可靠地給 module＋offset；不能由 nearest exported symbol 命名內部函式、payload 或 patch-generation phase。

詳見 [pointwise attribution](analysis/growth-attribution.json)、[實際 HR 一幀 attribution](analysis/custom-frame-attribution.json)、[ELF 邊界](analysis/elf-symbol-boundaries.json)、[libc thread proof](analysis/libc-thread-boundary-proof.json)。OOM 的實際 HR probe **沒有完成該幀**，不作 packet 等價或 frame coverage PASS。

## 4. 對照確認配置存活時間與排隊有關

下列使用同一 constructor binary、同一 engine、同一 `__myl` filter 與相同 zero-input F03 暖機；未帶 observer 的兩組命令只差 `--force-blocking-launches`：

| 工具設定 | 結果 | cgroup RAM |
| --- | --- | ---: |
| 原生排隊、1 worker | 約 7.55 秒 OOM | 6 GiB 硬限額 |
| 強制逐次 blocking、1 worker | 19.19 秒完整 constructor／close，0 hazards | **1,162,190,848 B（1.08 GiB）** |
| 原生排隊、4 workers | 約 5.31 秒 OOM | 6 GiB 硬限額 |

所以增加 worker 數沒有消除此例的峰值；逐次完成 launches 則明顯降低同時存活量。帶 observer 的重複對照亦同方向：強制 blocking 的 tracked heap peak 為 **686,468,044 B**，collection caller peak **528,935,864 B**，原生版本已觀察超過 5 GB 便 OOM。

正常退出的 no-match control、完整自訂 constructor 及 pointwise blocking 對照，最後 tracked heap 都回到 **93,358,024 B**，collection caller live 都是 **28,309,272 B**。可確認這些 bounded 對照釋放了 transient 大配置；不能把 OOM 中止的案例或任意長跑改稱 leak-free。見 [sync comparison](analysis/growth-sync-comparison.json)、[獨立因果核對](analysis/independent-causal-assessment.json)。

Constructor 的完整 custom-filter inventory 另驗證了 306 launches：31 selected（27 scan＋4 control／tail warmup）、275 TensorRT filtered，沒有錯配。Scan core 單獨插樁可完成；各個 helper 單獨插樁及完整自訂 constructor 的峰值不同，不能把整個來源簡化成 scan core。`__myl` pointwise 單獨就能重現快速 OOM；xmma-only 是 90 秒 timeout，不能判為完整通過。見 [filter 驗證](analysis/custom-filter-validation-full.json)。

## 5. 無模型的最小重現：成本隨 blocks 放大

新增 CUDA-only probe，沒有 TensorRT、JPEG、scan 或 tracker。固定 256 threads／block，只配置 `4 × blocks × 256` device bytes；每個 launch 後同步、D2H，驗證所有結果。Global kernel 的 SASS 只有 global store；shared kernel 保留 shared store／barrier／shared load，未被編譯器消去。

比較 blocks 由 1 增為 1,024，同一 collection 配置呼叫點的差值為：

| Kernel | 新增配置次數 | 新增**累計 requested bytes** |
| --- | ---: | ---: |
| Global store | 1,023 | **512 KiB × 1,023** |
| Shared／barrier | 3,069 | **704 KiB × 1,023** |

同樣 1,024 blocks，shared 路徑相對 global 路徑每 block 多兩次配置、合計 192 KiB；未記錄每筆大小，不能宣稱各為 96 KiB。四組都 exit 0，最後 tracked heap 82,282,607 B／collection caller live 22,234,872 B 完全相同。

這直接示範了 **grid 規模相關的暫時工具配置成本**，即使 global-only kernel 也存在；表中的累計 bytes 不能當作同時 resident RAM。完整暖機的 inventory 有 154 個 `__myl` launches、合計 187,761 個 logical blocks，故大量 launch／blocks 與暫存存活重疊足以形成很高的工具峰值；不把最小探針的固定比例直接當成所有 TensorRT kernels 的精確公式。

見 [block scaling](analysis/block-scaling-attribution.json)、[CUDA source](control/block_memory_probe.cu)、[SASS](control/block_memory_probe.sass)。

## 6. 證據與結論邊界

原始檔：[probes／per-process maps／telemetry](raw/probes.tar.gz)、[allocator logs／offline resolutions／selftests](raw/allocator-traces.tar.gz)、[observers 與 probes binaries](raw/observer-and-probe-binaries.tar.gz)。[ARTIFACTS.json](ARTIFACTS.json) 記錄檔案 hashes；raw probes 與原檔逐一核對。重跑入口為 [run_probe.py](control/run_probe.py)，實際每組 command/environment 已封存。

Allocator 數字是攔截到的 ≥64 KiB requested bytes，不是總 RSS；小配置、aligned／private allocator、直接 syscalls 等可能未涵蓋，observer 亦會擾動時間。Below-threshold realloc 的並行 pointer reuse 可能使計數偏低，因此僅作診斷歸因，不作正式 allocator accounting 或整體 RAM 的精確百分比。OOM 前最後一個 snapshot 不等於真實 peak。CUDA driver 的 100+ GB mmap requests 為稀疏虛擬位址保留，沒有當作實體 RAM、沒有加進 heap。

**本輪可確認**：主要 RAM 增長位於 target 的工具 collection worker 大配置；實驗支持 grid 規模及非同步排隊／暫存存活量放大的解釋；逐次完成 launches 可降低此例的峰值。**仍未確認**：sanitizer 私有 buffer 的具體資料結構、是否為 NVIDIA 已知 bug、任意長跑有無洩漏。沒有修改部署 kernel／排程、沒有降低 hazard cap；強制同步只作來源對照。[原生完整 HR racecheck](../../hr-sanitizer/2026-11-05-a00ab3a/REPORT.md) 仍為未完成，不能用本輪 scoped／同步化探針關閉。
