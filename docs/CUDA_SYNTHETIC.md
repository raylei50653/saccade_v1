# 合成資料 CUDA 執行骨架

日期：2026-10-28。基線：`a895ba27bf2b74c0552e7f9941b6af81015474be`；開始時 fetch 後的 `origin/main` 與基線相同。這個 profile 只驗證實際 CUDA 的執行、同步、提交與狀態隔離。YOLO／Mamba、完整 phase correlation／GMC、tracker、MOT writer、P01 模型 artifact 與完整 P08 設定不在本次實作範圍。

程式：[`synthetic_runtime.cu`](../cuda/synthetic_runtime.cu)、[`synthetic_runtime.hpp`](../include/saccade/synthetic_runtime.hpp)、[`synthetic_config.cpp`](../src/synthetic_config.cpp)、[`cuda_synthetic_tests.cpp`](../tests/cuda_synthetic_tests.cpp)。實測與尚未通過項目記於 [GPU 證據](../evidence/cuda-synthetic/2026-10-28/REPORT.md)；不能把工具未啟用時的測試程式成功當作 sanitizer 成功。

## 建置與重跑

CPU 建置不啟用 CUDA 語言，也不尋找 toolkit：

```sh
cmake -S . -B build-cpu -DSACCADE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel 4
ctest --test-dir build-cpu --output-on-failure
```

下列 GPU 命令對應本次 SM 12.0 裝置；其他裝置須指定適當 architecture，未列入本次驗證：

```sh
cmake -S . -B build-cuda -DSACCADE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --parallel 4
ctest --test-dir build-cuda --output-on-failure
build-cuda/cuda_synthetic_tests build-cuda/full-rerun
build-cuda/cuda_synthetic examples/synthetic/serial.cfg build-cuda/serial
build-cuda/cuda_synthetic examples/synthetic/pipelined.cfg build-cuda/pipelined
build-cuda/packet_compare build-cuda/serial/packets.bin build-cuda/pipelined/packets.bin
```

CTest 的 GPU test 會保存到 `build-cuda/gpu-evidence`。CLI `cuda_synthetic CONFIG OUTPUT_DIRECTORY` 的 exit code：0 為正常完成，1 為 sticky failure 或取消，2 為設定／I/O／CUDA API 錯誤。`examples/synthetic/failure.cfg` 是預期 exit 1 的固定尾端實驗。每次重跑使用新的 output directory，避免保留前次更多案例的殘留目錄。

每次 CMake configure 嵌入當時 `git rev-parse HEAD` 與 dirty 旗標；要取得乾淨 commit 的身分，先 commit，再重新 configure/build。各案例的 identity 記錄 compiler、nvcc、CUDA runtime／driver API、cuFFT、GPU 型號／UUID／PCI／compute capability 及 plan workspace 大小；driver 的完整版本另保存在證據的 `environment.txt`。本次小型 plan 所需 workspace 可以是 0 bytes，仍關閉 cuFFT 自動配置並設定獨立的預配置 work area。

## Schema 1 與比較契約

設定是 ASCII `key=value` 的 LF records；空 record 忽略，末尾 LF 可省略。無空白、註解、引號或隱式型別轉換；整數僅接受 canonical 十進位，拒絕 `+1`、`01`、`-0`、小數及溢位；布林只接受 `true`／`false`。未指定欄位使用 typed defaults；`schema` 可省略，提供時必須為 1。未知／重複欄位均拒絕。`run_synthetic` 也驗證直接建立的 typed config，驗證完成之前不建立 CUDA session。

下表順序也是 `canonical_config` 的固定輸出順序，輸出包含所有 defaults，最後一行也有 LF。hash 是這些原始 bytes 的 SHA256，小寫 64 位 hex；CPU 測試包含 SHA256 已知答案。此 schema 不處理模型或 tracker 設定。

| 欄位 | default | 合法值／意義 |
| --- | --- | --- |
| schema | 1 | 唯一支援版本 |
| mode | pipelined | serial／pipelined |
| n_mi、n_cs、n_h、n_rp | 2 各一 | 各自 1..16；依序為 MotionImage、候選、host staging、host packet rings |
| width、height | 8 各一 | 各自 8、16、32；session 內固定 shape，float32 R2C |
| frames | 16 | 1..100000，處理序號從 0 開始 |
| epoch | 1 | 非零 uint64 |
| seed | 7 | uint32 範圍，決定合成輸入 |
| delay_seed | 11 | uint32 範圍，決定固定整數 mixer 的延遲排程 |
| delay_cycles | 20000 | 0..1000000，`clock64` 延遲上限；不是時間／效能保證 |
| fail_frame、fail_stage | -1、none | frame 在 [0,frames) 並搭配 before_spectrum／after_spectrum／after_predict／after_update，或 -1 搭配 none |
| cancel_frame、cancel_cut | -1、none | frame 在 [0,frames)；切點見下節，或 -1 搭配 none |
| fixed_submit | false | 測試專用固定提交範圍；普通執行器讀到 FAILED 後停止新增幀 |
| terminal_snapshot | false | 測試專用 terminal drain 後 CTL／TS／SPEC 讀回 |

不支援 fail 與 cancel 同時啟用；非法 token、shape、容量與配對皆在 session 前拒絕。合法設定空間大於已測有限範圍。

完整設定 hash 保留所有欄位。`behavior_config`／`behavior_hash` 的明列 allowlist 僅移除 `mode`、`n_mi`、`n_cs`、`n_h`、`n_rp`、`delay_seed`、`delay_cycles`、`terminal_snapshot`。shape、frame 範圍、epoch、seed、failure／cancel 與 fixed_submit 都必須相同。比對的裝置、build 身分另行固定，hash 相同不取代封包全 bytes 比較，也不允許裁掉尾端封包。

## 資源、同步與控制

| 資源 | 所有權／生命週期 |
| --- | --- |
| STAGE[N_H] | pinned host Meta（epoch、frame、seed）；D 的 H2D 最後讀取後 record staged，H 重用前等對應 generation |
| DESC[1] | D 專用 device Meta，靠 D 順序 |
| MI[N_MI] | 獨立 float32 影像及 header；D record ready，M wait；cuFFT 完成與最後 header 核對後才 record consumed，D 下次覆寫前明確 wait |
| CS[N_CS] | header 與固定 2 個候選空間；D record ready，M wait；複製到 DB 及最後 header 核對後 record consumed，D 重用前明確 wait |
| DB、FFT_TMP、plan/work area | M 專用；不 alias TS／SPEC／packet；cuFFT 綁定 M，auto-allocation 關閉，初始化時暖機同一 plan |
| CTL、TS[2]、SPEC[2] | M 專用；committed_slot 初值 1、working_slot 0；所有 payload 初始化，初始 frame=-1；成功 commit 才共同切換版本 |
| RP[1]、HP[N_RP] | RP 是 M 專用 12368 bytes；每幀一次完整 async D2H 到 pinned HP，record pkt；H 等 pkt 後保存／decode 原始 bytes，再允許 HP 重用 |
| D／M terminal events | 停止 enqueue、消費全部已排入 packet 後各自 record，等待兩者完成才釋放資源；D-only 尾幀亦受保護 |

實作在消費全部 packet 後 record terminal events；等待兩條 stream 都完成後才作可選診斷或釋放。事件帳本只有在 record API 成功後更新 generation；每次 wait 檢查最新 generation 等於預期值，不等待未成功 record 的 event。H 固定依 D(k)→M(k)→D(k+1) enqueue。測試會獨立重播帳本，核對這個順序、各 ring 自己的 consumed wait、host leases、全部 packet 消費及雙 terminal drain。

D、M 均為 `cudaStreamNonBlocking`。所有 device／pinned buffer、events、plan 及 workspace 都在初始化配置，清零與 header invalid 初始化後作一次 device 同步。pipelined 幀路徑只有 async copy、明確 stream launch／wait／record；沒有 default stream、同步 memcpy/memset、GPU 臨時配置或額外 stream/device 同步。H 對 STAGE／HP 的必要 event 等待仍保留。serial 使用相同 kernels／cuFFT／依賴，在每個明確階段邊界 `cudaStreamSynchronize`，只作 correctness 對照，不作效能基線。

M 每幀的 begin／prepare／commit 都是前序單 thread 控制 kernel。計算 grids 只讀 `run_stage`，只以 `atomicOr` 累積錯誤，從不再用正在改寫的 frame_status 決定本 grid 的執行；下一個 prepare 才固定新的旗標。packet 只有一個 block，先平行清空全部 12368 bytes，再以 block barrier 分隔逐欄位 wire 寫入。未用 rows 與 reserved 每次都清零，count/status 留在 device，沒有先 D2H count 再變長複製。

錯誤注入分別為頻譜交付前 NUMERIC=8、交付 grid 內 OVERFLOW=4、predict grid 內 COMPUTE=16、update grid 內 PROTOCOL=1（下一控制邊界才觀察）。首次失敗保留 committed TS／SPEC、fail_frame／fail_code；後續 begin 仍更新 frame 身分，控制 kernel 額外累積不同的診斷 bits=10，檢查首次失敗不被覆寫。後續自訂計算跳過，但每次 cuFFT 都合法讀 MI、寫 FFT_TMP，直到最後讀取後才歸還 MI。

取消切點：after_h2d、after_mi、after_cs（D-only），after_begin、after_fft、after_predict、after_update（部分 M），after_packet（完整一幀）。不合成缺失的 consumed／packet record，不對缺失 record 等待；最後一個部分幀沒有下一輪重用，terminal events 覆蓋其剩餘工作。可回復的 host/API 例外也先排空已建立的 streams；fatal CUDA 錯誤不能標記為成功 drain 或完成。

可選診斷使用額外 pinned allocation，僅在雙 terminal drain 後讀 CTL、TS 兩份及 SPEC 兩份，再回傳 committed TS 的完整原始 bytes、committed SPEC 的所有 float bits、控制版本與 FFT 完成計數。這些額外 D2H 不屬於正常每幀資料面。TS 診斷 bytes 是本 build 的內部布局（含已初始化 padding），不是跨 build wire ABI；ResultPacket 始終使用逐欄位 v1 ABI。

## 獨立合成答案與有限驗收

令 k 為 frame、P=width×height、a=seed mod 13、b=seed mod 7。MI 全部像素為精確整數 float `a+k+1`，所以實際 R2C FFT 的 DC=`P*(a+k+1)`，其餘 words 全 +0。本次測試用解析答案檢查整份 committed spectrum，不使用另一個 GPU 執行器作唯一答案。

偶數 k 為成功空結果，奇數 k 有兩筆 observed rows（i=0,1）：UID=`2*k+i+1`，output_id=`100+UID`，class=i，flags=1，xyxy=`(x,x+1,x+2,x+3)`，x=`b+2*k+1+10*i`，scores 為 0.5、0.25。候選生成時 x 尚不含 predict_count；update 讀 working TS 的 predict_count 加到座標。成功提交到 k 時 predict_count=update_count=k+1，input_sum=`P*((k+1)*(a+1)+k*(k+1)/2)`。

例：預設 seed=7、8×8：frame0 的 packet 是 OK/count0，DC=512；frame1 的兩個 UID=3,4、output_id=103,104，座標 `(3,4,5,6)`／`(13,14,15,16)`，DC=576。成功提交 frame2 後，TS 計數均為3、input_sum=1728、SPEC DC=640。若 frame3 失敗並排入到 frame7，最後 committed 仍為2，FFT 完成8次、scratch_frame=7，且完整 committed TS／SPEC 與只成功執行0..2的獨立 baseline 一致。

測試的 CPU encoder 只產生上述獨立預期值；GPU 原始 D2H bytes 不經 host 重建就保存、全 bytes 比較及交給既有 decode_packet。所有失敗／取消案例保留自身 frame 身分和全部已完成 packet。

完整 suite 有72個 GPU sessions：

- 成功：4組 rings `(1,1,1,1)`、`(2,2,2,2)`、`(1,2,3,4)`、`(2,1,3,4)` × delay seeds 11、29、97 × serial兩次＋pipelined，每次16幀；最大 ring=4，至少初次使用加三輪重用。
- 失敗：獨立成功0..2 baseline；4種階段各 serial兩次＋pipelined 的固定0..7範圍，及各自首幀失敗；普通 stop-on-observed-FAILED 一例。固定失敗測試 N_H=N_RP=8，全部0..7先 enqueue 才讀 FAILED，核對4..7的cuFFT全完成，不裁掉SKIPPED。
- 取消：frame7 的8種切點各 serial／pipelined，共16例，含已重用 rings 的D-only尾端。
- 額外 shape：16×8、32×32，8幀，seed=uint32最大值。

這不證明所有合法 N／shape／seed／frame 數皆無競態，不涵蓋 GPU fatal／OOM／API 失敗注入、header 損毀負測試、效能加速比、跨 GPU／跨 build 數值一致性或完整 tracker 行為。

Sanitizer 的代表 subset 為11個 sessions（mixed rings成功serial兩次／pipeline、成功baseline、update後失敗serial兩次／pipeline、after_h2d與after_fft取消各兩模式），逐個執行：

```sh
for tool in memcheck initcheck synccheck racecheck; do
  /opt/cuda/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
    build-cuda/cuda_synthetic_tests --sanitizer "build-cuda/sanitizer-$tool" \
    > "build-cuda/$tool.log" 2>&1
  result=$?
  printf '%s exit=%s\n' "$tool" "$result"
done
```

racecheck 的 shared-memory 範圍不能證明跨 stream global-memory 正確；依 [NVIDIA 工具文件](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)，WDDM debugging interface 亦須啟用才能進行檢查。實際工具結果以證據報告為準，不因應用程式自行檢查通過而宣告 sanitizer 通過。
