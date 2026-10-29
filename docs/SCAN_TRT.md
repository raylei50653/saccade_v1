# S03 TensorRT FP32 scan plugin 與 frame-end harness

S03 將 [S02 CUDA FP32 forward](SCAN_CUDA.md) 接到 TensorRT 10.16.1.11 的 `IPluginV3One`，並用最小 engine 驗證 status 到 frame 結尾的交接。scan／Cross-Scan 直接連結 S02 library；FP32 算法、支援限制與逐元素誤差界維持 [SCAN_CUDA §3–§5](SCAN_CUDA.md#3-fp32-算法s02-基準)。這不是完整 YOLO26m head、真實權重、FP16、效能或 P01 模型驗收；這些留在 S04／P01 OPEN。

本文件描述實作契約。S03 是否通過，以對應 source commit 的 [S03 證據](../evidence/s03/2026-10-29/REPORT.md) 為準；沒有完整 capture／sanitizer／mutation 證據時不得宣告 PASS。

## 1. Plugin 身分與 I/O

三個 creator 都使用 namespace `saccade`，必須在 build／deserialize 前呼叫 `register_scan_trt_plugins(*getPluginRegistry())`。沒有隱式靜態註冊。

| plugin | version | serialized fields | FP32 execution inputs | output 0 |
| --- | --- | --- | --- | --- |
| `SelectiveScan` | **2** | `has_d`：INT32，0 或 1 | `u, delta (B,L,Dc)`、`A (rows,N)`、`B (B,L,N)`、`C (B,L,cw)`、有 D 時 `D (Dc)` | `y (B,L,Dc)` |
| `CrossScanExpand` | 1 | 無 | `x (B,C,H,W)` | `tokens (4B,HW,C)` |
| `CrossScanMerge` | 1 | `height, width`：正 INT32 | `y (4B,HW,C)` | `out (B,C,H,W)` |

每個 plugin 的 **output 1** 都是固定長度 **INT32[2]**，內容為 S02 `ScanStatus` 的 `{flags, done}` 兩個 32-bit words。它是一般 execution output（`isShapeInferenceIO == false`），不是 shape tensor。全部 I/O 都是 dense／linear、device memory；payload 與 inputs 只接受 FP32。workspace request 為 **0 bytes**，沒有 workspace 對齊需求。TensorRT execution-context memory 由 engine 另行計算，不能把它當作 plugin workspace。

SelectiveScan 舊 v1 只有一個 output；新增 status 已改變 ABI，所以 v2 使用不同 identity，不能讓舊 v1 engine 或舊欄位誤配。creator 僅接受上述完整欄位集合：未知、重複、缺少、非 INT32、length 非 1、null data、非法值都拒絕。序列化只存語義欄位；不存 stream、指標、recurrence state 或 CUDA context identity。

`N` 必須是 1、2、4、8、16 或 32；`rows` 為 1 或 Dc，`cw` 為 1 或 N。L 可為 0。其餘非空／容量／alignment／重疊限制繼承 S02 host 契約。沒有 FP16、非連續 stride、in-place、initial state 或 final state 支援。

## 2. 初始化、context 與執行生命週期

1. 選定 CUDA device／context，註冊 creators，build 或 deserialize engine。
2. `createExecutionContext(kUSER_MANAGED)` 時，TensorRT 對每個 plugin 呼叫 `attachToContext`，建立獨立 clone，載入 S02 全部 kernels，記住 current CUDA context 的 `cuCtxGetId`。build 階段 `configurePlugin` 也會載入 kernels，允許 TensorRT 在 builder context 做 timing。
3. 為每個 execution context 配置獨立 context memory、I/O、status、pinned staging 與 events。所有 slot warmup 一次，再將 warmup status retire 為 `{0,0}`。若使用 graph，每個 slot 都捕捉自己的 context／固定 I/O addresses，instantiate 並 upload。初始化末尾 device synchronize。
4. frame 階段 `enqueueV3(explicit_stream)` 只做 host descriptor／context 檢查與 S02 async calls。它不載入 kernel、不配置記憶體、不同步、不 D2H。stream 必須屬於 current CUDA context，且 context ID 與初始化所綁定者相同；拒絕 default／legacy／per-thread implicit stream。
5. 結束時 drain D、M streams，才讀診斷 snapshot 或釋放 graph、execution contexts、context memory、I/O、pinned memory 與 events。正常路徑呼叫 `close()` 檢查 release 錯誤；例外路徑保留 `last_run()` 的已收到 outcomes，析構時嘗試 drain／清理。全部 contexts／engines 必須在其 runtime、plugin creators、logger 與 recorder 之前釋放；應明確安排 cache cleanup，不依賴跨物件的 static 析構次序。程序退出成功也屬驗收條件。

kernel readiness 屬於 CUDA context，不能以單一 process-global 布林值替代。CUDA lazy loading 可能讓第一次 launch 等待已在 device 上的工作，因此需要在 frame 工作之前載入。跨程序 engine run 在新 context、任何 warmup 之前，用被 host callback 擋住的 stream 測第一次 `enqueueV3`；測試同時在 `CUDA_MODULE_LOADING=LAZY` 下執行。

`onShapeChange` 只做 host descriptor 檢查，因為 TensorRT 可能在 `enqueueV3` 內呼叫它。warmup 與 graph capture 必須在正式 frame 前完成；context、addresses 或 dynamic shapes 改變後，須重新完成相應初始化／capture。這個 harness 僅接受 static engine shapes。

## 3. 最小 engines 與數值判定

engine builder 關閉 TF32、optimization level 設 0、auxiliary streams 設 0、builder workspace 上限 16 MiB，不使用外部 timing cache。strongly typed network 只含上述 scan plugins。

| engine | profile／用途 |
| --- | --- |
| SelectiveScan，has D／no D 各一個 | dynamic B≤8、0≤L≤2048、Dc≤512、N≤32；A rows／C width 仍須滿足 plugin 規則 |
| Expand | dynamic B≤8、C≤512、H,W≤64 |
| Merge | 固定 H,W，dynamic B≤8、C≤512 |
| Expand→SelectiveScan→Merge | static shape，三個 status 都成為 engine outputs；部署 m 的 P5／P4／P3：C=256、N=16、H=W=5／10／20 |

測試重用 `tests/scan_cuda_suite.hpp` 的 S02 fixtures、數值／失敗／Cross-Scan groups。每次 plugin call 另與 S02 API 的相同輸入、相同 CUDA backend 結果逐位元比較；binary64 reference 與誤差界仍使用 SCAN_CUDA 的政策，不以平均誤差或新 tolerance 替代。engine save／readback 檢查 SHA256；另一個程序註冊並反序列化。未註冊 creator 的程序必須在可執行 context 形成前失敗：本機 TensorRT 10.16.1.11 將 plugin 反序列化延後到 `createExecutionContext`，所以 `deserializeCudaEngine` 可先回傳 engine，但 context 建立被拒絕，不能執行。工具分別記錄這兩個邊界。

## 4. Frame 結尾、status ownership 與封包

每個 slot 擁有自己的 execution context、I/O、status、staging、`staged`／`ready`／`consumed` events。每個 frame 使用以下依賴：

- D stream：若重用 slot，先等該 slot 上一代 `consumed`；H2D inputs → `enqueueV3` 或 slot graph → `ready[slot]`。
- M stream：begin → wait `ready[slot]` → status gate → guarded publish → record `consumed[slot]` → commit → ResultPacket → D2H → `pkt[ring]`。
- host：重用 pinned staging 前等待相應 `staged`，重用 packet ring slot 前等待該代 `pkt`；只按 frame 順序讀取已完成 packet。

status gate 一定執行，讀取 engine 的所有 status：`flags != 0` 設 `NUMERIC`；`done != kScanDone` 設 `PROTOCOL`；未知 flag bits 另設 `PROTOCOL`。未知非零 flag 因此得到 `NUMERIC|PROTOCOL`。讀完後將每個 status 清成 `{0,0}`，每個 completion 只能被消費一次。既有 sticky fail 或當 frame 有錯誤時，`run_stage` 關閉，publish 不得搬移 payload。

publish 只在 gate 放行時寫 working result version；commit 只在無錯誤且確實 publish 後切換 committed version。首次失敗固定 `fail_frame`／`fail_code`，不能推進 committed frame；已排入的後續 frame 會收到 SKIPPED。預設 host 遇到 FAILED packet 即停止新提交並 drain；`fixed_submit` 僅供測試故意觀察失敗後的 tail。

**ResultPacket v1 ABI 不變**，count=0、rows／reserved bytes 為零。host 只有收到 epoch 與 frame 都符合這次提交的 **OK packet** 才接受該 frame 的結果；enqueue 成功、完成 event、有限 payload 或先前的 OK packet 都不能替代。狀態傳遞不在 host 先讀 plugin status；D2H frame 流量只有 ResultPacket。gate records、publish digest、committed payload 與 slot/status snapshots 屬診斷，只能在兩個 stream terminal drains 後讀。

## 5. 三種失敗通道

| 通道 | 結果 |
| --- | --- |
| host 可立即判定的 descriptor／shape／stream／context／pointer 錯誤 | plugin 回傳非 0，`enqueueV3` 失敗；訊息送 `IPluginResourceContext` 提供的 `IErrorRecorder`，沒有 recorder 時送 TensorRT logger。runtime／execution context 的 recorder identity 不保證 plugin resource context 必定提供同一 recorder；測試另列實際 diagnostic channel。拒絕前不提交 S02 工作，不能為該 frame 發出 ready／OK packet。 |
| kernel 數值失敗 | status flags 與 `done`，S02 finalize 將 payload 覆成 quiet NaN；M gate 轉成 frame error，guarded publish 拒絕。 |
| CUDA 非同步 fault | 後續 CUDA API／event wait 回報錯誤，harness 拋例外並停止接受；`last_run().published` 只保留 fault 前已收到且有效的 OK packets。fatal fault 用獨立程序測，該 context 不再重用。 |

## 6. 重跑與證據

```sh
cmake -S . -B build-s03 -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_TENSORRT=ON \
  -DSACCADE_TENSORRT_ROOT=/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-s03 --parallel 4
ctest --test-dir build-s03 --output-on-failure
build-s03/scan_trt_tests build-s03/report.tsv
build-s03/scan_trt_tests --group api_trace
build-s03/scan_trt_tests --group async_fault
build-s03/scan_trt_engine build build-s03/pipeline.engine
CUDA_MODULE_LOADING=LAZY build-s03/scan_trt_engine run build-s03/pipeline.engine build-s03/engine-report.txt
build-s03/scan_trt_engine run-unregistered build-s03/pipeline.engine
```

預設 GPU suite 包含數值、contract、reset、context isolation、frame-end、slot ownership、graph、held stream。`api_trace` 與 `async_fault` 只在明確指定時執行。CUPTI trace 在 warmup 後 arm，檢查三次 pipeline `enqueueV3`：每次 9 個 kernels、3 個 async memsets；launch／memset 必須在指定 stream；禁止配置／釋放、同步、module／library loading、同步版 memcpy／memset 與 D2H。graph launch 另外 trace。graph nodes 只允許 kernel／memset，檢查 expand→scan→merge 與各 finalize 的遞移依賴，再重播比較每個 frame digest。

compute-sanitizer memcheck／racecheck／synccheck 執行預設 suite；initcheck 加 `--no-held-stream`，只略過會因工具插樁而等待 callback-held stream 的項目。CUPTI 與 fatal fault 測試不在 sanitizer 下執行，原生獨立程序的結果另列。四個 sanitizer 都以 S02 的 [positive control](../evidence/s02/2026-10-29/sanitizer-control.cu) 植入缺陷，必須回報。racecheck／synccheck 的適用範圍與 shuffle-mask 限制同 SCAN_CUDA §6。

[Mutation script](../evidence/s03/2026-10-29/mutation-check.sh) 在私有 source copy 改動一個條件並 incremental rebuild；未修改的測試必須拒絕 mutant。涵蓋 flags／done／未知 bits、status retire、consumed ownership、失敗 commit、guarded publish、context binding、serialization、kernel preload、錯誤 stream 與 enqueue 同步。compile error 只算無效 mutant，不算辨識成功。

完整 capture 使用 [capture.sh](../evidence/s03/2026-10-29/capture.sh)，要求 clean HEAD 等於受測 source commit，再以 `git archive` 重建 CPU 與 CUDA＋TRT targets；保留命令、exit code、raw logs、reports、source archive hash、binary hashes、sanitizers 與 mutation logs。測試 source 與後續 evidence／docs 分兩個 commits；GPU 結果與僅 CPU 的 remote CI 分開回報。
