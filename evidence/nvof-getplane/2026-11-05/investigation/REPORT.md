# NVOF cuArrayGetPlane investigation — 2026-11-05

結論：已把原始 HR／lifecycle 的全部診斷定位到 NVIDIA `libnvcuvid` 內部對非多平面 CUDA arrays 呼叫 `cuArrayGetPlane`。74 次發生於 `nvOFInit`，另 12 次僅發生於每個 session 的第一次 `nvOFExecute`。同型錯誤可完全脫離 TrackingSession、TensorRT、nvJPEG 和專案 CUDA kernels 重現。

這是實際 CUDA API 失敗回傳，CUPTI 在未啟用 memcheck 的程序也觀察到；不是已證實的 sanitizer 假報。外層 NVOF API 仍全部回傳成功。NVIDIA 是否有意以失敗呼叫探測格式、這種處理是否無害、何版本會修正，仍為 OPEN。

## 範圍與來源

- 實際 worktree：`/home/ray/developer/ai/saccade_v1-worktrees/g1-ofa`，HEAD `e56bfd914596b6e8917b687e42946b233bdd3867`。
- 原始兩份 log 保持不變：`../hr-memcheck-full.log`、`../lifecycle-memcheck-full.log`。
- 原有 `build/development/track_sequence`、`ofa_audit` 完全未重編。HR binary 自報 `source_commit=4382e5c9505b1537d87eb01dd5ab57b883e2f852`、`source_dirty=true`；不把目前 HEAD 當成該 binary 的完整來源證書。
- 僅新建此 ignored build 目錄內的診斷 source、binary、logs、JSON 和報告。沒有修改 runtime、NVIDIA headers、driver、驗收腳本或其既有分類規則。
- 進入 worktree 前後皆為原有三份 dirty scripts：`check_stream_placement.py`、`run_tracking_session_acceptance.sh`、`tracking_session_acceptance.py`。未 commit／push。
- GPU：RTX 5070 Ti Laptop；WSL2；NVIDIA-SMI 615.78.08、KMD 617.42、CUDA UMD 13.4；Compute Sanitizer 2026.3.0.0。完整命令、環境與 source／binary／library SHA-256 見 [manifest.json](manifest.json)、[manifest-v2.json](manifest-v2.json)。engine、實際 prefix JPEGs、seqinfo、libcuda／libcupti 與相關 runtime source hashes 另存 [additional-provenance.json](additional-provenance.json)。

## 原始紀錄的精確歸屬

| libnvcuvid 分歧位址 | API | HR | lifecycle |
|---|---|---:|---:|
| `0x16a76f` | Init | 12 | 144 |
| `0x16a89c` | Init | 20 | 240 |
| `0x16a85b` | Init | 20 | 240 |
| `0x16a11f` | Init | 18 | 216 |
| `0x16ab3e` | Init | 2 | 24 |
| `0x14fc08` | Init | 2 | 24 |
| `0x16ccff` | Execute | 4 | 40 |
| `0x16cd57` | Execute | 4 | 40 |
| `0x16ce2f` | Execute | 4 | 40 |
| 合計 | | **86** | **1008** |

兩份紀錄只有 `cuArrayGetPlane / CUDA_ERROR_INVALID_VALUE`；沒有其他 API、device memory 或未分類診斷。分類器保留所有未知診斷，不以不認識的訊息當成功，詳見 [classify_logs.py](classify_logs.py) 與 [classification.json](classification.json)。

既有 binary 的 `track_sequence+0x516f6` 位於 `call *0x50(%rbx)`，API table 的該 slot 為 `nvOFInit`，後接 `initialize session OFA`。lifecycle 的 `tracking_session_tests+0x54e96` 相同。`track_sequence+0x4d8a9` 對應 `nvOFExecute`。反組譯與 exact commands 另存 `*-disassembly.txt`、[disassembly-commands.json](disassembly-commands.json)。

因此建構路徑精確對應 `cuda/flow_motion.cu:151`，早於 `:152` 開始建立專案 OFA buffers。它不可能由尚未執行的 gray／cells kernels、buffer pitch 使用或 MI slot 重用造成。

lifecycle source 每個 graph 模式建立 6 個 sessions，兩種模式共 12 個；兩個 destructor-drain cases 只 enqueue frame 0，沒有 Execute，另 10 個有 Execute。原始錯誤數因此精確為 `74 × 12 + 12 × 10 = 1008`。HR 30 幀共 29 次 Execute；lifecycle 共 40 次 Execute，並非每次 Execute 都新增錯誤。

## 獨立重現與控制

先以原有 `ofa_audit` 執行完整 240 trials 加兩次基準 Execute：A1–A4 synthetic 檢查全部通過，仍報相同 **86 次／9 種 library stacks**；沒有隨 242 次 Execute 累積。命令與 raw log hashes 見 [ofa-audit-run.json](ofa-audit-run.json)。

接著以 [nvof_api_probe-v1.cpp](nvof_api_probe-v1.cpp) 直接呼叫 CUDA Driver 和 NVOF API。沒有 Saccade library、TrackingSession、TensorRT、nvJPEG 或自訂 CUDA kernel；使用合法 GRAYSCALE8／SHORT2、grid4、SLOW、相同非預設 nonblocking IO stream。輸入為固定 `(8,4)` 位移 texture，觀察 flow 中位數 `(8,4)`，重跑 raw flow bytes 相同。

| 測試 | Init 診斷 | 第一次 Execute 診斷 | 後續 Execute 診斷 | 合計 |
|---|---:|---:|---:|---:|
| 只 Init，不配置應用 buffer | 74 | — | — | 74 |
| 配置 buffers，不 Execute | 74 | — | — | 74 |
| 1 次 Execute | 74 | 12 | — | 86 |
| 2 次 Execute | 74 | 12 | 0 | 86 |
| 30 次 Execute | 74 | 12 | 0 | 86 |
| 2 個 sessions，各 2 次 Execute | 148 | 24 | 0 | 172 |
| default IO stream，2 次 Execute | 74 | 12 | 0 | 86 |

phase markers 精確包住每次 host API，所有 NVOF 回傳均為 `NV_OF_SUCCESS`，建立／buffer 操作／釋放本身沒有其他診斷。完整 Execute cases 的九種 library stack sequences 與兩份原始紀錄完全相同。

各次 memcheck 均使用 `--report-api-errors explicit --print-limit 0 --error-exitcode 99`，在 GPU lock 下執行。它們因 API 診斷全部 **exit 99**；正常輸出不改寫成 sanitizer PASS。

## 實際 array 格式與呼叫參數

先以 v1 CUPTI observer（不開 memcheck）看到相同 86 個實際 `CUresult=1`：Init 74、Execute#0 12、Execute#1 零。v1 callback 沒有 CUDA 呼叫，但無法從 array create callbacks 找到這些內部 arrays 的 descriptors，故該版一律記 `UNKNOWN`。

v2 在相同 observer 之外，於 NVOF 呼叫返回的普通 host 邊界查詢 `cuArray3DGetDescriptor`；**不在 CUPTI callback 內呼叫 CUDA**。完整 [probe-cupti-v2.log](probe-cupti-v2.log) 與 [array-trace-analysis.json](array-trace-analysis.json) 確認：

- 43 個不同的有效 arrays，每個正好被查 plane 0 和 plane 1，各回 `CUDA_ERROR_INVALID_VALUE`，共 86 次。
- Init 涉及 37 個 arrays：`37 × 2 = 74`；第一次 Execute 新增 6 個：`6 × 2 = 12`。
- 30 個 arrays 是 `CU_AD_FORMAT_UNSIGNED_INT8`，1 channel；13 個是 `CU_AD_FORMAT_UNSIGNED_INT32`，4 channels；全部 `Depth=0`、`Flags=0x2`。
- 這些皆為非多平面格式；全部 43 個 handles 都取得成功 descriptor，共 123 次成功查詢，跨觀測邊界的 descriptors 一致，沒有未知或查詢失敗。
- 包含 Init 的私有 pyramid 尺寸、首次 Execute 的 1920×1088 等對齊尺寸；這些不是應用所建立的 CUDEVICEPTR buffers。
- v2 和 v1 一樣觀察 86 次 GetPlane 失敗、外層 NVOF 全部成功、flow `(8,4)` 且重跑 raw bytes 相同。v2 exit 0，是普通觀察程式成功，不是 memcheck PASS。

CUDA 官方契約明定：對非多平面格式 array 呼叫 `cuArrayGetPlane` 會回 `CUDA_ERROR_INVALID_VALUE`。因此直接失敗原因已具體定位為 **libnvcuvid 對非多平面 arrays 進行 plane extraction**。其內部設計意圖與可接受性仍需 NVIDIA 說明。[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html#group__CUDA__MEM_1g3247d31ed2e8e659d24331e8d5a7ff016)

## HR 輸出對照

以同一個原有 `track_sequence` 和 engine，對原始 30-frame prefix 各執行一次普通模式與 memcheck，另寫新 output。普通模式 exit 0；memcheck exit 99，仍是相同 86 次／9 種 stacks。三種完整產物逐位元相同：

| 產物 | bytes | 兩次共同 SHA-256 |
|---|---:|---|
| MOT `.txt` | 23,460 | `d3972f8d5320edeac63b5b95de2a1a1a8db232fb7336b524835d9d67e729dfc9` |
| 完整 packets | 371,040 | `04e44fe63f7fbf9bc1bae512d361d6f8e5a1dd01bdf9fe162c05151d5c239cce` |
| MotionContext `.scmo` | 1,364 | `cc911417c3ed02d6556ec321d46db12b9b38c18d03991042b12efe0141ef32c6` |

這只證明此 30-frame 工作負載的一般執行／插樁執行結果相同，不延伸為全資料集品質、所有排程、race safety 或完整 sanitizer 無錯保證。本次未重跑 lifecycle；其 1008 次原始診斷全數完成分類。

## 使用方式與證據保管

已建好的直接重現程式：

```sh
flock ~/.cache/saccade-v1/gpu.lock timeout -k 10 300 \
  /opt/cuda/bin/compute-sanitizer --tool memcheck \
  --report-api-errors explicit --print-limit 0 --error-exitcode 99 \
  build/nvof-memcheck/investigation-20261105/nvof_api_probe --init-only
```

預期 Init-only 74 次並 exit 99。改用 `--executes 2` 時預期 86 次。若要看 actual descriptors，可直接執行 `nvof_api_probe-v2 --cupti --executes 2`，同樣取得 GPU lock。

`manifest.json` 保存初次編譯及 matrix exact commands。初版編譯時的 `nvof_api_probe.cpp`、`array_trace.hpp`、`run_probe.py` 已分別封存為 `nvof_api_probe-v1.cpp`、`array_trace-v1.hpp`、`run_probe-v1.py`，封存 bytes 與初版 manifest 原 hash 完全相同。未更動初版 binary。現行無版本尾碼 source 是 v2，編譯／執行 commands 與 hashes 在 `manifest-v2.json`。如需重建，使用對應 manifest 的 compiler argv 和對應 source 版本。

`--report-api-errors explicit` 不會排除第三方 shared library 的明確 CUDA API 呼叫；關閉 API reporting 只改變觀察範圍，不修復底層呼叫。[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#cuda-api-error-checking)

NVIDIA forum 另有獨立 `cuvidCreateDecoder` 在 `libnvcuvid` 內出現相同 GetPlane 錯誤的使用者報告，但沒有 NVIDIA 確認根因、無害性或修正。[公開原始報告](https://forums.developer.nvidia.com/t/errors-reported-by-compute-sanitizer-for-cuvidcreatedecoder/317461)

後續可向 NVIDIA 提供此獨立重現程式、兩版 callback 證據、原始 library stack offsets 與 library hashes，請其確認內部 plane probing／fallback 契約。本次未對外送出資料。現有證據不支持為了歸零而修改 session 同步、換 buffer 格式、清零或放寬驗收判定；原始 API 診斷仍保留 FAIL，driver 意圖及修復狀態為 OPEN。
