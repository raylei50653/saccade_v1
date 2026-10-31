# F03 producer status 聚合器：有界開發驗證

日期：2026-10-31。**`f03-producer/1` 已實作，受測範圍開發驗證 PASS**。本次使用尚未提交的工作樹，base HEAD 為 `491535612ab8e6aa5796263f79b8bd775b1df7dd`；`source_dirty=true`、`allow_dirty=true`，不稱 clean-source 正式結案，也不將 base HEAD 當成包含新實作的 source commit。

## 實作與判定

[`model_producer_cuda`](../../../include/saccade/model_producer_cuda.hpp) 在 detect stream 以 initialize → finite → aggregate 三個 kernels，完成完整 `images [1,3,640,640]`／六個 raw outputs 的 finite checks、九個 plugin status 的角色檢查及全部十一個 source statuses 的退休，再寫 F04 既有八位元組 aggregate 的 flags／done。scratch 為 16 bytes，不改 engine、F04 decode／排序／NMS 或候選 ABI；語義與 caller 所有權義務見 [MODEL_PRODUCER](../../../docs/MODEL_PRODUCER.md)。

任何 source flags 非零累積 `kPostprocessNumeric`；缺少 done、未知或錯誤角色的 bits 累積 `kPostprocessProtocol`，全部來源仍被讀取並退休。aggregate 失敗亦發布完成 marker；F04 將任何 aggregate flags 轉為 upstream failure，完整清零候選／統計，保留本幀身份。這不是 session sticky-failure 或 identity 判定。

## 固定身分與保存

| 項目 | 本次觀測 |
| --- | --- |
| Build-input manifest digest | `9252b012ffcefb4518e96ab8fb60af36c9baf7712893a85fb4b42c11c50aeb19` |
| F03 engine SHA-256 | `c6cff1936194e679ce11c3736cab552e4a73a2b1a9ae5508bc3e5d1b3fbab9c5` |
| 六個輸入的 concatenated FP32 SHA-256 | `14dda6867e23ed8a2d83086224d71a445a7fdcb606c7507a3f39c0e5faf60aa0` |
| GPU／driver | RTX 5070 Ti Laptop、sm_120、617.42、WSL2 |
| Compiler／CUDA／TensorRT | GCC 16.2.1、nvcc 13.4.92、runtime／driver API 13.4、TensorRT 10.16.1.11 |
| Compute Sanitizer | 2026.3.0.0 |
| 新 adapter flags | `--ftz=false --fmad=false -lineinfo` |

F03 engine／輸入沿用 [歷史乾淨 F03 capture](../../t01/2026-10-30/f03/REPORT.md) 的 artifacts，本次執行的是新 build 的 plugins／adapter／F04；沒有重建 engine，也不重新宣稱 F03 數值政策驗收或模型品質。

GPU 前記錄 [source manifest](capture/source-files.sha256)、[source 狀態](capture/source-status.txt)、[tracked diff](capture/source.diff) 與 [build-input archive](capture/source-build-inputs.tar.gz)；archive 包含本次新增的 untracked build inputs，不只 tracked diff。每個 capture 步驟前後核對 build inputs 不變。全部 source 身分以 [source.txt](capture/source.txt) 為準；不宣稱後續報告／導覽文件亦屬該凍結的 build-input 集合。

[Binary hashes](capture/binaries.log) 在測試前固定，[測試後 hashes](capture/binaries-after.log) 逐 byte 相同。受測 actual executables／archives 與 CPU／CUDA build 留在本機 `build-f03-producer-evidence/`；另外保存本機 actual binary bundle，身分見 [local-artifacts](capture/local-artifacts.sha256)。不宣稱 CUDA rebuild 逐位元可重現。[CMakeCache](capture/CMakeCache.txt)、[compile commands](capture/compile_commands.json)、[environment](capture/environment.log)、各命令／退出碼／time 與 [capture status](capture/capture-status.txt) 一併保存。capture 於 MemoryMax=12 GiB、MemorySwapMax=0 的 scope 內執行。

## 結果

| 檢查 | 結果與範圍 |
| --- | --- |
| 預設 CPU suite | **12/12 PASS**，[CPU CTest](capture/cpu-ctest.log) |
| CUDA／F04／live CTest | **6/6 PASS**，[CTest](capture/ctest.log)／[完整輸出](capture/LastTest.log)；含既有 F04 reference／CUDA／lazy 與新增 adapter／lazy／live |
| 獨立 adapter 正常 suite | **1,282 checks PASS**；九個來源逐一 known failure、缺 marker、錯 marker、未知 bit、錯角色 bit；混合失敗、來源退休與 failure→success 重用 |
| Finite checks | input 與全部六個 raw tensors 的首／末元素，NaN／±Inf 全部拒絕；有限 FP32 極值不誤判；包含非 person／低分位置 |
| Host arguments | 錯 counts、空／host／未對齊指標、實際 allocation 不足、重疊、default streams 拒絕；aggregate／來源 sentinel 保持不變；本機單 GPU，跨 device 實例未執行 |
| API trace | CUPTI **90 calls、3 launches**；無 allocation／free／transfer／memset／event／synchronization，stream-bearing calls 全在指定 stream |
| 正常所有權 gates | direct held stream、pending upstream ready、pending consumer consumed 均 PASS；fresh lazy process **30 checks PASS** |
| Graph | fresh status publication→adapter 的 success／failure／recovery replay，以及完整 engine→adapter→F04 的三次 replay 均 PASS |
| Live F03→adapter→F04 | **6 inputs、3,810 checks PASS**；全部 raw SHA 與歷史 bytes 相同，完整 candidates 對 CPU reference、重跑 bytes、歷史 kept counts、NaN failure、success after failure、來源退休及 checked close 通過 |
| Adapter Memcheck／Initcheck | 各 **1,167 checks、0 errors**，[memcheck](capture/sanitizer-memcheck.log)／[initcheck](capture/sanitizer-initcheck.log) |
| Adapter Racecheck／Synccheck | 各 **1,167 checks**；**0 hazards（0 errors、0 warnings）／0 errors**，[racecheck](capture/sanitizer-racecheck.log)／[synccheck](capture/sanitizer-synccheck.log) |
| Sanitizer 正控制 | **4/4 DETECTED**，各自預期診斷與 exit 99；沿用獨立 test-only F04 sanitizer controls，非 production fault injection |
| Live Memcheck／Initcheck | 各 **1 input、1,630 checks、0 errors**；另含 NaN failure／recovery／graph，[live memcheck](capture/live-memcheck.log)／[live initcheck](capture/live-initcheck.log) |
| 無 TensorRT／nvJPEG 的獨立建置 | `SACCADE_ENABLE_CUDA=ON`＋`SACCADE_ENABLE_MODEL_PRODUCER=ON` 的 library build PASS，[configure](capture/standalone-configure.log)／[build](capture/standalone-build.log) |

Standalone sanitizer 使用 `--no-held-stream --no-api-trace`：**sanitizer 下 host callback gates 與 CUPTI trace 為 UNEXECUTED**；正常與 lazy 測試另有實跑。四種 sanitizer 只覆蓋獨立 adapter／合成 raw→F04 路徑，**不涵蓋 TensorRT engine racecheck／synccheck**。Live memcheck／initcheck 限第一個固定真實輸入與其失敗／重跑／graph 序列，不能擴張為六個輸入皆經 sanitizer。

Live 原始 kept counts 為 **32／28／39／2／3／1**，完整 rows 為 **237／246／285／7／5／3**；kept／suppressed pool、header、candidate identity、順序、座標及未用 rows 逐 byte 對照，score 沿用 F04 既有 `4*FLT_EPSILON` 規則。同 binary／input 的 repeat／graph 比較完整 output bytes。此處的失敗後成功驗證是無 session 狀態的 primitive 重用，不能解讀為允許完整 session 在 sticky failure 後繼續提交結果。

## 邊界與重跑

本次完成 F03 producer 聚合器與有界 engine→F04 接續開發驗證。**完整 async session、native F01／F02→F03→F04、CS ready／consumed ring、epoch／frame identity、first-error latch、F04b／GMC／tracker／ResultPacket／MOT writer 與 P01 仍 OPEN**。Head／F03 racecheck／synccheck 仍 UNEXECUTED；未做效能驗收、跨 GPU／SDK 驗收、語義 mutations 或 clean-source 正式驗收。T01／F04 既有正式證據不改寫。

在含此變更的 source 上，準備 pinned SDK／engine／六輸入 corpus，指定不存在的 root `build-*`：

```sh
SACCADE_TENSORRT_ROOT=/absolute/path/to/pinned-sdk \
SACCADE_MODEL_PRODUCER_ENGINE=/absolute/path/to/model.engine \
SACCADE_MODEL_PRODUCER_INPUTS=/absolute/path/to/f03-inputs.f32 \
  bash scripts/run_model_producer_acceptance.sh build-f03-producer-new
```

尚未提交的工作樹需明確附 `--allow-dirty`，只產生開發證據。工具與輸入身分、判定條件及未執行項目不因該選項放寬。隨本報告保存檔案的 SHA-256 見 [SHA256SUMS](SHA256SUMS)。
