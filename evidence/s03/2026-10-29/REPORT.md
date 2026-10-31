# S03：TensorRT FP32 scan plugins 與 frame-end harness — 2026-10-29

**S03 PASS，範圍限 [SCAN_TRT](../../../docs/SCAN_TRT.md)**：SelectiveScan v2、CrossScanExpand／Merge v1、S02 FP32 library 的 TensorRT 包裝、最小 engine 序列化與 frame-end status／ResultPacket 交接。正式結果來自下列 clean-source archive 的完整重跑，包含 process teardown、四種 sanitizer 與 12 個 mutation。FP32 數值政策維持 [SCAN_CUDA §3–§5](../../../docs/SCAN_CUDA.md#3-fp32-算法s02-基準)。FP16、完整 head、真實模型權重與效能留在 S04；P01 仍 OPEN。

## Source 與執行身分

| 欄位 | 本輪記錄 |
| --- | --- |
| 起點 | `s03-trt-plugin`，由 **`e9dfa51`** 建立。交接時已有 `7aed310`：只把 S02 cases／error-bound／共用 groups 移至 `tests/scan_cuda_suite.hpp`。搬移前後的開發 report byte-identical，364 checks，`cmp` exit 0（[development comparison](capture/development-shared-suite/)）；兩份都自述 e9dfa51／dirty=true，只作搬移比較，不當作正式 source 驗收。 |
| 受測 source | **`e468044f0ab003fe5f76d6d2ef7260b618121516`**。clean HEAD 與受測 SHA 一致（[source](capture/source.log)），[working-tree](capture/working-tree.log) 為空；archive SHA256 **`ee29c139876a38ce4fc3ca7638bd6f4be543661444d019157e3433a4e7e5f0bf`**（[archive-hash](capture/archive-hash.log)），[source-file-hashes](capture/source-file-hashes.log) 另列 source bytes。第一次 attempt `190a3b91b39bdf835b2b0cf6999a463aaeca0522` 因 teardown SEGFAULT，CTest 21/22，**不通過**；只保留為 failed-attempt provenance，不與正式結果混用。 |
| 交付身分 | 本報告、raw captures 與文件更新為後續 evidence／docs commit；最終 SHA、remote equality 與 CI 由交付時記錄，不把開發建置當成 clean-source 證據。 |
| 環境 | WSL2 Linux **6.18.33.1**、GCC **16.2.1**、CMake **4.4.4**、glibc **2.44**、TensorRT **10.16.1.11**、nvcc **13.4.92**、driver **617.42**、compute-sanitizer **2026.3.0.0**；NVIDIA GeForce RTX 5070 Ti Laptop GPU，SM **12.0**，UUID `GPU-825ae551-ba16-539a-e444-22726b59ee9a`（[environment](capture/environment.log)）。CUPTI 用 native Toolkit API callback，不依賴 nsys。實際 build 命令與 SDK probe 見 [trt-configure](capture/trt-configure.log)／[trt-build](capture/trt-build.log)。 |
| S02 推送狀態更正 | 交接時 local main、origin/main 與遠端 main 已是 **`e9dfa51`**，S02 已推送。S02 原報告中的「本輪未推送」是當時記錄，保留原檔，不改寫歷史。 |
| 指令與工具事實 | repository 沒有 `AGENTS.md`；本 session 使用對話中提供的指令。先前交接 session 曾以 `python3` 編輯兩個新 network 檔案一次，之後停用；本次續作使用 C++／CUDA、CMake 與 shell／正常檔案編輯，未執行 Python 或舊模型。 |

## 契約與開發發現

1. **SelectiveScan 使用 v2 identity。** 舊 v1 僅單一 output；新增 INT32[2] status 已改變 ABI，不以 v1 冒充相容。所有 plugins 的 status 是普通 execution output，workspace=0；Serialize 僅存 has_d／height／width，creator 嚴格拒絕錯誤欄位。
2. **kernel readiness 綁定 CUDA context。** `attachToContext` 載入 S02 kernels 並記錄 `cuCtxGetId`；`enqueueV3` 的 stream 必須屬於 current context 且 identity 一致。初始化、warmup 與 capture 不在 frame 流量中；frame enqueue 禁止 allocation、synchronization、D2H 與 lazy module loading。新程序在任何 warmup 前執行 LAZY held-stream 第一個 enqueue，避免先跑 oracle 掩蓋未載入問題。
3. **未註冊 creator 的拒絕點與 brief 不同。** 本機 TensorRT 10.16.1.11 會延後 plugin materialization：`deserializeCudaEngine` 先回傳 engine，但 `createExecutionContext` 因缺少 creator 而失敗，不能形成可執行 context。工具如實記錄 `unregistered_deserialization=deferred`、`unregistered_context=rejected`，不宣稱前一個 API 已回傳 null。
4. **status 必須逐次消費。** M gate 讀 flags／done 後 retire 為 `{0,0}`，unknown flags 同時觸發 NUMERIC 與 PROTOCOL。失敗 payload 即使仍是有限值也不得 publish；首次失敗鎖定，後續 fixed-submit frames 僅 SKIPPED。ResultPacket v1 ABI 不變，只有該 epoch／frame 的 OK packet 被 host 接受。
5. **slot ownership 在最後一次 M 讀取後交還。** `consumed` 位於 guarded publish 後，D 重用同一 execution context／I/O 前等待上一代 consumed；staging 與 packet ring 另有 host 等待。延遲 M 的 ownership 測試對每一 frame 比對 S02 oracle digest。
6. **診斷與失敗路徑補強。** 正常／部分初始化失敗路徑都 drain 後清理；context memory、graph 與 buffers 依所有權釋放。分配、shape volume 與 aggregate counts 使用 checked multiplication，拒絕超大 max_frames 造成 size_t overflow；有 prework rejection regression。error recorder 的越界 getter 與 expected diagnostics 均有明確契約，預期錯誤只在定向失敗測試中消費，成功路徑另查無非預期錯誤。測試分別查 execution context 的 recorder identity，並記錄每次 plugin 診斷實際來自 recorder 或 logger；不以 runtime／execution-context recorder identity 推導 `IPluginResourceContext::getErrorRecorder` 必定傳入同一 recorder。本機有診斷經由 logger fallback 的情形，正式 channels 見 capture。
7. **原始數值政策未放寬。** 所有 plugin 數值測試沿用 S02 cases／bound，再逐次與 S02 CUDA API 比較 output/status bytes；未以新的全域 atol／rtol 取代，未變更 S02 kernels 或 S01 reference。
8. **完整 process exit 也是驗收的一部分。** 第一次 archive 重跑雖列出 841 checks passed，scan_trt 程序隨後 SEGFAULT；因此不能以最後一行數值檢查宣告成功。靜態檢查發現 global engine cache 會晚於 function-static plugin creators 析構，存在 lifetime UB；gdb 重跑未重現 crash、沒有取得 fault stack，故不把該順序問題說成已由 stack 證實的唯一原因。修正採成功輸出前明確清空 engine／runtime，且例外 unwind 有 RAII cleanup；正式 source 重跑完整 CTest 22/22 與四種 sanitizer 都正常退出。失敗 attempt 的 source／archive hashes、CTest exit 與修正 diff 保留於 [failed-attempt](capture/failed-attempt/)，不當作 PASS 證據。

以上是開發決策與本機觀察；下表全部結果來自正式 clean-source archive 重跑，失敗 attempt 與先前 dirty development reports 分開保留。

## 驗證結果

| 驗證 | 正式結果與證據 | 適用範圍 |
| --- | --- | --- |
| 預設 CPU build＋CTest | **9/9 PASS**（[cpu-ctest](capture/cpu-ctest.log)）。 | 預設 CUDA／TRT OFF |
| CUDA＋TRT build＋CTest | **22/22 PASS**（[trt-ctest](capture/trt-ctest.log)），包含完整 scan_trt 程序正常退出；[trt-build](capture/trt-build.log) 保留 compiler／linker 命令。 | pinned SDK、本機 SM 12.0；CPU 與既有 GPU regressions |
| 共用 S02 數值／失敗 groups | **842 checks PASS**、**91** 次 plugin／S02 API 的 bit-exact output／status 比較；報告共 **71** 個逐元素比較案例、**1,325,402** 個元素、超界元素 **0**，最大誤差／門檻 **0.295**（`shared_a_rank1_c_d_N1`）。build identity 為受測 SHA，`build_dirty=false`（[report](capture/scan-trt-report.tsv)）。 | 相同 FP32 inputs；含新增 frame／contract 數值比較，S02 support limits 不變 |
| plugin contract／creator | **PASS**（[report](capture/scan-trt-report.tsv)）：L=0、合法 shape recovery、錯誤 state／broadcast／互相不符的 dimensions、FP16、input count、profile 邊界、implicit stream、foreign CUDA context、unattached plugin、fields／duplicate／old-v1 identity、workspace=0，及超大 frame capacity 拒絕。engine statuses 全為 device linear INT32[2]、shape-inference I/O=false（[engine report](capture/engine-run-report.txt)）。定向 plugin 診斷實際均走 logger；execution-context recorder identity 另有斷言。 | host 可判定的錯誤在提交前拒絕，sentinels 不變；不宣稱 plugin-resource recorder 已繼承 |
| reset／isolation | **PASS**（[report](capture/scan-trt-report.tsv)）：A→B(fail)→A→B(valid)→A 三次 A bits 相同；同一 engine 的兩 contexts、兩 streams 交錯，結果各自符合 reference。 | 每次呼叫重設；context 狀態隔離 |
| frame-end | **PASS**（[report](capture/scan-trt-report.tsv)）：有效四 frames 全接受；A→NaN-delta B→A 得 OK／FAILED(NUMERIC)／SKIPPED，committed 仍為 A；packet_slots=1 遇 FAILED 停止新提交。有限 payload 搭配非零 flags、缺 done、unknown bit、expand flag 均被拒；status snapshots 已 retire。 | flags／done→NUMERIC／PROTOCOL、sticky fail、guarded publish／commit、精確 frame 接受 |
| m P5／P4／P3 | **PASS**（[report](capture/scan-trt-report.tsv)）：C=256、N=16、L=25／100／400，pipeline 最大 abs 分別 **2.17e−6／2.81e−6／4.08e−6**，最大誤差／門檻 **0.0375／0.0416／0.0495**。 | 最小 Expand→Scan→Merge，非完整 head／真實模型 |
| ownership | **PASS**（[report](capture/scan-trt-report.tsv)）：slots=1／2，各 10 frames，M gate delay **4,000,000 cycles**，逐 frame publish digest 都等於 S02 oracle。 | execution-context／I/O slot 重用前等待 consumed |
| graph | **PASS**（[report](capture/scan-trt-report.tsv)）：實際 **12 nodes＝3 memsets＋9 kernels**；依賴由 memset→check→expand→finalize→memset→check→scan→finalize→memset→check→merge→finalize；所有遞移依賴正確，重播 digest 與非 graph 相同，fail／skip 亦通過。 | 固定 slot／addresses／shape 的 graph |
| held-stream | **PASS**（[report](capture/scan-trt-report.tsv)）：enqueueV3 與 graph launch 在 callback-held stream 放行前返回，完成 event 仍未完成；放行後結果正確。 | 指定 stream、無隱式同步 |
| API trace | **PASS，729 checks**（[CUPTI trace](capture/scan-trt-api-trace.log)）：enqueueV3×3 的 **213** runtime／driver callbacks，runtime kernels **27**、async memsets **9**；graph launch **2** callbacks。所有 stream-bearing calls 的 stream 正確；未見禁止的同步／配置／釋放／module loading／同步 memcpy／memset／D2H。 | warmup 後實際捕捉的 API entries，不重複計算對應 driver launches |
| engine save／cross-process | **PASS**（[engine report](capture/engine-run-report.txt)、[CTest](capture/trt-ctest.log)）：P4 plan **33,516 bytes**，SHA256 `ed7228a5f18d2905fa36f187032925c619db47010ed05da9e65b9a63bbe11e39`，save/readback 一致；新 context、warmup 前的 LAZY held-stream 首 enqueue 返回（記錄 **780 μs**，僅診斷），S02 bit-exact；frame OK／FAILED／SKIPPED 與 committed payload 比對成功；未註冊 creator 的 context 被拒絕。 | plugin 序列化／註冊與首次載入；不作效能宣稱 |
| async fault | **PASS，6 checks**（[CTest](capture/trt-ctest.log)）：frame 3 trap，event wait 回報 `unspecified launch failure`、run 拋例外；測試要求 published 恰為 **{0,1,2}**。fatal context 清理的 TensorRT CUDA diagnostic 以 `expected_fatal_teardown_error` 保留。 | 獨立 process；faulted context 不重用 |
| memcheck／racecheck／synccheck | **各 842 checks、exit 0、0 errors／0 hazards**（[memcheck](capture/sanitizer-memcheck.log)、[racecheck](capture/sanitizer-racecheck.log)、[synccheck](capture/sanitizer-synccheck.log)）。 | 預設 suite；不含 explicit-only API trace／fatal fault |
| initcheck | **827 checks、exit 0、0 errors**（[initcheck](capture/sanitizer-initcheck.log)）；略過項目見 [sanitizer-skips](capture/sanitizer-skips.txt)。 | `--no-held-stream` 只略過 callback-held stream 檢查；API trace／fatal fault 另跑 native |
| sanitizer 正控制組 | **四個 planted defects 都被回報，工具各 exit 99**：memcheck **37 errors**、initcheck **32 errors**、racecheck **1 hazard**、synccheck **2 errors**（[memcheck control](capture/control-memcheck.log)、[initcheck control](capture/control-initcheck.log)、[racecheck control](capture/control-racecheck.log)、[synccheck control](capture/control-synccheck.log)）。wrapper 確認 exit=99 後 exit 0。 | S02 control 的越界／未初始化／race／非法 syncwarp mask；不宣稱工具完整檢查 shuffle mask |
| mutation | **12/12 DETECTED**，全部 mutant 重新編譯成功，未修改的 targeted tests 各 exit 1（[mutation log](capture/mutation-check.log)、[results](capture/mutations/results.tsv)）；baseline 842 checks 與 LAZY engine run 都通過。preload mutant 由 warmup 前第一次 held-stream enqueue 的 **30 秒 watchdog** 抓到；enqueue 同步 mutant 在 stream group 的 graph capture 被拒，default stream mutant 被 stream group 拒絕。 | flags、done、unknown bits、retire、consumed、commit、publish、preload、context、serialization、stream、sync |
| 既有 evidence／source | S02 kernels／S01 reference diff 為空（[source unchanged](capture/s02-kernels-and-reference-unchanged.log)）；歷史 evidence diff 為空（[evidence unchanged](capture/historical-evidence-unchanged.log)），包括 S02 的全部歷史 SHA256SUMS **通過**（[hashes](capture/historical-evidence-hashes.log)）。 | 不改寫歷史；共用測試抽取不改 S02 算法 |
| 文件與腳本 | **PASS**（[verification](verification/)）：四份交付 Markdown 的本機連結／anchors、authored-text whitespace 與 shell syntax 全部通過。raw captures／diffs 保留原始 bytes。 | 只驗證 repository 檔案，不代表 GPU 或模型驗收 |
| 遠端 CI | 交付 commit 的 push CI 於最終回報附 run URL；workflow 僅跑 CPU。 | 不將 CPU CI 當 GPU 驗收 |

## 證據與重跑

[capture.sh](capture.sh) 要求受測 commit 等於 clean HEAD，保留各命令、raw logs、exit codes、source archive／file hashes 與 binary hashes；CPU build、完整 CUDA＋TRT CTest、數值報告、CUPTI trace、四種 sanitizer、正控制組與 [mutation-check.sh](mutation-check.sh) 都從 archive 建置。mutation 改私有 source copy 並重新編譯，compile failure 不算 detection；mutant 的編譯期 Git identity 仍是 baseline parent metadata，實際變異 bytes 以各 `.diff` 與 build log 為準，不能把其 `build_dirty=false` 解讀成未變異。

```sh
git switch --detach e468044f0ab003fe5f76d6d2ef7260b618121516
bash evidence/s03/2026-10-29/capture.sh build-s03-replay
```

api_trace 與 async_fault 是明確指定的獨立程序，不與 sanitizer 混跑。initcheck 略過 callback-held stream 項目，原因同 S02：工具插樁會使 enqueue 等待被 callback 擋住的 stream；一般 native／其他工具仍執行該項。實際 skip 見 [sanitizer-skips](capture/sanitizer-skips.txt) 與 initcheck log 的 `skipped streams --no-held-stream`。synccheck 不檢查 shuffle mask 的既有工具限制仍適用；未驗證項目不列為 PASS。
