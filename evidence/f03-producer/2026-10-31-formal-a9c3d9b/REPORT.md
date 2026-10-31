# F03 producer：乾淨 a9c3d9b 正式驗收

2026-10-31，**clean-source 正式驗收 PASS（受測範圍）**。受測 source commit 為完整 `a9c3d9bfecb61fdea8850b3fdf2308f09aace629`，在獨立 detached worktree 重新配置、建置及執行，`allow_dirty=false`。本報告只結案 `f03-producer/1` primitive 與固定 F03 engine→producer→F04 接續，不將此驗收擴張為完整 async session、模型品質、P01 或效能驗收。

## 身分與封存

| 項目 | 固定身分／本次觀測 |
| --- | --- |
| Source | `a9c3d9bfecb61fdea8850b3fdf2308f09aace629`；[前後 clean](capture/source-status-after.txt)、[source identity](capture/source.txt) |
| Build-input manifest digest | `9252b012ffcefb4518e96ab8fb60af36c9baf7712893a85fb4b42c11c50aeb19` |
| F03 engine SHA-256 | `c6cff1936194e679ce11c3736cab552e4a73a2b1a9ae5508bc3e5d1b3fbab9c5` |
| 六個 FP32 inputs SHA-256 | `14dda6867e23ed8a2d83086224d71a445a7fdcb606c7507a3f39c0e5faf60aa0` |
| GPU／driver | RTX 5070 Ti Laptop、sm_120、617.42，WSL2 |
| Compiler／CUDA／TensorRT | GCC 16.2.1、nvcc 13.4.92、TensorRT 10.16.1.11 |
| Compute Sanitizer | 2026.3.0.0 |
| Capture scope | `MemoryMax=12G`、`MemorySwapMax=0`；[launcher](capture/scope.json) |

完整命令、退出碼、起始時間及 `/usr/bin/time -v` 保存在 [capture](capture/)。[source manifest](capture/source-files.sha256) 的 117 個 build inputs 已與 [source archive](capture/source-build-inputs.tar.gz) 逐一核對。每步前後由原始 capture script 核對 source commit 與 build-input digest；整體前後 `git status --porcelain` 皆空。執行結束為 [exit=0](capture/capture-status.txt)，[獨立核對摘要](capture/verification.json) 記錄 hash 與各步分類。主工作樹後續 async session 變更不屬於此 source。

[受測 binaries](capture/binaries.log) 與 [測試後 binaries](capture/binaries-after.log) 逐位元相同，engine／inputs 的 [前](capture/artifacts.log)／[後](capture/artifacts-after.log) hash 亦相同。[CMakeCache](capture/CMakeCache.txt)、[compile commands](capture/compile_commands.json) 與 [runtime dependency paths／hashes](capture/runtime-dependencies.json) 保留。實際 CPU／CUDA／standalone build 位於本機 `build-f03-producer-formal-a9c3d9b/source/build-capture/`；43 個實際 executables／libraries 另封裝於本機 `build-f03-producer-formal-a9c3d9b/actual-binaries.tar.gz`，bundle hash 見 [local-artifacts](capture/local-artifacts.sha256)，逐檔身分見 [actual-binaries](capture/actual-binaries.json)。此處不宣稱 CUDA rebuild 的逐位元可重現性。

使用既有固定 engine／六輸入 corpus，沒有重建 engine；來源同 [歷史 F03 evidence](../../t01/2026-10-30/f03/REPORT.md)。本次 clean capture 重新執行 producer 與 F04，未重做 F03 訓練或數值政策的候選選擇。

首次 launcher 誤用 main cwd，於 CPU build 期間主動停止，尚未進入 CUDA configuration 或 GPU 測試；其不完整紀錄保留於 [aborted-main-cwd](aborted-main-cwd/README.md)，不納入 PASS。正式結果全部來自重新啟動的獨立乾淨 worktree。

## 正式結果

| 項目 | 本次結果與範圍 |
| --- | --- |
| 預設 CPU suite | **12/12 PASS**，[log](capture/cpu-ctest.log) |
| CUDA／F04／live CTest | **6/6 PASS**，[summary](capture/ctest.log)／[完整輸出](capture/LastTest.log) |
| Producer primitive | **1,282 checks PASS**；九個來源逐一 known failure／缺 marker／錯 marker／未知 bits／角色錯誤、mixed failure、全部退休、failure→success 重用、input 與六個 raw tensors 的 NaN／±Inf、host arguments／overlap／allocation extent |
| Stream／ownership | 正常 direct held stream、upstream ready、consumer consumed gates PASS；fresh lazy process **30 checks PASS** |
| API trace | **90 calls、3 launches**；無 frame-path allocation／free／transfer／memset／event／synchronization，stream-bearing calls 使用指定 stream |
| Graph replay | 合成 success／failure／recovery publication→producer，完整 engine→producer→F04 三次 replay PASS |
| Live F03→producer→F04 | **6 inputs、3,810 checks PASS**；raw SHA 對歷史 bytes，完整 candidates 對 CPU reference，repeat bytes、NaN failure、failure→success 重用、status 退休及 checked close |
| Adapter Memcheck／Initcheck | 各 **1,167 checks、0 errors**，[memcheck](capture/sanitizer-memcheck.log)／[initcheck](capture/sanitizer-initcheck.log) |
| Adapter Racecheck／Synccheck | 各 **1,167 checks**；**0 hazards（0 errors、0 warnings）／0 errors**，[racecheck](capture/sanitizer-racecheck.log)／[synccheck](capture/sanitizer-synccheck.log) |
| Sanitizer 正控制 | **4/4 DETECTED**；各 exit 99 且符合預期 memcheck／initcheck／racecheck／synccheck 診斷；非僅以 nonzero exit 判定 |
| Live Memcheck／Initcheck | 各 **1 input、1,630 checks、0 errors**，含 NaN failure／recovery／graph；[memcheck](capture/live-memcheck.log)／[initcheck](capture/live-initcheck.log) |
| 無 TensorRT／nvJPEG 的獨立 library | **PASS**，[configure](capture/standalone-configure.log)／[build](capture/standalone-build.log) |

Live kept counts 固定為 **32／28／39／2／3／1**，全部 rows 為 **237／246／285／7／5／3**。完整 kept／suppressed pool、header、candidate identity／順序、座標及未用 rows 依既有 F04 reference 比較，score 沿用既定 `4*FLT_EPSILON`；相同 binary／input 的 repeat／graph 比較完整 bytes。

## 驗收邊界及重跑

四種 adapter sanitizer 僅涵蓋獨立 producer／合成 raw→F04；**TensorRT head／F03 racecheck 與 synccheck 為 UNEXECUTED**。Live memcheck／initcheck 各限第一個固定輸入與其 failure／recovery／graph，不能稱六個輸入皆經 sanitizer。Sanitizer 使用 `--no-held-stream --no-api-trace`，因此 **sanitizer 下 host callback gates／CUPTI trace 為 UNEXECUTED**，正常與 lazy CTest 的 gates 已實跑。單 GPU 環境未執行跨 device 實例，未做語義 mutations、效能或跨 GPU／SDK 驗收。

Primitive 的 failure→success 只是無狀態資源重用，不能據此允許 session 在首次失敗後繼續發布成功。Native F01／F02→F03→F04、雙 STAGE／candidate ring、frame／epoch identity、session first-error latch、EOF／cancel／D-only tail drain 及跨 session 生命周期均由各自 async session 變更及證據判定，本報告不涵蓋。F04b／GMC／tracker／ResultPacket／MOT writer 與 P01 亦不在此驗收範圍。

在相同乾淨 source、pinned SDK／engine／inputs 上，用不存在的 `build-*` 目錄重跑：

```sh
SACCADE_TENSORRT_ROOT=/absolute/path/to/pinned-sdk \
SACCADE_MODEL_PRODUCER_ENGINE=/absolute/path/to/model.engine \
SACCADE_MODEL_PRODUCER_INPUTS=/absolute/path/to/f03-inputs.f32 \
  bash scripts/run_model_producer_acceptance.sh build-f03-producer-new
```

隨本報告保存的檔案雜湊見 [SHA256SUMS](SHA256SUMS)；從本目錄執行 `sha256sum -c SHA256SUMS` 可核對。原始 [開發報告](../2026-10-31/REPORT.md) 保留原判定。
