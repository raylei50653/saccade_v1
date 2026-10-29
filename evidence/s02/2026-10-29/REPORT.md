# S02：SelectiveScan／Cross-Scan 的 CUDA FP32 forward — 2026-10-29

**S02 PASS，範圍限 [SCAN_CUDA](../../../docs/SCAN_CUDA.md) §1–§5 的 FP32 forward：`selective_scan_cuda`、`cross_scan_expand_cuda`、`cross_scan_merge_cuda` 在指定環境對 S01 binary64 reference 的逐元素比較，以及 status、重設、stream、host 拒絕與 sanitizer 行為。** 支援限制（FLT_MAX 附近幾個 ulp 的範圍判定灰區）列於 SCAN_CUDA §4，不在 PASS 內。FP16 I/O、TensorRT plugin（S03）、完整 head 與真實權重（S04／P01）未開始；P01 仍 OPEN。

本輪沒有執行 Python、舊程式或舊模型，沒有修改 S01 reference、fixtures 或任何既有 runtime（[s01-and-runtime-unchanged](capture/s01-and-runtime-unchanged.log) 為空），既有 evidence 未變動。

## Source 與執行身分

| 欄位 | 本輪記錄 |
| --- | --- |
| 起點 | `HEAD=origin/main=60d722cfa45bb6d9763740cf6c8103c9defa9c0d`，工作樹乾淨；本輪在本機分支 `s02-cuda-forward`，未推送。 |
| 政策先行 | `7ab618f`：[SCAN_CUDA](../../../docs/SCAN_CUDA.md) 的呼叫契約、FP32 算法、支援範圍與比較政策，在第一次 GPU 執行前提交。§3–§5 之後未修改（§2 新增初始化列、§6 修正 sanitizer 說明，原因見下節）。 |
| 受測 source | **`7f7bfa73d53d5ee5d41b416e8abf0b20b1d8e434`**：kernel、host API、GPU 測試、CMake 與本目錄的 capture／mutation／control 腳本。[capture.sh](capture.sh) 要求 HEAD 等於受測 commit 且工作樹乾淨，再以 `git archive` 解出建置；archive SHA-256 見 [archive-hash](capture/archive-hash.log)。GPU 報告的 `build_commit` 為此 commit、`build_dirty=false`（[report](capture/scan-cuda-report.tsv)）。 |
| 交付身分 | 本報告、captures 與文件更新為下一個 commit；其 SHA 與推送狀態由交付時的 Git 狀態與最終回報記錄。 |
| 環境 | WSL2 Linux 6.18.33.1；GCC 16.2.1、CMake 4.4.4、glibc 2.44；nvcc 13.4.92、CUDA runtime／driver API 13040、driver 617.42；compute-sanitizer 2026.3.0.0；NVIDIA GeForce RTX 5070 Ti Laptop GPU，SM 12.0，UUID `GPU-825ae551-ba16-539a-e444-22726b59ee9a`（[environment](capture/environment.log)）。建置 `-DCMAKE_CUDA_ARCHITECTURES=120`，scan 旗標 `--ftz=false --prec-div=true --prec-sqrt=true --fmad=true -lineinfo`（[cuda-build](capture/cuda-build.log) 的實際 nvcc 命令）。 |

## 開發過程的發現與處理

政策提交後的開發執行（`build-s02-dev`，未封存）依序遇到下列情況；封存證據全部來自上述乾淨 commit 的重跑。

1. **第一次 GPU 執行：數值全部在門檻內**（最大 誤差／門檻 0.295），失敗的只有一項 host 拒絕測試：測例把兩個不重疊的 buffer 當成「tokens 與 x 重疊」。這是測試構造錯誤，已修正測例；數值政策沒有改。
2. **lazy loading 造成的隱式同步。** 單獨執行 stream 測試時，新 process 的第一次 `selective_scan_cuda` 會在提交呼叫內等待被 host callback 擋住的 stream；`CUDA_MODULE_LOADING=EAGER` 時正常。CUDA 13 預設 lazy loading，第一次 launch 才載入 kernel，而載入會等待裝置上已在執行的工作。這是 ASYNC_CONTROL A06 禁止的隱式同步，所以新增 `selective_scan_cuda_load_kernels()`，列入 SCAN_CUDA §2 呼叫契約；CTest `selective_scan_cuda_lazy_streams` 在 lazy loading 下單獨執行 stream 測試，mutation 證明不載入時會卡住。
3. **initcheck 讓提交呼叫等待被擋住的 stream**（即使 kernels 已載入），host callback 測試在該工具下必然死結。initcheck 以 `--no-held-stream` 只略過這一項；其他三個工具與一般執行都照常跑。
4. **synccheck 不檢查 `__shfl_sync` mask。** 原先文件把它列為 shuffle mask 的檢查，以 control 實測後更正（下節）。
5. **default stream mutation 最初是被早期案例的競態抓到**，不具決定性；新增 `--group`，要求 host callback 測試與 graph capture 測試各自單獨抓到。

## 驗證結果

| 驗證 | 結果與證據 | 適用範圍 |
| --- | --- | --- |
| 預設 CPU build＋CTest | CUDA OFF，**9/9 PASS**（[cpu-ctest](capture/cpu-ctest.log)）。 | 預設建置不受影響 |
| CUDA build＋CTest | **14/14 PASS**：既有 9 項 CPU、`selective_scan_cuda`、`selective_scan_cuda_lazy_streams`、三項既有合成 GPU 測試（[cuda-ctest](capture/cuda-ctest.log)）。 | 本機 SM 12.0 |
| 數值對照 | **364 checks PASS**（[report](capture/scan-cuda-report.tsv)）。70 個比較案例、1,294,286 個元素全部在 2×E 內；最大 誤差／門檻 **0.295**（`shared_a_rank1_c_d_N1`）。m 的三個形狀：S01 ABI 資料 max abs 9.54e−7、0.061；實際尺度隨機資料 L=25／100／400 max abs 1.16e−5／1.28e−5／1.75e−5，0.099／0.117／0.132。 | GPU 對 binary64 reference，同一份 FP32 輸入 bytes |
| softplus 與辨識力 | review 測例（delta=−18、B=1e8）GPU 1.52299809、reference 1.5229979629，0.066；host 模擬舊式 `logf(1+expf(x))` 得 0，超出門檻 7.6×10⁵ 倍。52 點掃描（−1100…1000，含 −80、20 與相鄰 float）全部通過，最大 0.073；舊式在其中 18 點超出門檻。 | softplus 三個分支與門檻 |
| 中間量 | drive 2¹⁰⁰·2¹⁰⁰·2⁻¹⁰⁰ 得精確 2¹⁰⁰；delta=−100／−150 配大 B·u（一般 FP32 的 Δ 為次正常數或 0）相對誤差 2.5e−8／2.1e−8；B 為次正常數 2⁻¹⁴⁰ 時 3.8e−8；butterfly 部分和溢位、部分和由 D·u 拉回、merge 部分和溢位都得到 reference 的有限結果。`extreme_subnormal_input` 輸出最小次正常數而非 0（`--ftz=false` 實際生效）。 | SCAN_CUDA §3 的縮放處理 |
| 相消 | `cancellation_two_states` 相對 \|y\| 最大 0.415，誤差／門檻 0.036：門檻隨項量級 S 縮放，這類案例不需另設全域容差。 | SCAN_CUDA §5 |
| 失敗行為 | 15 個案例：六個 tensor 的 NaN／±Inf（輸入位元恰為該 tensor）、L=0 的 A／D、六類範圍失敗（位元恰為 decay／drive／state／C·state／D·u／output）、十二個 lane 中一個失敗；全部 `done` 已寫入、輸出全為 quiet NaN，reference 以相同診斷拒絕。 | status 契約 |
| 重設與決定性 | 同一組 buffers「A → B（失敗）→ A → B（有效）→ A」，三次 A 輸出逐位元相同且 status 有效；m L=400 在新 buffers 重跑逐位元相同；graph 重播與直接呼叫逐位元相同。 | 每次呼叫重設、同一 build／裝置的決定性 |
| stream | host callback 擋住非預設 stream 時，scan 的完成 event 未完成，放行後結果對應放行後才寫入的輸入；global-mode capture 成功，graph 恰為 1 memset＋3 kernels、無其他節點，換輸入重播通過；兩個 stream 同時執行不同案例各自正確；`0`／`cudaStreamLegacy`／`cudaStreamPerThread` 皆拒絕。 | 工作在指定 stream、無隱式同步／配置／D2H |
| host 拒絕 | S01 的 shape／容量條件（診斷相同）、null／未對齊指標、輸出與輸入重疊、status 重疊；拒絕後 status sentinel 不變，證明沒有 enqueue。L=0 時 u／delta／B／C／y 可為 null。 | 提交前拒絕 |
| Cross-Scan | 9 種形狀（含 S01 手列 2×3、非對稱 3×5／1×4／4×1／7×3、batch 2、m 的 128×20²／10²／5²）展開與 reference 逐位元相同；合併最大 0.245；展開→scan→合併串接 0.030／0.056；非有限值與合併溢位通過。 | 翻轉、展平、還原順序 |
| kernel 資源 | registers：scan 34、merge 34、expand 28、check 20、finalize 16；local（spill）、shared 皆 0（report 的 identity 列；靜態用量見 [kernel-resources](capture/kernel-resources.log)）。 | 參考資訊，非驗收條件 |
| memcheck／racecheck／synccheck | 完整測試 364 checks，**皆 exit 0、0 errors／0 hazards**（[memcheck](capture/sanitizer-memcheck.log)、[racecheck](capture/sanitizer-racecheck.log)、[synccheck](capture/sanitizer-synccheck.log)）。 | 本測試執行到的路徑 |
| initcheck | `--no-held-stream`，360 checks，**exit 0、0 errors**；報告記錄 `skipped stream_gate`（[initcheck](capture/sanitizer-initcheck.log)）。 | 同上，少一項 stream 測試 |
| sanitizer 正控制組 | [sanitizer-control.cu](sanitizer-control.cu)：越界寫、未初始化讀、跨 warp shared race、`__syncwarp` 非法 mask 分別被 memcheck／initcheck／racecheck／synccheck 回報（各 exit 99）。排除自身的 `__shfl_sync` mask 不被 synccheck 回報（exit 0，已記錄的工具限制）。 | 工具在本機確實啟用 |
| mutation | 未變異 baseline 364 checks 通過，lazy loading 下 stream 群組通過；**17 個單點 CUDA mutation 全部被測出**，default stream 另被 host callback 與 graph capture 測試各自單獨抓到，不載入 kernels 在 lazy loading 下卡住而逾時（exit 124）（[mutation-check](capture/mutation-check.log)、[diffs 與 logs](capture/mutations/)）。 | 測試能分辨所列實作條款 |
| 既有 evidence | 與基線比較，P01-A、兩組 CUDA synthetic、scan-contract、S01 的 evidence 沒有任何差異（[unchanged](capture/historical-evidence-unchanged.log)），全部 SHA256SUMS 重新通過（[hashes](capture/historical-evidence-hashes.log)）。 | 不改寫歷史 |
| 文件與腳本 | [verification](verification/)：本機連結（含 anchor）、空白與腳本語法。 | 只驗證本 repository 檔案 |
| 遠端 CI | 本輪未推送；CI 只跑 CPU 建置，不涵蓋 GPU。 | — |

## 重跑

```sh
git switch --detach 7f7bfa73d53d5ee5d41b416e8abf0b20b1d8e434
bash evidence/s02/2026-10-29/capture.sh build-s02-replay
sha256sum -c evidence/s02/2026-10-29/SHA256SUMS   # 在交付 commit 上
```

需要 C++20 compiler、CMake、CUDA Toolkit 13.4（預設 `/opt/cuda`，可用 `NVCC`、`SANITIZER` 指定）、SM 12.0 GPU，以及 Windows 端已啟用的 GPU debugger interface（見 [closeout 證據](../../cuda-synthetic/2026-10-28-closeout/REPORT.md)）。輸出寫入新的 `build-*` 目錄，不覆蓋封存 captures；本目錄的 `capture/` 由 `build-s02-evidence/capture/` 複製，mutation 只保留 logs 與 diffs，不封存二進位。

## 未涵蓋與下一步

- **支援限制**：FLT_MAX 附近幾個 ulp 的範圍判定可能與 binary64 reference 不同（SCAN_CUDA §4），未列入 PASS；次正常數結果只保證 η 項的絕對精度。
- **不在 S02**：FP16 I/O、plugin、MambaBlock 其他算子、完整 head、真實權重、效能與跨 GPU／跨 build 的逐位元一致性。數值界依 NVIDIA 公布的 `expf`／`log1pf` ulp（廣泛但非窮舉測試），只在本機實測。
- **status 的消費者**：TensorRT 內的下游 layer 讀不到 status，失敗時只能看到 NaN 輸出。S03 須決定 plugin 的 status 通道或由 F04 檢查非有限值。
- **權重**：與 S01 相同，部署 ONNX 對應 `…_gpu_decode/best_recall.ckpt`，不是 T1 主線；S04 須先選定同一份權重。

**下一步：S03，以 v1 自建 plugin 接上同一個 kernel，記錄完整 build 身分、shape／dtype 拒絕、顯式 stream 與 session workspace 隔離，並決定 status 的回報方式。**
