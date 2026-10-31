# F04 偵測後處理：本機開發驗證

日期：2026-10-31。**F04 已實作，本機開發驗證通過；clean-source 正式驗收未執行。** 本次使用未提交的工作區，binary 明列 `source_dirty=true`；不能將這些結果改稱已提交 source commit 的正式 F04 PASS。完整 source bytes、逐檔 hashes、編譯與測試命令保存在 [capture](capture/)。

範圍為固定 batch-1／640／FP32 的六個 raw cls／ltrb tensors，到完整 `DetectionCandidates`；包含 kept 與 suppressed pool、穩定 candidate IDs、容量／狀態與全部 padding。**不關閉 P01 或完整 async session。** API 與重跑入口見 [POSTPROCESS](../../../docs/POSTPROCESS.md)，固定算術與判定見 [POSTPROCESS_POLICY](../../../docs/POSTPROCESS_POLICY.md)。

## 身分與 GPU 前固定

| 項目 | 實際紀錄 |
| --- | --- |
| Git base | `b0dfffcbd25244c0e24f83bba5abcf4a0a55619a`；工作區未提交 |
| Capture | `build-f04-evidence-final`，`--allow-dirty`；[source](capture/source.txt)、[status](capture/capture-status.txt) |
| Build-input manifest digest | `416a30fbec8ea3f9c2098aa04c2f6ea6771f80922833ceb4eb1e20c829e301b8` |
| Policy | `f04-postprocess-fp32/1`；SHA-256 `96f1985dd7f90367af51139758eaab46ed92d82408ab63879e3ab641980e26f7` |
| CPU reference | SHA-256 `71c7868681f20ba69688be369d31e4e6a089604d1f51dc9cb622248266232ed4` |
| CPU cases | SHA-256 `74e7f3801657042df55a28cb5d66c120267a1c8ded3c2a1bed50c05f1fa73061` |
| 最終 GPU suite | SHA-256 `2295397df880a48c8975b6779b9c54791455e3d6a5b6ffd69930818b2e16046e` |
| CUDA implementation | SHA-256 `e10351a5c9fa30833cf2cea2db1851383bcf008c16969e2d19091ecbcc51e13b` |
| GPU／driver | RTX 5070 Ti Laptop、sm_120、driver 617.42、WSL2 |
| Compiler／CUDA | GCC 16.2.1、nvcc 13.4.92、CUDA runtime／driver API 13.4 |
| Kernel flags | `--ftz=false --prec-div=true --prec-sqrt=true --fmad=false -lineinfo` |
| Device ABI／scratch | 9,664-byte output；202,812-byte caller-owned scratch |

[首次比較前 hashes](development-attempt-1/SHA256SUMS) 與 [case list](development-attempt-1/cases.txt) 在第一個 F04 GPU 比較前保存。政策、reference、CPU cases 與 CUDA implementation 此後未改。測試程式先修正 worker context 初始化，後續獨立審查再發現 F03 dump 讀取順序錯誤；兩項修正與原始失敗均保留，詳見下節。最終 suite 在新的 GPU capture 前保存 [hashes](capture/frozen-policy-reference-cases.sha256)，新增既有訓練側 kept counts 作獨立回歸控制。沒有依 GPU 誤差改變 tolerance 或 runtime 離散決策。

最終 capture 對列入 manifest 的 build inputs，在每一步前後重新核對 SHA-256，全部相同；[source snapshot](capture/source-snapshot.tar.gz) 保存該批完整 bytes。[binary hashes](capture/binaries.log)、[CMake cache](capture/CMakeCache.txt)、[實際 compile commands](capture/compile_commands.json)、[environment](capture/environment.log)、[ldd resolution](capture/ldd.log) 與 [resolved library hashes](capture/resolved-libraries.sha256) 分別保留。library 紀錄是同環境的 loader resolution，未宣稱對每個受測 process 另行追蹤 `/proc/maps`。

## 執行結果

| 檢查 | 結果與實際範圍 |
| --- | --- |
| Default CPU Release | **12/12 CTest PASS**；[logs](cpu/ctest.stdout.log)，含 F04 reference **442 checks**；未設定可選真實 checkpoint，因此不含兩項候選權重 tests |
| CPU ASan／UBSan | **F04 1/1 PASS**；[log](asan/postprocess-only.stdout.log)。額外全 CPU Debug suite 中途停止，沒有列全套 PASS |
| F04 CTest | **4/4 PASS**：reference、完整 CUDA suite、lazy-loading streams、六份 F03 replay；[log](capture/ctest.log) |
| 固定合成案例 | **31/31 PASS**；每案完整 header／300 rows 比較、同 slot 重跑及 graph replay 逐位元相同 |
| CPU fixture 不變量 | **53 checks PASS**；在 CUDA 初始化前執行，[log](capture/fixtures.log) |
| 全套＋六份 replay | **13,983 checks PASS**；[log](capture/full-replay.log)，包含獨立歷史 kept-count 檢查 |
| 所有權與 context | held stream 非同步提交、pending producer-ready、pending consumer-consumed、64-bit epoch／frame 回傳與 frame 變更、兩個獨立 instances；正常 suite **PASS** |
| Host 拒絕 | null／不對齊／host pointers、shape／capacity／mapping、alias、實際 allocation 小於聲稱範圍、default streams；拒絕後原 output sentinel 不變 |
| Memcheck | **0 errors**；31 cases＋六份 replay，**13,678 checks**；[log](capture/sanitizer-memcheck.log) |
| Initcheck | **0 errors**；相同 suite／13,678 checks；[log](capture/sanitizer-initcheck.log) |
| Racecheck | **0 hazards（0 errors、0 warnings）**；相同 suite／13,678 checks；[log](capture/sanitizer-racecheck.log) |
| Synccheck | **0 errors**；相同 suite／13,678 checks；[log](capture/sanitizer-synccheck.log) |
| Sanitizer 正控制 | **4/4 DETECTED**；各 process exit **99**，分別匹配 global OOB write、uninitialized read、shared race、invalid barrier 診斷 |
| 語義變體 | **10/10 DETECTED**；各 exit **1** 且匹配預先指定完整 test diagnostic；[results](mutations/results.tsv) |
| 舊 parser 故障控制 | **DETECTED**；僅改 staged test 的 file parsing loop，正確 kernels 不變，exit **1** 且匹配獨立歷史 kept-count 診斷；[result](mutations/legacy-parser.result.tsv) |
| 前處理 header 相容性 | rebuild 與指定 `resize_cuda`／`native_preprocess` **2/2 PASS**；[log](preprocess-compat/f04-header-compat-ctest.stdout.log)。沒有重分類既知 nvJPEG initcheck 診斷 |

四種 sanitizer 均使用 `--no-held-stream`，其 callback gate 測試明列 **UNEXECUTED**；event-order／independent-instance 測試仍執行。不能將 sanitizer 結果擴張為「sanitizer 下也測過所有 pending gates」。正常、lazy-loading suite 的 gates 已實跑通過。sanitizer 都是獨立 process、exit 0，且須有對應零錯誤 summary 與測試 PASS；故障正控制也各自獨立，沒有把故意 planted failure 算成正常 runtime 錯誤。

完整結果比較要求 header、IDs、classes、排序、kept／suppressed、全部座標 bits 與未用 rows exact；只有 score 依事先固定的 `4*FLT_EPSILON = 4.76837158203125e-7` 絕對界。實測最大 score 差 **1.19209e-7**。相同 binary／GPU 的重跑與 graph 要求整份 bytes exact，沒有以容差容許不同候選集合。

10 個變體為 `IoU>=0.5`、suppressed 繼續抑制、sigmoid 前 argmax、反轉 candidate ID ties、錯 stride、left／top 交換、x／y scale 交換、只交 kept、忽略 upstream flags、不清 rows。正常 baseline、copied-source baseline 與還原後 baseline 均 PASS；變體 source、diff、binary hashes、命令、stdout／stderr 與各 exit 原樣保存在 [mutations](mutations/)。編譯失敗、timeout、crash 或非預定診斷不算 DETECTED。

另以 disposable copy 只恢復舊的 interleaved parser，正確 runtime kernels 保持不變；[diff](mutations/legacy-parser.diff) 僅交換 test 的 kind／level 兩層 loops，獨立 guard 確實以 `F03 replay disagrees with independent historical kept count: input 0` 拒絕。還原 parser 後重新編譯並重播六份 F03，**13,678 checks PASS**；[control status](mutations/legacy-parser-status.txt)。此附加回歸控制與原本 10 個 runtime 變體分開記錄。

## F03 錄製輸出重播

大型 raw bytes 來自既有正式 F03 capture 的 `model-dump/input-{0..5}.f32`，每份 **2,822,400 bytes**。先核對程式固定的全部 SHA-256，再按 **cls_p3、cls_p4、cls_p5、reg_p3、reg_p4、reg_p5** 解讀；FP32 element offsets 分別為 **0、512000、640000、672000、697600、704000**，並非 cls／reg 交錯。此順序直接核對 [原始 payload](../../t01/2026-10-30/f03/capture/model-run-report.txt)，沒有用 `native/dev` 產物替代。[來源 artifact hashes](capture/f03-source-artifacts.sha256) 同時重新核對 engine bytes；對應原 [F03 report](../../t01/2026-10-30/f03/REPORT.md)。

來源 engine 為 `c6cff1936194e679ce11c3736cab552e4a73a2b1a9ae5508bc3e5d1b3fbab9c5`，T01 bundle 為 `4fccfda6342b88d43fc2bf676228810b974b9e9956f45fa53352124098f046a3`。本次**沒有重跑 TensorRT**，只重播其已核對 raw bytes；caller 注入明確 aggregate producer status。

| 固定 input | 完整 rows | Kept | Suppressed |
| --- | ---: | ---: | ---: |
| MOT17-02-SDP/1 | 237 | 32 | 205 |
| MOT17-10-SDP/1 | 246 | 28 | 218 |
| MOT17-13-SDP/400 | 285 | 39 | 246 |
| zeros | 7 | 2 | 5 |
| ones | 5 | 3 | 2 |
| random/1 | 3 | 1 | 2 |

raw 與完整 F04 output SHA-256 在 [full replay](capture/full-replay.log) 中逐份保存。前三項以原圖 1920×1080 的 mapping，後三項以 640 identity mapping 測試。六個 kept counts 額外要求與預先存在的 [訓練側 native-raw diagnostic](../../t01/2026-10-30/f03/capture/f03-diagnostics.json) 的 **32、28、39、2、3、1** 一致；這是獨立的 layout／解碼回歸檢查，避免 CPU 與 CUDA 一起誤讀仍互相 PASS，不宣稱其餘欄位已與歷史 evaluator 完整等價，也不是重新計算 T01 detection AP 或模型品質。

## 保留的失敗與尚未執行

[第一輪開發結果](development-attempt-1/first-gpu.log) 的 31 numerical／repeat／graph cases 及 argument tests 先通過，之後新的 host worker 未呼叫 `cudaSetDevice`，driver allocation-range 查詢回 `CUDA_ERROR_INVALID_CONTEXT`，整個 suite exit **1**。此 attempt 不算完整 PASS。修正為每個 submitter 初始化自己的 primary context；[第二輪](development-attempt-2/second-gpu.log) 程序 exit 0，但其 F03 結論後來因下述 parser 錯誤撤回。沒有修改 kernel／reference／policy／cases 的數值語義。首輪 executable 未另外保存，保留其 pre-run hash 與原始 log，不宣稱可由該 log 恢復原 binary。

該第二輪及其後首次 capture 的 F03 replay 尚有 **dump layout 誤讀**：parser 交錯讀 cls／reg pairs，CPU 與 CUDA 同時收到錯資料，因而互相比較通過。其 F03 結論與相應 sanitizer 的 intended-replay 聲明均撤回；原 logs／snapshot／mutation results／當時錯誤報告草稿原樣保存在 [misparsed attempt](development-misparsed-replay/README.md)。修正 parser 後重新跑整份 capture 與四種 sanitizer，之後新增獨立歷史 kept-count assertion，再跑本報告採用的最終 capture。runtime kernels／CPU reference／數值政策始終未改；原錯誤不被刪除或追溯改稱 PASS。

額外全 CPU Debug ASan suite 在未修改的 head 測試期間停止，保留 [INTERRUPTED 紀錄](asan/ctest.interruption.json)。停止是因 F04 必要 ASan／UBSan 與完整 Release regression 已通過、結束額外範圍；不是測試 failure，也沒有把未完成的整套列為 PASS。

尚未執行／不在本項範圍：clean-source 正式 F04 acceptance、F03 全部 statuses 的 native aggregate adapter、F01／F02→F03→F04 的 live 接續、F04b、GMC、tracker、ResultPacket／MOT writer、session 首錯鎖定與完整 CS ready／consumed ring、全量 5,316 幀或跨 GPU／SDK 重驗、效能與模型品質驗收。F04 的單次呼叫不代替 caller 的 session failure latch。既有 T01 closeout、nvJPEG initcheck 診斷與 head／F03 未執行的 racecheck／synccheck 均不改寫。

重跑使用 [capture script](../../../scripts/run_postprocess_acceptance.sh) 與 [mutation script](../../../scripts/check_postprocess_mutations.sh)。前者預設要求 clean tree；`--allow-dirty` 明確留下 development 身分。此 capture 未建立 commit、push 或外部發布；後續提交程式與證據，不改變 capture 當時的 dirty 身分或正式驗收狀態。
