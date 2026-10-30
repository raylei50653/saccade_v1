# S04：完整 YOLO26m T=1 Mamba-FPN head 的 TensorRT FP32 驗證 — 2026-10-30

**S04 head PASS，限本報告的固定候選、固定介面、受測輸入與已執行檢查範圍。** S04 範圍限 [HEAD_TRT](../../../docs/HEAD_TRT.md)：指定 T1 checkpoint 的 native 權重交付、完整 head 的 native TensorRT 建圖、P3／P4／P5 features 到六個 raw cls／reg outputs 的獨立數值對照，以及共同發布、失敗傳遞與生命週期。比較政策為 GPU 比較前固定的 [HEAD_POLICY](../../../docs/HEAD_POLICY.md) `s04-head-fp32/1`。backbone／neck 推論、影像前處理、decode／NMS、tracker、模型品質與效能不在本輪驗收；**P01 仍 OPEN**。

正式結果來自 **capture4（`build-s04-evidence4`）** 對 clean-source archive 的完整重跑，capture 腳本正常 exit 0，59 個步驟的 `.exit` 均為 0。正控制組及 mutation 的預期 raw 非零退出另行核對，不以 wrapper exit 0 隱去。前三次失敗 attempt 各自保存，不與本輪結果混用；完整 head 的 racecheck／synccheck **UNEXECUTED**。

## Source 與執行身分

| 欄位 | 本輪記錄 |
| --- | --- |
| 分支與起點 | `s04-m-head-fp32`，由 S03 evidence commit `f02dd7a58aed87adec6265bc27bd54899b4f48ed` 建立。 |
| 政策 commit | `4d38bcc51caf829afe737959e334d0c85b1eba0b`：native 權重交付、獨立 binary64 reference 與比較政策；提交時沒有任何完整 head GPU 比較結果。`docs/HEAD_POLICY.md`、`src/head_reference.cpp`、`include/saccade/head_reference.hpp` 此後保持不變（[policy unchanged](capture/policy-unchanged-since-policy-commit.log)）。 |
| 實作與 capture 修正 | `222af61a8edf46950ffb2b58542c31074454b793`：完整 head 與測試；`dc016edfc0704d840c862b9b022a7152e1ded12e`：sanitizer API 記錄分類；`0f5829ddae8366103e019df0d59a956e642637d9`：sanitizer 範圍、memory 取樣、engine bytes 比較與 mutation rejection 的 fail-closed 檢查（第二失敗 attempt）；`198ad634886ddc02cb538cb2ec227fc6ecedb717`：僅修正一行 TF32 拒絕診斷 matcher（第三失敗 attempt）；`3857e243e8fc32663aa688f073343002ad12cb30`：兩個 suites 統一 explicit API 錯誤報告，並各保留 default memcheck raw／共用分類，不改 GPU 實作。 |
| 正式受測 source | **`3857e243e8fc32663aa688f073343002ad12cb30`**；[source](capture/source.log)、空白 [working tree](capture/working-tree.log)、[archive hash](capture/archive-hash.log)（`d4923a780229254c2cc7c9123fdc30b82f0f9d0ab867136d52d300e270ff4750`）、[source-file hashes](capture/source-file-hashes.log)。最終檢查確認四份 build reports 的 revision 均等於該 SHA、`build_dirty=false`（[identity check](capture/build-source-identity.log)）；本次 source 修正僅為 sanitizer 執行／分類流程，沒有修改 GPU 程式或數值政策。 |
| 交付身分 | 本報告、raw capture 與文件是後續 evidence／docs commit；交付 SHA、local／tracking／remote equality 與遠端 CI 於交付回報列明。受測 source 與 evidence commit 分開。 |
| 環境 | WSL2 Linux 6.18.33.1、GCC 16.2.1、CMake 4.4.5、glibc 2.44、nvcc 13.4.92、compute-sanitizer 2026.3.0.0、driver 617.42、NVIDIA GeForce RTX 5070 Ti Laptop GPU（SM 12.0，UUID `GPU-825ae551-ba16-539a-e444-22726b59ee9a`）；[environment](capture/environment.log)、[TRT configure](capture/trt-configure.log)、[compiler/linker commands](capture/trt-build.log) 記錄實際身分與 flags。 |
| 首次失敗 attempt | 受測 `222af61a8edf46950ffb2b58542c31074454b793`，archive SHA-256 `41d732f7e12efedd179f54591980f0c42db5e6c158fcf26899b477f4ff59420e`；head memcheck exit 99，且 `build_dirty=true`，**不通過**。保留於 [failed-attempt](capture/failed-attempt/)，不作本輪 PASS 證據。 |
| 第二失敗 attempt | 受測 `0f5829ddae8366103e019df0d59a956e642637d9`，archive SHA-256 `2c42f695fcf7b834ad9ae3cc3b3c0fa1d493e9531435b635e405d5efaae1fa47`；26 mutants 完成，25 DETECTED，TF32 因 matcher 漏 prefix 列 INVALID_RUNTIME，mutation wrapper exit 1，**不通過**。完整已產生 logs／mutation artifacts 保留於 [failed-attempt2](capture/failed-attempt2/)，不與正式 capture4 結果混用。 |
| 第三失敗 attempt | 受測 `198ad634886ddc02cb538cb2ec227fc6ecedb717`，archive SHA-256 `25603b5c23bad4363884f093c8e1cea8cdc4c6edd0f891406e424b4bc5f40c4b`；初始工作樹乾淨、head report `build_dirty=false`。S03 memcheck 的 842 checks 通過，但 default extended 模式將唯一 driver 重新插樁 note 計為 1 error、raw exit 99，capture 中止，**不通過**。保留於 [failed-attempt3](capture/failed-attempt3/)。 |

## 權重、engine 與獨立對照

| 產物 | 身分與邊界 |
| --- | --- |
| T1 候選 checkpoint | 舊庫 `runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt`，126,802,369 bytes，SHA-256 `a5a0e7091cd854ad68210ae4bbd43af73c7d3d098ea119f02ea65fab59c0676c`，epoch 15，`selection.status=diagnostic_train_loss`（[checkpoint identity](capture/checkpoint-identity.log)）。使用者指定的工程候選，無 holdout 品質驗收，不重訓。 |
| Native bundle | `SCWB 1`，40,711,552 bytes，SHA-256 `e1a8660fb9ad510f20ccc3d56a4feac61c6a2e5f771aab531caa741e7677f310`；69 個 active、39 個 excluded checkpoint tensors 的映射／hash／排除理由見 [mapping](capture/t1-head-mapping.tsv)、[conversion](capture/weights-convert.log)、[verification](capture/weights-verify.log)。 |
| 正式 head engine | **41,471,380 bytes**，SHA-256 `ca74a50d04980b7b15ffd66a968e4851459c4fb8c3fa2db0724d9d55f69509c1`（本輪 [identity](capture/engine-identity.log)）。native network API、TensorRT 10.16.1.11、strongly typed FP32、TF32 關閉、builder optimization level 0；layer／tactic 名稱見 [build report](capture/head-build-report.txt)。兩次建置的逐 byte 比較：**cmp exit 0、plan bytes 相同**（[determinism](capture/engine-rebuild-determinism.log)）。 |
| Manifest | schema 2 canonical SHA-256 `f0fa9cd54acad380dab9b70f05cf24723ff8ebb312cb47d99da64b3da1f496d7`（[canonical JSON](capture/yolo26m-head-t1.canonical.json)、[validate](capture/manifest-validate.log)、[verify](capture/manifest-verify.log)）。宣告／檔案完整性均 verified；`model_content=not_checked`、`real_model_ready=false`，不等於本輪 head 執行結果或 P01 readiness。schema 1 fixture 仍獨立驗證（[schema 1](capture/manifest-schema1-fixture.log)）。 |
| Backbone／neck 結構 | 舊 `yolo26m_backbone_640.onnx`，75,900,061 bytes，SHA-256 `2154f937dec832dcefc4229a66dfc438c8fb99ee0c1547d80c00cc0143e56a6a`（[identity](capture/backbone-identity.log)）。ordered I/O 只查結構：`images [batch,3,640,640]` → `p3_feat [batch,256,80,80]`、`p4_feat [batch,512,40,40]`、`p5_feat [batch,512,20,20]`（[ordered I/O](capture/backbone-ordered-io.log)）；權重歸屬、與 T1 head 的配對及推論未驗。 |

每組輸入固定為 `p3 [1,256,80,80]`、`p4 [1,512,40,40]`、`p5 [1,512,20,20]`；六輸出依序是 `cls_p3, cls_p4, cls_p5, reg_p3, reg_p4, reg_p5`，合計 **705,600 元素／組**。三組驗收輸入 `asymmetric/1`、`random/1`、`random/2` 共 **2,116,800 個逐元素判定**。原始 feature hashes 與逐輸出誤差見 [run report](capture/head-run-report.txt)。reference 從同一份 FP32 features 與 bundle bytes，以獨立 binary64 C++ 運算路徑計算；不呼叫 TensorRT／CUDA／S02，不回灌 GPU 中間值，但共用唯讀 bundle decoder 與 feature 產生器。全部元素須滿足 `|gpu − ref| ≤ 16σ＋2⁻¹¹⁰`；σ 是含運算／捨入假設、事前固定的工程機率模型，不是無條件嚴格誤差界，first-order 最壞界 E 只報告。scan 步驟本身的誤差界須逐位元等於 S02 bound。

## 契約與八項開發發現

以下是交接保存的開發紀錄，與後節 clean-source 正式結果分開。遺失的 `/tmp` 開發 logs 不以本輪 logs 冒充補證。

1. **由最壞界改為機率尺度政策，先於政策 commit 與 GPU 比較。** 最初 first-order E 將輸入誤差以 `Σ|w|E` 傳遞，每層約放大 √K；到 cls／reg hidden 已達訊號十幾倍，PixelShuffle DCR 變體只到 2E 的 0.002 倍，無辨識力。只用 CPU 時改為 Higham–Mary 型 σ，政策固定 `|Δ| ≤ 16σ＋2⁻¹¹⁰`，E 留作報告。合成權重的 scan 分支耦合弱，三個 scan 內部變體不超界，故合成變體檢查只作資訊；候選權重下 13 個語義變體皆超界。正式數字見 [reference report](capture/reference-report.log)。
2. **builder level 3 選到 Winograd。** 第一次建置、任何 GPU 比較之前，IEngineInspector 顯示 P3 upsample 與三尺度融合 cls／reg 的 3×3 conv 選到 `…_fprop_wngd_…`；不屬 HEAD_POLICY 的直接求和模型，因此改 level 0，並加入建置後 tactic guard，拒絕 Winograd／FFT／FP16／BF16／TF32／HMMA／INT8。guard 初版誤把 `__myl_…_0x…4f16…` hash 視為 FP16，修成 token 比對並略過 `0x…` hash。正式 tactics 見 [build report](capture/head-build-report.txt)。
3. **第一個完整 GPU 執行在 graph 初始化中止。** S03 harness 的 `cudaGraphGetEdges` 未提供 edge data；完整 TensorRT graph 含 programmatic typed edges，CUDA 13 拒絕此 semantically lossy 查詢。當時工具僅於結尾輸出，因此先前計算的比較數字未輸出或被看到。修正會讀取並列出 typed edges，且 `head_trt run` 在失敗時也保留部分報告。正式 graph 結果見 [suite report](capture/head-suite-report.tsv)。
4. **mutation 找到兩個測試盲點。** converter 的 P4／P5 key 交換最初未被測出，因合成 archive 本身由同一份 schema 產生；新增獨立 level↔checkpoint index 與字面 key 對照後可測出。σ 移除內積局部捨入項仍無判定變化，因 SiLU／exp 的 2⁻¹⁷ 假設主導，政策偏保守，不能宣稱對此變異敏感；正式改用清除 dot-product σ 的 mutant。正式 mutation 同時要求重新編譯成功、exit 1 與對應 assertion／tactic guard 診斷；timeout、signal 與無關工具／I/O 失敗不算 detection。
5. **TF32 的主要防線是建置 guard。** TF32 tensor-core tactics 會被 guard 拒絕。開發中暫停 guard 的試驗只有 6 個輸出中的 5 個超出數值政策，最大約 2.22 倍，`reg_p4` 約 0.82 倍；不是 frozen HEAD_POLICY §7 所寫的「遠超 λσ」。政策檔保留原文字，如實記錄此限制，不以調整門檻修正結果。正式 `tf32_enabled` mutation 必須由 tactic guard 拒絕。
6. **第一次正式 capture 不通過。** 受測 `222af61`：CPU、ASan／UBSan、CUDA＋TRT CTest、轉換、head run、manifest、backbone I/O exit 0；head memcheck suite 640 checks 通過、未見記憶體錯誤，但預設 extended API 報告把 driver 記錄「Selective device code recompilation in progress」計為一個錯誤，exit 99；發生在 TensorRT session 初始化 `cuLibraryLoadData` 重新插樁 Myelin kernel library。該 attempt 另有 `build_dirty=true`，因 capture 期間修改文件；不能作 clean-source PASS。原 source／archive hash、exit codes、memcheck 與三份 CTest logs 保留於 [failed-attempt](capture/failed-attempt/)。正式 head memcheck 用 explicit API 報告，另保留預設模式逐項分類，不能忽略其他回報。
7. **受插樁程序的記憶體暴增已由使用者確認，WSL VM 當機根因仍未判定。** 開發機先以全範圍 head racecheck 執行約 19 核心、逾 16 分鐘仍在 atomicity，中止後改 ownership 組；約 11:41 WSL VM 當機重開，`last -x` 記為 crash、`/tmp` 被清空。使用者後續明確補充：當時由 Compute Sanitizer 執行的 `head_trt_tests` 程序吃爆記憶體，這是使用者觀察到的事實；目前仍無法區分工具追蹤資料的開銷與程式本身洩漏，也不能因此推導 WSL 當機的根因。主機 64 GB，WSL 上限 48 GB＋16 GB swap；主機未見 low-memory（2004）事件。主機 10:28、10:53、11:06 有 NVIDIA TDR `LiveKernelEvent 0x141`，時間對應含 S03 `scan_trt_async_fault` 的 CTest，11:41 無主機事件。[NVIDIA 的 memory footprint 說明](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memory-footprint) 記載，工具追蹤資料可能使 host allocation 失敗或 application crash；這提供一般機制，並未證明本例原因。正式 capture 保留系統 [memory samples](capture/memory.log) 與旁路 [process memory samples](capture/process-memory.log)，本輪取樣數字列於下節；它們只描述已執行範圍，不能排除洩漏或解釋先前 VM 當機。
8. **工具與舊資料的使用邊界。** 未執行 Python 程式；交接前一次 shell 命令誤加 `python3 -` 只等 stdin，未讀到程式即被停止。舊模型 source 只以 `git show` 讀 `mamba_head.py`／exporter；checkpoint 由本輪 native reader 只讀資料，設計前以 `unzip -lv`／`xxd` 查看 zip 結構。主機事件曾以 `powershell.exe Get-WinEvent` 唯讀查詢。不執行舊模型、不沿用舊 engine，也不把舊實驗結果作本輪驗收。

**第二次正式 capture（`0f5829d`）：不通過。** 此輪 CPU、ASan／UBSan、CUDA＋TRT、head 數值與所有已安排的 sanitizer／controls 均完成，沒有 head 數值失敗；26 個 mutant 中 25 個 DETECTED。TF32 mutant 的建置 guard 實際正確拒絕，但錯誤訊息由 network builder 的 `head engine build: ` 與 CLI 的 `head_trt: ` 合成前綴，分類器漏了中間前綴，將該 exit 1 記成 INVALID_RUNTIME；mutation wrapper 因此 exit 1。這是驗證分類器的錯誤，不能將整次 capture 改判 PASS。修正僅改 matcher 一行，要求完整前綴且 tactic list 含 `_tf32` token；另以所有 26 份既有 raw 診斷與 317 個正／負分類 fixtures 檢查（[script](verification/classify-mutation-fixtures.mjs)、[JSON](verification/mutation-classification-fixtures.json)）。原始 [mutation results](capture/failed-attempt2/mutations/results.tsv)、logs 與 commands 不改寫，第三次 capture 從新 source 全部重跑。

**第三次正式 capture（`198ad63`）：不通過。** CPU 13/13、ASan／UBSan 13/13、CUDA＋TRT 31/31 與 head 數值檢查完成，head memcheck 通過。接續的 S03 suite memcheck 完成 842 checks，但 default extended API 報告將唯一 `Selective device code recompilation in progress` driver note 計入 ERROR SUMMARY（1 error），raw exit 99；堆疊位於 `cuModuleLoadFatBinary`／TensorRT engine build。未見其他 sanitizer 錯誤，且 head report 為受測 SHA／`build_dirty=false`，仍須保留整次失敗。原始 [scan memcheck](capture/failed-attempt3/sanitizer-scan-trt-memcheck.log) 不改寫。`3857e24` 只將兩個 suites 的指定 sanitizer 執行統一為 explicit API 錯誤模式，並各加完整 default memcheck raw 保留與共用分類函式；沒有修改 GPU 實作。分類器另以 68 個 CPU fixtures 驗證（[script](verification/default-memcheck-fixtures.mjs)、[JSON](verification/default-memcheck-fixtures.json)）：歷史 head／scan 已知 note、clean run 可分別辨識，額外 API／memory 錯誤、crash、summary 不符及缺 success 均拒絕。第四次 capture 從新 source 全部重跑。

## 正式驗證結果

| 驗證 | 本次正式結果與證據 | 適用範圍 |
| --- | --- | --- |
| CPU build＋CTest | **13/13 PASS，exit 0**（[CPU CTest](capture/cpu-ctest.log)）。 | CUDA／TRT OFF，指定候選的 native 轉換與 reference 另執行。 |
| ASan／UBSan | **13/13 PASS，exit 0**（[CTest](capture/cpu-sanitize-ctest.log)）。 | CPU 權重交付、reference 與既有協定。 |
| CUDA＋TRT build＋CTest | **31/31 PASS，exit 0**；完整 head suite **824 checks PASS**（[CTest](capture/trt-ctest.log)、[verbose build](capture/trt-build.log)）。 | pinned SDK／本機 SM 12.0，含 S01–S03 regressions、所有程序正常退出。 |
| Native checkpoint／bundle | **PASS**：69 active tensors、bundle 40,711,552 bytes、預期 SHA 相符、全部有限值（[convert](capture/weights-convert.log)、[verify](capture/weights-verify.log)、[mapping](capture/t1-head-mapping.tsv)）。 | 指定候選；資料只讀、不執行 pickle。 |
| 獨立 CPU reference | **228 checks PASS**；scan 對 S01 max abs diff=0；候選 FP32 host 最差誤差／門檻 **0.00604**；候選 13/13 語義變體 detected，最弱 `delta_bias_after_softplus` 393,480/705,600 元素超界、最大 24 倍（[report](capture/reference-report.log)）。 | primitive 手算／bc、scan 與 S01 bit-exact、FP32 host；合成變體只作資訊。 |
| 三輸入 × 六輸出 | **2,116,800 元素、0 failures、0 nonfinite，PASS**；最大 abs **0.00051**、可報告 relative 最大 **0.00295**（近零 excluded 3,472），最大誤差／門檻 **0.00626**，位於 `asymmetric/1 cls_p4 (c=78,h=5,w=0)`：GPU `72.1010513`、reference `72.100886535339328`、tol `0.0263`（[run report](capture/head-run-report.txt)、[suite report](capture/head-suite-report.tsv)）。 | 固定權重／固定輸入；各輸入及各輸出的全部欄位均保留 raw report。 |
| 51 個中間 tensors | **全數 PASS，outside=0**；最大誤差／門檻 **0.0134**（`p3.input_proj`）（[suite report](capture/head-suite-report.tsv)）。 | 另建 debug engine 定位；不把 debug engine 當正式被驗收 engine。 |
| Engine save／新程序／重建 | **PASS**：新程序 LAZY、三輸入 rerun bit-identical、graph 三輸入對 enqueue bit-identical；A／NaN-P4／A 得 OK／FAILED(8)／SKIPPED、committed=0；重建 cmp exit 0（[run report](capture/head-run-report.txt)、[identity](capture/engine-identity.log)、[rebuild](capture/engine-rebuild-determinism.log)、[CTest](capture/trt-ctest.log)）。 | 未註冊 creator 拒絕與 checked close 亦在 CTest 正常退出。 |
| I/O／tactic contract | **PASS**：缺／多 status、duplicate、output／feature 次序、status/output role 錯置均拒絕，建置後 tactic guard 通過（[suite report](capture/head-suite-report.tsv)、[build report](capture/head-build-report.txt)）。 | 有序 I/O 名稱／role／dtype／shape／status、policy 外 tactics 的拒絕。 |
| 六輸出共同發布與 atomicity | **PASS**：11 status×3 注入、六輸出逐一 NaN、三輸入逐一 Inf 全通過（[suite report](capture/head-suite-report.tsv)）。 | 失敗與後續 SKIPPED 不發布、不推進 committed。 |
| ownership／isolation | **PASS**：slots 1／2 各 6 frames、gate 延遲 **4,000,000 cycles**，逐 frame 六輸出 digest 正確；兩 session 交錯及失敗後 reset 通過（[suite report](capture/head-suite-report.tsv)）。 | 回收 consumed 後重用 execution context／I/O slot。 |
| graph／指定 stream | **PASS**：**113 kernels＋9 memsets**、52 nodes with typed edges；重播與失敗路徑通過，enqueueV3／graph launch 均在 held stream 放行前返回（[suite report](capture/head-suite-report.tsv)）。 | 固定 addresses／shape 的 graph 與 native held-stream。 |
| API trace | **2,850 checks PASS**；三次 enqueueV3 共 **777 callbacks**、81 runtime kernel launches、27 runtime memsets，所有禁止 API／stream 檢查通過（[CUPTI trace](capture/head-api-trace.log)）。 | warmup 後 DirectHead 三次 enqueueV3；禁止同步、配置、module load、D2H，逐 call stream 正確；不含整個 session.run 的 reset／terminal 診斷。 |
| Head memcheck | **640 checks PASS，raw exit 0，0 errors**（[explicit memcheck](capture/sanitizer-head-memcheck.log)）。 | `--report-api-errors explicit`；全預設 head suite，僅排除另建 debug engine 的 intermediates；API trace 是獨立 native 程序。 |
| Head initcheck | **593 checks PASS，raw exit 0，0 errors**（[initcheck](capture/sanitizer-head-initcheck.log)）。 | `--report-api-errors explicit --no-held-stream`，其餘同預設 suite；保留實際 skips。 |
| Head 預設 memcheck 分類 | **640 checks PASS，raw exit 0、summary 0、notes 0；wrapper exit 0**（[classification](capture/sanitizer-head-memcheck-default-reporting.log)、[raw](capture/sanitizer-head-memcheck-default-raw.log)）。本次沒有重新插樁 note。 | 保留預設 API 報告；同 head memcheck 範圍，包含 held stream。分別記錄 raw exit／errors／known notes 與 wrapper exit，僅接受正確 suite success 及完全匹配的已知 note 分類。 |
| S03 預設 memcheck 分類 | **842 checks PASS，raw exit 0、summary 0、notes 0；wrapper exit 0**（[classification](capture/sanitizer-scan-trt-memcheck-default-reporting.log)、[raw](capture/sanitizer-scan-trt-memcheck-default-raw.log)）。本次沒有重新插樁 note；與第三失敗 attempt 的 raw exit 99 分開。 | S03 預設完整 suite（含 held stream；API trace／fatal fault 為 native explicit groups），保留預設 API 報告並由同一分類函式核對 scan success；不能將 head 的分類結果代替 scan。 |
| Head racecheck／synccheck | **UNEXECUTED**（[skips](capture/sanitizer-skips.txt)）。 | 未取得完整 head 的 race／sync 插樁證據，不能列 PASS。 |
| S03 suite 四種 sanitizer | **全部 raw exit 0**：memcheck／racecheck／synccheck 各 **842 checks PASS**、0 errors／0 hazards；initcheck **827 checks PASS**、0 errors（[memcheck](capture/sanitizer-scan-trt-memcheck.log)、[racecheck](capture/sanitizer-scan-trt-racecheck.log)、[synccheck](capture/sanitizer-scan-trt-synccheck.log)、[initcheck](capture/sanitizer-scan-trt-initcheck.log)）。 | 四種均 `--report-api-errors explicit`；initcheck 加 `--no-held-stream`，其他使用 S03 預設 suite；native API trace／fatal fault 另跑。 |
| 四種工具正控制組 | **四個 planted defects 均被回報，各 raw exit 99、wrapper exit 0**：memcheck **37 errors**、initcheck **32 errors**、racecheck **1 hazard（1 error）**、synccheck **2 errors**（[memcheck](capture/control-memcheck.log)、[initcheck](capture/control-initcheck.log)、[racecheck](capture/control-racecheck.log)、[synccheck](capture/control-synccheck.log)）。 | wrapper 必須確認對應工具 raw exit 99 才通過；不宣稱工具檢查覆蓋所有可能缺陷。 |
| 旁路 process memory 監測 | **完成**：283 個系統 samples、1,882 個程序 samples；最低 MemAvailable **41,588,536 KiB**，未觸發 LOW_MEMORY_STOP／TERM；monitor 已正常記錄 END（[samples](capture/process-memory.log)、[monitor script](capture/process-memory-monitor.mjs)）。各 comm 的 RSS／HWM／swap 見下節。 | 每 5 秒採樣 VmRSS／VmHWM／VmSwap、MemAvailable 等；若 MemAvailable < 8 GiB，停止本 capture 的 sanitizer 子樹並保留失敗，不放行為 PASS。只監測及中止本輪程序，不變動受測 source。 |
| Mutation sensitivity | **26/26 DETECTED**，各 targeted execution raw exit 1，且對應 expected assertion／tactic-guard regex 均匹配；全部 mutant 重新編譯成功，mutation wrapper exit 0（[capture wrapper](capture/mutation-check.log)、[results](capture/mutations/results.tsv)、[all diffs／commands／logs](capture/mutations/)）。 | 26 mutants；compile failure、未 apply、timeout、signal、無關 runtime／I/O exception 不算 detection；每項須 exit 1 與指定 assertion／guard 診斷。 |
| Manifest／backbone ordered I/O | **PASS，exit 0**：schema 2 宣告與檔案完整性 verified；schema 1 fixture 通過；backbone ordered I/O 如上表（[manifest verify](capture/manifest-verify.log)、[backbone I/O](capture/backbone-ordered-io.log)）。 | 宣告、檔案完整性與 ONNX 結構；不驗 matching 權重／backbone 推論／品質。 |
| Frozen policy、既有 kernels／plugins | **PASS，指定 diff 為空、exit 0**（[policy](capture/policy-unchanged-since-policy-commit.log)、[source](capture/s01-s02-s03-kernels-reference-plugins-unchanged.log)）。 | HEAD_POLICY／reference 自政策 commit 不變；S01–S03 指定 kernels／reference／plugin 相對 S03 baseline 不變。 |
| 歷史 evidence | **PASS，diff 為空、S03 all entries OK，exit 0**（[unchanged](capture/historical-evidence-unchanged.log)、[S03 hashes](capture/historical-evidence-hashes.log)）。 | 不改寫既有 evidence；本次實際 hash check 範圍明列 S03，其他以 diff 檢查。 |
| capture 完成與最終 build identity | **PASS**：capture 正常 exit 0，59 個 `.exit` 全為 0；head build／run／suite 與 scan report 的 SHA／`build_dirty=false` 均通過，最終工作樹 log 為空（[build identity](capture/build-source-identity.log)、[final working tree](capture/working-tree-final.log)、[binary hashes](capture/binary-hashes.log)）。 | 所有 required capture 步驟已完成；raw controls／mutation 的預期非零退出如各列所示。 |
| 文件／腳本／本輪 hashes | **PASS**：10 份文件、225 個本機目標、27 個 anchors；317 個 mutation 分類 fixtures、68 個預設 memcheck 分類 fixtures、authored whitespace 與腳本語法全部通過（[verification](verification/)）。本輪全部交付檔案的 SHA-256 見 [SHA256SUMS](SHA256SUMS)。 | 本機 links／anchors、authored whitespace、shell syntax、證據 bytes；不代表 GPU／模型驗收。 |
| 遠端 CPU CI | 交付 commit 的 CPU CI 於最終回報提供 exact commit／run URL。 | workflow 僅 CPU，無 checkpoint、不執行候選測試，不代替本機 GPU 結果。 |

## Sanitizer 範圍與未驗證邊界

本輪選擇 **不跑完整 head racecheck／synccheck**。開發期分析 TensorRT conv／GEMM 的 shared-memory 存取成本極高，且曾在試跑期間發生原因未判定的 WSL 當機；本輪保留全範圍 head memcheck、initcheck 與預設 API 模式分類，S03 suite 仍執行四種 sanitizer 及各自正控制組。S04 新增有限值／done check kernels 不使用 shared memory／block barrier；既有 packet kernel 有 block barrier，不能把整個 frame path 描述為沒有 barrier，也不能以 S03 工具結果宣稱完整 head race／sync 已驗證。native ownership／graph／atomicity tests 是另一層證據。工具實際參數與省略項目以 [sanitizer-skips](capture/sanitizer-skips.txt) 為準。

head memcheck／initcheck 與 S03 四種工具均用 `--report-api-errors explicit`；記憶體／race／sync／初始化檢查不因 API 記錄範圍而改為略過。另外，兩個 suites 各執行一次預設 API 模式的完整 memcheck，各自保留 raw log，再由同一 helper 分類。分類僅接受：raw exit 0、summary 0、known notes 0；或 raw exit 99、每一個 summary error 都對應一行完全匹配的重新插樁 note。兩者都要求相應 suite 恰有一個正確 success line；額外 sanitizer header、summary 不符、缺 success、crash／timeout 皆拒絕。default scan 的實際回報不能沿用 default head 的結果；正控制組仍使用工具預設 API 模式，逐工具檢查 planted defect 與 raw exit 99。

使用者已確認受 Compute Sanitizer 執行的 `head_trt_tests` 有記憶體暴增；本輪因此在受測 source 外加旁路 process monitor（本輪 raw MONITOR_BEGIN 記錄 PID 565562），從 capture4 CPU 階段開始每 5 秒記錄 VmRSS、VmHWM、VmSwap 與系統 MemAvailable 等欄位，原始資料見 [process-memory.log](capture/process-memory.log)，[監測 script](capture/process-memory-monitor.mjs) 本身隨 evidence 保存並納入 SHA256SUMS。若 MemAvailable 低於 **8 GiB**，監測程序停止本 capture 的 sanitizer 子樹，該次 attempt 保留為失敗，不將中止或未完成工具檢查記成通過。這是執行側觀察與中止條件，不變動 `3857e24` 的受測 source／archive，也不構成對工具追蹤開銷、程式洩漏或 WSL 當機根因的判定。監測於 2026-10-30 **04:33:16 至 04:56:47 UTC** 取樣，04:56:52 UTC 記錄 MONITOR_END。共 283 個系統 samples、1,882 個程序 samples；最低 MemAvailable 為 **41,588,536 KiB**，LOW_MEMORY_STOP 與 TERM 事件均為 0。下列為各 comm 跨程序實例的觀察最大值，並非同一時刻或程序樹的總和：

| comm | sampled RSS 最大值（KiB） | 取樣觀察到的 VmHWM 最大值（KiB） | sampled VmSwap 最大值（KiB） |
| --- | ---: | ---: | ---: |
| `head_trt_tests` | 3,132,088 | 3,165,636 | 0 |
| `scan_trt_tests` | 5,342,424 | 5,699,464 | 0 |
| `head_trt` | 2,975,480 | 3,038,280 | 0 |
| `compute-sanitiz`（Linux comm 截短名稱） | 9,004 | 9,004 | 0 |

RSS 是取樣瞬間的 resident bytes；VmHWM 是該次讀取時核心回報的程序 high-water counter。每 5 秒取樣仍可能漏掉短命程序，以及某程序最後一次取樣後至退出前的更高峰值。因此 sampled RSS／觀察到的 HWM **不等於未取樣全時段的真峰值**，也不能排除洩漏；本輪亦未執行完整 head racecheck／synccheck，不能將此資源紀錄外推至該負載。

intermediates 會另建 debug engine，排除於 sanitizer；api_trace 是獨立 native CUPTI process。initcheck 的 `--no-held-stream` 只省略 callback-held stream 測試，因插樁時 enqueue 可能等待被該 callback 擋住的工作；native 測試仍涵蓋。S03 `api_trace`／`async_fault` 是明確指定的 native groups；fatal GPU context 不重用。synccheck 對 shuffle mask 的既有限制仍適用。

## 完成回報五問

1. **指定候選權重下完整 T=1 head 是否 PASS？** **PASS**，正式 capture4 完整正常退出。PASS 限指定權重、固定介面與受測輸入範圍內的 FP32 數值、執行與失敗傳遞；head racecheck／synccheck 未執行、模型品質未驗收。
2. **六個 raw outputs 如何獨立對照？** 同一份 FP32 feature bytes 與 bundle bytes 交給獨立 binary64 C++ 運算路徑計算六輸出與逐元素 σ；不呼叫 TensorRT／CUDA／S02、不回灌 GPU 中間值，唯讀 bundle decoder／feature 產生器共用。三組輸入的全部 2,116,800 元素須滿足 `|gpu − ref| ≤ 16σ＋2⁻¹¹⁰`。primitive 手算、scan 與 S01 逐位元對照、FP32 host、13 語義變體、51 中間 tensors 與 mutation 提供不同層的檢查。σ 是含假設、事前固定的工程機率模型，不是無條件嚴格誤差界；其保守性與 TF32 數值辨識限制已明列。
3. **status 與輸入／輸出檢查如何共同控制發布？** engine I/O 嚴格核對 FrameIoSpec；9 個 plugin status＋輸入／輸出兩個 GPU 有限值檢查，由 gate 全部讀取：flags→NUMERIC，缺 done／未知 bit→PROTOCOL，再 retire。六輸出同一 slot arena 由一次 guarded publish 全部發布或全部不發布；無錯誤且已發布才推進 committed；`consumed` 在最後一次 publish 讀取後。「共同／原子發布」指單一 frame gate、整 arena publish 與同 stream commit 定義的可見語義，不是 705,600 floats 的硬體原子指令。首次數值／協定失敗 sticky、後續 SKIPPED、ResultPacket v1 不變；不承諾 TDR／VM 當機後恢復。session reset 與 terminal raw-tensor 診斷會同步／D2H，沒有以 enqueue API trace 推導整個 run 都無同步。
4. **還缺什麼？** head 候選權重已 native 交付，但以 training loss 選取、無 holdout，品質未驗；matching backbone／neck 只查舊 ONNX ordered I/O，權重歸屬、native engine 與推論未驗；影像前處理、decode／NMS 與候選容量未交付。**P01 仍 OPEN。**
5. **commits、推送、CI？** 政策 `4d38bcc`、首次受測 `222af61`（失敗）、API 報告分類 `dc016ed`、第二受測 `0f5829d`（分類器失敗）；第三受測 `198ad63`（S03 memcheck driver note 中止）；正式受測 `3857e243e8fc32663aa688f073343002ad12cb30`（capture4 全輪完成，exit 0）；本報告屬後續 evidence／docs commit，最終 SHA 於交付回報提供以避免自指。交付 commit 的推送／remote equality 與 CPU CI 於最終回報提供；本報告不預宣稱遠端 CI 已通過。

## 證據與重跑

[capture.sh](capture.sh) 要求 clean HEAD 等於指定受測 source，以 `git archive` 固定 source bytes，逐步記錄命令、raw logs 與 exit codes。capture 期間不得修改工作樹；產物寫入 ignored `build-*`。本輪附加旁路 process monitor，script 與 [process-memory.log](capture/process-memory.log) 隨 evidence 交付，5 秒取樣、MemAvailable < 8 GiB 時停止本輪 sanitizer 子樹並保留失敗；受測 source 不變。系統／程序記憶體數值與使用者觀察各自標示，不以一般工具文件或單次記憶體數值推導 WSL 當機根因。任何 capture 步驟未完成或非零退出都不能作 PASS，失敗 attempt 保留；需要修改 source 時以新的 source commit／新輸出目錄重跑。

```sh
git switch --detach 3857e243e8fc32663aa688f073343002ad12cb30
bash evidence/s04/2026-10-30/capture.sh build-s04-replay
```

重跑需要指定 checkpoint、CUDA／compute-sanitizer、pinned TensorRT SDK、實際 GPU，以及固定 hash 的 ONNX proto。實際路徑與所有命令見 capture 下 `.command`；不需 Python 或執行舊模型。

[mutation-check.sh](mutation-check.sh) 在 private source copy 單項修改並重新編譯。mutant 的編譯期 Git identity 仍來自 parent metadata，不能將其 `build_dirty=false` 視為沒有變異；實際 mutant bytes 以各 `.diff` 與 build log 為準。正式 sensitivity 判定同時核對 exit 1 與 `.command` 記錄的 expected rejection，不把 compile failure、timeout 或崩潰當成測出。

```sh
bash evidence/s04/2026-10-30/verification/run.sh
sha256sum -c evidence/s04/2026-10-30/SHA256SUMS
```
