# F04 adaptive Top-300：實作與本機開發驗證

日期：2026-10-31。decode 融合壓實與三路 device 排序已實作，**本機開發驗證通過**；驗證執行時 source 尚未提交，binary 明列 `source_dirty=true`，不是 clean-source 正式 F04 acceptance。政策仍為 [`f04-postprocess-fp32/1`](../../../docs/POSTPROCESS_POLICY.md)，CPU scalar reference、CPU cases、9,664-byte output ABI 與浮點算術均未改。舊 [F04 evidence](../2026-10-31/REPORT.md) 保持原樣。

## 實作與身分

`M` 是全部 8,400 個模型位置在 `score>=0.001` 後的數量，計算時尚未篩 person／退化框。decode 沿用原 qualified atomic 的回傳 index 寫入 first-512 compact keys，同時寫滿 8,400 個 full keys；不另作壓實 launch、memset、host count readback。原始 candidate ID 保留於唯一 46-bit key，atomic 到達順序不決定輸出排序。

| Device 分支 | 執行方式 |
| --- | --- |
| M=0 | 排序 kernel 直接返回，finalize 交付合法空結果 |
| 1≤M≤100 | CTA0 對 compact keys 做 shared-memory rank，只比較 M 個候選 |
| 101≤M≤512 | CTA0 以 CUB `BlockRadixSort` 排 512 keys，未用位置補 sentinel |
| M>512 | 9 個 1,024-item tiles 各取前 300；第二 kernel 對其他 sorted prefixes 做 lower_bound，計算唯一 global rank，保留前 300 |

一個全域 Top-300 候選必在自己的 tile Top-300 中；其他 tile 若有至少 300 個更小 keys，此候選必不入選。因此局部前綴足以精確決定最後 300 個，未把任意 atomic first-512 當作 dense 候選全集。key 為 `(~score_bits<<14)|original_id` 升序，與 score 降序／原始 ID 升序等價，沒有 score 量化。invalid sentinel 不參與 rank，避免重複寫入。

每次固定五個 kernels：initialize、decode、adaptive prefix、dense merge、finalize。sparse 分支的 merge 直接返回；所有 collective 分支 CTA-uniform。producer／numeric failure 的排序 kernel 不讀未初始化 scratch，原 finalize 的完整清零與失敗規則保留。normal enqueue 的 counts 留 device，不新增傳輸、配置、同步或 event。

| 項目 | 本次紀錄 |
| --- | --- |
| Git base | `a7e104131aff88c5ba21a28cc623897be4d294b6`，dirty development source |
| 最終 capture | `build-f04-adaptive-evidence-final`，`--allow-dirty`；[source](capture/source.txt)／[status](capture/capture-status.txt) |
| Source manifest digest | `0c787d7d94adeb15525c0c68a3621fb7681814cf53eca1af772464d172697dc8` |
| CUDA source SHA-256 | `daf6b1ae75c666782e23d9d84de58bd271e6c97911ac2eacf819caf6df86a4c1` |
| GPU suite SHA-256 | `1b44cafb4059de6c951c5d27fd7323141705c9e08b75a2430741b16580a6cc40` |
| Policy／reference／CPU cases | 原 SHA-256 `96f1985d…e26f7`／`71c78686…32ed4`／`74e7f380…73061`，完整值見 [frozen hashes](capture/frozen-policy-reference-cases.sha256) |
| Device scratch | **295,712 bytes、8-byte alignment**；舊版為 202,812 bytes，caller 須重新查詢／配置 |
| GPU／driver | RTX 5070 Ti Laptop、sm_120、driver 617.42、WSL2 |
| CUDA／compiler／CUB | nvcc 13.4.92、CUDA runtime／driver API 13.4、GCC 16.2.1、CUB／CCCL 3.5.0 |
| Flags | `--ftz=false --prec-div=true --prec-sqrt=true --fmad=false -lineinfo` |

[完整 source snapshot](capture/source-snapshot.tar.gz)、逐檔 manifest、編譯命令、binary hashes 與 GPU 身分保存在 [capture](capture/)。新舊 public function signatures、輸出 ABI 與 semantic policy 相同；workspace 大小／alignment 有明確變更，不宣稱舊 scratch allocation 可直接重用。

## 正確性與故障控制

| 檢查 | 結果 |
| --- | --- |
| CPU scalar reference | **442 checks PASS**，數值 reference bytes 未改 |
| CPU fixture 不變量 | **2,805 checks PASS**；CUDA 初始化前檢查固定 49 cases |
| F04 CTest | **4/4 PASS**：reference、CUDA、lazy streams、六份 replay；[log](capture/ctest.log) |
| 完整 GPU suite＋replay | **34,915 checks PASS**；[log](capture/full-replay.log) |
| 分界與 dense | M=0／1／99／100／101／299／300／301／511／512／513／8400；同分 IDs、全類別 top-k、dense 高 ID winners、tail tile、非 person 與退化框不回補 |
| 重用與身份 | 同一 output／scratch 的 **10-phase** dense→sparse→失敗→empty→radix→dense→overflow→rank→empty→dense，變更 epoch／frame，不重 poison；完整 bytes 比對 |
| 所有權／graph | repeat、graph、held enqueue、pending producer／consumer、兩個獨立 instances **PASS**；workspace+4 在實際 extent 足夠時仍明確拒絕 |
| Memcheck | **0 errors**；[log](capture/sanitizer-memcheck.log) |
| Initcheck | **0 errors**；[log](capture/sanitizer-initcheck.log) |
| Racecheck | **0 hazards（0 errors、0 warnings）**；[log](capture/sanitizer-racecheck.log) |
| Synccheck | **0 errors**；[log](capture/sanitizer-synccheck.log) |
| Sanitizer 正控制 | **4/4 DETECTED**，各 exit 99 且匹配對應 planted diagnostic |
| 語義變體 | **13/13 DETECTED**，各 exit 1 且匹配預期完整 diagnostic；[results](mutations/results.tsv) |

四種 sanitizer 各執行 **34,610 checks**，包含六份固定 F03 dumps；使用 `--no-held-stream`，其 callback gates 明列 UNEXECUTED，不擴張為 sanitizer 下完整 pending-gate coverage。正常與 lazy-stream suite 的 gates 已實跑。失敗／overflow／合法空結果分開判定；全部 header／rows／padding 與離散決策 exact，只有 score 沿用預先固定 `4*FLT_EPSILON` 界，最大實測差 **1.19209e-7**。

13 個變體包含原 NMS／class／stride／mapping／upstream／reset 錯誤，以及 packed-ID tie 反轉、person-only top-k、dense 漏末 tile、dense 誤用 first-512。最後還原 production CUDA source 並重新編譯，baseline **34,610 checks PASS**。編譯失敗、crash、timeout 或不符預期的 diagnostic 不算 DETECTED。

首次 mutation capture 的前 12 項匹配，末項 first-512 錯誤已使程式 exit 1，但先被較早的 `deterministic-dense-random` 以 score-bound diagnostic 攔下，分類器原先預期較晚的高-ID fixture，故整個 capture **exit 1／complete=false**。完整 [failed classifier attempt](development-classifier-attempt/README.md) 保留。只修正該 variant 的預期完整 diagnostic，再重跑 13 項與 restored baseline；runtime、測例與 tolerance 均未因此更動。最終 capture 另行固定更新後的 helper hash 並重跑 suite／sanitizers，不以舊 attempt 取代。

## 六份 F03 原始輸出

raw dumps 仍來自既有正式 F03 capture，每份 2,822,400 bytes，全部 hash 固定。讀取順序是 **cls_p3、cls_p4、cls_p5、reg_p3、reg_p4、reg_p5**；既有獨立歷史 kept-count guard 仍要求 **32、28、39、2、3、1**。

| 固定 input | M／完整 pool | Kept | Suppressed |
| --- | ---: | ---: | ---: |
| MOT17-02-SDP/1 | 237 | 32 | 205 |
| MOT17-10-SDP/1 | 246 | 28 | 218 |
| MOT17-13-SDP/400 | 285 | 39 | 246 |
| zeros | 7 | 2 | 5 |
| ones | 5 | 3 | 2 |
| random/1 | 3 | 1 | 2 |

這六份資料的 M 恰等於 person pool 大小，不將此偶然關係當作一般資料契約；M 的程式定義始終在 class 篩選前。六份完整 output hashes 與舊 a7e1041 版本逐份相同，包含 scores 與 padding；[replay log](capture/full-replay.log)。沒有重跑 TensorRT 或宣稱新的模型品質。

## 整合後完整 F04 的效能

工具為 [postprocess_benchmark.cpp](../../../tools/postprocess_benchmark.cpp)。**計時完整 F04 API 所排入的 initialize、decode、排序與 finalize／NMS**，輸入已 resident；載檔／hash／CPU reference／H2D／D2H 與 allocation 不在 timer 內。普通模式包含 host 驗參／提交造成的 stream gaps；graph 模式先 capture／instantiate 32 個完整呼叫，timer 量 replay，故不含 capture 時的 host 驗參。每輪之後仍檢查完整 output bytes 與 reference。

同一個工具分別連結 clean detached worktree 的 a7e1041 舊 library 與 adaptive library；順序為 **old-before → adaptive → old-after**，各 32 iterations×7 rounds。舊版 graph 有 4 kernels，新版 5；均無額外 memset／copy nodes。36 個 measured rows 全部 correctness PASS，同 input 的新舊完整 output hash 一致。[summary](performance/summary.json)、[binary hashes](performance/binaries.sha256)、[舊版前測](performance/baseline-before.csv)、[新版](performance/adaptive.csv)、[舊版後測](performance/baseline-after.csv)。

以下為 graph-batch 每段均攤 GPU 中位時間，單位 **µs**；舊版列前／後兩次中位數範圍：

| Input | 舊 F04 | Adaptive F04 | 倍率 |
| --- | ---: | ---: | ---: |
| MOT17-02-SDP/1 | 1831.36–1834.10 | 941.75 | 1.94–1.95× |
| MOT17-10-SDP/1 | 1811.88–1816.26 | 923.72 | 1.96–1.97× |
| MOT17-13-SDP/400 | 2138.25–2138.41 | 1210.66 | 1.77× |
| zeros | 932.16–932.88 | 40.20 | 23.19–23.21× |
| ones | 965.76–966.40 | 39.26 | 24.60–24.62× |
| random/1 | 929.40–929.42 | 36.65 | 25.36× |

三個真實 frame 的完整 F04 改善為 **1.77–1.97×**，沒有將合成 inputs 的約 23–25× 推廣為真實 frame、整個模型或 tracking pipeline 的倍數。完整 F04 包含未改動的 NMS／finalize，因此不能把先前 sort-only microbenchmark 的倍率當作整段加速。

這是 repeated resident inputs、固定 request 的短跑診斷；round-level p95 是七個 batch 平均值的分位數，**不是單幀 p95**。此處沒有 M>512 dense 的完整 API 效能樣本，也沒有全量 sequence／跨排程吞吐驗收；dense correctness 由獨立 cases 驗證。最終 performance binary 連結最終 capture 的 adaptive library，binary 與 library hashes 分別保存；早期成功 capture 的量測另保留於 classifier-attempt 資料夾，不以其取代本表。

## 邊界與重跑

上述 capture 執行時 source 尚未提交，未建立 commit／push；後續發布程式與證據不改變 capture 當時的 dirty 身分。clean-source 正式 acceptance **UNEXECUTED**。F03 producer 聚合器、F01／F02→F03→F04 live session、F04b、GMC／tracker、ResultPacket／MOT writer、完整 first-error latch／CS ready-consumed ring 仍依各任務補齊，P01 **OPEN**。既有 T01、nvJPEG 診斷與 head／F03 sanitizer 邊界均不改寫。

入口見 [POSTPROCESS](../../../docs/POSTPROCESS.md)：[capture](../../../scripts/run_postprocess_acceptance.sh)、[mutation controls](../../../scripts/check_postprocess_mutations.sh) 與完整 API benchmark。新增 `postprocess_benchmark` 需 CUDA postprocess build，預設 CPU build 仍不需要 SDK。
