# llama.cpp 啟發的完整 F03 加速與共用讀取試驗

2026-11-04。現行 HR 1280×768 FP16 F03 在既有 N16 特化之上，採用 **部署 shape 編譯期特化＋128 threads/block**。兩輪完整 engine ABBA 的 Graph interval 分別改善 **1.15% 與 1.64%**，六個 raw outputs 與九個 plugin statuses 在四張固定輸入上逐位元相同。這是未提交工作樹的開發證據，`formal_acceptance=false`。

將更多 states 放進每個 thread、leader load＋shuffle，以及 B／C shared copy 都已實測，因完整模型變慢而未採用。llama.cpp 的具體來源比較、適用條件與原型 review 見 [來源筆記](LLAMA_SOURCE_NOTES.md)。

## 1. 固定來源與量測範圍

- Saccade HEAD 為 `f4865cdb27f0e624b8cc26219e78f4aa938690df`，起點已有 F02 fusion、N16 特化、benchmark 與相關文件的未提交修改。開始狀態、diff 與 323 個 frozen source files 的 hashes 見 [初始狀態](capture/initial-status.txt)、[diff](capture/initial.diff)、[manifest](capture/baseline-source-sha256.json)。本次保留原有修改；目前 runtime 新增差異僅為 [scan patch](source/change-from-N16.patch)。
- llama.cpp `/home/ray/developer/src/llama.cpp` 為 `aa588a3463357ee990b314f41dda242869130704`，調查時工作樹乾淨；未修改其來源。先讀文件與普通檔案搜尋，未使用 Graphify。
- RTX 5070 Ti Laptop GPU、WSL2、driver 617.42、CUDA 13.4.92、sm_120、TensorRT 10.16.1.11。原始工具版本見 [environment](capture/environment.txt)。所有 GPU 工作使用 `~/.cache/saccade-v1/gpu.lock` 串行；未鎖 GPU clocks。
- 同一份部署 engine，沒有 rebuild：`~/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine`，SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`。
- 四張 FP32 HR 輸入：`~/.cache/saccade-v1/hr-deploy/refs/model-images.f32`，SHA-256 `0d2ad63d8d74b02858099d48b100887667d0b6040cca2bfd39d1faf7edec5611`。
- baseline／candidate 共用 CMake 建置的 host objects、TensorRT libraries 與模型工具；替換 CUDA static library 來隔離 scan 差異。[連結命令](capture/block128-link-model.command)、[binary hashes](capture/binaries-sha256.json) 與 [來源 archive](source/baseline-source.tar.gz) 保留。

**完整 F03** 是 CUDA events 包住單次完整 TensorRT engine 執行，包含 backbone／neck、head、scan plugins；排除 H2D／D2H、初始化、暖機與外部驗證，亦不含 F01／F02、producer aggregator、F04 或 tracking。計時固定 input 0；數值驗證另覆蓋四張輸入。scan API interval 包含 status clear、finite check、scan core、finalize，不是 core-only 時間。

## 2. llama.cpp 的啟發與採用項目

`ggml/src/ggml-cuda/ssm-scan.cu` 的 Mamba1 path 把 state count 放入模板、短序列按長度分派，並以128-thread blocks執行。其另一個做法是每 thread 保管全部 states，B／C 放 shared memory。這些做法各自獨立驗證，沒有直接移植 llama 的數值公式或順序加總。

採用版 [selective_scan.cu](../../../cuda/selective_scan.cu)：

1. 保留既有 `N=16` 特化；符合 `channels=256, A_rows=1, C_width=1, has_D=true` 才進一步固定這些編譯期條件。batch 與 L 仍動態；不符合的 shape 保留 generic／N16 fallback。
2. N16 scan 改為128 threads/block；HR 三層各從64 blocks×256 threads變為128 blocks×128 threads，仍是16,384 threads／512 warps。finite／finalize／expand／merge 的 block 大小不變。
3. 預載新增 deployment kernel，保留 lazy loading／指定 stream 契約。浮點運算、16-state butterfly 次序、scaled overflow fallback、flags、done、零 workspace 與 plugin ABI 均保留。

增加 block 數使工作在46個 SM間有機會分布得較均勻；這是 launch geometry 的解釋，沒有將理想 block 分配當成 achieved occupancy 的量測。

## 3. 算術與 launch 候選

以下 primitive 對照起點均為本次凍結的既有 N16 版本，每種 direct／Graph 模式三輪 ABBA、每 entry 暖機20次、量測100次。表列 HR 三尺度 scan API Graph interval。

| 候選 | baseline µs | candidate µs | 時間變化 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 4 states/thread，4 threads/channel | 767.768 | 1040.080 | +35.47% | 不採用 |
| 2 states/thread，8 threads/channel | 768.724 | 956.405 | +24.41% | 不採用 |
| 只固定部署 shape | 773.526 | 755.994 | −2.27% | 完整模型收益很小 |
| shape＋block32 | 769.251 | 723.798 | −5.91% | 與64／128接近 |
| shape＋block64 | 769.146 | 721.818 | −6.15% | 與32／128接近 |
| shape＋block128 | 768.464 | 722.351 | −6.00% | 採用；再驗證完整模型 |

三種 block 大小的差距小於2 µs，沒有宣稱128是唯一最優值。原型、逐輪 samples、全部 shape groups 與 identities 分別保存在 [source](source/) 和 [primitive](primitive/)。更多 states/thread 的4-state版本使用55 registers/thread且無 spills；既有N16及deployment kernel為32 registers/thread。減少共同量重算，沒有在本機換到更好的完整模型時間。

## 4. 完整 F03 的比較

每組外層 baseline→candidate→candidate→baseline；每個程序內 ordinary→Graph→Graph→ordinary。第一輪每 entry 暖機30次、量測100次；確認輪為40／200次。第一輪每個 implementation／mode共400個 samples，確認輪為800個。每個 fresh process 都驗證四張輸入的 ordinary／repeat／Graph bytes；各版本再互相比對六個 raw outputs 與九個 statuses。

| 候選 | baseline Graph ms | candidate Graph ms | 時間變化 |
| --- | ---: | ---: | ---: |
| 4 states/thread | 4.791572 | 4.982015 | +3.97% |
| 2 states/thread | 4.797566 | 4.905862 | +2.26% |
| 只固定部署 shape | 4.813631 | 4.803245 | −0.22% |
| shape＋block128，第一輪 | 4.810033 | 4.754482 | **−1.15%** |
| shape＋block128，加長確認 | 4.899585 | 4.819165 | **−1.64%** |

確認輪 baseline 前後為4.871997／4.927173 ms，candidate 為4.806209／4.832121 ms；存在 clocks／環境漂移，但兩個 candidate 均快於相鄰 baseline。此結果支持本機固定 engine／輸入的有限收益，不作任意輸入或平台的固定百分比承諾。完整 samples、各輪 mean／percentiles、身份與四張輸出的 hashes 見 [model](model/)；數值一致的逐檔證據見 [第一輪](capture/block128-model-exact-comparison.json) 與 [確認輪](capture/block128-confirm-model-exact-comparison.json)。彙整資料為 [summary.json](capture/summary.json)。

## 5. 共用讀取與 copy 方式

採用版不新增資料 copy：u／delta／B／C 直接從device global memory讀取；A／D在迴圈前讀入，h保留於register。相同位址的lane請求可由warp memory coalescing及cache合併，不能把原始程式的16次lane讀取算成16倍DRAM流量。

以部署版128-thread block為起點，另外只改搬運、保留算術，測兩個獨立版本：

- **raw load＋shuffle**：每16-state group由n0讀u／原始delta／rank-1 C，再用3次float shuffle傳原bits；softplus與split仍按原方式執行。沒有把「算好後廣播」混入這次比較。
- **B／C shared copy**：每block含同一direction的8個channels。前16個threads讀B[16]，thread0讀C，放68-byte shared storage；每t兩次block barrier保護讀取與覆寫。失敗channel停止算術但持續參與barriers，保留其原first-failure flags與其他channel的後續檢查。

| 搬運版本 | HR scan API Graph µs | 時間變化 | 完整 F03 Graph ms | 時間變化 |
| --- | ---: | ---: | ---: | ---: |
| raw load＋shuffle | 721.229→792.149 | +9.83% | 4.754143→4.814856 | +1.28% |
| B／C shared copy | 723.353→856.552 | +18.41% | 4.777672→4.881617 | +2.18% |

兩者的252 cases、完整status words、ordinary／repeat／Graph／changed-input／reset bits全部相同；四張完整F03輸出亦相同。兩個版本因效能退步而保留為負控制，沒有套入runtime。

### 暖機後 P3 scan core 的 NCU counters

NCU 2026.3.1，application replay、cache-control none、clock-control none，選取20次暖機後的一個L=960 scan core；grid128×block128。這是隔離單次kernel的counter，不取代上表未插樁的ABBA時間。

| kernel | global load warp requests | L1 global load sectors | L2 read sectors |
| --- | ---: | ---: | ---: |
| 採用版直接讀取 | 1,967,104 | 2,459,136 | 508,718 |
| raw load＋shuffle | **1,967,104** | **2,459,136** | 508,798 |
| B／C shared copy | 1,229,824 | 1,353,216 | 508,730 |

leader load沒有減少warp global requests或sectors，支持原同址lane讀取已合併。shared版本讓global requests減少約37.5%、L1 sectors減少約45.0%，但L2 read sectors幾乎不變，支持被移除的請求主要原本已由L1吸收。新增copy／shared讀取與barriers後，未插樁完整模型仍較慢；未進一步將成本分解為各種stall比例。

本機DRAM bytes counters為`n/a`；L2 hit-rate counter約100.05–100.90%，超出可合理解讀的比例，因此保留原值而不拿來推導DRAM流量或hit rate。原始 [baseline](ncu/memory-baseline-ncu.csv)、[shuffle](ncu/memory-shuffle-ncu.csv)、[shared](ncu/memory-shared-ncu.csv) 與 [metrics JSON](ncu/metrics.json) 可核對。第一次application replay因輸出目錄必須為新而失敗，第二次template-name filter未匹配，均保存於ncu／capture；成功版本改為每個replay程序使用新目錄及實際demangled名稱。

## 6. 檢查與保存

| 檢查 | 結果與範圍 |
| --- | --- |
| 每個獨立候選與其baseline的scan比較 | 各252 cases、每次主要比較5,517,204元素；完整FP32 bits與status words相同 |
| 每個受測完整F03 binary | 四張輸入、六outputs＋九statuses，ordinary／repeat／Graph與新舊binary均相同 |
| 採用版binary64／CUDA scan suite | 368 checks PASS |
| 採用版lazy stream-only | 21 checks PASS；含generic及N16 streams |
| 採用版memcheck／racecheck／synccheck | 各368 checks，0 errors；racecheck 0 hazards／warnings |
| 採用版initcheck | 360 checks、0 errors；依既有契約略過held-stream callback cases |
| 原始碼一致 | 工作樹scan與受測block128 snapshot逐byte相同；patch保留原N16修改 |

這些sanitizer結果限scan suite；未重新執行完整tracking、七序列品質重評、GT讀取、semantic mutations、新的clean-source gate或TensorRT內部sanitizer。既有正式證書保留各自來源；本次不改寫它們。

本機可執行檔與完整raw outputs保存在`build/f03-llama-20261104/`；evidence保存source archive、失敗候選、輸出hashes、samples、工具命令及退出碼。重跑方法見 [REPLAY.md](REPLAY.md)。
