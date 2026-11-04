# 完整 F03 的 backbone 與 head 瓶頸分析

2026-11-04。現行 HR 1280×768 FP16 F03 的主要剩餘耗時是 **backbone／neck，占 kernel duration 加總的66.12%**；head占33.88%。按工作類型，convolution／GEMM占63.73%，scan core占15.63%。因此下一個主要研究目標是TensorRT的卷積／矩陣乘法，而不是繼續只改scan資料共用。

本輪已測原engine的NVTX／persistent cache，以及重新build的optimization level、workspace、tiling及causal padding。**沒有找到可套用到現行部署的穩定加速**：cache初輪約快0.7%，加長確認輪反而慢1.20%；causal padding相對重建control快1.51%，但直接對現行engine只有0.014%差異，且輸出bits改變。本輪保留現行engine與runtime設定，新增可重跑profiling工具及開發證據，`formal_acceptance=false`。

## 1. 實際 GPU 工作的分布

完整F03的邊界是GPU上的`images`輸入到六個raw outputs及九個plugin statuses；不含H2D／D2H、F01／F02、producer aggregator、F04或tracking。計時固定input0，數值比對另覆蓋四張固定輸入。

使用Nsight Systems只擷取20次暖機後的原CUDA Graph重播。6040個kernel calls、302個不同kernel nodes，均完整對回engine inspector的286個layers；九個scan／Cross-Scan plugins各有三個kernels，兩個cache-policy control layers不產生kernel。所有nodes各出現20次，未知分類0。[完整對映](trace/kernel-layer-map.csv)、[摘要及對映驗證](trace/kernel-summary.json)、[SQLite](trace/f03-graph.sqlite)與[原始trace](trace/f03-graph.nsys-rep)均保存。

| 區域 | 每次重播的 kernel duration 加總 | 占比 |
| --- | ---: | ---: |
| Backbone／neck，含model input cast | **3.0456 ms** | **66.12%** |
| P3 head | 1.1044 ms | 23.98% |
| P4 head | 0.3334 ms | 7.24% |
| P5 head | 0.1227 ms | 2.66% |
| 合計 | **4.6060 ms** | 100% |

| 工作類型 | duration 加總 | 占比 |
| --- | ---: | ---: |
| Convolution／GEMM | **2.9354 ms** | **63.73%** |
| Layout／cast／SiLU／pooling／softmax等 | 0.8764 ms | 19.03% |
| Scan core | 0.7198 ms | 15.63% |
| Cross-Scan expand／merge及其檢查 | 0.0466 ms | 1.01% |
| Scan finite／finalize檢查 | 0.0277 ms | 0.60% |

上表的layout桶不是全部copy，convolution桶也不包含以xmma命名的pooling。對映使用tactic名稱及序列核對，將`_by_fusion_tactic`對應到實際`_fused` kernel；區域使用ONNX metadata、P3／P4／P5名字及tensor producer關係判斷。[分析程式](replay/analyze_trace.py)可從原SQLite重算。

這是kernel durations的加總，不是完整frame延遲。trace span÷20為4.7302 ms，是重播processing interval；另有180次status memset，平均合計約2.33 µs／重播。單stream timestamps有少量相鄰區間重疊，最大128 ns；摘要保留原值，沒有由此宣稱實際並行執行。

## 2. 應優先處理的 layers

| 實際工作 | 每次重播平均 |
| --- | ---: |
| P3 scan整個plugin，含finite／finalize | 566.11 µs，其中core 547.00 µs |
| Backbone `/layers.3/conv/Conv` | **220.92 µs** |
| P3已融合的`reg.conv0 + cls.conv0` | **209.47 µs** |
| Backbone `/layers.5/conv/Conv` | **208.06 µs** |
| Backbone `/layers.1/conv/Conv` | 133.65 µs |
| Backbone `/layers.2/cv2/conv/Conv` | 120.52 µs |
| Neck `/layers.16/cv1/conv/Conv` | 98.27 µs |
| Backbone兩個大型SiLU | 91.08＋85.21 µs |
| P3 upsample | 59.24 µs |
| P3 in-projection後的transpose／slice | 50.40 µs |

P3 cls／reg第一個3×3 convolution已融合，不能再把合併這兩分支當作新收益。head的Conv1d causal slice、reshape、transpose、SiLU及cast也已有融合，gate的SiLU／multiply與residual add／cast同樣不是全部分離的kernels。完整inspector包含於[現行engine資訊](model/graph-layer-profile/engine-layers.json)。

兩個大型backbone SiLU各有約30 MiB的logical activation，合計耗時176.30 µs，約占本trace kernel sum的3.83%。Conv＋SiLU epilogue是有明確上限的後續fusion候選，但本輪未實作；融合仍需保留原卷積輸出先捨入FP16、再進activation的語義，不能直接假設FP32 accumulator上做SiLU會有相同bits。

## 3. 同一份 engine 的設定試驗

所有設定在warmup及capture之前套用，並讀回actual值。每組外層baseline→candidate→candidate→baseline，每個程序內仍是ordinary→Graph→Graph→ordinary。初輪每entry暖機30次、測80次，確認輪為40／200次。四張輸入的六raw outputs與九plugin statuses在新舊設定、repeat及Graph間逐位元相同。

| 設定 | baseline Graph ms | candidate Graph ms | 時間變化 |
| --- | ---: | ---: | ---: |
| NVTX detailed／engine default改為none | 4.806812 | 4.817620 | +0.225% |
| Persistent cache 0→4 MiB | 4.828119 | 4.791479 | −0.759% |
| Persistent cache 0→8 MiB | 4.828680 | 4.792955 | −0.740% |
| **4 MiB加長確認** | **4.797633** | **4.855034** | **+1.196%** |

Cache確認輪的方向反轉，因此沒有將4／8 MiB設成runtime預設。NVTX none也沒有觀察到收益，ordinary enqueue平均只由6.175713變成6.160670 ms。[逐組比較](capture/comparison-summary.json)、[samples／identities](model/)與四個`*-exact.json`保存完整bits比對。

`setPersistentCacheLimit`是execution-context的activation cache設定；它不同於下一節需要rebuild的builder tiling。setter及getter成功不單獨證明此architecture有性能收益，本輪以完整F03實測判定。

## 4. 重新 build 的 backbone 與 head 候選

使用相同ONNX、head weights、TF32 off、FP16旗標、FP32 model I/O與scan邊界、aux streams 0。所有engine只寫新build目錄，不覆寫部署檔。

| 隔離候選 | 相對原builder的變化 | 初篩 Graph ms |
| --- | --- | ---: |
| Rebuild control | 原opt3／256 MiB workspace／tiling NONE | 4.729229 |
| Optimization level 5 | 只將FP16 opt3→5 | 4.835168 |
| Workspace 1 GiB | 只放寬FP16 workspace上限 | 4.766223 |
| Tiling FAST | 只啟用builder on-chip caching strategy | 4.814943 |
| Causal padding | Conv1d只做left padding 3、right padding 0，直接產生L | 4.715505 |

這些初篩是不同程序的個別量測，不能作為已確認的paired收益。前三個設定沒有提供足以採用的新收益；causal padding另做較長的外層ABBA：

| 直接比較 | baseline Graph ms | candidate Graph ms | 時間變化 |
| --- | ---: | ---: | ---: |
| 現行engine→相同設定重建control | 4.829399 | 4.852168 | +0.471% |
| 重建control→causal padding | 4.906002 | 4.832028 | **−1.508%** |
| **現行engine→causal padding** | **4.772548** | **4.771886** | **−0.014%** |

每entry暖機40次、測160次；每個implementation／mode合計640個samples。causal相對現行engine只差0.662 µs，沒有可採用的收益。它與重建control相比，81個可用ONNX metadata配對的backbone convolutions中有60個tactic改變，[具體差異](capture/causal-backbone-tactic-changes.json)已保存。因此不能將對重建control的1.51%歸因於head少產生三個padding outputs。

候選來源在[source-variants](source/source-variants/)，FP32原路徑均保持原設定；各build report記錄actual workspace、optimization level、tiling與預設L2 tiling limit。timing cache未匯入／匯出；TensorRT仍有建置期間的local timing cache，這不是完全沒有cache。

## 5. FP16 數值與採用邊界

同設定重新build，也不保證同一組tactics或相同FP32 raw output bits。下表每個數值是**相對現行部署engine**，在四張輸入、六個outputs中最大的per-tensor relative L2；不是相對binary64 reference的模型誤差或品質分數。

| 新engine | 最大per-tensor relative L2 | 對應tensor |
| --- | ---: | --- |
| Rebuild control | 0.004899 | reg_p3 |
| Opt5 | 0.005730 | reg_p3 |
| Workspace 1 GiB | 0.004804 | reg_p3 |
| Tiling FAST | 0.007850 | reg_p3 |
| Causal padding | 0.007801 | reg_p3 |

五個新engine均有raw bits差異，不能直接替換現行engine。各engine自身的ordinary／repeat／Graph則逐位元相同，輸出有限、statuses成功；九個statuses也與現行engine相同。[全部120筆逐tensor比較](capture/numeric-differences.csv)保留bit differences、max abs與relative L2。CSV中`engine=baseline`指**新重建control**，其reference是**現行部署engine**。

本輪沒有讀GT、重評品質、重新選模型，亦未將同權重／同FP16旗標視為已通過數值等價。由於同engine的cache收益未重現，而新engine沒有超過現行版本的可信收益，本輪只交付工具及證據，保留上一輪的runtime與engine。

## 6. 工具、來源與驗證

擴充[model_benchmark.cpp](../../../tools/model_benchmark.cpp)：

- `--persistent-cache-bytes`、`--nvtx engine|none|detailed`，記錄requested及actual值；預設仍為cache 0／engine NVTX設定。
- `--layer-iterations`在原測時完成後，另capture啟用IProfiler的Graph，寫`layers.csv`、inspector與`layer-profile-summary.json`。
- `--trace-iterations`在原測時之後、IProfiler之前，用CUDA Profiler API包住原Graph重播，供Nsight擷取真正kernel時間。

Trace與layer profiling都不混入原ABBA samples。IProfiler插樁後Graph平均10.0694 ms、layer means加總9.1532 ms，未插樁的同次Graph平均4.7624 ms；這些插樁時間沒有拿來計算§1占比。§1完全使用正常Graph的Nsight kernel timestamps。

第一版profiler嘗試`setProfiler(nullptr)`被本SDK拒絕，原始失敗保留於`layer-profile`及`screen-baseline`；已修成profiler壽命超過context，不再傳nullptr。成功的`graph-layer-profile`、`screen2-*`及所有ABBA另存，不將失敗紀錄改成PASS。最終工具來源見[source/model_benchmark.cpp](source/model_benchmark.cpp)，包含Graph RAII與noexcept callback錯誤處理。

- 起點HEAD `f4865cdb27f0e624b8cc26219e78f4aa938690df`，已有前幾輪未提交修改；[初始狀態](capture/initial-status.txt)、[diff](capture/initial.diff)、[321檔source manifest](capture/source-baseline-sha256.json)與[source archive](source/baseline-source.tar.gz)保留。
- 本輪CUDA scan使用前一輪已測的shape特化／128-thread block library，沒有再修改scan kernel；host libraries共用固定CMake objects，以隔離builder／context設定。所有[命令及退出碼](capture/)均保存。
- 現行engine SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`；四張輸入 SHA-256 `0d2ad63d8d74b02858099d48b100887667d0b6040cca2bfd39d1faf7edec5611`。ONNX／bundle hashes見[artifact inputs](capture/artifact-inputs.json)，新engine及binary hashes見[manifest](capture/binaries-and-engines-sha256.json)。
- RTX 5070 Ti Laptop GPU、WSL2、CUDA 13.4.92、TensorRT 10.16.1.11；工具版本見[environment](capture/environment.txt)。GPU工作均持有既有gpu.lock，未鎖clocks。無法把小幅、方向反轉的測時當成穩定收益。
- 最終benchmark以C++20及warnings-as-errors編譯；逐層Graph、Nsight trace、四張raw／status驗證與SQLite完整對映均通過。runtime未改動，本輪未另跑新CTest、sanitizer、tracking session、semantic mutations或clean-source正式gate，不改寫既有證書。

完整engine、binaries及raw outputs留在`build/f03-backbone-head-20261104/`。重跑方法見[REPLAY.md](REPLAY.md)。

## 7. 尚待研究的主要方向

最有意義的下一步，是針對`/layers.3/conv`、`/layers.5/conv`、P3 fused cls／reg convolution及兩個大型SiLU做tactic或epilogue的受控試驗。先固定並保存可重用timing cache，避免每次改head時連backbone也大幅重新選tactics，才能歸因收益。任何新卷積實作、fusion或重建engine，仍需逐tensor數值比較；本輪沒有證明存在可保留現行bits的大幅加速。
