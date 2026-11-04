# F03 scan kernel 試驗：N=16 特化與既有 CUDA Graph

2026-11-04。這是已完成的 **development optimization**，受測來源有未提交修改，`formal_acceptance=false`。scan、完整 F03 與 tracking 比較通過，candidate CTest 49/49 通過。

## 1. 已觀察結果與採用方向

保留既有 CUDA Graph，搭配 **N=16 編譯期特化的 scan kernel**。第一版 leader broadcast 較慢，不採用；第二版固定 N／log2(N)，保留原 FP32 算術、state 遞推與 butterfly 次序。

| 測項 | 同起點 baseline | N16 特化 | 時間變化 |
| --- | ---: | ---: | ---: |
| HR 三尺度 scan API pipeline，direct | 1094.65 µs | 783.964 µs | −28.38% |
| HR 三尺度 scan API pipeline，Graph | 1083.52 µs | 770.103 µs | −28.93% |
| T01 三尺度 scan API pipeline，Graph | 462.687 µs | 332.949 µs | −28.04% |
| HR 完整 F03，ordinary enqueue | 6.435416 ms | 6.072526 ms | −5.64% |
| HR 完整 F03，Graph | 5.050734 ms | 4.761608 ms | −5.72% |
| HR tracking，Graph 單幀 enqueue→receive 平均 | 7.012872 ms | 6.646921 ms | −5.22% |
| HR tracking，Graph queued wall time/frame | 5.360608 ms | 5.068107 ms | −5.46% |

scan 數字涵蓋每次 API 的 status clear、finite check、scan 與 finalize，**不是 scan core 單獨 busy time**。完整 F03 數字是 CUDA events 包住單次 engine 執行的 interval，包含 plugins 的檢查與提交空檔，排除 H2D／D2H、外部驗證、初始化及暖機。不能把這兩種時間或 pipeline 吞吐混為單幀端到端延遲。

完整模型的 ordinary 與 Graph 都有改善；Graph 下仍保有約 0.289 ms 收益。這次採用的 kernel **仍執行原有重複 softplus／split(u)**，不把收益歸因於第一版的共同量去重。

## 2. 固定來源、模型與環境

- HEAD：`f4865cdb27f0e624b8cc26219e78f4aa938690df`，開發工作樹。開始時已有 F02 tile fusion 與相關文件／工具修改，見 [初始狀態](capture/initial-status.txt)。本次沒有回復或覆蓋那些修改。
- 完整模型／tracking baseline 為開始時目前工作樹的獨立 source snapshot；candidate 具有同一份 F02，其 runtime 差異只有 scan kernel。baseline 的新增 `model_benchmark` 與 CMake target 只是共用量測工具。完整 baseline snapshot manifest 見 [source-baseline-final](capture/source-baseline-final.json)。
- 獨立 scan harness 以 HEAD 的 scan source 為 baseline，與 frozen candidate source 共同連結至同一程序；這些 scan source 在試驗開始時等同 HEAD。兩版均使用相同 Release FP32 arithmetic flags。
- GPU：RTX 5070 Ti Laptop GPU，WSL2，driver 617.42；CUDA 13.4.92，sm_120；TensorRT 10.16.1.11。
- 所有 GPU 工作經 `~/.cache/saccade-v1/gpu.lock` 串行執行；沒有鎖定 GPU clocks。結果限本機、固定輸入與本次 ABBA 條件。
- 部署 engine 沒有 rebuild：`model-hr-fp16.engine`，SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`。
- 四張 HR 固定輸入：`~/.cache/saccade-v1/hr-deploy/refs/model-images.f32`，SHA-256 `0d2ad63d8d74b02858099d48b100887667d0b6040cca2bfd39d1faf7edec5611`。
- [來源修改](source.diff)、[最終 source hashes](source-sha256.json)、[binary hashes](capture/binaries.sha256) 分別記錄；未以 commit hash 代替未提交 source 身分。

## 3. 兩個 kernel 試驗

### Trial 1：leader broadcast，不採用

每個 N16 state group 的 leader 計算 `softplus_scaled(delta)` 與 `split(u)`，透過四次 shuffle 廣播 mantissa／exponent。其他 recurrence 與 failure checks 保持原式。

精確比較通過，但 HR scan direct 1091.85→1193.84 µs（+9.34%），Graph 1160.36→1216.22 µs（+4.81%）。完整 F03 ordinary 6.484729→6.569776 ms（+1.31%），Graph 5.148380→5.146118 ms（−0.04%，無可採用收益；baseline 前後有明顯漂移）。原始來源、資料與編譯紀錄保留在 [broadcast trial](trials/broadcast/benchmark/benchmark.json) 及 [完整模型比較](capture/model-trial1-comparison.json)。

減少 active lanes 重複算術並不保證減少 SIMT 執行時間；此版本新增的分支、shuffle／依賴鏈是可疑成本，但本次沒有以硬體 counters 對其逐項歸因。

### Trial 2：N16 編譯期特化，採用

`scan_kernel<true>` 固定 `N=16`、`log2_state=4`，讓 state 索引、mask 與四級 butterfly 可以在編譯時決定。host 僅在已驗證 `state==16` 時分派；其他 N 仍執行 generic kernel。

- 原 `softplus_scaled`、scaled drive、FMA、term、固定加總樹、range checks 與失敗時整體 poison 保留。
- 預載 generic 與 N16 kernels，避免 lazy loading 首次載入造成隱式同步。
- 沒有新增 workspace、global state history、跨幀狀態或 kernel launches；API／plugin ABI、Graph 節點數與部署模型不變。
- CUDA API 回報 generic 34 registers/thread，N16 32 registers/thread；兩者 local／shared bytes 均為 0。這是資源紀錄，不單獨用作加速原因證明。

見 [N16 原始結果](trials/n16/benchmark/benchmark.json)、[逐輪 timings](trials/n16/benchmark/timings.csv) 與 [完整 F03 比較](capture/model-trial2-comparison.json)。

## 4. 比較方法與數值結果

### Scan

同程序雙重連結新舊 kernel，逐一比較完整 status words 與每個 FP32 output bits；ordinary、repeat、Graph、changed inputs、同 buffers 的 A→失敗 B→A 都有覆蓋。

252 cases，每輪主要比較合計 **5,517,204 output elements**，全部相同。包含 N=1／2／4／8／16／32、shared／per-channel A、rank-1／full C、D 有無、L=0／1／tail、HR L=960／240／60、T01 L=400／100／25、softplus 分支及相鄰 float、scaled delta／drive、六個輸入的非有限值與 range controls。[驗證 JSON](trials/n16/verify/verification.json)、[逐案例紀錄](trials/n16/verify/verification.csv)。

每個 timing group、每種 direct／Graph 模式使用四輪 ABBA，每個 entry 暖機 20 次、量測 100 次。metric 是多次 replay 的 event interval 除以 replay 次數。

### 完整 F03

外層 baseline→candidate→candidate→baseline；每個程序內 ordinary→Graph→Graph→ordinary，各 entry 暖機 20 次、量測 80 次。每種 implementation／mode 合計 320 個單次 event samples。timing 固定使用第 0 張輸入；數值比較另外覆蓋四張輸入。

全部六個 raw outputs、九個 plugin statuses，四張輸入在新舊版本、repeat、ordinary／Graph 間 **逐位元相同**；所有輸出有限，statuses 成功。具體 output hashes、每輪摘要、均值與 pooled percentiles 見 [完整比較](capture/model-trial2-comparison.json)。原始 outputs 與 statuses 保留在本機 `build/f03-scan-20261104/model-trial2-*/`；本 evidence 保存 hashes、驗證表與測時 CSV。

### 完整 tracking

MOT17-10-SDP 的固定前 140 張 JPEG，20 張暖機＋120 張測時，Graph 模式。外層 baseline→candidate→candidate→baseline；每個 binary 的 fresh sessions 各有兩次 single 與兩次 queued，共 **16 sessions**。預載 JPEG，排除磁碟讀檔、初始化與輸出檔案寫入。每種 implementation 的 single 合計 480 個樣本；queued 合計四個 run 的吞吐 interval。

所有 sessions 的 140 幀完整固定容量 packet bytes 在 single／queued、新舊 binary 與重跑之間全部相同，SHA-256 `35ec34474f692e5e398790d5d81edbff3f94bf3bed431695d8b33714153775a9`。single p50 6.968079→6.610496 ms，p95 7.432971→6.942817 ms。queued 的 5.068107 ms/frame 是吞吐換算，**不是 queued 單幀交付延遲**。見 [完整 tracking 比較](capture/tracking-trial2-comparison.json) 與 [逐輪資料](tracking/tracking-trial2-0-baseline/runs.csv)。

## 5. 檢查狀態

| 檢查 | 結果與範圍 |
| --- | --- |
| 原 binary64／CUDA scan suite | candidate 368 checks PASS |
| Lazy stream-only test | N8 與 N16 held streams、N16 Graph 與併行 generic／N16；21 checks PASS |
| memcheck | scan suite，368 checks；0 errors |
| initcheck | scan suite，360 checks；0 errors；依既有契約略過 held-stream callback cases |
| racecheck | scan suite，368 checks；0 hazards／errors／warnings |
| synccheck | scan suite，368 checks；0 errors；既有文件的 shuffle-mask 工具限制照舊 |
| Candidate CTest | 此新 build 已配置的 49/49 tests PASS，234.22 秒；含 scan／head TRT、HR profile、preprocess、producer、tracking primitives；未配置的七序列／optional reference suites 不列為本次執行 |
| 完整 tracking packet／timing 比較 | 16 fresh sessions，同一段 140 幀全部完整 packets 相同；single／queued 測時見 §4 |
| Strict build／script | C++ warnings-as-errors 建置、`bash -n scripts/run_scan_optimization.sh`、`git diff --check` 通過 |

這些 scan sanitizer 結果沒有擴張為 TensorRT 內部、完整 HR session 或 nvJPEG 的 sanitizer 驗收。沒有執行新的 clean-source 正式 gate、semantic mutations、七序列品質重評或讀取 GT。

## 6. 重跑與保存

```sh
# 新輸出目錄；保留編譯前 source snapshots、命令、exit status、CSV 及 hashes。
bash scripts/run_scan_optimization.sh build/scan-replay-NEW \
  f4865cdb27f0e624b8cc26219e78f4aa938690df

# 完整模型，使用 CMake MODEL_TRT target 所提供的工具；使用固定部署 engine。
flock ~/.cache/saccade-v1/gpu.lock build/YOUR-BUILD/model_benchmark \
  ~/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine \
  ~/.cache/saccade-v1/hr-deploy/refs/model-images.f32 \
  build/model-replay-NEW --profile hr-1280x768 \
  --iterations 80 --warmup 20 --rounds 2
```

完整 source snapshot、trial binaries、raw outputs 與 packet bytes 保留在本機 `build/f03-scan-20261104/`。兩份獨立 scan capture 保留在 `build/scan-broadcast-trial1-20261104/` 與 `build/scan-specialize-trial2-20261104/`。本 evidence 中的 `capture/run_model_abba.py`、`compare_models.py`、`run_tracking_abba.py` 記錄此次跨 binary replay 的具體命令與比較方式。

同目錄 `SHA256SUMS` 覆蓋本 evidence 的來源、命令、logs、tables 與報告；以 `sha256sum -c SHA256SUMS` 核對。受測 binaries 另保存在本機 baseline／trial2 目錄，未因後續 CTest 重新 build engine 而改寫。

初始 harness build 因 `++seed` 與同一呼叫讀取 seed 的 warnings-as-errors 失敗；修正為分開敘述後才執行 GPU 工作。[失敗紀錄](failed-attempts/initial-harness-build/compile-harness.log) 保留。Trial 1 的 scaled-drive fixture 沒有真正造成中間乘積溢位；Trial 2 已修正為 Δ=2¹⁰⁰、B=2¹⁰⁰、u=2⁻¹⁰⁰，並重新完整比較，不把舊 fixture 當成該控制已覆蓋。
