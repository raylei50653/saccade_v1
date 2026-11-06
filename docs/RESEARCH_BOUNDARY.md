# 正式發布與獨立研究

2026-11-06。正式來源與研究分開建置；本輪整理前基線 `r0` 固定
`dacb91abd810d9349450df2725a6d4b952843687`。HR engine 仍以 SHA-256
`8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876` 固定。
engine 原 dirty build provenance 與 runtime 基線各自保存。

## 正式依賴

- `saccade_tracking_core` 提供 tracker 設定與穩定型別；CUDA tracker 直接連 core。
- `saccade_tracking_reference` 保留 stock CPU 演算法，供 offline、測試及需要的 benchmark。
- `saccade_gmc_g1` 只保留 G1 mask 與 component median reference；CUDA flow motion
  連 flow records，GPU parity tests 額外連 G1 reference。
- 正式驗收使用的 `gmc_records`、`flow_sequence`、`ofa_audit` 保留。
- `saccade_tracking_numeric_policy` 明確傳遞 CXX `-ffp-contract=off`；CUDA
  flags 仍由各正式 target 的既有政策定義。

正式 tracking header 不提供實驗 control/cost hooks。OAO、tracking events 與
九臂 GMC 實驗移至平行、獨立 Git 的
[研究入口](../../saccade_research/README.md)。需要 hooks 的實驗只在自己的固定
checkout 套用保存 patch，全部重建相依程式；純參數 relink 直接使用公開設定。

## 保存與驗證

[保存收據](../../saccade_research/docs/preservation.json)涵蓋三庫 Git bundles 與
dirty 狀態、15 個明確選定資料單位（35,950,887,105 bytes 原始檔案），每個單位
已完整解壓核對。其餘舊 results/runs、訓練 caches 與資料集留在原地並依索引列出；
這不是全部磁碟的備份，沒有刪除原始資料。缺件不以新檔案或新實驗替換。

[本輪驗證](../../saccade_research/docs/VALIDATION.md)記錄 R0/R1 CPU suites、固定
records 的完整 ResultPacket、逐欄位 committed state、MOT bytes、遷移後的實驗、
受依賴變動影響的 GPU tests 及隔離結果。這些檢查不重發完整 GPU 正式驗收，
既有 memcheck／native racecheck 的 OPEN 與 forced-sync／filtered 範圍仍依原報告。

舊 `evidence/` 保留原 bytes；其中指向當時 source/build 的連結與 dirty 身分是歷史紀錄。
已移走 source 可在 R0 的固定 Git 版本、來源 bundle 或研究遷移收據取得。

隔離檢查需全新 configure 與已建置產物：

```sh
mkdir -p build/isolation/.cmake/api/v1/query
touch build/isolation/.cmake/api/v1/query/codemodel-v2
cmake -S . -B build/isolation -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/isolation --parallel 6
python3 scripts/check_formal_isolation.py --source . --build build/isolation
python3 tests/formal_isolation_tests.py
```

檢查 target closure、編譯來源、實際 include dependencies 與未移除符號的產物。
物理上移開研究根目錄後的 configure/build 另外保存執行收據。
