# 原版 YOLO26m head 的 scan 契約與 S01 reference — 2026-10-29

**scan 契約已由原版 m head 的實作、checkpoint 容器與部署 ONNX 核實；S01 PASS（限 [SCAN_CONTRACT](../../../docs/SCAN_CONTRACT.md) §3–§5 的 `SelectiveScan` 與 Cross-Scan）。P01 與真實 artifact readiness 仍 OPEN；S02–S04 未開始。**

本輪依使用者決策以 YOLO26m backbone／neck＋原版 m T3→T1 Mamba-FPN head、單幀 T=1 為 v1 模型基線。沒有執行 Python、舊程式或舊模型；checkpoint 只以本輪撰寫的 C++ pickle 容器讀取器解析，不匯入任何類別。舊實驗成績未轉作 v1 驗收，既有 evidence 未更動。

## Source 與執行身分

| 欄位 | 本輪記錄 |
| --- | --- |
| 開始／fetch | [initial-session.txt](initial-session.txt)：`HEAD=origin/main=2aef5f8feaec6126f814cd1ed1cabf85f8994c9b`，main 無後續提交；起始工作樹乾淨（檔中的 `?? evidence/s01/` 是建立本目錄後才記錄的狀態）。 |
| 文件交付基線 | `2aef5f8`（含 `88a5c9d` 的 s／D1 查核）；P01-A 交付 `9f2c11a`、既有實作 `622a33b` 保留為歷史證據。 |
| 實作／實測 source | `97f9b8c43f4127feb4bd1056893648f9b6cfc6c5`：reference、tests、fixtures、CMake、CI 步驟名稱與 SCAN_CONTRACT（含事先固定的比較政策）。以 `git archive` 重新解出後建置，archive SHA-256 `a5b51fdcbe9e62009cbe19ad886a16f5156c7f89b83a2ed5d8f7f92fd2181401`（[archive-hash](reference/capture/archive-hash.log)、[source hashes](reference/capture/source-file-hashes.log)）。 |
| 交付身分 | 本報告、audit captures 與其餘文件更新為下一個 commit；其 SHA 與推送狀態由交付時的 Git 狀態與最終回報記錄，避免報告自含其 commit hash。 |
| 舊庫 | `/home/ray/developer/ai/saccade`，HEAD `e48f3b39`，tracked 工作樹乾淨，GitHub 公開、Apache-2.0（[legacy-identity](audit/capture/legacy-identity.log)、[metadata](audit/capture/legacy-repository-metadata.log)）。 |
| 環境 | 2026-10-29 Asia/Taipei；WSL2 Linux 6.18.33.1 x86_64、GCC 16.2.1、CMake 4.4.4、glibc 2.44、GNU bc 1.08.2、protoc／protobuf 36.1、UnZip 6.00（[audit](audit/capture/environment.log)、[reference](reference/capture/environment.log)）。 |

## 本輪新增的核實

1. **目標 head 的 source、設定與 checkpoint 對上。** 固定 `e48f3b39` 的 11 個來源檔 blob／size／SHA-256（[source-blobs](audit/capture/source-blobs.log)）與契約逐條 locator（[source-locators](audit/capture/source-locators.log)）。T1 主線 checkpoint 的 `mamba_args`、`args`、`selection` 與 108 個 tensor 的 shape／stride／SHA-256 由 native 讀取（[mamba_args](audit/capture/ckpt-t1-mamba_args.log)、[student](audit/capture/ckpt-t1-student.log)）：cross-scan、PixelShuffle、shared A＝ln 1…16、C rank 1、`legacy_n1_scan=false`。
2. **scan 精確語義來自兩份一致的實作。** PyTorch JIT 與 CUDA forward kernel 的 recurrence、softplus 門檻 20、D skip 與零初始狀態相同；差異只在 FP32 精度與 reduction 次序，列入 S02（SCAN_CONTRACT §7）。
3. **訓練期與整理期的 scan 定義相同。** `09eb1747`（T1 args 鍵首次出現）到 `e48f3b39` 之間，scan 相關定義只多出 N 驗證與預設關閉的 `_bypass_reduction`；kernel 差異為 launch 形狀、N 驗證與 D skip 位置（[functions](audit/capture/training-era-scan-functions.log)、[kernel diff](audit/capture/training-era-kernel-diff.log)）。checkpoint 未記錄訓練 revision，這不是逐位元 provenance。
4. **部署圖的算子身分與 Cross-Scan 次序。** `mamba_head_26m.onnx` 的完整 node／attribute／initializer dump（[onnx-dump](audit/capture/onnx-dump.log)）確認 3 個 `saccade::SelectiveScan`（opset `saccade` 1、6 輸入、STRING attribute `domain`）、step −1 Slice 的四方向順序、ReduceMean 合併與 DepthToSpace `CRD`。
5. **部署 ONNX 的權重來源。** 65/65 initializer 逐 byte 對上 `…_gpu_decode/best_recall.ckpt`（epoch 11），對 T1 主線與同 run `best.ckpt` 各只有 17 個凍結參數相符（[match](audit/capture/onnx-checkpoint-match.log)）。這解決舊 inventory 標為 ambiguous 的 ONNX 歸屬，但 engine bytes 仍未歸屬；plugin 的 workspace 修正 commit `9101b81d` 只比 engine mtime 晚 4 分鐘（[capture](audit/capture/plugin-workspace-fix.log)），engine 實際使用的 plugin 版本未知。
6. **S01 reference 與 fixtures。** 獨立 binary64 C++ 實作 [selective_scan.hpp](../../../include/saccade/selective_scan.hpp)；expected 由 bc closed form／明確和、手算與契約等式產生，253 項檢查（[reference report](reference/capture/reference-report.tsv)）。

## 驗證結果

| 驗證 | 結果與證據 | 適用範圍 |
| --- | --- | --- |
| 來源與 artifact audit | [capture.sh](audit/capture.sh) 全部步驟 exit 0；每步 `.command`／`.exit`／`.log` 在 [audit/capture](audit/capture/)；audit 工具與二進位 hash 見 [tool-hashes](audit/capture/tool-hashes.log)。 | 結構與 bytes 查核；沒有推論或數值比較 |
| 預設 CPU build＋CTest | Release `-O3`，CUDA／TensorRT OFF；**9/9 PASS**（[cpu-ctest](reference/capture/cpu-ctest.log)）。`selective_scan_tests` 只連結 libstdc++／libm／libgcc／libc（[ldd](reference/capture/cpu-linkage.log)）。 | 不需 CUDA、TensorRT、protobuf 或 Python |
| ASan＋UBSan | RelWithDebInfo、`-fsanitize=address,undefined -fno-sanitize-recover=all`；**9/9 PASS**（[sanitizer-ctest](reference/capture/sanitizer-ctest.log)）。 | 全部預設 CPU 測試 |
| 數值結果 | 全部案例在事先固定的容差內；最大 abs 8.72e−15、最大 rel 1.25e−14（SCAN_CONTRACT §5 的 `unit`／`relative` 皆為 1e−10）。 | reference 對契約；不是 CUDA／TensorRT 或真實模型對照 |
| expected 重產 | `derive.sh` 重跑與提交版逐 byte 相同，SHA-256 `88e8e4db…2ff2434e7`（[cmp](reference/capture/expected-reproduce.log)、[hash](reference/capture/expected-hash.log)）。 | 推導可重現 |
| fixture 敏感度 | 8 個單一語義 mutation（softplus 門檻、D skip、狀態重設、per-channel A、full C、recurrence 次序、方向翻轉、mean）全部使測試失敗（[mutation-check](reference/capture/mutation-check.log)）。 | 測試能分辨所列契約條款 |
| 共用 runtime 與既有 evidence | `2aef5f8..97f9b8c` 對既有 source、hash、設定、manifest、CUDA 與 cmake 無差異（[shared-runtime-unchanged](reference/capture/shared-runtime-unchanged.log) 為空）；既有 evidence 未變動，四組 SHA256SUMS 通過，scan-contract 中四份 repo 文件以其交付 commit `88a5c9d` 核對（[historical-evidence-hashes](reference/capture/historical-evidence-hashes.log)）。 | 不需重跑 80 份 GPU 封包 |
| 文件連結與格式 | [verification](verification/) 保存本機連結（含 anchor）、空白與腳本語法檢查。 | 只驗證本 repository 檔案 |
| GPU／TensorRT／80 sessions | **本輪未重跑**：只新增獨立 CPU reference 與文件，未觸及共用 hash、設定編碼或 runtime。 | 既有 GPU 結論維持原範圍 |
| 遠端 CPU CI | 推送後由 GitHub Actions 執行；結果在最終回報中與本機結果分開記錄。 | — |

## 重跑

```sh
bash evidence/s01/2026-10-29/audit/capture.sh build-s01-audit-replay
bash evidence/s01/2026-10-29/reference/capture.sh build-s01-reference-replay
bash evidence/s01/2026-10-29/reference/historical-hashes.sh
sha256sum -c evidence/s01/2026-10-29/SHA256SUMS
```

audit 需舊庫的 Git objects 與本機 artifacts（可用 `LEGACY_ROOT` 指定）、`gh`、curl（下載固定 ONNX schema）、protoc／protobuf、unzip；reference 需 C++20 compiler、CMake、bc。兩者都寫入新的 `build-*` 目錄，不覆蓋已封存 captures。

## 剩餘缺件與下一步

逐項見 [SCAN_CONTRACT §7](../../../docs/SCAN_CONTRACT.md#7-來源差異衝突與缺件) 與 [readiness §7.3](../../../docs/MODEL_ARTIFACT_READINESS.md#73-缺件更新)：v1 可 native 載入的 m 權重與 backbone／neck component I/O（P01／S04）、舊 engine／plugin 身分與部署精度（P01／S03）、decode 候選容量（P02／S04）、manifest schema 升版（P01），以及 S02 的 FP32 kernel 容差。新訓練的起點、cache、split 與選模依 TRAINING_WORKFLOW 另定，S01 不依賴訓練權重。

**最直接的下一步：S02，以同一契約實作 CUDA forward，先依 kernel 的 FP32 運算訂定對照本 reference 的容差。**
