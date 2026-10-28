# P01-A 模型輸入契約與 native SDK 驗收 — 2026-10-28

**Manifest 工具 PASS、native TensorRT 小網路路徑 PASS；目標真實 artifact readiness OPEN，整個 P01 與 S01 保持 OPEN。**

開始前 `git fetch origin` 後確認 `HEAD=origin/main=437c99edbae1a34854aa6966dc210b5053069929`，無後續提交、工作目錄乾淨。正式驗收 source commit 為 **`622a33b8bd2a1a593fcfbc8a12f6b6e30e770d8a`**；從乾淨 source 重新 configure/build。封存內 source-status 為空，80 份合成 GPU identity 及 native execution reports 均對應此 commit、`build_dirty=false`。後續 evidence commit 不變更實作。

## 三項獨立結果

| 驗收項目 | 結果 | 證據與限制 |
| --- | --- | --- |
| Manifest 工具 | **PASS** | 純 C++ 嚴格 ASCII JSON schema 1；75 項 unit checks、CLI 正／負案例、獨立 canonical hash oracle、ASan／UBSan 全通過。宣告與檔案驗證成功不升格成內容核對／readiness。 |
| Native SDK 路徑 | **PASS，限此實機／小網路** | TensorRT 10.16.1.11、fixed FP32 `[1,1,1,256]` affine：build → serialize → checked save/read → deserialize → 新 user-managed context → explicit nonblocking stream enqueue → 同步讀回全部輸出，256/256 exact，max abs error 0，清理成功；memcheck 0 errors。 |
| 真實 artifact readiness | **OPEN／部分查核** | 全新官方一般 YOLO26s ONNX 的來源、size/hash 已確認，native inspector 核對 I/O／圖 inventory；尚無新 Mamba-FPN head、公開 split interface 或其精確 scan 契約。不能以工具／SDK 成功取代。 |

契約／實作導覽：[MODEL_MANIFEST](../../../docs/MODEL_MANIFEST.md)、[NATIVE_TENSORRT](../../../docs/NATIVE_TENSORRT.md)、[MODEL_ARTIFACT_READINESS](../../../docs/MODEL_ARTIFACT_READINESS.md)、[MODEL_FEASIBILITY](../../../docs/MODEL_FEASIBILITY.md)。

## 實際測試與回歸

| 執行 | 結果 |
| --- | --- |
| CPU-only | [8/8 CTests](acceptance/cpu-ctest.log)，含既有 protocol／packet／config／checked I/O／cleanup 與新增 manifest unit／CLI。Manifest 75 checks；既有 config 88、I/O 23、cleanup 79 checks。 |
| Parser sanitizer | [ASan＋UBSan，75/75 checks](acceptance/manifest-sanitizer.log)，exit 0，無 sanitizer 診斷。 |
| CUDA synthetic | [11/11 CTests](acceptance/cuda-ctest.log)；原有完整有限矩陣 80 sessions／43,782 checks，cleanup integration 7 sessions／91 checks。 |
| 舊驗收回歸 | [80/80 完整 packet streams 逐 bytes 相同](acceptance/packet-regression.log)，全部 config／behavior hashes 亦相同；原 closeout 外層 SHA256SUMS 完整通過，未改寫舊 evidence。共用 hash 函式只原樣移出，synthetic 編碼不變。 |
| TensorRT build | [10/10 CTests](acceptance/trt-ctest.log)，含 8 CPU tests＋native roundtrip／CLI I/O。負案例驗證同路徑 alias、usage、缺失 parent、engine/report/stdout 的 `/dev/full` 錯誤。 |
| SDK 拒絕 | 缺失 SDK root、獨立 private fixture 的錯誤 header minor version，均 configure exit 1，具有指定診斷；沒有 fallback 或 skip。原下載 SDK 未修改。 |
| 獨立 native run | [exit 0／完整報告](acceptance/native-report.txt)，含所有 256 項 input／expected／actual、engine inspector JSON／實際 tactic、載入 libraries 及 cleanup。 |
| Native GPU sanitizer | [compute-sanitizer memcheck](acceptance/native-memcheck.log)，exit 0，完整 affine 流程完成，`ERROR SUMMARY: 0 errors`；沒有停用檢查。 |
| 真實候選 manifest | [宣告＋實際檔案驗證 exit 0](acceptance/candidate-verify.log)，仍回報 `model_content=not_checked`、`real_model_ready=false`。外部結構查核另見 artifact 文件。 |

本次修改共用 hash，因此重跑受影響的完整合成 CUDA 回歸；沒有修改 synthetic kernels。新增 native 路徑執行 memcheck，未聲稱對其執行 initcheck／racecheck／synccheck。舊骨架的四工具結果仍只屬於原驗收範圍。沒有執行真實 YOLO／Mamba 推論、scan、custom plugin、F03、tracker 或效能評估。

## 身分、精度與 hashes

- SDK：新下載 NVIDIA native `10.16.1.11-1+cuda13.2` headers／runtime，原始 package rows、實際 package hashes、授權宣告見 [SDK 來源紀錄](native-exploratory/README.md)。未用 Python 或舊專案。
- 實際載入：獨立 SDK root 的 `libnvinfer.so.10.16.1`、`libnvinfer_builder_resource_sm120.so.10.16.1`；`dladdr` 核對 primary library 必須等於 configure 路徑，`/proc/self/maps` 記錄實際 library 清單。[Native libraries／headers hashes](acceptance/sdk-native-binaries.sha256)。
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，UUID `GPU-825ae551-ba16-539a-e444-22726b59ee9a`，SM 12.0；driver 617.42，CUDA runtime／driver API 13040，native libcudart 13.4.92。
- Toolchain：GNU C++ 16.2.1 20260810、CMake 4.4.4、CUDA compiler 13.4.92、WSL2 kernel 6.18.33.1。[完整環境](acceptance/environment.log)。Arch／GCC16 不在 NVIDIA 所列的測試平台，結論限定本機實測。
- 執行前固定：strongly typed FP32、TF32=false、optimization level 0、builder workspace limit 16 MiB、aux streams 0、沒有外部 timing cache import/export；`x[i]=(i-128)/8`、獨立 C++ `2*x+1`、atol=rtol=0。Engine 要求的 execution workspace 為 0，程式明確處理零需求；I/O device/pinned buffers 在 enqueue 前配置。
- Engine 12,356 bytes，SHA-256 **`e59b00754e82853bbe1ef49ea903e111ba042b99d32d014a05ac191dcdf1a65f`**。Inspector JSON hash `f9130c5a536da1d615cb023446d3bf5a244bb566e07f8103c85a2e9f1a3d8e51`，實際 fused tactic `__myl_MulAdd_0x3f242f4e15f28f48dfccb2005c35cfc4`。這些是此產物身分，不承諾跨環境 engine bytes 重現。
- Parser fixture canonical hash **`97d83da29be2693ba86a210a10c8a5454de8a135b5b31985feb3aee07152b0b7`**，payload 為 `abc`，不是模型。
- 真實一般 YOLO26s ONNX：38,291,130 bytes，SHA-256 **`d26b65c432111eb95798cd2320603d4d75627605dbec6c6b7f98c499a80e7321`**；候選 manifest canonical hash **`999f7a5ba64fa05d34d2e8e978a610e65c78142f9a1754a060dc8d7f9f3da63c`**。此候選明列 upstream_detector／head=null／scan unverified。

## 原始紀錄與重跑

[raw-runs.tar.gz](acceptance/raw-runs.tar.gz) 保存命令／exit／stdout-stderr、三種 build 的 CTest logs、CMake 設定／native link flags、80 個完整 GPU sessions 的 packets/config/hash/identity/帳本、CLI fixtures 的普通檔案、native engine／完整讀回報告、memcheck 與獨立 artifact inspector 重跑紀錄。跳過 symlinks（包含 `/dev/full`）及大型上游權重／SDK binaries；權重保存在新 cache，重跑可重新下載。內附 `raw-evidence.sha256` 逐檔驗證；[外層 SHA256SUMS](SHA256SUMS) 覆蓋本次所有 evidence 普通檔案。

[commands-summary.log](acceptance/commands-summary.log) 列出整體 exit；[binaries.log](acceptance/binaries.log) 記錄實際工具 binary hashes。正式 native／GPU 結論只引用上述乾淨 source run；[開發過程](native-exploratory/README.md) 保留第一輪 SDK builder resource 載入失敗。驗收腳本最初的錯誤版本 fixture 未改中 alias 巨集，configure exit 0 被 harness 判為失敗，未列為拒絕成功；原記錄在封存 `development/`，修正後的正式 `wrong-sdk.exit=1` 與診斷另行保存。

```sh
# 從 source commit 或後續 evidence commit 都可跑；後者 build_commit 會如實不同。
scripts/run_p01_acceptance.sh \
  /home/ray/.cache/saccade-native-trt/10.16.1.11/sdk build-p01-rerun

sha256sum -c evidence/p01-a/2026-10-28/SHA256SUMS
mkdir -p build-p01-archive-check
tar -xzf evidence/p01-a/2026-10-28/acceptance/raw-runs.tar.gz -C build-p01-archive-check
(cd build-p01-archive-check && sha256sum -c raw-evidence.sha256)
```

SDK 首次準備見 [native SDK 文件](../../../docs/NATIVE_TENSORRT.md#重跑)。完整 harness 需要 C++20、CMake、curl、native protobuf/protoc/pkg-config、CUDA Toolkit／cuFFT、指定 native TensorRT SDK 及可用 GPU；default CPU-only build 不需要這些 GPU／protobuf 依賴。

## S01 接續輸入

現有一般 YOLO26s 只確認 `images FLOAT[1,3,640,640]`、`output0 FLOAT[1,300,6]` 及圖 inventory；內部 P3/P4/P5 候選不是已匯出的新 head 介面。沒有已核實、屬於新 Mamba-FPN head 的 scan 精確公式，故 **S01 尚未實作**，不自行選通用 Mamba 公式。所需新 head／native 權重途徑、ordered tensor／decode 契約、scan attributes／方向／initial state／精確公式來源與數值比較依據，已列於 [具體缺件清單](../../../docs/MODEL_ARTIFACT_READINESS.md#4-三層查核狀態與接續輸入)。取得後直接接續獨立 C++ scalar reference 與數值 fixtures。

遠端 CI 與本機 GPU 結果分開回報；GitHub default workflow 僅驗證 CPU build，不能代替此處的 native GPU evidence。
