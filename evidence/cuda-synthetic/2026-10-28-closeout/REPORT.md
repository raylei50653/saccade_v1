# CUDA synthetic error propagation closeout — 2026-10-28

**三處錯誤傳播已修正並通過回歸；啟用主機 debugger interface 後，四項 sanitizer 均有效執行並通過。** 本次合成 CUDA 協定骨架的有限驗收已完成，範圍如下；不擴張成所有合法設定或完整 tracker 的驗證。原有 [2026-10-28 證據](../2026-10-28/REPORT.md) 完整保留，本目錄追加收尾與啟用前後的證據。

開始時 fetch 後 `HEAD=origin/main=8a5eb5961f73e5e32db6f20fc0ff9276b5ba78bc`。本次實測 source commit 為 **`1c5135abaf9f157e1f65916f171ce81fb02ebbf4`**，在乾淨 worktree 重新 configure/build；封存內232份 session identity 均為此 commit，`build_dirty=false`。主機設定啟用後使用同一份執行檔，沒有重新編譯。後續 evidence commit 不修改程式。

## 修正與測試

| 審查問題 | 實作及實際證據 |
| --- | --- |
| 讀取失敗被當成合法空設定 | 共用 `read_text_file` 以分段 `ifstream.read` 區分 EOF、bad／fail，讀取與關檔錯誤均傳出；合法空檔保留預設設定。CPU 測試涵蓋空檔、目錄、缺失檔案與 chunk 邊界；實際 CLI 的目錄／缺失設定皆 exit 2，沒有先建立輸出目錄 |
| 緩衝寫入／關檔錯誤未傳出 | 共用 text／binary／append writers 都明確 close 並檢查；CLI 與 GPU evidence writer 全部使用同一 helper。CPU `/dev/full` 小型 text／binary／append 寫入皆拒絕；實際 CLI 的 resolved.cfg、identity.txt、packets.bin 三處 `/dev/full` 皆 exit 2 並核對對應診斷 |
| 清理失敗仍可能回傳成功 | `Session::close` 在雙 terminal drain 及可選 diagnostics 之後，檢查 cuFFT／device／pinned／event／stream 全部釋放，任何錯誤都阻止成功回傳。destructor 保留 noexcept 兜底；handle 在嘗試前移出所有權，已嘗試的 free/destroy 不重試。每次成功回傳最後都有 `finalize resources_released` 帳本記號 |

清理 API 可能回報先前非同步工作的錯誤，因此錯誤不等於 handle 仍有效；釋放結果不確定時保留原錯誤並嘗試其餘獨立資源，不冒稱已成功釋放或安全重試。[CUDA event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html) 明列先前非同步錯誤與 destroy 後 handle 的使用界限。

檔案 I/O 修正限定在一般錯誤傳播，沒有加入 fsync、斷電 durability、整組輸出的原子發布保證或設定 schema 變更。

## 執行結果

| 檢查 | 結果 |
| --- | --- |
| CUDA OFF | **6/6 CTests 通過**；既有216配置／15類同步邊變異、封包／狀態630 checks、設定88 checks保持通過；新增 I/O **23 checks**、cleanup collector **79 checks** |
| CUDA ON | **9/9 CTests 通過**，含五個 CLI I/O 負測試與獨立清理整合測試 |
| GPU 原始矩陣 | 原有72 sessions全保留；新增8 sessions觀察開關對照；合計 **80 sessions／43,782 checks**，保存892個原始ResultPackets，共11,032,256 bytes |
| 舊證據回歸 | 新 build 前72個案例的全部原始封包，逐檔與先前 `c453cec` 封存比較；**72/72檔完全一致**。舊證據外層25項hash亦重新通過，不改寫歷史紀錄 |
| terminal_snapshot=false/true | 成功與update後失敗，各serial／pipelined，4對共8 sessions；同一0..7提交範圍的原始封包逐bytes一致，behavior hash相同、完整config hash不同。false沒有額外diagnostic buffers／snapshot檔，true在雙drain後讀回 |
| 正常清理整合 | Linux linker wrappers 跑 **7 GPU sessions／91 checks**：正常、五種釋放API各一例、再正常；釋放結果錯誤傳至caller，所有獨立資源繼續釋放，handle不重試 |
| sanitizer啟用前 | 四工具皆exit99／BLOCKED；WDDM debugger interface初始化失敗，無有效檢查結果。原紀錄不改寫，應用層成功不列作工具通過 |
| sanitizer啟用後 | **四工具皆exit0**；每工具完成19 sessions／8,057應用層checks。memcheck、initcheck、synccheck各 **0 errors**；racecheck **0 hazards、0 errors、0 warnings**。有效結果與啟用前分開保存 |

清理整合 wrappers 先呼叫真正的 `cufftDestroy`、`cudaFree`、`cudaFreeHost`、`cudaEventDestroy`、`cudaStreamDestroy`，確認真實釋放成功後，只修改選中類別的第一次回傳碼。每個 session 真實釋放數一致：plan=1、device=14、pinned=2、event=14、stream=2。五個注入案例分別取得帶有正確 API 名稱的例外，且之後的新 session 能正常完成。這是錯誤傳播測試，不是實際 driver／記憶體故障注入，不等同 sanitizer。

CPU、GPU 完整輸出見 [cpu-tests.log](cpu-tests.log)、[gpu-tests.log](gpu-tests.log)；CTest摘要見 [cpu-ctest.log](cpu-ctest.log)、[gpu-ctest.log](gpu-ctest.log)。[cases.tsv](cases.tsv) 明列80個案例的設定／行為hash與byte數；[previous-72-compare.log](previous-72-compare.log) 保留舊封包完整比較結果。

## 身分與原始輸出

- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，UUID `GPU-825ae551-ba16-539a-e444-22726b59ee9a`，SM12.0。
- Driver：617.42（NVIDIA-SMI工具615.78.08）；CUDA runtime／driver API13040，nvcc13.4.92，cuFFT12400。
- Toolchain：GNU C++16.2.1 20260810，CMake4.4.4，Release、C++20／CUDA20、architecture120、CUDA `-lineinfo`。
- Sanitizer：2026.3.0.0 build38855100；WSL2 6.18.33.1-microsoft-standard-WSL2。

完整身分與啟用前唯讀Windows registry query在 [environment.txt](environment.txt)；當時query找不到 `HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger` 的 `EnableInterface` 設定，exit1。使用者確認既有授權後，以Windows管理員程序將該DWORD設為1；[enable-debugger.log](enable-debugger.log) 記錄更新exit0，[debugger-registry-after.log](debugger-registry-after.log)讀回確認0x1。原樣執行腳本存為 [enable-debugger-executed.ps1](enable-debugger-executed.ps1)，僅作歷史紀錄：本次key不存在，但該版本的 `New-Item -Force` 不宜對既有key重跑。另附 [enable-debugger.ps1](enable-debugger.ps1) 作重跑版本，先檢查key不存在才建立，保留其他既有values，再設定EnableInterface；這份guarded版本未再次執行。執行檔hash見 [binaries.sha256](binaries.sha256)。

[raw-runs.tar.gz](raw-runs.tar.gz) 無損封存80個完整suite、啟用前4×19及啟用後4×19個工具下sessions，共232個案例，含原始packets、resolved config、behavior config、hash、identity、事件帳本及啟用時的terminal snapshots。`sanitizer-*`為啟用前，`enabled-*`為有效工具執行，兩套各自保留。另有5個CLI I/O案例的stdout／stderr／exit、設定及identity寫入失敗前的一個原始完整packet。**不封存 `/dev/full` symlinks**；只有普通檔案。`raw-evidence.sha256` 對1847份payload逐檔列hash。GPU bytes不由host重建或替換。

## 重跑命令

從乾淨source commit執行下列命令可重建本次身分；若在後續evidence commit重跑，build_commit會如實不同。完整設定與協定契約在 [CUDA_SYNTHETIC](../../../docs/CUDA_SYNTHETIC.md)。

```sh
cmake -S . -B build-closeout-cpu -DSACCADE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-closeout-cpu --parallel 4
ctest --test-dir build-closeout-cpu --output-on-failure

cmake -S . -B build-closeout-cuda -DSACCADE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build-closeout-cuda --parallel 4
ctest --test-dir build-closeout-cuda --output-on-failure

for tool in memcheck initcheck synccheck racecheck; do
  /opt/cuda/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
    build-closeout-cuda/cuda_synthetic_tests --sanitizer "build-closeout-cuda/enabled-$tool" \
    > "build-closeout-cuda/enabled-$tool.log" 2>&1
  result=$?
  printf '%s exit=%s\n' "$tool" "$result"
done
```

CPU機也能驗證封存與新增觀察對照：

```sh
sha256sum -c evidence/cuda-synthetic/2026-10-28-closeout/SHA256SUMS
mkdir -p build-closeout-evidence
tar -xzf evidence/cuda-synthetic/2026-10-28-closeout/raw-runs.tar.gz -C build-closeout-evidence
(cd build-closeout-evidence && sha256sum -c raw-evidence.sha256)
build-closeout-cpu/packet_compare \
  build-closeout-evidence/gpu-evidence/case0073/packets.bin \
  build-closeout-evidence/gpu-evidence/case0074/packets.bin
```

case0073/74為serial成功的snapshot off/on；75/76為pipelined成功，77/78為serial固定失敗，79/80為pipelined固定失敗。要重跑72檔舊證據比較，先解開舊archive到 `build-evidence`，再執行：

```sh
for original in build-evidence/gpu-evidence/case*/packets.bin; do
  build-closeout-cpu/packet_compare "$original" "build-closeout-evidence/${original#build-evidence/}" || exit 1
done
```

## 有效工具結果與驗證界限

[memcheck.log](memcheck.log)、[initcheck.log](initcheck.log)、[synccheck.log](synccheck.log)、[racecheck.log](racecheck.log) 保留啟用前的WDDM debugger interface初始化失敗與Device not supported。racecheck當時另回報process未成功終止、可能host Unified Memory存取錯誤與2 hazards；原樣保留，不用後續PASS重寫舊紀錄或聲稱已精確確定舊診斷成因。

依 [NVIDIA Windows debugging interface要求](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior) 啟用後，使用相同build、相同19-session subset及相同 `--error-exitcode 99`，取得 [enabled-memcheck.log](enabled-memcheck.log)、[enabled-initcheck.log](enabled-initcheck.log)、[enabled-synccheck.log](enabled-synccheck.log)、[enabled-racecheck.log](enabled-racecheck.log) 的有效exit0結果。沒有使用停用檢查的選項。先前racecheck異常在有效執行中未重現。racecheck不涵蓋跨stream global-memory順序，仍依賴協定、時序擾動及完整輸出比較共同驗證。

四工具驗證範圍為19-session代表subset，並非80-session整個矩陣或任意設定。此次未驗證實際GPU fatal／OOM／API釋放失敗、配置註冊時host allocation failure、跨裝置／跨build一般性逐位元一致性或所有合法ring／shape／seed；YOLO／Mamba、完整GMC、tracker、MOT writer與P01仍由各自任務接續。
