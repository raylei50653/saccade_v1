# 自建 GCC 16.2.0 與重驗（2026-11-13）

**結果：PASS。G0–G5 全部通過。** 以自建的 GCC 16.2.0 重建全部 11 個 feature 選項，結果如下：CTest 63/63、ASan／UBSan 21/21；五個 benchmark、七個序列的 MOT 輸出與 primary156 逐位元組相同，42/42 個指標完全相等；FPS 與 2026-11-12 的驗證相差在 ±0.4% 內。

## 為什麼要固定編譯器

系統編譯器原本是 Arch `core/gcc` 16.2.1（`+r23+gd564253eb6c8`），會跟著滾動更新升級。nvcc 13.4 的 `crt/host_config.h` 只接受 `__GNUC__ <= 16`，所以 GCC 一升到 17，CUDA 建置就會失敗。若 Arch 改用 `NVCC_CCBIN` 把 nvcc 指向較舊的 GCC，`.cu` 和 `.cpp` 又會被兩個不同版本編譯，而且不會出現任何錯誤。固定編譯器可以同時避免這兩種情況。

## 編譯器

- **來源**：GCC 16.2.0 官方 release `gcc-16.2.0.tar.xz`，SHA-256 `e6738e29…085c3e`。GPG 簽章由 GCC release manager 的金鑰 `7F74F97C103468EE5D750B583AB00996FC26A641` 簽署，以 gpgv 驗證為有效；SHA-512 與另一台伺服器 gcc.gnu.org 公布的值一致。GMP 6.3.0、MPFR 4.2.2、MPC 1.3.1、ISL 0.24 由 `contrib/download_prerequisites` 下載，以其內建的 SHA-512 校驗，並編進編譯器內部（靜態連結）。詳見 `gcc-build/gcc-source.json`。
- **建置**：`scripts/build_gcc.sh`，32 執行緒，耗時 23:53。三階段 bootstrap 的 stage 2／stage 3 比對結果為 `Comparison successful.`。完整 log 在 `gcc-build/build-gcc-16.2.0.log.gz`。
- **configure**：凡是會影響產生程式碼或預設選項的設定，都和 Arch 一致（PIE、SSP、CET、threads、clocale、hash style、checking=release）。不同之處只有以下幾項，都不影響產生的程式碼：只編 C／C++、不開 multilib、用一般 bootstrap 而非 `bootstrap-lto`、prerequisites 編進編譯器內部。
- **安裝**：本體在 `~/.local/opt/gcc-16.2.0`（293 MB），另有 `~/.local/bin/gcc-16.2.0`、`~/.local/bin/g++-16.2.0` 兩個連結。`cc1plus` 執行時只依賴系統的 libz、libzstd、libm、libc。
- **專案使用方式**：`-DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-gcc-16.2.0.cmake`。這會同時設定 C++ 編譯器與 `CMAKE_CUDA_HOST_COMPILER`，而 CMake 會把它以 `-ccbin=` 傳給 nvcc，所以系統的 `NVCC_CCBIN=/usr/bin/g++` 不會生效。

## 範圍與身分

- **原始碼**：commit `3f1f9d2f`（main `9af0e969` 加上 build script 與 toolchain file），在 worktree `../saccade_v1-worktrees/gcc16` 執行，git porcelain 為空。相對 `9b0d14f8`（11-12 eval 驗證的 C++ 基準），C++／CUDA 原始碼沒有變動，只多了一個 toolchain file。
- **工具與硬體**：nvcc 13.4.92、CMake 4.4.5、TensorRT 10.16.1.11、driver 621.14（與 11-12 相同）、RTX 5070 Ti Laptop（SM 120）。engine SHA-256 為 `8292cc23…7876`，未重建。
- **建置**：
  - `build/gcc16`：Release，全部 11 個 `SACCADE_ENABLE_*` 選項，與 `build/stream-view` 相同。
  - `build/gcc16-asan`：Debug，CPU-only，`-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer`。
- **輸入檢查**：16/16 通過，包括 engine、七個序列的 GT／seqinfo 對 primary156 凍結清單的 hash，以及 `per-sequence.csv`。見 `receipts/eval-preflight.json`。

## Gates

| Gate | 結果 | 內容 |
| --- | --- | --- |
| G0 編譯器身分 | PASS | 全部專案 `.o` 檔共 170 個（完整建置 126、ASan 建置 44），`.comment` 都只有 `GCC: (GNU) 16.2.0`，沒有缺標記的檔案。106 個 C++ 編譯命令都用固定的 g++；18 個 `.cu` 都帶 `-ccbin=` 指向同一份 g++。在 `-O3` 下比對 target、common、optimizers、c++、warnings 五類有效選項共 2,443 行，與系統 GCC 16.2.1 完全相同。 |
| G1 完整 CTest | PASS | 63/63，162 秒，涵蓋 CPU、CUDA、TensorRT、preprocess、postprocess、tracking、flow／OFA、video、streaming；包含刻意觸發 GPU fault 的 `scan_trt_async_fault`。 |
| G2 ASan／UBSan | PASS | 21/21，1130 秒。sanitizer runtime 使用 GCC 16.2.0 自己的 `lib64`（`LD_LIBRARY_PATH`），與 instrumentation 同版本。 |
| G3 MOT bytes | PASS | 7/7 序列的 `mot.txt` SHA-256 等於 primary156 main（期望值取自 11-12 `parity.json` 的 G1）。 |
| G4 逐序列分數 | PASS | 42/42 指標完全相同：HOTA、DetA、AssA、IDF1、MOTA 以解析後的浮點數做 exact equality，IDSW 以整數比對。 |
| G5 速度 | PASS | 7/7 summary 的 status 為 PASS，每個序列 480 個 single samples。與 11-12 的比較見下表。 |

`track_sequence` 的 binary 與 11-12 不同（`3954eeaf…` 對 `6e0844b6…`），這是換編譯器的預期結果；輸出則逐位元組相同。

## 速度（與 2026-11-12 run 2 相比）

測量條件與 11-12 相同：queued FPS，預先載入 JPEG，frames 21–140，時脈未鎖定。

| 序列 | FPS 11-12 | FPS GCC 16.2.0 | 差 | single p95 ms 11-12 | GCC 16.2.0 | 差 |
| --- | --- | --- | --- | --- | --- | --- |
| MOT17-10-SDP | 196.4 | 195.9 | -0.23% | 6.984 | 6.870 | -1.63% |
| MOT17-02-SDP | 194.2 | 194.3 | +0.07% | 7.045 | 6.921 | -1.76% |
| MOT17-04-SDP | 195.6 | 194.8 | -0.38% | 7.065 | 7.069 | +0.07% |
| MOT20-01 | 194.8 | 194.5 | -0.18% | 7.122 | 7.103 | -0.27% |
| ADL-Rundle-6 | 197.9 | 197.1 | -0.37% | 6.717 | 6.746 | +0.43% |
| dancetrack0004 | 195.5 | 195.1 | -0.20% | 6.758 | 6.799 | +0.60% |
| uid_vid_00008.mp4 | 193.7 | 194.0 | +0.14% | 7.031 | 7.029 | -0.04% |

FPS 的差異在 −0.38% 到 +0.14% 之間，p95 在 −1.76% 到 +0.60% 之間，有正有負，看不出一致的變快或變慢。作為對照，11-12 同一組 binary 在 attempt-1 與 run 2 之間，MOT17-10 的 p95 就差了 1.17%，與這裡的幅度同一量級。時脈未鎖定，因此這只能說明沒有可觀察到的速度退化，不能據此宣稱哪一版比較快。GPU telemetry 峰值：CTest 期間 59 °C、77 W；eval 期間 78 °C、215 W。

## 執行期函式庫

只固定編譯器。程式執行時仍載入系統的 `/usr/lib/libstdc++.so.6` 和 `libgcc_s.so.1`，見 G0 的 `runtime_libraries_track_sequence`。libstdc++ 保證向後相容，所以系統 GCC 升級後照常可用。反過來，如果用 rpath 綁死 16.2.0 的 libstdc++，日後由新版 GCC 編譯的系統函式庫（例如 FFmpeg 相依的 C++ 函式庫）可能會因 symbol version 不足而無法載入。

## 驗證程式本身的缺陷（已修正，未重跑 GPU）

1. G0 第一版比對預設選項時傳了 `-x c++` 卻沒有輸入檔，g++ 直接以 `no input files` 結束，兩邊都輸出 0 行，造成假的 PASS。修正後改成要求兩邊行數都大於 0，並把 GCC 為 `-o` 印出的隨機暫存檔名遮蔽掉。只重跑 G0 的檢查，沒有重新建置。
2. 使用 toolchain file 時，CMake 不會把 `CMAKE_CXX_COMPILER`／`CMAKE_CUDA_HOST_COMPILER` 寫進 `CMakeCache.txt`，第一版因此讀到空值。改為讀 `CMakeFiles/*/CMake*Compiler.cmake`，並把它納入 G0 的判定。
3. CMake 4.x 在全部通過時印的是 `100% tests passed out of N`，沒有 `, 0 tests failed`，摘要的 regex 對不上，把 G1／G2 誤判為未通過。改正 regex 並逐項確認每個測試都是 `Passed` 後，從原 log 重新判讀，測試沒有重跑。

另外，第一次 GCC 建置在 bootstrap stage 2 時被中止，原因是安裝路徑由 `~/.cache/saccade-toolchain` 改為 `~/.local/opt`。configure 會把 prefix 寫進編譯器，所以從頭重建。這不是建置失敗。

## 未涵蓋

- compute-sanitizer（memcheck／racecheck／initcheck／synccheck）、nsys，以及各 stage 的正式 acceptance 都沒有重跑。
- 11-12 的 G3 scorer parity 和 Python 測試與編譯器無關，沒有重跑。
- `docs/` 未更新（文件整理階段暫停中）。`docs/NATIVE_TENSORRT.md` 等文件仍記載 GCC 16.2.1。

## 檔案

- `methods/validate.py`：驗證程式，分 build／g0／ctest／ctest-summary／eval 五個階段。
- `gcc-build/`：GCC 來源與建置的身分紀錄，以及建置 log。
- `receipts/`：每一步的 command、result、log（gz）、junit、telemetry，以及 G0–G5 的 JSON。
- `runs/<benchmark>/`：eval runner 輸出的 RESULTS.md、results.csv、results.json、environment.json。
- `SHA256SUMS`
