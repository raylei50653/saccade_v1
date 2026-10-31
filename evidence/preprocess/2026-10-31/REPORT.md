# F01／F02 native 前處理驗證報告（2026-10-31）

**結論：F02 在固定 cases 與受測 CUDA 環境下 PASS；F01 的數值與有限 API／生命週期測試 PASS，但 initcheck FAIL（8,151 errors），原因 OPEN。整體 F01／F02 不列為完整 sanitizer PASS。**

原始 clean-source capture 在 `native-initcheck` 以 exit 99 停止；其失敗結果保留。後續以相同未改 source 與執行檔補跑未到達的工具，不能將補跑改寫成原 capture exit 0。補跑的真實 case 0 initcheck 在同類 SDK 報告後達 300 秒上限，exit 124，未取得完整結束 summary；不列為 PASS。

範圍：`t01-input/1` 的 nvJPEG HARDWARE batch-1 planar RGB decode（F01）、FP32 `/255` 後的 antialiased bilinear stretch（F02），以及獨立 API 的 stream／容量／錯誤／生命週期測試。比較政策為 `t01-preprocess-fp32/1`，在首次 native GPU 比較前寫定。最初的開發比較曾失敗，修正 coefficient 計算的 FP32 contraction 後，在同一政策下重跑；政策、門檻及固定 cases 未放寬。

## 身分

| 項目 | 值 |
| --- | --- |
| 受測 source | `2b39ddaf7cefcbdfb8a7fc143ecfd83466618cb3`；formal capture `allow_dirty=false`、build `source_dirty=false`，source digest `30cb22e943e27ec87a35eb5ee376bf10c31f7015353be8010b872f649d1b690f`；capture 停止時 source 未改，無 source-drift log |
| 政策 SHA-256 | `e9d67b6ade582a51b7687473afc6ff91a8348c2a20b4fabf863a1d3802ddca6e`（`docs/PREPROCESS_POLICY.md`） |
| reference script SHA-256 | `0500765ca0a03a7c2191930127f9dfb3a6332e45014df50a4e9acce7602d8999`（`training/t01/t01_preprocess_ref.py`） |
| native nvJPEG | `13.2.3`；formal `build-identity.log` 核對實際 library `/opt/cuda/targets/x86_64-linux/lib/libnvjpeg.so.13.2.3.58`，SHA-256 `c8fb7a95fa58f20eeb58c21760ad4017845fd051c6a70672ff25736696404338` |
| 訓練 reference nvJPEG | torchvision 0.26.0 內附 `libnvjpeg.36e11081.so.13`（13.0.1），SHA-256 `1a359ba735541f0b21d9dd41166b5af05c0609476771ddc688f3cc80e38f1d74`；torch `2.11.0+cu130`，Python 3.12.13 |
| 獨立 CPU decoder | libjpeg-turbo 3.2.0（build 20260701），`/usr/bin/djpeg` SHA-256 `b8f6099ac96661b9bcdcb176519e9b9e33f72c3c3863546fa998aa2b61661824`；`-dct int -nosmooth -rgb` |
| 環境 | RTX 5070 Ti Laptop（sm_120，driver 617.42）、CUDA runtime／driver API 13.4、nvcc 13.4.92、GCC 16.2.1、CMake 4.4.5、WSL2 Linux 6.18.33.1；Compute Sanitizer 2026.3.0.0；不啟用 TensorRT |
| 執行檔 SHA-256 | `preprocess` `41fbeea381448745d4f37cafe1e80c05fc75b68ac1d9d27a7b555849f553925f`；`preprocess_tests` `053e4d027b1621a825f259521b70b5dea4ace6ca42d90fb12e253c759e8fdf24`；`resize_cuda_tests` `4485ec75f9c253bcad29b4c2bc80293eccaf0ee8f83f365d79fa49666631b02e` |
| capture | `bash scripts/run_preprocess_acceptance.sh build-preprocess-evidence`，2026-10-31 09:41:13–09:44:46 +08:00，**exit 99**，停止於 `native-initcheck`；補跑命令封存於 [supplement.sh](supplement.sh)，於 09:53:04 +08:00 結束，exit 124 |

native 與訓練 nvJPEG 版本不同，因此本次重新執行逐位元對照；不沿用 T01 舊全量 decode 比較的結論。

## 固定資料與判定

7 張真實 JPEG 是 `MOT17-02-SDP/1`、`10/1`、`13/400`、`04/1`、`05/1`、`09/1`、`11/1`，涵蓋七個 sequences；6 張 1920×1080 與 1 張 640×480。F01 合計 **38,246,400 RGB bytes**。10 個合成 cases 是 zero／one、色彩 ramp、邊界 impulse、odd-size checkerboard、stripes、固定 random、identity、單列與單欄。17 cases 合計 **13,522,182 FP32 輸出元素**；尺寸、來源與 hashes 固定於 reference manifest。

F01 要求全部 RGB bytes 相同。F02 使用政策固定的 absolute bound `2*gamma(4*(Nx+Ny)+24)`（`u=2^-24`、每軸 N 是最大 filter span 的保守上限），本次門檻為 5.7221×10⁻⁶–1.0014×10⁻⁵，且所有值有限。同 session 重跑要求 RGB 與模型輸入逐位元相同。

reference generation 已在 formal capture 重新執行，7 張 JPEG 的 torchvision batch／single／repeat 與獨立 djpeg 輸出全部逐位元相同；6 種預定語義變體均被固定門檻偵測：

| 變體 | 超界元素總數 |
| --- | ---: |
| 無 antialias | 7,781,126 |
| BGR | 7,280,164 |
| HWC 當 CHW | 11,062,288 |
| 未除 255 | 12,286,009 |
| resize 後 uint8 取整 | 9,797,871 |
| `align_corners=True`／無 AA 對照 | 9,919,492 |

「先 resize 再除 255」最大差僅 2.3842×10⁻⁷，位於本政策捨入預算內；除法順序與不 clamp 另由 source／stage tests 核對，不宣稱數值門檻單獨證明所有語義。

## 結果

| 項目 | 結果 |
| --- | --- |
| CPU 預設 CTest | 11/11，開發驗證 `build-preprocess-cpu/Testing/Temporary/LastTest.log`；正式 CUDA build 亦包含這些 regression tests |
| CUDA＋preprocessing CTest | **19/19 PASS**，39.93 秒；沒有啟用 TensorRT |
| F01 7 真實影像 | **數值 PASS**；全部 38,246,400 RGB bytes 與 reference 相同 |
| F02 17 cases | **數值 PASS**；全部 13,522,182 元素逐位元相同，max abs＝0，0 超界／非有限值；獨立 Python comparator 亦 PASS |
| 重跑與 mapping | 17/17 `repeat_exact=1`，separate x／y scales 通過 |
| 三張 F03 既有輸入 | formal native 輸出的 hashes 恰為 `a99482f804318d2348a49903b69f85bc0a3aa610d2880e223f591e3118fedfa2`、`ca9388d1e5042961d708d103681a5e2fc54c5e4f20940c655a775df936e2cb57`、`4a30d9421e3c414bfab6d08f7db83b020da0046abd8a49158e32b3394fc1f738`；與 F03 已固定的 PyTorch inputs 相同 |
| F02 primitive 行為 | 68 cases PASS：binary64 mathematical reference／手算、指定 stream、重跑、graph capture／replay、參數／容量／overlap 拒絕；對 binary64 真實座標的數學 reference 最大差 1.41507×10⁻⁵（獨立的 mathematical check，其座標精度與相同 FP32 coefficients 的訓練比較不同） |
| F01／F02 session 行為 | 2,977 checks PASS：odd-size JPEG、不同尺寸切換／reuse、caller JPEG copy、independent sessions、metadata／格式／容量／frame order 拒絕、failure latch、capture 拒絕、checked close；功能測試通過不消除下列 initcheck FAIL |
| 單 JPEG CLI | PASS，產生 `frame.rgb`、`frame.f32`、`mapping.txt` |

## Sanitizers

下列狀態均須讀到每個 `.exit` 與 sanitizer summary 後才填 PASS；timeout、未執行、記憶體門檻中止均不得列為 0 errors。capture 在各 sanitizer 前保存 MemAvailable，低於 4 GiB 即停止；每次 command 限時 300 秒，保存 `/usr/bin/time -v` RSS。各次預檢的最低 MemAvailable 為 47,122,904 KiB；`time -v` 所報全部命令最大 RSS 為 1,314,304 KiB（reference generation），sanitizer 最大為 1,139,036 KiB（resize racecheck）。這些是 wrapper 記錄的 RSS；timeout 的 detached worker 未被完整彙計，不作整棵 process tree 的峰值保證。

| 工具 | F02 `resize_cuda_tests` | F01／F02 `preprocess_tests` | 真實 case 0 reference run |
| --- | --- | --- | --- |
| memcheck | 0 errors，exit 0 | 0 errors，exit 0 | 補跑 0 errors，exit 0，RGB／FP32 exact |
| initcheck | 0 errors，exit 0（`--no-held-stream`） | **FAIL：8,151 errors，exit 99** | **未完成：300 秒 timeout，exit 124；已有 SDK 未初始化讀取報告，無完整 summary** |
| racecheck | 補跑 0 hazards／errors／warnings，exit 0 | 補跑 0 hazards／errors／warnings，exit 0 | 未排入本次 real-case sanitizer |
| synccheck | 補跑 0 errors，exit 0 | 補跑 0 errors，exit 0 | 未排入本次 real-case sanitizer |

F02 initcheck 的 `--no-held-stream` 僅略過刻意以 host callback 擋住指定 stream 的那個 ordering case；其他 tools 與普通 suite 仍執行。完整 JPEG preprocessing 不支援 CUDA graph；graph 證據限低階 F02 primitive，不能套用至 nvJPEG。

### F01 initcheck 問題：原因 OPEN

原始 `native-initcheck.log` 的 100 筆已印出紀錄均位於 SDK kernel `nvjpeg::rgba2rgb +0x170`，回報未初始化的 3-byte global read；總數 8,151，另 8,051 筆受 print limit 限制未印出。相同 command 的功能測試 2,977 checks 通過，memcheck 0 errors，這些結果不能抵銷 initcheck FAIL。

獨立控制只使用 CUDA／nvJPEG HARDWARE batch-1 decode，完全不連結 saccade 或 F02，對 repo 35×27 baseline JPEG 解碼兩次亦重現 initcheck **1,890 errors**，而 normal exit 0、memcheck 0 errors、全部 2,835 RGB bytes 與 CPU reference exact，重跑 exact。此控制把問題縮至 nvJPEG hardware 路徑與 sanitizer 的交互，尚未證明是程式錯誤、SDK 錯誤或工具追蹤限制。控制 source／commands／logs 與說明封存在 [initcheck-control/README.md](initcheck-control/README.md)；原始完整工作目錄為 `build-preprocess-initcheck-control/`。全部成功 HW outputs 的 SHA-256 是 `761809bdcad84f79c1b01f0d2369bb4cf52f280cb2dff92be87c3ef9f11c7162`。

嘗試 V1／V2 allocator callbacks 的 dirty／zero／0xa5 初始化方案，實際 device callback 呼叫數都是 0，各模式仍然 normal exit 0、memcheck 0 errors、initcheck 1,890 errors，因此沒有對 SDK 的實際配置產生有效補救，不能據此宣稱已初始化相關 buffer、poison independence 或已修正。有效 V2 控制同時提供 device 與 pinned allocator，命令為 `v2-*-try3-*`；先前未提供 pinned allocator 的 V2 status 2 與編譯失敗 attempt 都由控制 README 分開保留，不算有效 GPU 實驗。HYBRID backend 只作診斷：initcheck 0 errors、重跑 exact，但 RGB 有 93 bytes 與指定 reference 不同，最大 2 LSB，未採用為 fallback。

目前沒有找到 NVIDIA 明確確認 nvJPEG hardware decode 此現象為 initcheck false positive 的官方依據。保留 **F01 initcheck FAIL／原因 OPEN**；獨立重現只能支持進一步向 SDK／工具定位，不能據此將錯誤抑制、改成 PASS 或提高整體驗收層級。

## 開發失敗、修正與 mutation

初版 development run 的 build 身分是 `88f906809c68111d8a132c2f9fe08f45b4adecc4`＋`source_dirty=true`，不是 clean-source acceptance。真實 JPEG 已逐位元相同，但 synthetic `checker-odd-downsample`（1281×721→640×640）有 **48,087 個元素超出固定 7.629423635×10⁻⁶ 門檻**，最大差 **1.490116119×10⁻⁵**；原始 `build-preprocess-dev/preprocess-reference/report.tsv` 與 `comparison.json` 保留失敗。

原因是 FP32 coefficient 計算把 center 的乘法先獨立捨入，而 PyTorch CUDA 的對應 expression 允許 FMA contraction。修正只對齊取樣座標與權重建構的運算次序／contraction；沒有變更 filter、輸入 bytes、policy 或容差。修正後 `build-preprocess-fix-all/report.tsv` 的 17 cases 全部 exact。

另以固定測例重建「強迫 center 先獨立捨入」的 mutation，測試明確失敗：`odd-ratio coordinate contraction differs from fixed PyTorch reference`（`build-preprocess-contraction-mutation/result.log`）。這是測試捕捉錯誤變體的證據，不是正常執行的 PASS，也不是 sanitizer 結論。原始失敗、修正後 development run 與 formal capture 各自保留身分，不能將早期 dirty attempt 改稱 clean。

## 邊界與未完成項目

- 只驗本次 7 張真實 JPEG；T01 曾有 5,316 張 reference decode 比較，但本次 native 13.2.3 不據此宣稱全量已驗。
- 嚴格 baseline 8-bit YCbCr 4:2:0、HARDWARE batch 1；沒有 progressive／CMYK／gray／其他 subsampling 支援或 decoder fallback。
- 目前一份 host staging／nvJPEG state，重用前等待前幀 F02 event；RAW／模型輸入由同一 D stream 順序保護。尚未整合 ASYNC_CONTROL 的雙 STAGE decoupled host／transfer／device overlap。
- 尚未把 F05／F03 排入同一完整 session；三張 native tensor 等於既有 F03 輸入 hashes，只證明輸入 bytes 對齊，不代表完成 F01→F03 stream integration／ResultPacket。
- F04 decode／NMS 與容量、完整 tracker／GMC／MOT writer、P01 與模型品質仍 OPEN。未重跑 F03／head sanitizers，不擴張其既有覆蓋。
- 結果限本機 GPU、driver、CUDA／nvJPEG 與固定 cases；其他裝置／版本另驗。

## 重跑與檔案

```sh
bash scripts/run_preprocess_acceptance.sh build-NEW-OUTPUT
```

需乾淨已提交 source、CUDA／hardware JPEG GPU、Compute Sanitizer、T01 pinned export venv／MOT17 與 djpeg。腳本每步前後核對 HEAD、tracked／untracked nonignored source digest，保留 `.command`／`.log`／`.exit`／`.time`；任何步驟失敗中止且保留 completed prefix。`--allow-dirty` 僅可產生 development evidence。

大型 `.jpg`／`.rgb`／`.f32` 與建置執行檔留在 `build-preprocess-evidence/`，hash 見 formal `reference-hashes.log`／`build-identity.log`。封存目錄為 `evidence/preprocess/2026-10-31/`：`capture/` 保存原始失敗 capture，`supplement/` 保存補跑與 timeout，`refs/` 保存固定 reference manifest／policy／case list，`native/` 保存完整比較與身分，`initcheck-control/` 保存獨立 SDK source／命令／logs，`attempts/rounded-center/` 保存開發失敗、修正後比較與 mutation。`LastTest.log`／`CPU-LastTest.log` 保存完整測試輸出。此 bundle 的實際檔案以 `SHA256SUMS` 核對；control 內的 `FILES.sha256`／`output*.sha256` 是原始工作目錄 manifest，亦列出未封存的 binaries／輸出。

在受測 SDK 上重跑 acceptance 預期仍停於相同 initcheck 問題；退出碼不應被改為成功。若要獨立重跑 SDK control，先將 `initcheck-control/repro.cpp` 與 `control3.sh` 複製到新的 `build-preprocess-initcheck-control/`，從 repo root 執行該腳本並逐項讀取 `.exit`；其 driver script 本身不以 exit 0 表示每個子實驗成功。沒有向外部服務發送 bug report。
