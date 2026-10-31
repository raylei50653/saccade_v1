# F01 nvJPEG initcheck 追查（2026-10-31）

**後續結論：本機已取得 trace、私有 surface poison 與 allocation identity 的直接機制證據，支持 initcheck 未辨識 nvJPEG HARDWARE surface 寫入。** 尚未獲得 NVIDIA 官方分類、跨 GPU 或原生 Linux driver 驗證；本次沒有變更原有 **F01 initcheck FAIL／原因 OPEN** 的驗收標記，也沒有將診斷 poison 當作 runtime 修法。新材料與離線核對見下方「私有 surface 的後續干預證據」；本報告其餘原始實驗結果保留。

**原生 Windows 對照已補齊**：同一張 GPU、相同 standalone source／fixture、對齊的 nvJPEG／CUDA／sanitizer 版本，Windows PE 執行檔同樣報 initcheck 1,890 errors、RGB exact、memcheck 0。**WSL 不是重現問題的必要條件。** 詳見 [Windows 實測報告](windows-native/REPORT.md)；仍沒有原生 Linux driver 對照，驗收狀態未改。

**已向 NVIDIA 回報 [Bug 7021871](https://developer.nvidia.com/bugs/7021871)**（2026-10-31），standalone repro、完整 945-error log、Windows log 與去除本機路徑的 identity 以 email 附件寄至 NVSDKIssues。回報只請 NVIDIA 分類，尚無回覆；送出內容與附件 hash 見 [nvidia-report/](nvidia-report/README.md)。

這是接續[原始 F01／F02 驗證](../2026-10-31/REPORT.md)的獨立 SDK 診斷；沒有連結 Saccade、F02、TensorRT 或 Python runtime。未變更 native runtime、比較政策或錯誤抑制；原始 capture 的失敗不改寫。這次不重驗 F02、F03、完整 async session 或 P01。

## 身分與範圍

- 診斷時 repository HEAD：`40c904061c17f0b1ea6afa60aa084a84cea34407`，開始與結束的 tracked worktree 均乾淨；診斷 source／outputs 當時位於 ignored `build-f01-investigation/`，其後才封存本報告。這不是新一輪 application acceptance。
- RTX 5070 Ti Laptop（SM 12.0）、WSL2 Linux 6.18.33.1、driver 617.42、CUDA toolkit 13.4.92、Compute Sanitizer 2026.3.0.0；與原報告相同裝置。環境與 dynamic library 解析見 [identity.log](capture/identity.log)。
- 受測 source [`repro.cu`](repro.cu) SHA-256：`c3b0bbf242218a22d86568b527df363978588f9ee1d56f7c65891e4e0cdbc977`。受測 binary SHA-256：`58e2c0ea589fd64ef9e334793c2af8230f4e580d6a303888de769313b5da27c3`；原執行檔留在 ignored 工作目錄。
- 同一執行檔以 `LD_LIBRARY_PATH` 只切換 nvJPEG：native 13.2.3 SHA-256 `c8fb7a95fa58f20eeb58c21760ad4017845fd051c6a70672ff25736696404338`；訓練 torchvision 內附 13.0.1 SHA-256 `1a359ba735541f0b21d9dd41166b5af05c0609476771ddc688f3cc80e38f1d74`。兩種解析均記錄於 identity，程式另查詢實際 nvJPEG version。
- nvcc 使用靜態 CUDA runtime；`libcudart_static.a` SHA-256 `428c653a077388ef8aee4a961aa9cec1e552dad7b87766bb2d384a5695415ed9`，見 [post-identity.log](capture/post-identity.log)。identity 亦列出 shared libcudart hash，但該 shared library 並非本 harness 的 runtime linkage。
- 固定輸入：repo `fixtures/preprocess/baseline.jpg`，35×27 baseline 8-bit 4:2:0。獨立 `djpeg -dct int -nosmooth -rgb` 產生 2,835-byte planar [baseline.rgb](baseline.rgb)，SHA-256 `761809bdcad84f79c1b01f0d2369bb4cf52f280cb2dff92be87c3ef9f11c7162`。
- 每組正常執行、memcheck、initcheck 均各自新 process；所有 GPU 工作依序執行。每命令 60 秒 timeout、至少 4 GiB MemAvailable，保存 `.command`／`.exit`／`.log`／`.time`／`.memory`。沒有 timeout，也未使用 suppressions、kernel filter 或關閉 API memory checks。

## API 與所有權核對

[`src/preprocess.cpp`](../../../src/preprocess.cpp) 在改寫 pinned JPEG staging 前等待前幀 event；input pointer、length、destination、state 都由 instance 持有。RGB 的三個 plane 各為 width×height，pitch＝width；nvJPEG、F02 和完成 event 位於同一 explicit non-default stream；close 先 drain stream 再釋放資源。核對未找到能解釋這次報告的 caller lifetime／pitch 違規。

獨立 harness 在每次 decode 後完整同步，再以自己的 `inspect_caller_output` kernel 逐 byte 讀取 caller destination，最後複製到 host。這排除了 F02、Saccade 重用、下游 overlap 為重現此問題的必要條件，但不等於證明所有 application API 行為均正確。

## 控制矩陣

下表除特別標示外，均解碼同一張圖兩次；HARDWARE RGB／RGBI 輸出逐 byte 與 CPU reference 相同且 repeat exact。RGBI 只在 host 對照時轉成 canonical planar RGB。

| 控制 | normal exit | memcheck errors／exit | initcheck errors／exit | 數值或 API 結果 |
| --- | ---: | --- | --- | --- |
| 13.2.3 HARDWARE RGB | 0 | 0／0 | 1,890／99 | RGB exact |
| 13.0.1 HARDWARE RGB，同一 binary | 0 | 0／0 | 1,890／99 | RGB exact |
| 13.2.3 RGB，`HW_DECODE_NO_PIPELINE` | 0 | 0／0 | 1,890／99 | RGB exact |
| 13.2.3 RGB，不呼叫 PreAllocate | 0 | 0／0 | 1,890／99 | RGB exact |
| 13.2.3 RGB，每次 decode 前 memset caller output＝0 | 0 | 0／0 | 1,890／99 | RGB exact；未初始化 SDK 私有 surface |
| 13.2.3 HARDWARE RGBI | 0 | 0／0 | 1,890／99 | RGB exact；仍經 `rgba2rgb<5>` |
| 13.2.3 HARDWARE Y | 0 | 0／0 | 108／99 | repeat exact；未做獨立 Y 數值驗收 |
| 13.2.3 HARDWARE YUV | 0 | 0／0 | 164／99 | repeat exact；未做獨立 YUV 數值驗收 |
| 13.2.3 HYBRID RGB，僅診斷 | 0 | 0／0 | 0／0 | 每次有 93 RGB bytes 不同，最大 2 LSB；不符合契約 |
| HARDWARE RGB，單次，unlimited print | 未另跑 | 未另跑 | 945／99 | RGB exact；完整逐像素分析如下 |
| 跳過 decode，未初始化 output，單次 | 未另跑 | 0／0 | 2,835／99 | inspector 正控制；沒有 RGB 等價 |
| 跳過 decode，output memset＝0，單次 | 未另跑 | 未另跑 | 0／0 | inspector 負控制；沒有 RGB 等價 |

HARDWARE Y／YUV 的已印出報告改為 SDK 私有 source 在 `cuMemcpy2DAsync_v2`／cudaMemcpy 路徑遭讀取；仍在 `nvjpegDecodeBatched` 內。這表示問題不依賴 RGB converter 的那一條 CUDA load instruction。不同 API 與 kernel 的錯誤計數單位不同，不以 108／164 比 1,890 小推論改善。

另有 **失敗的 NV12 控制**，完整保留：35×27、Y pitch 35、UV pitch 36 時，普通 decode 就返回 nvJPEG status 6；memcheck 顯示 SDK `cudaMemcpy2DAsync` 的 `cudaErrorInvalidPitchValue`，exit 99／1 error，initcheck exit 99／68 errors。此路徑未成功完成，可能涉及 odd-width NV12 的 pitch／SDK 限制，沒有在本次解決；不把它當作成功輸出或 F01 根因證據。

所有實際退出碼、API failures、結果、output／log hashes 見 [`summary.json`](summary.json)；normal exit 0 只表示診斷程式完成，不抵銷 initcheck 非零或 RGB mismatch。

## 全像素定位與 instrumentation 控制

[`new-rgb-once-full-initcheck.log`](capture/new-rgb-once-full-initcheck.log) 使用 `--print-limit 0`，只解碼一次，取得全部 945 筆錯誤。離線 [`analyze.py`](analyze.py) 核對：

- 依 kernel 的 32×8 block 座標重建 `x=32*block.x+thread.x`、`y=8*block.y+thread.y`，恰好覆蓋 x＝0…34、y＝0…26 的 **945 個有效像素，每個一次**；沒有缺漏或超出影像的像素。
- 全部為 `nvjpeg::rgba2rgb<3> +0x170` 的 **3-byte** read。錯誤地址均滿足 `0xf0e200000 + 512*y + 4*x`；相鄰像素 stride＝4 bytes，跨列 stride＝512 bytes。
- 私有讀取 surface 與 caller output `0xf0e400000` 是不同地址。沒有 `inspect_caller_output` 的錯誤；read 錯誤不只出現在 padding 或 alpha byte。
- 這個 GPU inspector 在跳過 decode、output 未初始化時，正確報出每 byte 一筆，共 2,835 errors；output 清零時為 0。工具確實有檢查下游自訂 kernel。

因此「1,890＝35×27×2」現在有完整單次逐像素紀錄支持，不再只依總數猜測。RGB 全 bytes exact、memcheck 0、完整同步與重跑一致，加上 Y／YUV 在私有 copy source 的報告，共同支持「硬體 surface 寫入沒有被 initcheck 視為初始化」的假說。**這仍不是對 SDK 內部寫入、同步或工具缺陷的正式證明。**

## 官方依據與剩餘問題

[NVIDIA Initcheck 文件](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#what-is-initcheck)描述 device writes 與 CUDA memcpy／memset 的初始化追蹤；沒有在此說明 nvJPEG 私有 hardware surface 如何被追蹤。[nvJPEG 文件](https://docs.nvidia.com/cuda/nvjpeg/index.html#using-jpeg-decoding)規定各 output format 的 channel／pitch。[Compute Sanitizer 已知限制](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations)雖列出其他可產生 false positives 的機制，這次沒有證據顯示私有 nvJPEG surface 使用那些機制，不能套用其分類。

仍需 NVIDIA 對這份最小重現確認：HARDWARE surface 的寫入是否應被 initcheck 追蹤、是否為 SDK／driver 同步或初始化缺陷，以及保留原 RGB 數值契約的受支援修正。其他 driver、原生 Linux、GPU、sanitizer 版本尚未對照。本機兩個 nvJPEG 版本都重現，僅排除「只在 13.2.3 發生」；不能代表所有版本。

**狀態維持 F01 initcheck FAIL／原因 OPEN、P01 OPEN。** HYBRID、Y／YUV／NV12 都沒有被採用為 runtime fallback；清零 caller output 與改變 buffering／預配置亦沒有修好本問題。

## 私有 surface 的後續干預證據

已將使用者提供的 scratch 診斷逐 byte 封存到 [surface-poison/](surface-poison/README.md)：47 個原始檔案，包括 source、binary、14 份 logs、6 份 surface dumps、nsys report／SQLite／CSV 與小型輸出。來源分析另存 [USER_ANALYSIS.md](surface-poison/USER_ANALYSIS.md)，匯入身分見 [IMPORT.json](surface-poison/IMPORT.json)。本次只做離線核對，未重跑 GPU。

- **Trace**：SQLite 確認每次 decode 的 CUDA 可見工作是 1,124-byte H2D（等於 JPEG 大小）和 `rgba2rgb`；之後另有 harness inspector 與 2,835-byte D2H。沒有觀察到產生私有 surface 像素的其他 CUDA kernel／memcpy／memset。
- **Poison 與配置身分**：RGB preallocate 控制第一次 buffer ID 1065→1079，同位址、同 16 KiB；只有第一次報 945 errors。之後兩次在 ID 保持 1079 時對私有 surface poison，無新增錯誤且 RGB exact。Y 控制同樣只有第一次的 54 errors；正確定位私有 surface 的 no-preallocate 控制亦只有第一次的 945 errors。
- **Dump**：6 份 dump 的 945 個有效像素全部符合 reference、alpha 全為 0；32 列的後 256 bytes 均保留 poison，共 8,192 bytes。原分析「所有其他區域都保留 poison」不成立：另有有效像素以外的區域被改寫，詳見補充核對。dump-enabled run 總數仍有 1,457 errors，包含首次 decode 與整塊 allocation dump 引入的 API 報告，沒有將它列成全程 PASS。
- **來源界線**：部分實驗沒有保存 `.command`／`.exit`，且 source 在實驗間演進；匯入時 final source／binary hash 不能追溯套用至全部 run。錯誤地址、未成功 poison 的普通／memcheck run 均保留並另列，不補造退出碼。

這些觀察把機制說明由原先排除控制推進到**直接干預與 allocation identity 對照支持**；沒有因此證明 nvJPEG 內部同步在所有排程下正確，亦未改變驗收分類。Poison 依賴 SDK 私有位址，只作診斷，禁止納入 native runtime。完整離線分析由 [analyze_import.py](surface-poison/analyze_import.py)重建為 [analysis.json](surface-poison/analysis.json)；官方分類與跨環境驗證仍待完成。

## 檔案與重跑

`capture/` 保留實際命令與原始輸出；本目錄保留全部小型 output `.bin`、CPU reference、source、runner、離線分析與 hashes；`SHA256SUMS` 綁定封存內容。原始 F01／F02 evidence bundle 保留原樣。

診斷編譯使用 `-std=c++20 -O2 -lineinfo --allow-unsupported-compiler -arch=sm_120`、`-lnvjpeg` 與 nvcc 靜態 CUDA runtime。後續重編成功，但 binary SHA 與原受測檔不同（[`build-equivalence.json`](build-equivalence.json)）；重編檔保留為 `build-f01-investigation/repro.rebuilt`，未執行 GPU 測試。原受測檔已保留／恢復為 `repro`；所有上述結果只屬於原受測 hash，不繼承到重編檔。

以下將 source／scripts 複製到**新的** ignored 工作目錄；需上述 CUDA、nvJPEG、djpeg 及本機 pinned torchvision library。重編的新 binary 必須自行重跑，不沿用本次判定。

```sh
mkdir -p build-f01-rerun/capture build-f01-rerun/old-nvjpeg
cp evidence/preprocess/2026-10-31-f01-investigation/{repro.cu,run.py,analyze.py} build-f01-rerun/
ln -s "$HOME/.cache/saccade-v1/t01/venv-export/lib/python3.12/site-packages/torchvision.libs/libnvjpeg.36e11081.so.13" \
  build-f01-rerun/old-nvjpeg/libnvjpeg.so.13
/opt/cuda/bin/nvcc -std=c++20 -O2 -lineinfo --allow-unsupported-compiler \
  -arch=sm_120 -I/opt/cuda/include build-f01-rerun/repro.cu \
  -L/opt/cuda/lib64 -Xlinker -rpath -Xlinker /opt/cuda/lib64 -lnvjpeg \
  -o build-f01-rerun/repro
python3 build-f01-rerun/run.py identity
python3 build-f01-rerun/run.py baseline
python3 build-f01-rerun/run.py formats
python3 build-f01-rerun/run.py full-pixels
python3 build-f01-rerun/run.py reader-controls
python3 build-f01-rerun/run.py hashes
python3 build-f01-rerun/analyze.py
```

runner 會保留非零 child exits 後繼續矩陣，runner 自身 exit 0 **不代表 sanitizer PASS**；須逐項查 `.exit`／summary。這次未向外部服務送出 bug report。
