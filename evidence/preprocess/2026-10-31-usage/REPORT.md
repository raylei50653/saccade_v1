# F01／F02 正常使用輸出等價（2026-10-31）

**使用驗收 PASS：本機固定七張真實 JPEG、四個持續 session、208 次使用，全部 RGB bytes 與 FP32 tensor bits 對固定 reference 相同；有限值、重跑、尺寸切換、frame／mapping、排隊 consumer 與 checked close 均通過。** 本次依先固定的 [`t01-preprocess-usage/1`](../../../docs/PREPROCESS_USAGE_POLICY.md)，沒有變更 runtime、decoder、references 或數值門檻。原 nvJPEG initcheck FAIL 仍保留為診斷紀錄，不作本次正常使用結案門檻。

## 受測使用方式與結果

`NativePreprocessor` 使用 HARDWARE batch 1、baseline 8-bit YCbCr 4:2:0、`NVJPEG_OUTPUT_RGB`；F02 為原有 FP32 `/255` 後 antialiased stretch 到 640×640。每個 instance 都由同一個 host submitter 呼叫，其 stream 明確、non-default，生命週期長於 instance。

每個 session 固定四個 cycles，每 cycle 的 case index 為：

```text
0,4,1,4,2,4,3,4,5,4,6,4,0
```

case 4 為 640×480，其餘為 1920×1080。dual-b 使用逐項 `(index+1)%7` 的固定不同影像次序，避免兩條 stream 同圖時掩蓋跨 session 資料混淆。

| session | readback window | 通過 frames | next enqueue 時 consumer 尚未完成 | 最大 pending |
| --- | ---: | ---: | ---: | ---: |
| serial | 1 | 52/52 | 0 | 1 |
| queued | 3 | 52/52 | 2/51 次 query | 3 |
| dual-a | 3 | 52/52 | 1/51 次 query | 3 |
| dual-b | 3 | 52/52 | 2/51 次 query | 3 |

總共逐 byte 比較 **891,187,200 RGB bytes**，逐 bit 比較 **255,590,400 FP32 elements**，全部 exact；沒有非有限值、repeat／cross-session hash 漂移或 mapping 不符。四個 session 各自 checked close，cleanup failures 均為 0。[逐幀結果](capture/run/report.tsv)、[實際提交／query](capture/run/submissions.tsv)、[session summary](capture/run/summary.tsv)、[獨立離線核對](capture/verification.json)。

完整 `memcmp`／有限值檢查在正常 native harness 執行時進行；每幀另對完整 RGB／FP32 bytes 計算 SHA-256。離線 verifier 核對保存的逐幀 hashes、reference manifest、case 次序、身分與覆蓋條件，**不會重讀 208 份 native tensors**，這些大型輸出未另行封存。可重建工具與 local ignored executable 的 binary hash 保存於 [build metadata](build-3/build.json)；manifest 不要求 ignored executable 必須隨 repo 交付。

## 排隊 consumer 的覆蓋

每個 session 有三份互相獨立的 pinned RGB／FP32 readback slots 與 completion events。每次呼叫按以下順序：

1. 呼叫原 `enqueue`，取得當前 borrowed RGB／model-input 與 metadata。
2. 在同一 detect stream、前處理完成 event **之後**，排入單 thread 的 **10,000,000 個 `clock64` cycles** bounded delay，再排入兩個 pinned D2H consumers 與其 completion event。
3. 呼叫者覆寫本次 JPEG 容器，核對 `enqueue` 複製 bytes 的合約；在下一次 `enqueue` 前已把所有 borrowed-buffer consumers 排入同一 stream。
4. 只有 readback slot 重用或最後 drain 才等待 consumer event。queued／dual 的下一次 enqueue 前先 query 最新 consumer event，並要求每個 window-3 session 至少實測一次 `cudaErrorNotReady`。

delay kernel 在受測 phases 前已用 `cudaFuncGetAttributes` preload。以上 2／1／2 次 NotReady 是實際量測，不只依 kernel 存在推論有 pending consumer；其餘 query 已完成的情況也原樣記錄。不將本次五次 pending 觀察外推為所有排程皆已證明。`cudaErrorNotReady` 後另記錄／清除同一 expected last-error 狀態，任何其他 CUDA error 仍會失敗，不會誤列為數值 mismatch。

## 身分與重跑

- runtime 來自歷史 clean source **`2b39ddaf7cefcbdfb8a7fc143ecfd83466618cb3`** 的既有靜態 archives；逐檔 pin archive hash，並在 compile 前及 GPU 前後核對相關 runtime source／headers 與該 commit bytes 相同。新增 harness 有獨立 source／binary hash，不宣稱整個新增文件工作樹是 clean-source acceptance。
- 原 active build 為 Release、CUDA `sm_120`、preprocess ON；沒有重建 runtime／CMake。原 formal `preprocess` binary hash 與原始證據相同。
- 診斷 harness binary SHA-256：**`1bda7a872fe2e33368ccee9063248252722de13e12379a209d90927c2fcfddb2`**。使用政策 SHA-256：`18538830e174bbe740f082e9af3936d94d35ffa6359943367d4e7a755a129298`。固定 PLAN SHA-256：`cc2798a6fdd8c5b798aa32ee39ea0ad189b20b80b155d5c275f1ebfaa1197986`。
- 七張 JPEG／RGB／FP32 都先核對已封存 reference manifest，SHA-256 **`a3b08bdfe20f13f122e63a0314969832622daf8cedf0e29330e4b325471af5ff`**；本次不重產 reference、不依 native 結果挑 cases。reference 原有獨立 CPU decode 對照沿用其 pinned 證據，不宣稱本次又執行 CPU decoder。
- RTX 5070 Ti Laptop、driver package **617.42**、CUDA runtime／driver API **13.4**、nvJPEG **13.2.3**；實際 nvJPEG library SHA-256 `c8fb7a95fa58f20eeb58c21760ad4017845fd051c6a70672ff25736696404338`。每個 session 保存實際 live `/proc/self/maps`，mapped library paths 於正常執行後另計 file hashes；不是 live executable pages 的完整 dump。[GPU identity](capture/gpu-identity.stdout.log)、[實際模組](capture/loaded-modules.json)、[native identity](capture/run/queued-identity.txt)。
- 正常 GPU process **exit 0**，無 timeout；實際執行約 **6.637 秒**，限制 180 秒，執行前 MemAvailable 超過 4 GiB。source、binary、references、archives、dynamic libraries、政策及 capture／verifier source 的前後身分均相同。[命令／退出紀錄](capture/normal-usage.status.json)、[身分狀態](capture/status.json)。

從 repo root 使用新目錄重跑，需保留原 pinned historical archives 與 refs：

```sh
python3 evidence/preprocess/2026-10-31-usage/capture.py prepare --out build-preprocess-usage-NEW
python3 evidence/preprocess/2026-10-31-usage/capture.py run \
  --build build-preprocess-usage-NEW --out build-preprocess-usage-capture-NEW
```

prepare 只做 CPU preflight／建置；run 先重新驗 hashes／記錄 GPU 環境，再直接執行正常 native harness。沒有 sanitizer、LD_PRELOAD、poison、CUPTI／NVTX hook 或全裝置同步。單獨離線重驗現有 capture，不執行 GPU：

```sh
python3 evidence/preprocess/2026-10-31-usage/verify.py evidence/preprocess/2026-10-31-usage/capture
```

準備過程另保留兩個非 GPU 失敗：最初 preflight 的錯誤 source path（`preparation-attempt-1.json`，該次例外未另存完整 raw stderr）；`build-2/` 的 nvcc-generated `#line` 被 GCC 16 `-Wpedantic -Werror` 拒絕。只移除新 harness compile 的 `-Wpedantic`，仍保留 `-Wall -Wextra -Werror -ffp-contract=off`；`build-3/` compile exit 0，只有這個 binary 進入正常 GPU capture。未放寬輸出判定或變更 runtime。

## 保證邊界

本次證據支持七張固定真實影像、208 次正常使用與本機版本的有限使用結案；**不是全量 5,316 張、所有影像／排程／GPU／SDK 的形式化保證**。既有 F02 十個 synthetic cases 的 primitive 邊界未在本次重跑。

下游使用 pinned D2H 作為 caller-owned consumer，以驗證 borrowed-buffer 同 stream 重用合約；未把 F03／F05、完整 async session／ResultPacket、F04、tracker／GMC／MOT writer 或模型品質納入。P01 仍依各自原列缺件判定。本次也不宣稱 nvJPEG initcheck 已修復或已被 NVIDIA 確認為誤報。
