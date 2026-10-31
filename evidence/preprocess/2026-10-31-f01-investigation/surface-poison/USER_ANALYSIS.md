# 使用者提供的分析原文

以下保留匯入前的來源敘述；封存後的離線核對、dump 區域修正與 provenance 限制以 [README.md](README.md) 為準，不以此原文單獨改變驗收狀態。

## 原因

這些 initcheck 報告是工具的覆蓋盲點，不是真的讀到未初始化記憶體。 HARDWARE backend 的 RGBA 私有 surface 是由 NVJPG 硬體引擎寫入的。Initcheck 只會在兩種情況把記憶體標成「已初始化」：經過 instrument 的 SM kernel 寫入，或 CUDA memcpy／memset API。硬體引擎的寫入兩者都不是，所以它寫進去的每個 byte 在 initcheck 眼中都還是未初始化。接著 nvJPEG 自己的 rgba2rgb（RGB／RGBI）或內部的 cudaMemcpy2D（Y／YUV）去讀那塊 surface，就被報成錯誤。

原報告只能用「排除法」推測這一點；這次的實驗直接證明了它。

## 新證據

我在 scratchpad 做了一個 harness 變體，repo 和 evidence 都沒有動：

1. 沒有任何 CUDA 可見的操作寫入 surface。 用 nsys trace，每次 decode 在 GPU 上只有兩個操作：一次 1,124-byte 的 H2D copy（剛好等於 JPEG 檔案大小，也就是上傳 bitstream）和 rgba2rgb。沒有任何 kernel、memcpy 或 memset 寫入那塊 16 KiB 的 surface，但 decode 完成後 surface 裡就是正確的像素，所以寫入者只可能是 NVJPG 引擎。
2. Poison 實驗。 每次 decode 前，先用 initcheck 追蹤得到的 cudaMemset(0xA5) 填滿整塊 surface：
   - 只要 buffer 沒被換掉，那次 decode 就是 initcheck 0 errors，RGB 也逐位元相同。
   - Dump 出的 surface 中，945/945 個像素都被引擎蓋成 reference 值（alpha＝0），其餘區域仍是 0xA5。
   - 如果 rgba2rgb 讀在硬體寫入之前，或讀到真正未初始化的資料，輸出就會混入 0xA5 的值；實際上沒有。
3. Y 路徑也一樣。 Poison 後只有第一次 decode 報 54 筆，之後的 repeat 都乾淨。這對應到原報告中 memcpy-source 那一類錯誤。
4. HYBRID 是 0 errors，和上面一致：HYBRID 由 CUDA kernel 解碼，寫入看得到。

## 為什麼每次 decode 都報 945 筆，而 poison 對第一次 decode 無效

- 沒有 poison 時，surface 永遠不會被標成已初始化，所以每次 decode 都報全部 945 個像素。2 次 decode 就是 1,890 筆。
- 有呼叫 PreAllocate 時，第一次 decode 會在同一個位址、以同樣大小重新配置 surface（buffer_id 1065→1079），這會把 initcheck 的 shadow 重設回未初始化。所以 decode 前的 poison 只會在第一次 decode 失效，之後就有效。
- 不呼叫 PreAllocate 時，surface 在第一次 decode 中才配置，結果相同。

## 仍未證明的部分

- 還沒有 NVIDIA 官方文件說明 initcheck 會漏掉 NVJPG 引擎的寫入。上面是根據 initcheck 機制與實驗得出的結論，不是官方確認。
- 引擎寫入與 rgba2rgb 之間的先後順序由 nvJPEG 內部 stream 處理，racecheck／synccheck 一樣看不到。我們每次執行都觀察到順序正確，但這不構成正式證明。
- 只在本機這張 GPU 和這版 driver 上驗證過。
- 重新配置的現象只在 initcheck 底下觀察過；正常執行時沒有查 buffer_id。

對 F01 的意義： 原因已經從「最符合的假說」變成有直接證據的機制。要不要把 F01 從 FAIL 改成「已解釋的工具誤報」，是你們驗收政策的決定，我沒有改任何狀態。注意 poison 是依位址寫進 SDK 私有記憶體，只能當診斷手段，不能放進 runtime。

診斷 source 和 logs 在
`/tmp/claude-1000/-home-ray-developer-ai-saccade-v1/1c3d56aa-2e17-433d-b405-33b28355d3f3/scratchpad/f01-alloc/`，這是暫存位置。
