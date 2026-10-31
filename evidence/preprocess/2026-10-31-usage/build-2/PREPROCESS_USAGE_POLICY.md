# F01／F02 使用驗收增補（`t01-preprocess-usage/1`）

固定日期：2026-10-31。使用者決策：「目前不應該追求不報錯，只需要讓使用時保證輸出相同即可」。本增補將目前的使用驗收與結案目標固定為**正常執行的輸出等價、重跑一致與資源重用正確**。數值仍適用既有 [`t01-preprocess-fp32/1`](PREPROCESS_POLICY.md)；該政策的 reference、固定 cases、演算法與門檻均不變。

## 1. 輸出判定

- **F01**：每張受測真實 JPEG 的全部原尺寸 planar RGB bytes 與固定 reference **逐位元相同**。reference 仍須對上獨立 libjpeg-turbo `djpeg -dct int -nosmooth -rgb`；不採像素容差或換 decoder 取代。
- **F02**：完整 FP32 tensor 全部有限，每個元素仍須滿足原政策的 `atol = 2*gamma(4*(Nx+Ny)+24)`，不另加相對容差、不放寬門檻。既有 17 組結果逐位元相同是實測結果；不將其改寫為已證明所有尺寸／環境都 bitwise。
- **重跑與重用**：同 source、設定、輸入與 stream 的 RGB／模型輸入重跑必須逐位元相同；同 session 在不同輸入間切換後重跑，仍須符合各自 reference，不得混入前幀資料。frame 身分與 `ImageMapping` 必須對應當前輸入。

記錄完整輸出比較、reference／native hashes、source／binary、GPU／driver／CUDA／nvJPEG 與設定身分。缺失、錯置、非有限值、任一超界元素或重跑漂移都屬失敗；失敗 case 與實際退出碼保留。

## 2. 正常使用路徑與 API 合約

使用驗收以原 [`NativePreprocessor`](PREPROCESS.md) 為對象：HARDWARE batch 1、baseline 8-bit YCbCr 4:2:0、`NVJPEG_OUTPUT_RGB`，再依 `t01-input/1` 做 FP32 正規化與 antialiased stretch。依 API 原有 stream／event 與生命週期合約執行；診斷用的 poison、私有 surface 標記、preload／CUPTI hook 或額外全裝置同步產物，不作正常使用等價的替代證據。

一個 instance 對應一個 host submitter 與一條 explicit non-default detect stream。JPEG bytes 由 `enqueue` 複製；下一次重用 staging 前等待前幀 F02 event。借用的 RGB／模型輸入，其下游 consumer 須在下一次 `enqueue` 前排入同一 detect stream，stream 須比 instance 活得久。檢查 host 輸出時依 API 等待完成，`close()` 須 checked drain 與釋放資源。

CUDA／nvJPEG／檔案 API 錯誤仍須回報；格式、容量、metadata 或 frame order 違約仍須拒絕。`enqueue`／`wait` 的錯誤維持 failure latch，不能沿用失敗序列或把錯誤當作正常 EOF。使用驗收不豁免這些行為。

## 3. Initcheck 與使用結案

原始 nvJPEG initcheck **FAIL**、完整 diagnostics 與 [NVIDIA Bug 7021871 回報](../evidence/preprocess/2026-10-31-f01-investigation/nvidia-report/README.md)保留為診斷紀錄，不改寫成 0 errors 或已確認誤報。**此既知 initcheck 報告不再作為目前正常輸出／重用使用驗收的結案門檻，也不要求先取得 NVIDIA 分類。** 本增補適用於這項門檻決策；舊報告反映當時的驗收分類。

可使用「**F01 使用驗收 PASS（明列受測範圍）；nvJPEG initcheck FAIL 保留為診斷項**」記錄有證據支持的結案。數值不符、API 失敗、重跑漂移或不正確重用仍會阻擋使用結案；不得用 annotation 消除報告代替正常輸出驗證。

## 4. 保證範圍

使用保證以受測 GPU／SDK、固定輸入、容量與呼叫合約為界。有限 cases 通過不等於全部影像、所有排程或其他版本都已證明；更換 GPU／nvJPEG 等環境須重新比對，不繼承舊數值結果。七張真實影像的既有證據亦不擴張為全量 5,316 張已驗。

本增補涵蓋 F01／F02 的正常輸出與獨立 API 使用；完整 async session、F01→F03／F05 整合、F04、tracker／GMC／MOT writer、P01 與模型品質仍依各自驗收項判定。
