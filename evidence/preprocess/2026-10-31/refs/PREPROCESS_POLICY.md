# F01／F02 native 前處理比較政策（`t01-preprocess-fp32/1`）

固定日期：2026-10-31。本檔在首次 native F01／F02 GPU 數值比較前寫定；政策與 reference 的 SHA-256 隨產物保存。不得依 native GPU 的觀測誤差放寬門檻；若運算假設失效，保留失敗結果並另立版本。

適用契約為 [T01 §2.1](T01_TRAINING.md#21-輸入前處理契約-t01-input1) 的 `t01-input/1`：nvJPEG hardware backend、`NVJPEG_OUTPUT_RGB` planar uint8；先以 FP32 除以 255，再以 FP32 antialiased bilinear、`align_corners=False` stretch 到 640×640，RGB CHW，不取整、不 clamp。測試介面另允許非 640 尺寸，以覆蓋邊界、identity 與 upsample。

## 1. Reference 與 F01 判定

離線工具 [`t01_preprocess_ref.py`](../training/t01/t01_preprocess_ref.py) 使用 T01 pinned export venv（torch 2.11.0、torchvision 0.26.0），直接呼叫 `torchvision.io.decode_jpeg(device="cuda")` 與 `F.interpolate(mode="bilinear", antialias=True, align_corners=False)`。Python 僅產生／核對測試資料，不進 native runtime。

每個真實 JPEG 的 F01 輸出必須與 reference 的全部 RGB bytes **逐位元相同**，不可用像素容差取代。reference 本身要求 batched／single／重跑 decode 相同，並與獨立 `djpeg -dct int -nosmooth -rgb` 的 libjpeg-turbo RGB 輸出逐位元相同。記錄實際 decoder library、GPU、CUDA、torch／torchvision 與 `djpeg` 身分；不同 nvJPEG 版本／GPU 必須重新比較，不繼承舊結果。沒有 hardware backend 或不支援的 JPEG 不可靜默切換 CPU／hybrid backend。

## 2. F02 算法與判定

每軸 scale 為 FP32 `input_size/output_size`，support 為 `max(scale,1)`。output index i 的中心是 `scale*(i+0.5)`，三角權重是 `max(0,1-abs((j-center+0.5)/support))`；截斷至合法 source 範圍後重新正規化。先 horizontal、再 vertical，channel 間獨立。FP32 scale、center、span 與權重建構必須對齊 PyTorch CUDA 實作，reference source 是 [UpSample.cuh @ v2.11.0](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/UpSample.cuh) 的 antialias helpers 與 [UpSampleBilinear2d.cu @ v2.11.0](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu) 的 antialias forward。相同尺寸的軸可直接複製。

所有輸出元素必須有限，並滿足以下**絕對**界（不另加相對容差）：

```
u = 2^-24
gamma(n) = n*u/(1-n*u)
Nx = ceil(2*max(input_width/output_width, 1)) + 1
Ny = ceil(2*max(input_height/output_height, 1)) + 1
atol = 2*gamma(4*(Nx+Ny)+24)
abs(native - torch_reference) <= atol
```

此界是針對 [0,1] 非負輸入、相同 FP32 取樣權重的保守捨入預算：每軸涵蓋最多 N 個權重的求和、權重正規化、乘加與兩次 pass，另保留 24 次 FP32 操作給輸入除法與 pass 間捨入；兩份實作的預算相加。不允許以此界替代不同取樣座標／filter 的語義核對。這不是跨所有尺寸的實數插值誤差定理；適用於本次固定 cases 及同 coefficient construction 的實作。

以完整 tensor 判定，報告 max／mean absolute error、最差元素座標、超界數、輸出範圍、reference／native hashes。同 source、同 stream 的重跑必須逐位元相同；native graph replay 與原 enqueue 的一致性由 CUDA tests 獨立檢查。

## 3. 固定輸入與語義對照

真實幀：`MOT17-02-SDP/1`、`10/1`、`13/400`（與 F03 已固定的三張影像相同），另加 `04/1`、`05/1`、`09/1`、`11/1`，涵蓋七個既有 sequences 的尺寸。這些影像只作數值對照，不進行選模或模型品質評估。

合成 RGB uint8：zero、one、色彩 ramp、邊界 impulse、checkerboard、stripe、deterministic random、identity，以及單列／單欄。涵蓋不整除 downsample、非正方形、upsample、singleton 與邊界。每個 case 的尺寸與原始 bytes hash 由 manifest 固定；不得以 native 結果選取 cases。

reference 預檢要求以下語義變體各在至少一個固定 case 有元素超出 §2 門檻：無 antialias、BGR、HWC 誤讀為 CHW、未除 255、uint8 resize 後取整、`align_corners=True`（無 antialias 對照）。另記錄「resize 後才除 255」的最大差；此差可能位於 FP32 預算內，因此除法先後、不 clamp 與不取整亦須由 native source／stage tests 核對，不宣稱單靠容差能區分所有這些語義。

## 4. 證據邊界

F01／F02 的數值與生命週期結果限固定 cases、受測 SDK／GPU；七張 decode 不等於全量 5,316 張已驗，未執行的 sanitizer 必須明列。此任務不關閉 F04 decode／NMS、完整 async session 整合、P01 或模型品質。發生 EOF、讀檔錯誤、JPEG 錯誤、metadata／尺寸不符與不支援格式時，不得把錯誤當作正常結束；native API 的拒絕及 teardown 另由其測試驗證。
