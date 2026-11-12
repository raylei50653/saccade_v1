# TensorRT 三方性能與品質比較（2026-11-11）

同條件 TensorRT 下，主方案相對微調 YOLO 的整體 MOTA +2.9244 pp、HOTA +2.0527 pp、IDF1 +0.1991 pp；模型 Graph GPU 平均耗時增加 25.76%。應依下面完整路徑與資料角色判讀，不能將此整體品質差距視為未見場景的普遍收益。

對微調原 YOLO，本機完整 tracking 的平均 single 延遲增加 **14.79–17.42%**，queued 吞吐下降 **15.13–15.52%**；engine 檔案由 42.85 MB 增至 60.75 MB（+41.77%），execution-context scratch bound 則同為約135.17 MB。主方案未呈現速度、檔案大小或 context scratch 的優勢。MOT17-10 的 single p95 反而微幅較低（7.1388 vs 7.1776 ms），兩者相近；MOT17-02 主方案 p95 較高，不能宣稱尾延遲在所有場景都較好或較差。

只看 02＋10，主方案／微調版 MOTA 為 **55.0000／56.9828**，HOTA **45.0860／44.3321**，IDF1 **56.0397／55.8413**。因此這是一項品質與成本的取捨，尚未建立對未見場景的普遍優勢；若優先追求吞吐、平均延遲與較簡單的原始 YOLO 架構，微調 YOLO 是這次資料中較有效率的選擇。本次維持部署不變。

## 給決策者的主要表格

以下完整追蹤計時使用 MOT17-10 前 140 幀、20 幀暖機、四次 fresh session／480 個量測樣本；CUDA Graph 開啟、G1、candidate／motion／result slots=2／2／4。JPEG 預載，排除檔案讀取、初始化、暖機與寫檔；single 是 enqueue→receive 交付延遲，queued FPS 是整段完成吞吐，兩者不互換。品質為三方實際 TRT 原生全七序列各自 G1 結果。

| 方案 | F03 Graph GPU mean ms ↓ | 完整 tracking single mean ms ↓ | single p95 ms ↓ | queued FPS ↑ | 七序列 MOTA ↑ | HOTA ↑ | IDF1 ↑ | engine MiB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 官方 YOLO26m raw6 | 3.744346 | 5.734325 | 6.938432 | 245.175 | 45.2755 | 46.0107 | 55.5969 | 42.506 |
| 微調 YOLO26m raw6 | 3.809107 | 5.937670 | 7.177594 | 230.123 | 73.2914 | 61.0302 | 71.3287 | 40.868 |
| 目前 HR Mamba | 4.790332 | 6.815614 | 7.138781 | 194.417 | 76.2157 | 63.0829 | 71.5278 | 57.939 |

## 完整路徑的兩個場景

| 序列 | 路徑 | 方案 | single mean ms | single p95 ms | queued interval ms/frame | queued FPS | queued delivery mean ms | queued delivery p95 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MOT17-10-SDP | detect | 官方 YOLO26m raw6 | 7.840573 | 13.616808 | 5.015996 | 199.362 | 19.766573 | 21.873518 |
| MOT17-10-SDP | detect | 微調 YOLO26m raw6 | 6.876933 | 7.405998 | 5.068365 | 197.302 | 19.965705 | 22.022040 |
| MOT17-10-SDP | detect | 目前 HR Mamba | 8.847712 | 11.774563 | 6.589398 | 151.759 | 25.989777 | 28.459847 |
| MOT17-10-SDP | tracking | 官方 YOLO26m raw6 | 5.734325 | 6.938432 | 4.078723 | 245.175 | 16.075161 | 16.953768 |
| MOT17-10-SDP | tracking | 微調 YOLO26m raw6 | 5.937670 | 7.177594 | 4.345494 | 230.123 | 17.130361 | 19.908373 |
| MOT17-10-SDP | tracking | 目前 HR Mamba | 6.815614 | 7.138781 | 5.143574 | 194.417 | 20.281105 | 21.309439 |
| MOT17-02-SDP | detect | 官方 YOLO26m raw6 | 6.853928 | 7.390089 | 5.010287 | 199.589 | 19.739822 | 21.578470 |
| MOT17-02-SDP | detect | 微調 YOLO26m raw6 | 7.012852 | 7.786508 | 5.113811 | 195.549 | 20.147762 | 22.015258 |
| MOT17-02-SDP | detect | 目前 HR Mamba | 8.551140 | 9.071566 | 6.632608 | 150.770 | 26.136182 | 28.885932 |
| MOT17-02-SDP | tracking | 官方 YOLO26m raw6 | 5.631624 | 5.881876 | 4.056629 | 246.510 | 15.983438 | 16.760612 |
| MOT17-02-SDP | tracking | 微調 YOLO26m raw6 | 5.771343 | 6.023891 | 4.373497 | 228.650 | 17.238364 | 19.983974 |
| MOT17-02-SDP | tracking | 目前 HR Mamba | 6.776592 | 6.999336 | 5.152909 | 194.065 | 20.316156 | 21.215789 |

DetectionSession 目前只支援 ordinary enqueue，因此 detect 表是相同原生 ordinary detection 路徑；tracking 才開啟部署 D/M Graph。這兩條是分別直接量測的範圍，不能以兩個均值相減推算各 stage 的精確耗時。

## 模型層與原始 one2one 控制

模型計時包含同一 stream 的 TensorRT enqueue 或 Graph launch，GPU events 與 CPU wall 分列；四張相同真實輸入的 H2D、D2H、檢查、初始化、暖機及磁碟均排除。每組四次 fresh process、每 mode 1,280 筆單次樣本；另外 2,048 次 back-to-back replay 量吞吐。

| 比較組 | 方案 | mode | GPU mean ms | GPU p95 ms | CPU wall mean ms | queued GPU interval ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| raw6 | main | ordinary | 6.398946 | 6.839392 | 6.446679 | 6.373147 |
| raw6 | main | graph | 4.790332 | 4.892928 | 4.837373 | 4.794414 |
| raw6 | official | ordinary | 4.793083 | 5.164032 | 4.843115 | 4.779443 |
| raw6 | official | graph | 3.744346 | 3.842496 | 3.789450 | 3.737483 |
| raw6 | finetuned | ordinary | 4.846997 | 5.183712 | 4.899112 | 4.837291 |
| raw6 | finetuned | graph | 3.809107 | 3.893696 | 3.855261 | 3.792729 |
| original-default | main | ordinary | 6.448510 | 6.899296 | 6.502607 | 6.540059 |
| original-default | main | graph | 4.871181 | 4.955104 | 4.928590 | 4.874392 |
| original-default | official-default | ordinary | 4.849758 | 5.197568 | 4.895865 | 4.833807 |
| original-default | official-default | graph | 3.861927 | 3.915904 | 3.905611 | 3.877900 |

official-default 是官方原始 one2one/end2end、output0[1,300,6]，engine 內含 decode／TopK，與 raw6 表的輸出終點不同，另外與主方案做平衡對照。沒有把它冒稱成 raw6／同一 F04 的品質等價替代；本次其品質及完整追蹤未重評。

## TRT 品質與資料角色

| 序列 | 官方 MOTA | 微調 MOTA | 主方案 MOTA | 官方 HOTA | 微調 HOTA | 主方案 HOTA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MOT17-02-SDP | 38.625478 | 49.652871 | 46.784350 | 36.760927 | 37.709811 | 40.565286 |
| MOT17-04-SDP | 45.072229 | 84.734109 | 92.463780 | 49.473983 | 70.671622 | 74.802259 |
| MOT17-05-SDP | 40.378777 | 69.799046 | 68.960532 | 43.149881 | 59.769679 | 60.827415 |
| MOT17-09-SDP | 57.539906 | 81.164319 | 81.840376 | 45.118201 | 57.251399 | 57.734295 |
| MOT17-10-SDP | 53.656827 | 67.590934 | 66.889945 | 44.735061 | 52.200411 | 50.868904 |
| MOT17-11-SDP | 48.844850 | 76.473082 | 77.903773 | 51.454402 | 67.119273 | 65.607966 |
| MOT17-13-SDP | 41.882838 | 66.457653 | 67.471225 | 41.466679 | 56.901456 | 56.306295 |
| COMBINED | 45.275475 | 73.291361 | 76.215749 | 46.010671 | 61.030214 | 63.082927 |

04／05／09／11／13 用於模型訓練、10 用於選模、02 是 M4 後重用的原 held-out；訓練序列占整體有效 GT 約72%。`quality-subgroups.json` 保留真實 TrackEval 分組聚合；不得以七序列總分代替未見場景的判定。`quality-scores.json` 另保留相同 main-derived G1 的 fixed-motion CPU replay，用於分開看 detector 與 detector-conditioned GMC。

## Engine、精度與驗證

- 原始權重、微調 primary EMA 固定；每個 ONNX 的 542／542 initializer 逐 byte 對到來源權重，四張 FP32 PT／ONNX 數值核對通過。相同 SDK 10.16.1.11、batch1、1280×768、FP16 allowed、TF32／BF16／INT8 關閉、opt3、aux0、workspace256MiB；I/O FP32。實際 layers／formats 與 builder readback 已保存，沒有僅憑 FP16 flag 宣稱全部算子為 FP16。
- 主方案維持固定部署 engine，scan plugins 與 I/O FP32。研究 adapter 只讓 YOLO raw6 支援零 engine-plugin-status，仍保留全 input／output 有限值與真實 producer status。四個正常／NaN 定點負控制通過，無合成 scan status 或關閉品質檢查。主方案七序列 SCDR／motion／MOT／packets 重現原部署，三方全部 native 對 CPU 自身 motion 的 MOT＋packets 逐 byte 相同。
- FP16 原定 1% raw relative-L2 screen **FAIL 保留**：官方最大 2.406%、微調最大1.537%；實際 finite／repeat／Graph bitwise 檢查通過。相同官方 ONNX 的 FP32/no-TF32 控制最大6.497e-6，支持差異來自精度／tactic，而非錯載權重或接錯 graph。完整品質以本次 TRT 重新評分，不繼承上一輪 FP32 偵測分數。官方 decoded Top300 亦有集合／順序差異，未宣稱輸出等價。
- `model-memory.json` 的 getDeviceMemorySizeV2 為 execution-context scratch bound，explicit I/O／status allocations 另列；cudaMemGetInfo 與 NVML 是整張 device 的觀察，不能解讀為本行程私有顯存或單幀能耗。Engine 檔案大小不是完整推論顯存。
- 兩輪 raw M/O/F/F/O/M、另兩輪 M/default/default/M；模型共20 fresh processes，完整 detection／tracking 共96 fresh sessions。單一 GPU flock 橫跨兩個 campaign，量測時沒有建置、GPU 診斷或 profiler。未刪除長尾；以 nearest-rank 算分位數。沒有測試攝影機曝光到外部消費者的全系統延遲。
- 本次是本機、固定 GPU／SDK／模型／輸入的研究比較；沒有切換部署、提交或發布，也不是新的 sanitizer／跨 GPU 正式驗收。原有 OPEN／FAIL 保留。

TRT 檔案實體位置與 hashes 見 [ARTIFACTS.json](ARTIFACTS.json)，數值／建置交付見 [delivery.json](methods/export-tools/delivery.json)，品質、模型與 pipeline 的所有原始精度樣本／配置／hash 收據均在本目錄；大型原始建置與全量 dumps 留在 `build/yolo26m-trt-comparison-20261111/`。

獨立最終稽核：20 個模型程序與 96 個 pipeline 程序的完整矩陣、原始樣本、nearest-rank 分位數與吞吐公式通過核對；全部96份 pipeline 結果逐 byte 等於全序列原生品質輸出的前140幀（SCDR 僅調整檔頭幀數），包含暖機、unused／reserved bytes。見 [統計稽核](methods/pipeline-tools/benchmark-statistics-review.json)與[結果前綴稽核](methods/pipeline-tools/benchmark-prefix-review.json)。分組 vendor combine 對整數計數精確，浮點控制最大僅1ULP；過嚴統一絕對誤差檢查的先前 attempt 另存，沒有改動資料或重新量測。

本次比較不是只有 Mamba 架構一項因素不同的隨機消融：模型訓練配方、score calibration 與精度計算路徑亦不同，因此不能把全部品質差距歸因於 Mamba。單純微調原 YOLO 已取得大部分相對官方預訓練權重的品質提升。
