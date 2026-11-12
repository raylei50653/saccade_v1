# YOLO26m 三方固定追蹤比較（2026-11-11）

本機實際執行三個偵測器，各涵蓋 MOT17-10 654 幀與 MOT17-02 600 幀，共 1,254 幀／方案、3,762 幀。固定 1280×768、同一 tracker、G1 motion、MOT formatter 與 TrackEval。這是開發研究比較，未調整門檻或部署預設，不是新的正式驗收。

兩序列合併時，主方案相對微調 YOLO 的 HOTA +0.8759 pp、IDF1 +0.6230 pp，MOTA −1.7314 pp；FN 多 687、FP 少 132、IDSW 少 11。微調 YOLO 的偵測覆蓋較好、主方案的合併關聯指標較好；逐序列結果並不一致，沒有三項品質指標全面勝出的方案。

官方臂指官方預訓練權重在專案 one2many＋F04 路徑的結果；微調臂為 UA2 seed 1 e4 primary EMA，保留 YOLO head、關閉 gate；主方案為現行固定 HR Mamba TensorRT FP16 engine。官方／微調臂使用 PyTorch strict FP32，主方案 scan plugins 與 input／output 為 FP32。

## 追蹤結果

所有百分比分數以 0–100 表示。FP／FN 是 TrackEval CLEAR 的計數，並非偵測 AP50 的固定 0.25 門檻計數。COMBINED 為 TrackEval 正式聚合，不是兩序列的分數平均。

| 序列 | 方案 | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | MOTA ↑ | IDSW ↓ | Frag ↓ | FP ↓ | FN ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MOT17-10-SDP | 官方 YOLO26m 權重 | 45.0460 | 47.7778 | 42.6602 | 56.7807 | 53.9528 | 54 | 181 | 1372 | 4486 |
| MOT17-10-SDP | 微調 YOLO26m UA2 e4 | 51.5723 | 54.8257 | 48.7030 | 66.0612 | 66.5005 | 68 | 164 | 1224 | 3009 |
| MOT17-10-SDP | 目前主方案 HR Mamba FP16 | 50.8689 | 54.4073 | 47.7989 | 64.9748 | 66.8899 | 56 | 150 | 1060 | 3135 |
| MOT17-02-SDP | 官方 YOLO26m 權重 | 37.0254 | 34.0692 | 40.5253 | 47.2195 | 38.9161 | 59 | 91 | 686 | 10605 |
| MOT17-02-SDP | 微調 YOLO26m UA2 e4 | 38.0758 | 43.3334 | 33.9391 | 47.2544 | 49.9812 | 101 | 120 | 1595 | 7598 |
| MOT17-02-SDP | 目前主方案 HR Mamba FP16 | 40.5653 | 40.8957 | 40.5660 | 49.1544 | 46.7843 | 102 | 136 | 1627 | 8159 |
| COMBINED | 官方 YOLO26m 權重 | 40.7294 | 39.7771 | 42.0239 | 51.5510 | 45.0605 | 113 | 272 | 2058 | 15091 |
| COMBINED | 微調 YOLO26m UA2 e4 | 44.2101 | 48.0410 | 41.0320 | 55.4167 | 56.7314 | 169 | 284 | 2819 | 10607 |
| COMBINED | 目前主方案 HR Mamba FP16 | 45.0860 | 46.4103 | 44.0802 | 56.0397 | 55.0000 | 158 | 286 | 2687 | 11294 |

## 主方案相對微調 YOLO 的差值

| 序列 | ΔHOTA（pp） | ΔIDF1（pp） | ΔMOTA（pp） | ΔIDSW | ΔFP | ΔFN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MOT17-10-SDP | -0.7034 | -1.0864 | +0.3894 | -12 | -164 | +126 |
| MOT17-02-SDP | +2.4895 | +1.9000 | -3.1968 | +1 | +32 | +561 |
| COMBINED | +0.8759 | +0.6230 | -1.7314 | -11 | -132 | +687 |

## 同次偵測 AP50

| 序列 | 官方權重 | 微調 YOLO | 主方案 |
| --- | ---: | ---: | ---: |
| MOT17-10-SDP | 72.707970% | 87.462292% | 86.015605% |
| MOT17-02-SDP | 65.513911% | 76.305649% | 72.908554% |

## 驗證與解讀邊界

- 三方均保留 NMS kept 與 suppressed 的完整候選、原始 anchor candidate ID，經同一原生 CPU tracker 與 EOF 插值輸出。NSA／measurement box 皆關閉。官方與微調臂由 native scalar F04 產生 SCDR；主方案使用原生 CUDA F04。
- 新跑主方案兩序列 SCDR 與固定部署紀錄逐 byte 相同；本次來源 CPU replay 的 MOT 與完整 packets 均重現正式 G1 參考。每個新臂亦通過完整 SCDR 與 canonical packet 驗證，沒有排除失敗幀。
- 三方共用已凍結的 G1 SCMO。它的 person mask 源自主方案偵測器；本次控制固定 motion，比較的是偵測器接到相同追蹤器的結果，沒有各自重算 detector-conditioned GMC。
- MOT17-10 已用於選模；MOT17-02 是 M4 之後重用資料的比較，不再視為新的獨立 eval。微調 Teacher 是既有兩 seed 中選定的一個；這不是跨 seed 的估計。
- PyTorch FP32 與 TensorRT FP16 的精度／backend、模型訓練配方不同，因此不能將差距全部歸因於 Mamba 架構。官方臂不是 Ultralytics 預設 one2one/end2end 的官方原生 baseline。
- 原歷史 AP50 使用 PyTorch decode；本次統一採原生 F04 的候選產生與 tie／rounding 規則，小幅數值差異需以本次 AP50 表為準。
- 本次沒有量測可比較的三方 FPS／latency，也未執行新的 CUDA session 或 sanitizer 驗收；既有 OPEN 不變。
- 獨立計分稽核通過：9 列與 detailed CSV 一致，MOTA 公式與 COMBINED 計數正確；主方案兩序列各 275 個數值欄位與既有 G1 詳細表相同，見 [scoring-review.json](methods/replay-tools/scoring-review.json)。
- 官方／微調推論時段的 75 筆 GPU 取樣：使用率平均 44.4%、最高 100%，顯存最高 3,086 MiB，溫度最高 75°C。此時段亦含初始化、scalar F04 與資料搬移，不作三方效能比較；原始遙測及環境見 [environment.json](environment.json)。

完整原始精度見 [scores.csv](scores.csv)／[scores.json](scores.json)，事前設定見 [PLAN.json](PLAN.json)，驗證見 [validation.json](validation.json)。方法、逐序列收據、MOT、壓縮 SCDR／packets、GT／像素與權重 hashes 均保留於本目錄。
