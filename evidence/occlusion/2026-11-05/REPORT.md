# OAO duration-ramp＋front-occluder：MOT17-10 負結果

2026-11-05，本地開發來源 CPU replay；`formal_acceptance=false`。
依事前固定的 [PLAN](PLAN.md) 執行四臂，沒有掃描參數。

## 結論

**本輪不採用舊 preset 的遮擋懲罰。** 在目前 HR＋G1 NVOFA＋bridge 基準上，
OAO tau 0.50／ramp 25 使關聯指標退步；front 單獨幾乎持平但未改善。
兩者一起加也未優於 baseline。這個結論只適用本輪固定參數與 MOT17-10 val，
不能推論所有遮擋模型、權重或其他場景。

| 臂 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline（G1） | 50.869 | 54.407 | 47.799 | 64.975 | 66.890 | 56 |
| OAO only | 50.548 | 54.666 | 46.971 | 63.218 | 66.516 | 60 |
| front only | 50.864 | 54.396 | 47.799 | 64.915 | 66.812 | 56 |
| both | 50.634 | 54.716 | 47.103 | 62.998 | 66.680 | 67 |

相對 baseline，both 的 HOTA −0.235、AssA −0.696、IDF1 −1.977，IDSW +11。
OAO only 的 HOTA −0.321、IDF1 −1.757；front only 的 HOTA −0.005、IDF1 −0.060。
完整數值見 [summary.csv](run/summary.csv)、[summary.json](run/summary.json)，
每臂 `trackeval/results/v1/` 保留 TrackEval summary／detailed tables。

## 行為與判讀

懲罰確實有啟動：OAO only 在 24,805 次 cost evaluation 中有 20,629 次正 OAO 項；
front only 有 1,085 個 latched track-frames、1,354 次正 front 項。
Both 有 20,827 次正 OAO 項、1,683 次正 front 項。
這些是包含不同 association stages 的 evaluation 次數，不能當作獨立配對數或誤配數。

OAO only 的輸出 ID 數由 86 增至 99，both 為 104；TrackEval 的 CLEAR FN
由 3,135 增至 3,204／3,187，FP 由 1,060 降至 1,035／1,024。
DetA 小幅上升的同時，AssA、IDF1 下降。
這與懲罰切斷部分延續配對的解釋相符，屬輸出統計上的推論；
本輪未做逐次配對與 GT 的因果歸因，不能聲稱已證明每個退步事件的原因。

## 實作與驗證

- 新增 CPU-only `OcclusionExperiment` 與獨立 `occlusion_ablation` 工具。
  `associate_and_update` 的可省略 cost hook 保留原本預設路徑；
  native CUDA、tracker config／hash、wire format 與部署預設未加入遮擋項。
- 以所有存活 predicted tracks 計算 overlap，按 UID 保存 duration／TTL。
  公式、front latch、bridge／birth 的狀態選擇與舊來源差異均見 PLAN。
- 全新 Release CPU build；[CTest 19/19](validation/ctest.log) 通過，
  含 duration reset／saturation、lost／tentative、UID／epoch、front direction／TTL、
  指數整合、disabled bits、實際 matching hook 與 numeric rejection。
- baseline **654/654 幀完整 ResultPacket bytes** 與 stock `TrackerReference` 相同；
  MOT 與目前 G1 records baseline 逐位元相同。
- 四臂同一個新建 binary、同一份 SCDR／SCMO，各自執行一次。
  結束核對 source、binary、inputs、scoring dependencies 全部 unchanged。
  `py_compile`、工作樹 `git diff --check` 通過。

只讀 MOT17-10-SDP 的 GT；未開啟 MOT17-02、in-domain 或 PersonPath22 GT。
這是固定 detector／motion 的 CPU replay，沒有重跑 GPU detector、CUDA parity、
完整 session 或端到端效能測試。診斷中的 CPU 毫秒包含 harness，baseline 另含 oracle，
不得比較成 GPU／產品速度；本輪不作效能結論。
既有 v1-motion/2 驗收的 NVOF memcheck OPEN 仍依原證據，不受本輪影響。

## 來源身分與重跑

Base commit：`4a9b01f1e272bad0162f2e17b3b5ef16b5b99a90`，
`source_dirty=true`，修改與新增 source 保存在 [run](run/manifest.json) 中的
`source.diff`、`source-files.sha256.json`、`source-untracked/`。
建置設定、compile flags 與逐項測試 log 見 [validation](validation/commands.json)。

| 身分 | SHA-256 |
| --- | --- |
| HR SCDR | `98e2554f1b1db179bfd756fb1dbd37ee694e912d6debb2c7a11c3a610a433a6f` |
| G1 SCMO | `85dfd326b65a384c87fedd6aacfd9f107063543a1551163a3a773ba426c0cc15` |
| baseline MOT | `ccd1c2197e20d8a13a2695da8c07b2ee8e38ba46792c8f5f77d12e9882a8a35f` |
| 原計分 experiment binary | `a3b441c3bd594c31789b6f8f87e5135157da39f7dfde7d785f5f9af97cf58086` |
| canonical tracker config | `bc8ade7c5d05163aac7ac9659623f2371453c8c577d0e7da936fb8204c573f7f` |

舊公式與 preset 的來源為 sibling `saccade` 的乾淨 `e48f3b3`；
完整 commit 與三個參考檔 hash 見 [legacy-provenance.json](validation/legacy-provenance.json)。
舊專案的歷史增益不移作本輪數值。

```sh
cmake -S . -B build/occlusion-ablation/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/occlusion-ablation/cpu --parallel 8
ctest --test-dir build/occlusion-ablation/cpu --output-on-failure
python3 scripts/run_occlusion_ablation.py \
  --binary build/occlusion-ablation/cpu/occlusion_ablation \
  --records ../saccade_v1-worktrees/hr-records-v2 \
  --output build/occlusion-ablation/val-fresh
```

Runner 拒絕已存在的 output directory。加 `--skip-score` 可只驗證 replay，
不讀 GT。封存 run 中的絕對 build path 保留原始執行身分；搬移後以相對 artifact 路徑查閱。

實驗封存時為本地未提交來源；後續 Git 交付不改寫受測來源的 dirty 身分或正式驗收範圍。
若要探索其他 tau、contention 或局部 gate，應另訂 val 實驗；本輪沒有以多次試參挑選結果。

## 交付前整理與補驗（2026-11-05）

補上 CLI 輸出路徑檢查：MOT 與 diagnostics 解析為同一個路徑時，在讀取輸入前拒絕，
避免小工具自行覆蓋輸出。新增測試涵蓋相同路徑、`..` 別名與父目錄 symlink，
並驗證非法 arm。原 runner 的四臂都使用不同輸出路徑，原計分結果不受影響。

重建後完整 CPU CTest **20/20** 通過，見 [交付前驗證](publication-validation/ctest.log)。
再用 `--skip-score` 跑四臂，四個 MOT 均與原計分檔逐位元相同；
baseline 仍有 654/654 幀完整 packet 比對。
這次沒有讀 GT 或重新計分，上表品質由相同 MOT bytes 沿用。
OAO／front 公式與 tracker reference 未再修改，source 差異見
[source-delta.json](publication-validation/source-delta.json)。

交付前 binary SHA-256：`6a3bb5487f88991d1ffd93ee5045a7fdd264da59771c498ddb704bc0f0b20b25`。
命令、source snapshot、輸入／binary hash、四臂輸出與 unchanged 檢查保留在
[publication-replay](publication-replay/manifest.json)，
[replay-comparison.json](publication-validation/replay-comparison.json) 記錄四臂與原計分檔的比較。
原 `run/` 與 `validation/` 的 bytes、source_dirty 與 formal_acceptance 身分完整保留。
Source／文件的 staged whitespace 與本地連結檢查通過；完整 staged diff 對原始
logs、CSV、generated config 與 diff context 的 115 個 whitespace 診斷保留原 bytes，
分類見 [whitespace-check.json](publication-validation/whitespace-check.json)。
