# NSA v2 與量測框正式驗收（TRACKING §22.5，2026-11-09）

**結果：PASS。** §22.5 第 1–5 項全部通過，第 6 項即本報告。預設關閉時，80 序列 offline 與 76 序列 HR native 的 MOT 都不變。開啟時，80 序列 offline 等於研究 G2，CUDA tracker 對 CPU reference 逐位元相同，HR native pipelined／serial 等於 offline。四種 sanitizer 皆 0 errors／hazards。效能只記錄，不作判定。本次未改部署預設，也未合併 main。

## 1. 受測身分與保管

| 項目 | 內容 |
| --- | --- |
| Source | `feat/nsa-v2-measurement-box` 乾淨 commit `527ffe5a1daba5771265ea489d7031e8cc2f845a`，worktree `saccade_v1-worktrees/nsa-v2` |
| 命令 | `scripts/run_nsa_v2_acceptance.sh build/nsa-v2-acceptance-20261109`：整段持有 `gpu.lock`，外層 timeout 28800 s，每個命令 `timeout --kill-after=30` |
| 規格 | TRACKING.md §22 固定段 SHA-256 `1e7b734c…2757ad`（harness 的 preflight 驗證） |
| 環境 | RTX 5070 Ti Laptop（`GPU-825ae551-ba16-539a-e444-22726b59ee9a`）、WSL2、driver 617.42；CUDA 13.4.92、Compute Sanitizer 2026.3.0.0、TensorRT SDK 10.16.1.11、CMake 4.4.5、GCC 16.2.1 |
| Build | 全新 CPU Release build，以及全新 CUDA Release build（CUDA、PREPROCESS、POSTPROCESS、MODEL_PRODUCER、TENSORRT、MODEL_TRT、DETECTION_SESSION、FLOW 全開，sm_120） |
| HR engine | `model-hr-fp16.engine` SHA-256 `8292cc23…7817876`，profile `hr-1280x768`，Graph ON（`track_sequence` 預設） |
| 開啟設定 | `enabled.config` SHA-256 `f06d17eb…10b85d`：`nsa_enabled=true`、`nsa_s0=0.9305`、`rows.measurement_box=true` |
| 釘選輸入 | 研究 `selection.json`（80 序列）、stock run.json、G2 run.json `3ccda77d…72de3e6c` 與其 artifact-manifest、PP22 69 序列清單、研究 `nsa.patch`，hash 都對上預期值 |
| 保管 | 54,316 個輸入（含 53,768 張 native 影像、80×2 份 records、全部參考 MOT）在執行前後都 hash 過，完全相同。受測 binaries 前後 hash 相同。執行期間 source 檢查從未變更，環境中沒有 `SACCADE_CFG_*` 覆寫，啟動時 GPU 閒置 |

主要 binaries：`track_offline` `33c8caba…766262d2`（CPU 與 CUDA build 相同）、`track_sequence` `a3183148…877ea377`、`tracking_cuda_tests` `00b145a4…4402ee80`。完整清單見 capture 內的 `binaries.json`。

## 2. 各項結果

整次執行共 737 個命令、無非零 exit，耗時 2,042 s。

| §22.5 | 內容 | 結果 |
| --- | --- | --- |
| 1 | CPU CTest **21/21**（51 s）；CUDA CTest **70/70**（619 s）。新增測試涵蓋 multiplier 固定值與 clamp 邊界、CPU／CUDA 共 1,000,001＋11 個 s 逐位元比對、P08 欄位的解析／限制／canonical／hash、量測框 row 開關與 birth、非有限 `r_eff` 記 NUMERIC、事件 codec | PASS |
| 2 | 預設設定：`track_offline` 80/80 與 stock MOT 逐位元相同（含 MOT20 四條 A1）；HR `track_sequence` pipelined 76/76（MOT17 7＋PP22 69）與 §20.8 的 G1 參考相同 | PASS |
| 3 | 開啟設定：`track_offline` 80/80 等於研究 G2 `mot.txt`；CUDA 對 CPU recorded parity 80 序列、62,699 幀、**467,934,820 checks** 零差異（predicted、batch、update result、含 `measurement_xyxy` 的完整狀態、rows）；HR pipelined 76/76 及 `--serial` MOT17 7/7 等於 offline | PASS |
| 4 | 開啟設定下跑合成 suite 加 MOT17-10 recorded 前 60 幀，每種工具 9,072,792 checks：memcheck、synccheck、initcheck 皆 `ERROR SUMMARY: 0 errors`；racecheck `0 hazards displayed (0 errors, 0 warnings)` | PASS |
| 5 | 只記錄，見 §4 | 已記錄 |

預設 tracker config hash 依 §22.2 改變：舊 `bc8ade7c5d05163aac7ac9659623f2371453c8c577d0e7da936fb8204c573f7f`，新 `d846e2a5b447ce2c92f7dfda77f7fd52e27ee950e4065f93d1d0b3f99c960d6f`。舊值與既有證據 log 中的值相同，兩者都在 `tracking_reference_tests` 中釘選。這只改變 canonical 文字，不影響 packets 或 MOT，第 2 項已實際驗證。

## 3. 第 3 項與品質的關係

開啟後的 offline MOT 與研究 `nsa-gmc-gate` run `20261108-001` 的 G2 臂逐位元相同，因此品質數字直接沿用研究結果（研究 merge `c387929`）。四個判定組的 ΔHOTA／ΔIDF1 為：MOT17 held-out +1.03／+1.04、MOT20 +0.06／+0.61、PP22/moving +1.45／+1.65、PP22/static +2.26／+2.71。本次未重新計分，也未讀 GT。G2 是探索性重用資料的候選，限制見 §22.1。

## 4. 效能（只記錄）

`track_sequence` HR pipelined，每序列跑 3 次，下表取中位數（ms/frame）：

| 序列 | 預設 | 開啟 | 差 |
| --- | ---: | ---: | ---: |
| MOT17-02 | 5.232 | 5.288 | +1.06% |
| MOT17-04 | 5.232 | 5.275 | +0.82% |
| MOT17-05 | 5.035 | 5.066 | +0.61% |
| MOT17-09 | 5.205 | 5.228 | +0.46% |
| MOT17-10 | 5.263 | 5.288 | +0.48% |
| MOT17-11 | 5.230 | 5.248 | +0.34% |
| MOT17-13 | 5.251 | 5.270 | +0.36% |
| 依幀數加權 | 5.208 | 5.236 | +0.54% |

MOT17-04 records 的 tracker 單段 GPU 時間（1,050 幀平均）：預設 403.27 µs，開啟 406.07 µs。

判讀限制：harness 先跑完預設臂的 21 次，再跑開啟臂，兩臂沒有交錯（非 ABBA），所以約 0.5% 的差距無法和執行順序／溫度漂移分開。開啟設定每次 update 只多一次 multiplier 計算與量測框複製。這些數字不構成效能結論。

## 5. 執行經過

- 2026-11-08 曾在 `6fb2db5` 執行一次（`build/nsa-v2-acceptance`），在第 3 項 HR pipelined 第 18/76 序列時由使用者停止。當時第 1、2 項已通過；第 3 項的 offline＝G2 80/80 與 CUDA／CPU recorded parity 80/80 已通過；已跑完的 18 個 native 比對也都相同。該次未完成，不作為證據。
- 重跑前審查 branch，唯一的變更是 `527ffe5`：`docs/P08_TOOLS.md` 的事件 schema 名稱由 `tracking-events/1` 改為 `/2`。這個檔不在 harness 的 source hash 範圍內，tracker 程式碼與測試未變。
- 開發期的檢查與偏差（NUMERIC 修正 `614a021`、SDK 路徑修正、兩次 dry run）見 [2026-11-08/NOTES.md](../2026-11-08/NOTES.md)。

## 6. 未涵蓋

- `track_stream` 的串流 profile、改成預設之後的部署驗收、品質重新計分（依 §22.5 不涵蓋）。
- 完整 HR session sanitizer：§20.9、§20.10 的 OPEN 照舊，NVOF memcheck（Bug 7048273）仍 OPEN。本次第 4 項只涵蓋 tracker 的合成 suite 加 60 幀 recorded。
- 事件檔 schema 升為 `tracking-events/2`（track snapshot 多了 `measurement_xyxy`），既有 `/1` 事件檔需用原版 reader 讀取。ResultPacket 與 MOT 格式不變。

## 7. 檔案

- [SUMMARY.json](SUMMARY.json)：機器可讀的判定、hash 與各次計時。
- [raw/capture.tar.gz](raw/capture.tar.gz)：完整 capture，包含每個命令的 command／log／exit／`time -v`、`source.json`、`inputs.json` 與 `inputs-after.json`（54,316 個 hash）、`binaries.json`、`outputs.json`（全部輸出 MOT 的 hash）、`timings.json`、`status.json`，以及兩份 CTest `LastTest.log`。輸出 MOT 檔本身留在 build 目錄，未封存。
- [SHA256SUMS](SHA256SUMS)。
