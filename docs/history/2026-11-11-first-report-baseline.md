# 第一輪報告版本收尾

日期：2026-11-11。此版本供第一輪「原生 GPU 追蹤系統的實作與階段驗證」報告引用。核心 M0–M4 已完成，P01 在固定模型／環境範圍內 CLOSED；本輪收妥評分、共享資料與 base weights 路徑整理，建立本機 commit。各原始正式證書保留其受測來源與驗收範圍。

## 版本與部署身分

| 項目 | 身分／狀態 |
| --- | --- |
| 收尾前來源 | `e14c41ec4964b1be7a780ae1a973f496a1e59276`；另有六個 base weights 路徑／文件修改由本次 commit 收妥 |
| 已有本機提交 | `9d8cef0`：評分脫離舊 Saccade；`e14c41e`：訓練／驗證直連共享資料集 |
| 本輪報告 commit | 首次加入本文件的 commit；以下命令可取得完整 SHA |
| 發布範圍 | 僅本機 commit；本次未 push，未執行遠端 CI |
| 模型 | HR 1280×768 FP16；engine SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876` |
| Runtime 預設 | CUDA Graph ON，G1 NVOFA／`v1-motion/2`；NSA v2、量測框皆 OFF |

```sh
git log --diff-filter=A -1 --format=%H -- docs/history/2026-11-11-first-report-baseline.md
```

這個報告 commit 固定文件與工具收尾；不將其 SHA 追溯套用到較早的 GPU 正式驗收，也不把本輪既有建置目錄的 CPU 重建記為 clean-source 正式驗收。

## 報告引用入口

| 主題 | 證據與判讀範圍 |
| --- | --- |
| 架構與核心完成 | [DATAFLOW](../DATAFLOW.md)、[STATUS](../STATUS.md)、[M4 正式報告](../../evidence/m4/2026-11-03-formal-cad5298/REPORT.md)；M0–M4 完成，P01 有範圍結案 |
| 現行 G1 | [TrackingSession 報告](../../evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2/REPORT.md)；八步通過，完整 session memcheck OPEN |
| NSA v2／量測框 | [乾淨 `527ffe5` 正式報告](../../evidence/nsa-v2/2026-11-09-formal-527ffe5/REPORT.md) PASS：CPU 21/21、CUDA 70/70，80 序列 CPU／CUDA parity；預設關閉，品質沿用探索性研究結果，本次未重計 |
| F02／F03 正確性 | [乾淨 `23ade53` 整合報告](../../evidence/f02-f03/2026-11-05-clean-23ade53/REPORT.md)；原 `v1-motion/1`、七序列 5,316 幀，21/21 輸出與 M4 相同 |
| 效能 | [clean-source ABBA 報告](../../evidence/f02-f03/2026-11-05-clean-performance/REPORT.md)：queued 180.623→195.183 FPS，單幀平均 7.092724→6.626377 ms；固定 engine、舊 `v1-motion/1`、MOT17-10 140 幀 prefix，JPEG 預載，排除磁碟／初始化／暖機／輸出；queued interval 與單幀延遲分列 |
| 串流與工具 | [STREAMING](../STREAMING.md)、[P08](../P08_TOOLS.md)、[EVALUATION](../EVALUATION.md)；依各自開發證據與未涵蓋項目呈現 |

## 本輪整理與檢查

T01 recipe、HR export、外部路徑盤點與 T01 archive 程式的 base weights 入口一致改為 `/home/ray/developer/ai/dataset/yolo26m.pt`；大小 **44,255,705 bytes**、SHA-256 **`401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`** 實際核對不變。訓練文件補上新位置，既有 manifest、archive 與歷史命令內的來源路徑保持原樣。本輪未執行匯出或重製 T01 archives。

| 檢查 | 2026-11-11 本機結果 |
| --- | --- |
| `cmake --build build/cpu --parallel 2` | PASS；既有 CPU build 重建完成 |
| `ctest --test-dir build/cpu --output-on-failure` | **21/21 PASS** |
| `~/.cache/saccade-v1/eval/venv/bin/python tests/evaluation_tools_tests.py` | **12/12 PASS**；使用合成 GT／predictions，不執行模型 |
| `python3 scripts/audit_external_paths.py --include-artifacts` | exit 0；checkout、S04、T01、HR deploy、SDK 掃描均無失效連結或掃描錯誤；11 個 HR 歷史來源缺口仍列為缺失。`g1-ofa-base` 是驗收步驟建立的輸出，非事前缺件 |
| Python／JSON | 三個修改的 Python 程式通過記憶體內 `compile()` 語法檢查；recipe JSON 可解析；共享 base weights 大小／hash 符合 recipe |
| 舊評分收據核對 | 重新核對 2026-11-10 保存的 MOT17-10／PP22 輸入、reference 與輸出 CSV hashes；各序列 275 個數值欄位仍等於原 reference。本次未重跑這兩個真實序列的評分 |

本輪 [機器可讀收據](../../evidence/report1/2026-11-11-closeout/checks.json)、CPU／評分測試 logs 及原 2026-11-10 收據副本一併納入 Git；兩份原始詳細 CSV 以 gzip 保存，解壓後 bytes（含 CRLF）與原 hash 相同。原資料核對收據記錄 5,330 個檔案（5,316 張影格）符合 split manifest；本次未重做全量資料 hash。詳細本機盤點保留於 `build/report1-closeout-20261111/`。

## 保留的 OPEN 與接續

- 完整 G1／HR session memcheck、原生排隊／未過濾 Graph racecheck、nvJPEG initcheck 依 [BACKLOG](../BACKLOG.md#4-尚未完成與適用界線) 保持原判定；同步化、過濾或 tracker component PASS 不關閉完整 gate。
- P08 GPU 中間事件與觀察開關一致性、外部攝影機／RTSP 與重連、跨 GPU／SDK 支援、fatal recovery 仍依各自契約處理。
- NSA 改為預設須另做部署驗收；本次未改模型、tracker 預設或品質政策。
- 本輪沒有新增 GPU 推論、訓練、sanitizer、品質重評或效能實驗。後續報告使用上述歷史量測時，保留 source、profile、資料與計時範圍。
