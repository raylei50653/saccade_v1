# MOT17／PersonPath22 離線評分

`scripts/eval_mot17.sh` 與 `scripts/eval_pp22.sh` 使用本庫的 [TrackEval MOT 子集](../third_party/TrackEval/README.md)、獨立 Python 環境及共享資料集。這兩個入口與 `score_mot17_evaluation.sh` 不再讀取舊 `saccade/` checkout，也不使用 `SACCADE_LEGACY_ROOT`。native runtime 不需要 Python。

## 安裝與使用

先安裝 `uv`，再建立 CPU 評分環境（預設 Python 3.12）：

```sh
scripts/setup_eval.sh
scripts/eval_mot17.sh build/my-mot-results MOT17-10-SDP
scripts/eval_pp22.sh build/my-pp22-results uid_vid_00008.mp4
```

輸入目錄放 `<SEQUENCE>.txt`。省略序列名稱時，評分目錄內所有 `*.txt`。評分會重新建立該目錄的 `trackeval/`，輸出 `trackeval.log`、`results/v1/pedestrian_summary.txt`、`results/v1/pedestrian_detailed.csv`，標準輸出保留 `HOTA DetA AssA IDF1 MOTA IDSW` 六個欄位。結果或 GT 缺失時，入口會在重建輸出目錄前停止。

安裝只包含 [固定版本的 NumPy／SciPy](../scripts/requirements-eval.lock.txt)，不安裝模型或訓練套件。可以從任意工作目錄呼叫評分入口；本庫與預設資料位置由腳本位置決定。

| 變數 | 預設值／用途 |
| --- | --- |
| `SACCADE_EVAL_ROOT` | `~/.cache/saccade-v1/eval`；安裝環境位於其 `venv/` |
| `SACCADE_EVAL_BOOTSTRAP_PYTHON` | `3.12`；安裝腳本傳給 `uv venv --python` 的 Python 選擇 |
| `SACCADE_EVAL_PYTHON` | `$SACCADE_EVAL_ROOT/venv/bin/python`；評分 interpreter，可指定已有 NumPy／SciPy 的環境 |
| `SACCADE_DATASET_ROOT` | 本庫同層的 `../dataset`；共享資料集根目錄 |
| `SACCADE_MOT17_GT` | `$SACCADE_DATASET_ROOT/MOT17/train`；包含 MOT17 序列目錄的位置 |
| `SACCADE_PP22_GT` | `$SACCADE_DATASET_ROOT/PersonPath22/mot_test_kf`；包含 PP22 序列目錄的位置 |
| `SACCADE_AP50_PYTHON` | `~/.cache/saccade-v1/t01/venv/bin/python`；完整 M4 scorer 使用的 AP50 interpreter，需要 NumPy、PyTorch、torchvision |

自訂 GT 位置的例子：

```sh
SACCADE_MOT17_GT=/data/MOT17/train scripts/eval_mot17.sh build/my-mot-results
SACCADE_PP22_GT=/data/PersonPath22/mot_test_kf scripts/eval_pp22.sh build/my-pp22-results
```

`setup_eval.sh` 只安裝至 `$SACCADE_EVAL_ROOT/venv`；`SACCADE_EVAL_PYTHON` 用於執行時選擇環境。AP50 與 MOT 評分可使用不同 Python；`SACCADE_EVAL_PYTHON` 現在專用於 TrackEval，AP50 的覆寫改用 `SACCADE_AP50_PYTHON`。

## 固定評分規則

TrackEval 的算法檔保留既有 vendor snapshot 的 bytes；來源、LICENSE 與 SHA-256 見 [SOURCE.json](../third_party/TrackEval/SOURCE.json)。僅限縮 package imports，MOT 評分不需 matplotlib 或 mask 套件。

兩個入口均使用 `MotChallenge2DBox`、`BENCHMARK=MOT17`、`SPLIT_TO_EVAL=train`、pedestrian 與預設 `DO_PREPROC=True`，執行 HOTA／CLEAR／Identity（另自動加入 Count）。combined 指標由 TrackEval 聚合，使用詳細 CSV 可保留完整精度。

PP22 保留追蹤原始 FPS 與 frame numbers，只將 GT 出現過的 frame predictions 送入評分；不重編 frame，也不縮短 `seqLength`。`trackeval/keyframe-filter.csv` 記錄各序列的 keyframes、input rows、kept rows。兩個入口都停用繪圖，錯誤 log 寫在結果目錄。

`score_mot17_evaluation.sh` 保留原本的 gates、revision、dirty-mode、source hashes 檢查與 MOT17-02 最後評分次序；GT 的實體根目錄必須與 capture 記錄的 `sequences` 相同，再將這個已核對的位置傳給 MOT 評分入口。這支 scorer 對應歷史 M4 流程；`run_mot17_evaluation.sh` 仍依 [TRACKING §20](TRACKING.md)標示 retired，新版 session 驗收入口是 `run_tracking_session_acceptance.sh`。該驗收腳本其他固定 artifacts／資料路徑仍依其原契約管理。

## 一次執行品質與速度評估

`eval_run.py` 以系統 Python 標準函式庫協調 native capture、TrackEval 及速度量測；評分 worker 使用上表的 `SACCADE_EVAL_PYTHON`／`SACCADE_EVAL_ROOT`。例如：

```sh
python3 -I -B scripts/eval_run.py --benchmark MOT17 \
  --build build/eval-run --out build/my-evaluation \
  /home/ray/developer/ai/dataset/MOT17/train/MOT17-10-SDP \
  /home/ray/developer/ai/dataset/MOT17/train/MOT17-02-SDP
```

`--benchmark` 可選 `MOT17`、`MOT20`、`MOT15`、`DanceTrack`、`PP22`。`--build` 必須包含 `track_sequence` 與 `tracking_benchmark`；相對 build 路徑以本庫根目錄為準。`--out` 必須是尚不存在的 `build/[A-Za-z0-9][A-Za-z0-9._-]*`，相對於本庫根目錄。輸入序列必須同屬一個 GT parent、名稱唯一，且各有 `seqinfo.ini`、其中 `imDir` 指定的影像目錄與 `gt/gt.txt`。

預設 engine 為 `/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine`，可用 `--engine` 指定；profile 固定 `hr-1280x768`，capture 固定關閉 `tracker.kalman.nsa_enabled` 與 `tracker.rows.measurement_box`。GPU capture 與 speed 全程持有 `~/.cache/saccade-v1/gpu.lock`，每個 native 呼叫均以 `timeout -k 30s 3600s` 執行，並每秒記錄 `nvidia-smi` telemetry；釋放 lock 後才評分。工具接受 dirty tree，`environment.json` 記錄 HEAD、porcelain、binary identity、兩個 binary 與 engine hashes、GPU 與 driver；binary dirty 或 commit 不符 HEAD 時，`RESULTS.md` 首行會顯示警示。

品質遵循 2026-11-11 primary156 的凍結規則：MOT17／MOT20／MOT15 分別使用同名 TrackEval adapter；DanceTrack 保留 GT 前七欄並補 class 1、visibility 1，使用 MOT17 adapter；PP22 只保留 GT 出現過的原始 frame predictions，維持 frame numbers 與完整 `seqLength`，也使用 MOT17 adapter。GT 與 predictions 全部在輸出目錄的 `scoring/` 準備，不寫回資料集。所有序列在同一次 pedestrian HOTA／CLEAR／Identity 評分中聚合，`COMBINED` 直接讀取 TrackEval detailed CSV，沒有平均各序列分數。

速度固定使用前 140 frames、20 warmup、120 measured、2 rounds、graph，共 4 個 single 與 4 個 queued fresh sessions。single mean／p50／p95／p99 取自工具 summary 的 480 個 pooled samples；queued FPS 為 `1e6 / queued_us_per_frame_mean`。summary 的 queued mean／min／max 與每個 session 的 tracker config hash 比對結果保留在 `environment.json`；若暴露的 hash 與 capture 不符則失敗。少於 140 frames 的序列不量速，註記 `fewer than 140 frames`；`--no-speed` 只略過速度步驟。

`results.csv`、`results.json`、`RESULTS.md` 包含逐序列品質、速度及一列只有品質的 `COMBINED`。JSON／CSV 保留完整精度，Markdown 品質及毫秒顯示三位小數；另保留 `command.json`、`telemetry.csv`、capture／speed／scoring logs。queued FPS 使用預載 JPEG，排除磁碟 I/O 與輸出檔案；它不是逐 frame latency、live stream 或 camera FPS，量測不鎖 clocks。

CPU fixtures（stub GPU binaries，真實 TrackEval worker）：

```sh
python3 -I -B tests/eval_run_tests.py
```

## 驗證

```sh
~/.cache/saccade-v1/eval/venv/bin/python tests/evaluation_tools_tests.py
```

這組 CPU fixtures 驗證實際 shell 入口、評分與 PP22 過濾，不執行模型推論或正式 session 驗收。

2026-11-10 本機開發驗證：獨立環境（Python 3.12.13、NumPy 2.4.4、SciPy 1.17.1）的 12 項 fixtures 通過。另對既有 G1 的 `MOT17-10-SDP`、`uid_vid_00008.mp4` 輸出各重新評分一次，每個序列的 275 個數值欄位均與原始詳細 CSV 完全相同；執行時 `SACCADE_LEGACY_ROOT` 指向不存在的位置，其他評分位置使用新預設。這是評分工具遷移驗證，非重新執行模型或 session 正式驗收。本機收據與來源／結果 hashes 位於 `build/evaluation-tools-20261110/comparison.json`（不入 Git）。
