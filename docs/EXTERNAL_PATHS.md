# 外部路徑與符號連結

盤點日期：2026-11-06。本頁記錄本機檔案依賴與連結整理；GPU／模型／sanitizer 的驗收判定仍由 [STATUS](STATUS.md) 與各原始報告定義。

## 目錄分工

| 位置 | 用途與引用入口 |
| --- | --- |
| 本專案 `src/`、`cuda/`、`include/`、`tests/` | runtime、公開契約與測試；這些目錄目前沒有 symlink |
| 本專案 `build/` | 忽略於 Git 的建置、測試、replay 與本機盤點結果；其中的輸入連結可能指向庫外 |
| `~/.cache/saccade-v1/t01/` | T01 pinned source、兩套 venv、checkpoint、ONNX、訓練與 native artifacts；根目錄設定見 [T01](T01_TRAINING.md) |
| `~/.cache/saccade-v1/hr-deploy/` | HR engine、ONNX、head bundle、references 與 manifest；[部署操作](../training/hr_deploy/README.md) |
| `~/.cache/saccade-v1/s04/` | 歷史 head artifacts；`best.ckpt` 仍連到舊庫的訓練結果 |
| `~/.cache/saccade-native-trt/10.16.1.11/sdk/` | pinned TensorRT SDK；三個共享函式庫版本連結都在 SDK 內 |
| `../saccade/` | 舊庫來源、base weights、資料集、TrackEval 與評分 Python 環境；[T01 recipe](../training/t01/recipe.json)、[評分入口](../scripts/eval_mot17.sh) |
| `../saccade_v1_dev/` | HR teacher／student checkpoints、cache 與研究用 Python 模組；[匯出程式](../training/hr_deploy/export.py) |
| `../saccade_v1-worktrees/` | 驗收 records、flow、M4／G1 固定輸出及獨立建置；[TrackingSession 驗收入口](../scripts/run_tracking_session_acceptance.sh) |
| `../dataset/MOT17/` | MOT17 實體資料；目前由 `../saccade/datasets/MOT17` 的連結抵達 |
| `../dataset/PersonPath22/` | PersonPath22 實體資料；由 `../saccade/datasets/PersonPath22` 的連結抵達 |

上表的 `../` 以目前主 checkout 為基準。部分既有入口仍固定本機絕對路徑；從其他 checkout 執行前應查看該入口，不能只搬動 repository 就視為完整搬機。

## 專案內連結

2026-11-06 整理前的起始快照共 108 個，全部位於 `build/`，沒有 Git-tracked symlink。

| 類別 | 數量 | 目標與用途 |
| --- | ---: | --- |
| MOT17 圖片 | 19 | `../saccade/datasets/MOT17/train/MOT17-10-SDP/img1`；各 prefix 共用圖片，保留各自的 sequence 長度 |
| T01 產物與來源 | 7 | T01 cache 的 student／teacher checkpoint、backbone ONNX、export metadata、`runs/` 與 `src/` |
| 寫入失敗測試 | 45 | `/dev/full`；測試 write／close 錯誤傳播，建立方式見 [SyntheticIOTest](../cmake/SyntheticIOTest.cmake) |
| 內部 fixtures | 37 | manifest alias 與 parent-directory alias；其中七個搬移前的絕對路徑已修復 |

這 71 個外部連結的目標均存在。七個失效的 `alias.json` 已改為同目錄的相對目標 `contract.json`：`build/d2/` 兩個、`build/preprocess-evidence/` 一個、`build/t01-f03-evidence/` 兩個、`build/t01-head-evidence/` 兩個。修復前逐一核對同目錄 contract 與 tracked fixture 的 SHA-256 相同；新產生的 manifest alias 也使用相對路徑。舊測試 logs 與判定保留原樣。

## Cache 與環境的額外連結

- `s04/best.ckpt` → `../saccade/runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt`，目標存在。S04 的 manifest 是歷史記錄，保留原始 bytes。
- T01 cache 的六個 artifact links 原本有一個反向依賴本專案 `build/`。`t01/f03/stage/t1-head.swb` 已改連到同一 cache 內的 `../../native/dev/t1-head.swb`；兩份 bundle 的 SHA-256 都是 `4fccfda6342b88d43fc2bf676228810b974b9e9956f45fa53352124098f046a3`。其餘五個 artifact links 都在 T01 cache 內。
- T01 的 `venv`／`venv-export` 各有三個 Python 連結，最終抵達 `~/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`，另各有一個內部 `lib64 → lib`。兩套 venv 仍依賴這份 uv Python。
- `hr-deploy/` 本身沒有 symlink。其 manifest 以一般路徑記錄 teacher／student／base weights 與歷史來源。

## 歷史來源與新建輸出

HR `manifest.json` 的 17 個來源路徑有 11 個位於已移除的 `saccade_v1-worktrees/d1-a/`。目前 engine、teacher、student 與 base weights 仍存在；原 manifest 的路徑及 hashes 保留。這 11 個缺失列為歷史來源路徑缺口，沒有改指目前 checkout 或重新產生 manifest。恢復來源時須逐檔符合原 hash，另附恢復對照，不能用目前檔案替換歷史身分。

`saccade_v1-worktrees/g1-ofa-base` 是 [TrackingSession 驗收腳本第 8 步](../scripts/run_tracking_session_acceptance.sh)才用 `git worktree add --detach ... 23ade53` 建立的輸出。執行前不存在符合流程，應與必須事先存在的 records／engine 輸入分開判讀。

## 重跑盤點

```sh
python scripts/audit_external_paths.py
python scripts/audit_external_paths.py --include-artifacts
```

[盤點工具](../scripts/audit_external_paths.py)只讀取檔案系統，JSON 寫到標準輸出。第一個命令掃描整個 checkout（含隱藏與忽略目錄），不進入 symlink 目錄；第二個另掃描 S04、T01、HR deploy 與 TensorRT SDK，並列出固定的本機依賴路徑、連結祖先及 HR 歷史來源是否存在。它不掃描所有相鄰專案，也不重新執行模型或驗收。

工具支援 `--repo PATH`。`SACCADE_T01_ROOT`、`SACCADE_LEGACY_ROOT`、`SACCADE_HR_DEPLOY`、`SACCADE_TENSORRT_ROOT` 可覆寫盤點所用的相應外部位置；這不會改動既有腳本的設定或其固定輸入契約。其餘固定參照使用目前使用者的 `~/developer/ai/` 配置。

Exit 0 表示掃描範圍沒有失效連結或掃描錯誤；exit 1 表示發現失效連結或掃描錯誤；exit 2 表示工具無法完成盤點。缺少的一般輸入、未來輸出與歷史來源另列於 JSON，均不由 exit 0 推導為可重建或驗收 PASS。產物根目錄缺失或本身是 symlink 時，外部掃描會明確標示 skipped。

這次本機調整的前後完整清單與八個連結的舊／新目標記錄於 `build/external-paths-20261106/{before,after,repairs}.json`。該目錄不入 Git；清單是本機快照，後續 CTest 或新建驗收輸出可能增加連結數量。
