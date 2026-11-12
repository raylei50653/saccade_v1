# 既有 G1 trace 的 n／n+1 排程重建

2026-11-12 只讀分析 2026-11-05 的 Nsight Systems 原始 SQLite。沒有新 GPU 推論、測速、計分或 runtime 改動，也不重新發出正式驗收結論。

## 可以從這份 trace 看見什麼

- 第 n 幀的 GMC／F06 可與同一幀的 F03 偵測重疊。完整 F06 定義為 `flow_input` 起點至 `flow_output` 終點，包含最後影像寫回；不含之後的 predict／person mask。
- 第 n 幀的追蹤尾段與 n+1 幀的 D CUDA 計算確實有活動交集。追蹤尾段在本分析指 `flow_mask` 加九個 M Graph kernels，結束於 packet；圖稿將 mask 與 graph tail 分開呈現。
- F03 含 kernel 間的空檔；F06 還含等待與不可直接觀察的 OFA 執行。因此另外用實際 CUDA kernel 區間聯集計算交集，不把兩個 stage 的包圍區間相交直接稱為忙碌重疊。

穩態取零起 frame index **2–29**，亦即原 JPEG **第 3–30 幀**，共 28 幀。最前兩幀的 capture／首次 Execute 成本保留於 CSV，但不放進穩態統計；相鄰穩態幀有 27 對。

| 觀察 | 保存結果 |
| --- | --- |
| 完整 F06 區間完全落在同幀 F03 內 | 28／28 幀 |
| F03 區間中位數 | 4.953100 ms |
| 完整 F06 區間中位數 | 2.6624485 ms |
| 間接 OFA 包圍區間中位數 | 2.213867 ms；不是 OFA engine 延遲 |
| n 追蹤尾段／n+1 D 有實際 CUDA 活動交集 | 27／27 對 |
| 實際 kernel 聯集交集平均／中位數 | 0.937263667／0.962797 ms |
| stage 包圍區間交集平均 | 1.128582667 ms；含空檔 |
| 其中與 n+1 F03 kernel 聯集的交集平均 | 0.908776185 ms；其餘為少量 F05 |

F02、producer、F04 的跨幀 kernel 交集為零。這並不否定邏輯上的平行能力，只描述本次 trace 的實際活動。

## 檔案與重算

```sh
python3 -B evidence/report1/2026-11-12-schedule-trace/analyze.py
```

僅需 Python 標準函式庫，不需要 GPU、模型、資料集、nsys 或原 worktree。程式核對壓縮及解壓後 SHA-256，在臨時目錄展開 SQLite，透過 `mode=ro` 讀取。

- `inputs/track.sqlite.gz`：2,400,256 B 原 SQLite 的 511,963 B gzip 副本。
- `inputs/track-graph.sqlite.gz`：1,298,432 B 原 SQLite 的 272,239 B gzip 副本。
- `source-inputs.json`：原始位置、原 manifest 雜湊、壓縮副本身分、受測來源與環境。
- `inputs/nsys.sha256`：原始歷史 manifest。兩個 SQLite 與兩個原 `.nsys-rep` 均核對通過；`.nsys-rep` 不複製，位置及 hash 留在 provenance。
- `inputs/` 其餘小檔：原始 profile 命令、source／環境、30 幀 seqinfo、舊 OFA bound CSV、輸出比對與 HR engine 收據。
- `per-frame.csv`：完整 30 幀資料，保留 frame 零起與 source_frame 一起兩種編號；尾端無下一幀的欄位留空。
- `summary.json`：28 穩態幀／27 相鄰對統計、定義、方法與限制。
- `timeline.json`：零起 20／21／22，也就是 JPEG 第 21／22／23 幀，供報告畫圖。每段為 `{frame, source_frame, stage, role, start_ns, end_ns}`，`interval_kind=stage_envelope`；實際 CUDA 區間另存 `busy_intervals_ns`。`D2H` 指 memcpy 活動，沒有當成 CUDA kernel。

代表 n=20→21 的實際 CUDA 交集為 0.964779 ms，包圍區間交集為 1.171791 ms；n=21→22 則為 0.923563／1.140622 ms。

## 幀與 stream 怎麼對上

原命令使用 MOT17-10 前 30 幀，主機依序提交。Trace 有 30 組排序一致的 F02／F05；每個 F05 的 host API 呼叫窗口內恰有一個 D Graph（310 kernels）和一個 M Graph（9 kernels）。首幀額外的 TensorRT 初始化／capture 活動不混進後續圖表。

Node capture 中 graph kernels 出現在 profiler 合成 streams。分析先由內容分成 D／M graph，再把全部 60 個 graph launches 依 host 順序與 graph-level capture 一一配對，確認實際 D stream=13、M stream=14，graph ID 對應一對一。不能把 node trace 的合成 stream 29／30 等號碼當成實際新增 streams。

## 結果界線

受測來源是乾淨 `fae5cb7c56388195b5dda0c356383b3bd9963c4f`、driver 617.42、G1／`v1-motion/2`、固定 HR engine，而不是本日重測 `3cbaa518f0dc`。Nsight Systems node 插樁會改變時序，以上數字作為排程證據，不能代替無 profiler 的延遲／FPS。

OFA 固定功能引擎本體不在這份 CUDA trace 內。`flow_input` 完成至 `flow_cells` 起點只提供 OFA 的外包界限，不是 OFA engine utilization。重疊也不證明占用率、加速幅度或無競態；原 memcheck／racecheck／initcheck 的 OPEN／FAIL 保留。
