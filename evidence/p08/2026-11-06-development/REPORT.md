# P08 設定與 CPU 事件工具：本機開發驗證

2026-11-06。固定起點 `7dfd51319a1c2c0ae841b9bf777d20f63c06487b`，本輪為
**source_dirty=true、formal_acceptance=false** 的本機修改，沒有 commit／push。
已重建 CPU 與本文指定 native binaries；來源檔案與受測 executable 的 SHA-256／size
見 [provenance.json](provenance.json)。不回寫既有正式證書。

## 實作與判定

- 共用 `saccade-tracking-config/1` resolver 接上 `track_offline`、`track_sequence`、
  `track_stream`，保留各入口原本的有效預設。檔案、已宣告環境與 CLI 依固定優先序
  展開，輸入資格／組合在提交 frame 前驗證；記錄設定 hash、逐項來源與獨立 build 身分。
- SDK-independent `tracking_config` 提供 schema、解析、收據核對及明列 allowlist 的設定比較。
- `tracking-events/1` 編碼實際 CPU 決策點的事件；`tracking_events` 提供獨立 lifecycle／
  身份／提交模型、第一分歧及完整 ResultPacket byte 綁定。事件、packet、MOT 的比較分開。
- SCDR／SCMO 由同一份 immutable bytes 解碼與計算 input hash；新增 diagnostic／receipt
  使用 O_EXCL，檢查 MOT `.tmp` 與輸入／其他輸出的碰撞。

使用方式與完整契約見 [P08_TOOLS](../../../docs/P08_TOOLS.md)。本輪沒有修改部署參數、
Bridge 算法、CUDA kernels 或正常路徑的每幀 D2H。

## 實際驗證

| 檢查 | 結果與收據 |
| --- | --- |
| 最終 CPU CTest | **21/21 PASS**；[完整 log](cpu-ctest.log)、[各測例輸出](cpu-tests-detail.log) |
| 設定單測 | 300 checks；schema metadata、拒絕條件、預設／覆寫／來源、FP32 bits、hash、能力與收據 tamper |
| 事件單測 | 235 checks；合法／非法 trace、codec、同幀 birth＋confirm、tentative remove、lost／expiry、Bridge donor、失敗／skipped／overflow 回滾、packet reserved／unused bytes |
| CLI 合約 | 43 checks；真實 CLI 的設定覆寫、來源、收據、全 packet／MOT／事件重跑、輸出保護、第一分歧與失敗前綴 |
| Motion／MOT 單測 | 691／258 checks；含 immutable-byte decoder 與截斷拒絕；既有 tracker reference 1,750,999 checks |
| 正式依賴隔離 | CPU closure＋新增兩個 CLI 的 compiled source／include dependencies／symbols PASS；[收據](cpu-isolation.json)。本輪沒有重新執行物理移開研究根目錄的 build |
| 既有串流 CLI | **35/35 PASS**；[原開發收據](stream-cli-development.json)，對應其自己的 binary hash；含 CPU／NVDEC、H.264／HEVC 8／10-bit、Graph／serial bytes、取消／來源保密／輸出競態 |
| 新 P08 native CLI | 首輪 **18/18 PASS**：[收據](native-cli-development.json)。共用 header／decoder 收尾後重新 build 並執行最終 **18/18 PASS**：[收據](native-cli-final.json)、[CTest log](native-final-ctest.log) |

Native 驗證使用 `flock ~/.cache/saccade-v1/gpu.lock`；本機 RTX 5070 Ti Laptop、
CUDA 13.4.92／driver API 13040、TensorRT 10.16.1.11。固定 HR engine：
`8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`。
新 native 測試包含 source-open 前的錯誤拒絕、覆寫優先序與來源、相同有效值的 hash、
receipt self-parse、9 幀 stream packets／MOT 相同，以及真實 JPEG 9 幀的 HR sequence
入口與收據。兩輪是各自的測試收據，不加總為新的獨立樣本數。

最終 native executable 身分見 provenance；其 default／precedence／sequence 收據另存於
[native-receipts](native-receipts/)。完整本機 logs、影片與 private sequence view 留在
`build/p08/native-validation/`、`build/p08/native/p08-cli-tests/run-dp1k27gi/`。
較早一次串流測試因 dataset 目錄拼字於首張檢查即停止，沒有觸及 GPU，空 run
`build/p08/native-validation/stream/run-75o6vfkk/` 保留；之後用實際來源完成前述測試。

## 固定 records 回歸

完整命令、input／source／binary 前後 hashes 與各序列結果：
[records-validation.json](records-validation.json)。來源是既有 HR／G1 records，
**不讀 GT、不調參、不重新計分**。Baseline 從 HEAD 的選定 source paths 以 `git archive`
實體化在 `build/p08/baseline-source/`，獨立 build 為 `build/p08/baseline-cpu/`。

- MOT17 七序列共 **5,316 幀**：新版 MOT 與 HEAD baseline、七份 frozen MOT 全部逐 byte 相同。
- MOT17-10 共 **654 幀**：observer OFF／ON／repeat 的完整 packets 與 MOT 逐 byte 相同。
- ON／repeat 的 **60,056 筆事件**全 bytes 相同；模型、source／epoch／input frame count
  與完整 packet 綁定皆通過。設定比較只允許 `observer.event_dump` 差異。
- 事件 SHA-256：`c300b7614e5622fb5d5195bc4a925f11dbca084427bfdfef143015bce4e7a1c5`。
- 新單測另逐 frame 比較 observer ON／OFF 的全部 committed-state 欄位 bits，含未使用
  history；真實 records 的本輪對照範圍為事件、packets、MOT，沒有額外匯出完整 state。

完整 outputs／events／receipts／logs 保留於
`build/p08/records-validation/run-20261106T140900.941325Z/`；大檔未複製進 Git。
本報告的 manifest 只對應明列保存材料，不宣稱 build 目錄已完成外部備份。

## 重跑入口

```sh
cmake -S . -B build/p08/cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/p08/cpu --parallel 6
ctest --test-dir build/p08/cpu --output-on-failure
python3 tests/p08_cli_tests.py --bin-dir build/p08/cpu
```

隔離 checker 需要 configure 前建立 File API query（見 RESEARCH_BOUNDARY），再執行：

```sh
python3 scripts/check_formal_isolation.py --source . --build build/p08/cpu \
  --target tracking_config --target tracking_events
flock ~/.cache/saccade-v1/gpu.lock ctest --test-dir build/p08/native \
  --output-on-failure -R '^tracking_p08_native_cli$'
```

Native build 開啟 CUDA／TensorRT／MODEL_TRT／PREPROCESS／POSTPROCESS／MODEL_PRODUCER／
DETECTION_SESSION／VIDEO／NVDEC／STREAMING，使用既有 SDK root、HR engine 與 MOT17-10
sequence cache 設定。新 native 測試會保留自己的全新 run，不覆寫舊結果。

## 尚未涵蓋

**P08 未整體結案。** GPU 的完整中間 lifecycle／Bridge event dump 及其觀察開關一致性
仍 OPEN，CUDA `observer.event_dump=true` 明確拒絕，不能從 ResultPacket 推回未保存的轉移。
VideoSource 格式／timeout／transport／frame-limit 仍是獨立 typed CLI，另列收據，
不由 tracking config hash 覆蓋。

模型不重新計算 GMC／Kalman 的 x／P，也不重解 auction／Bridge 幾何與排名；CPU 合法性
與 byte equality 不能證明 GPU 無競態。本輪沒有完整 GPU CTest／sanitizer／效能或新品質
驗收；G1 memcheck、原生完整 HR racecheck 及既有 forced-sync／filtered 界線維持原判定。
