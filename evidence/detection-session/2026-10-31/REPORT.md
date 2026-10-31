# Native F01–F04 bounded async detection session

2026-10-31。新增 `f01-f04-session/1` 已實作，**正常使用與明列範圍開發驗證 PASS**，最終 capture exit 0。這是 **未提交工作樹的開發來源驗證**：base HEAD 為 `a9c3d9bfecb61fdea8850b3fdf2308f09aace629`，`source_dirty=true`、`allow_dirty=true`，不把 base HEAD 當成含有 session 的 clean-source commit。乾淨 `a9c3d9b` 的 F03 producer 正式驗收已 [另行封存](../../f03-producer/2026-10-31-formal-a9c3d9b/REPORT.md)；兩份證據的 source／判定互不替代。

## 實作與所有權

[`DetectionSession`](../../../docs/DETECTION_SESSION.md) 擁有兩條 explicit non-default D／M streams：D 依序執行原 HARDWARE nvJPEG F01、FP32 antialiased stretch F02、固定完整 TensorRT F03、producer 聚合器與 F04；M wait CS ready 後讀完整候選、核對身份、提交成功或鎖定首次失敗，最後交付固定容量 detection result。CS 的最後讀取後 record consumed，D 下次覆寫該 slot 前必須 wait。單份 device result 的 finalize／D2H 靠 M 順序；pinned HP slot 在 host receive 副本後才歸還。沒有中間 count／status D2H。

保留完整 kept／suppressed pool、candidate identity／排序與未用 rows，沒有用 person-only 或只交付 kept 的輸出替代 F04。首次 failure 記下 fail_frame／fail_code／最後 committed_frame，後續 queued 結果為 skipped、零候選，不能恢復成功。Host API／metadata／frame 錯誤停止新提交，可能留下 D-only tail；finish／cancel／close 各自 drain D 及 M，destructor 為 best effort。R 個未讀結果時以提交前的 recoverable length_error 施加 backpressure，不丟幀。

## 固定來源、輸入與保存

| 項目 | 身分 |
| --- | --- |
| Base HEAD | `a9c3d9bfecb61fdea8850b3fdf2308f09aace629`，新程式以 source archive 固定 |
| Build-input manifest SHA-256 | `5dea548b610a2db3e1f223b6371a7a7ee28d2714219e7521dafb133e4fc62af5` |
| F03 engine SHA-256 | `c6cff1936194e679ce11c3736cab552e4a73a2b1a9ae5508bc3e5d1b3fbab9c5` |
| Corpus | 七張真實 baseline 8-bit YCbCr 4:2:0 JPEG；各 JPEG／RGB／FP32 的 frozen hashes 固定於測試 source 與 [artifact manifest](capture/artifacts.sha256) |
| GPU／driver | RTX 5070 Ti Laptop、sm_120、617.42，WSL2 |
| Toolchain | GCC 16.2.1、nvcc 13.4.92、TensorRT 10.16.1.11、nvJPEG 13.2.3、Compute Sanitizer 2026.3.0.0 |
| Resource scope | MemoryMax=12 GiB、MemorySwapMax=0；[launcher metadata](capture/scope.json) |

以全新 `build-detection-session-evidence-final/` 重建，保存 [source status](capture/source-status.txt)、[tracked diff](capture/source.diff)、包含 untracked build inputs 的 [archive](capture/source-build-inputs.tar.gz) 及 [manifest](capture/source-files.sha256)，127 個 build inputs 已逐檔對上 archive。原始 capture 在每個步驟前後核對 source revision／digest，不能以 report 或後續導覽變更代替凍結 build inputs。啟動器輸出 log 亦隨本報告保存。

所有命令、起始時間、退出碼、`time -v`、[CMakeCache](capture/CMakeCache.txt)、[compile commands](capture/compile_commands.json) 與 [capture status](capture/capture-status.txt) 均保存。[Binary hashes](capture/binaries.sha256)／[測試後 hashes](capture/binaries-after.sha256)、engine／corpus 的前後 hashes 均須一致。dependency 路徑與 file hashes 由 [ldd resolution](capture/runtime-dependencies.json) 保存；這不是 live process maps 或 executable pages dump。

實際 CPU／CUDA build 及大型 engine／reference corpus 留於本機既有 ignored directories；actual executable／archive bundle 為本機 `build-detection-session-evidence-final/actual-binaries.tar.gz`，hash 見 [local artifacts](capture/local-artifacts.sha256)。bundle 不包含外部 SDK／driver／engine／JPEG corpus，不宣稱獨立自足或 CUDA rebuild 逐位元可重現。

## 結果

| 檢查 | 結果與範圍 |
| --- | --- |
| Default CPU CTest | **13/13 PASS**；包含新增 detection happens-before model |
| CUDA／session／preprocess／producer／F04 CTest | **10/10 PASS**；[summary](capture/ctest.log)、[完整輸出](capture/LastTest.log) |
| 最終正常 session suite | **4,293 checks PASS**；[log](capture/session.log) |
| 獨立 direct pipeline | 七張 JPEG 的全部 RGB bytes／FP32 bits 對 frozen hashes 相同，F04 完整 header／rows 對獨立 CPU reference；score 沿用既定 `4*FLT_EPSILON`，其餘 rows／座標／padding 逐 byte 相同 |
| 歷史反控 | 首三張 JPEG 的完整 raw SHA 與既有 F03 bytes 相同，kept counts **32／28／39** |
| Serial／queued 與重用 | CS=1／2／3，HP=1／5／5，各跑固定 13 幀序列，共 39 幀；候選身份另核對後，完整 candidate bytes 對 independent serial baseline，含尺寸切換、同影像穿插重跑、caller JPEG bytes 立即改寫 |
| 正常 suite 總範圍 | 23 個有效 runtime instances，73 次成功 enqueue，71 次 receive；兩次由 close／destructor 直接 drain 而不接收；另四次 F02 後 D-only 失敗提交 |
| 錯誤／首次失敗 | Test-only producer numeric→F04 upstream、候選 epoch 錯置→protocol、真實 output capacity=1 overflow；後續不同失敗／正常輸入均 skipped，首錯與最後 commit 不變，所有候選／統計清零 |
| Host 合約與生命週期 | Empty／malformed JPEG、metadata、missing／duplicate frame、結果 ring backpressure／重試不丟幀；empty EOF、finish／cancel、checked repeated close、pending close、destructor 及 fresh epoch 隔離 |
| 確定 pending 的時序擾動 | 穩態 M consumer gate 下 CS=1／2 的覆寫重用；D-only tail 的 finish／cancel／close／destructor 四種 terminal gates；皆在 watchdog 前完成 |
| 新 session control kernels | **78 checks PASS**；完整成功 bytes、四種 flags、首錯鎖定／skip／零輸出、十一種 metadata／count／frame 變體與 fresh control isolation |
| Control-kernel sanitizers | Memcheck／Initcheck／Racecheck／Synccheck 各 **78 checks PASS**；0 errors，racecheck 為 0 hazards（0 errors、0 warnings） |
| Sanitizer 正控制 | **4/4 DETECTED**，各自預期診斷與 exit 99，不只依 nonzero 判定 |
| 完整 JPEG→模型→F04 session memcheck | **2,037 checks PASS、0 errors**；首三張 independent inputs，21 個 session instances、36 次成功 enqueue 的 bounded 模式 |
| 完整路徑 initcheck | 最後 capture **UNEXECUTED**；[前次 FAIL](failed-initcheck-attempt/README.md) 的 12,704 reports、Myelin／CUDA 702 termination 與 cleanup errors 完整保留，根因未定 |

正常輸出保存七份 independent candidate binaries 與 49 份 session result binaries；包含完整成功／失敗 rows、reserved 與未用 bytes。最終 memcheck 的 sanitizer 模式核對全部七張 corpus 檔案 hashes，但僅執行首三張 independent direct pipeline，正常主序列縮為每 ring 三幀；21 個 runtime instances、36 次成功 enqueue，另四次 D-only 失敗，保留 functional error／lifecycle cases。這不是七張 JPEG 皆經完整 session sanitizer 的宣稱。

## 同步模型與證據邊界

CPU finite graph 覆蓋 CS=1..3、HP=CS+1..CS+3、正常 EOF／D-only tail，合計 **18 組**，每組十個完整 frames。Baseline 衝突存取有序且無環；ready／consumed／staging host wait／result host wait／D terminal wait 缺邊與舊 ready generation 六種變體皆被抓到，D terminal 缺邊在九個 partial-tail 設定被抓到。這是有限協定模型的敏感度控制，不是任意 ring 大小的證明，也不等於 GPU 無競態。

Sanitizer 下 callback gates **UNEXECUTED**，正常 suite 另有實跑。四種 sanitizer 的零報告只適用新 session control／finalize kernels；**完整 TensorRT head／F03 racecheck、synccheck 仍 UNEXECUTED**。完整路徑 memcheck 使用上表的 bounded sanitizer 模式，不能擴張為全量影像或所有時序；先前 initcheck 以同模式啟動，但第一個 independent reference 未完成即失敗，不計入成功覆蓋。

TensorRT 在建構時以零 input warmup 同 context，並重新 invalid 初始化 plugin statuses。正常 held-stream 測試另在同一 session 完成一個真實 frame／receive 後才插入 gates，證明範圍為穩態 pending consumer／D-only drain，**不承諾 nvJPEG 首張 JPEG enqueue 沒有 vendor 隱式同步**。原單 JPEG staging／host 等前幀 F02 的保守合約維持不變。先前兩次失敗 logs 與原因 [完整保留](development-attempts/README.md)，沒有將 watchdog 釋放後的結果算作 gates PASS。

**未涵蓋：** 雙 STAGE host-phase overlap、F05／MI、GMC、F04b、tracker、tracking ResultPacket／MOT writer、模型品質、效能驗收、跨 GPU／SDK／多 device 實例、sticky CUDA fault／實際 release-API failure 注入，以及 clean-source 新 session 正式驗收。P01 與完整 tracking session 保持 **OPEN**。nvJPEG initcheck 的既知診斷仍依 [使用政策](../../../docs/PREPROCESS_USAGE_POLICY.md) 保存，不能改稱原路徑零錯誤。

## 重跑及核對

```sh
SACCADE_TENSORRT_ROOT=/absolute/path/to/pinned-sdk \
SACCADE_DETECTION_ENGINE=/absolute/path/to/model.engine \
SACCADE_DETECTION_REFERENCES=/absolute/path/to/frozen-refs \
  bash scripts/run_detection_session_acceptance.sh build-session-new
```

完整 nvJPEG／模型 initcheck 預設另列未執行；可用 `SACCADE_DETECTION_INITCHECK_DIAGNOSTIC=1` 明確重跑診斷，應用程式或 API 失敗仍使 capture 失敗，不歸類為 PASS。未提交工作樹須明確加 `--allow-dirty`，只產生開發證據。模型與數值政策不因此放寬。從 repo root 執行 `python3 evidence/detection-session/2026-10-31/verify.py` 可離線核對 source archive／manifest、前後 hashes、command exits、sanitizer controls 與報告適用的結果；receipt 見 [verification.json](verification.json)。從本目錄執行 `sha256sum -c SHA256SUMS` 可核對完整封存。
