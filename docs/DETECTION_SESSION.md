# Native F01–F04 detection session（`f01-f04-session/1`）

[`DetectionSession`](../include/saccade/detection_session.hpp) 將 HARDWARE nvJPEG F01、原有 antialiased stretch F02、完整 TensorRT F03、[`f03-producer/1`](MODEL_PRODUCER.md) 及 [`F04`](POSTPROCESS.md) 接成一個 bounded async detection session。這裡的 M stream 是候選 consumer／session commit；不包含 F05、GMC、F04b、tracker、tracking ResultPacket 或 MOT writer。其驗證不能作為完整 tracking session 或 P01 的結案。

## 所有權與同步

一個 instance 對應一個 host submitter、一個 epoch 及一個連續零起始 frame 序列。session 擁有兩條 explicit non-default streams D／M、TensorRT runtime／engine／context、前處理器及所有 buffers。固定模型介面由 `model_io_spec()` 核對名稱、mode、dtype、shape、linear device layout 及完整 I/O 集合；本版要求 engine auxiliary streams 為零。建構時配置並初始化資源，載入 kernels，在任何 frame 前完成 D／M 初始化同步。建構時以零 input 在相同 TensorRT context 執行一次 warmup，只寫初始化 scratch，不算 accepted frame；完成後將 plugin statuses 重設為 invalid，避免首次模型載入混入正常 frame 的時序。

| 資源 | Owner／producer | Last use／重用 edge |
| --- | --- | --- |
| JPEG staging、nvJPEG state | D 的 `NativePreprocessor` | 下次改寫 host staging 前，host 等前幀 F02 event；沿用單 staging，沒有 host decode phase overlap |
| RAW、FP32 model input | D | RAW 最後由 F02 讀；input 最後由 F03／finite checks 讀；下一幀覆寫靠 D stream 順序 |
| 六個 raw tensors、九個 plugin statuses、aggregate、scratch | D | engine→aggregate→F04 同 stream；聚合器退休全部 plugin statuses，F04 讀完前不重用 |
| CS[N] 完整候選 slots | D 寫，M 讀 | D 重用前 wait 前次 `consumed`；F04 完成 record `ready`，M wait 此 generation 後 finalize／copy；最後讀完 record `consumed` |
| session control、單份 device result | M | finalize→consumed→D2H→下幀 finalize，靠 M 順序；control 只由 M 推進 |
| HP[R] pinned 結果 ring | M D2H，H 讀 | D2H 後 record `done`；`receive()` 等此 event 並回傳值的副本，完成才歸還 host slot |

CS 的 ready record 在 M wait 之前已 enqueue，consumed record 在下次 D wait 之前已 enqueue。事件不以未來的 record 滿足現有 wait，也不靠 header 取代同步。N 可配置 1..16、R 可配置 1..64；R>N 可減少 host 接收節奏遮蔽 consumed 缺邊的情形。有限實測與模型只覆蓋報告明列的組合。

正常 frame 路徑沒有中間 count／status D2H，只將一份固定容量 `DetectionSessionResult` 從 M 交給 host；包含所有 kept／suppressed candidates、候選 identity、統計及零初始化的未用 rows。這是 detection delivery ABI，不能解讀為 `ResultPacket`。收到結果後可立即重用 HP；未讀結果不會自動丟棄或覆寫。R 個未讀結果時 `enqueue()` 在提交前丟出可恢復的 `length_error`；先 `receive()`，再重試相同 frame。

## 錯誤、提交與生命週期

M finalize 檢查候選 epoch／frame、schema／capacity／done、known flags、count／kept／suppressed 及保留欄位，並核對尚未失敗時的下一個 committed frame。成功才推進 `committed_frame`。第一次失敗鎖定 `fail_frame`／`fail_code`，當幀為 `Failed`；已 enqueue 的後續結果為 `SkippedAfterFailure`，保留第一次失敗與最後成功 frame，全部候選 rows／統計清零。D 上已提交的 engine 工作仍可完成，只寫 D scratch／候選；不能使 M control 恢復成功。

Host 的 frame／JPEG／metadata／CUDA enqueue 錯誤會 latch session `Failed` 並停止新提交；可能只留下沒有結果的 D 段。host error 由例外回報，不偽造 device failure result。`receive()` 觀察到 device failure 後亦停止新增提交，已排入的結果仍可讀回。`state()` 是 host 已觀察狀態，不能把尚未接收的 device failure 當成已同步觀察。

`finish()` 停止提交，分別等待 D／M 終端 events，再檢查未讀結果；只有正常 EOF 且沒有失敗才為 `Finished`。`cancel()` 同樣 drain，但記為 `Cancelled`，不是正常 EOF；已有失敗保留 `Failed`。兩者都保留未讀結果供 `receive()`。`close()` checked drain 兩條 streams，然後嘗試所有獨立釋放；重複 close 安全。它釋放未讀結果，caller 若需保存須先 receive。destructor 僅 best effort，正式生命週期邊界使用明確 close。

失敗／取消後使用新 instance 開始下一序列；沒有在途 reset。D-only tail 即使沒有 HP／packet，仍受 D 終端等待保護。sticky CUDA error 若使 drain 無法成功，例外會保留，不得據此宣告 session 完成。

## 建置與驗證

預設 CPU build 包含獨立的 `detection_session_model`：18 組有限 happens-before 圖覆蓋 CS=1..3、HP=CS+1..CS+3、正常 EOF／D-only tail，每組十個完整 frames；檢查衝突存取有序及無環。六個敏感度控制移除 ready、consumed、staging host wait、result host wait、D terminal wait，或綁定舊 ready generation，皆須至少找到一個反例。這不是 GPU 無競態證明，也不證明任意 ring 大小。

GPU 使用以下可選建置；`120` 為本機 GPU 架構，其他環境應據實指定：

```sh
cmake -S . -B build-detection -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_PREPROCESS=ON \
  -DSACCADE_ENABLE_POSTPROCESS=ON -DSACCADE_ENABLE_MODEL_PRODUCER=ON \
  -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_ENABLE_MODEL_TRT=ON \
  -DSACCADE_ENABLE_DETECTION_SESSION=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSACCADE_TENSORRT_ROOT=/absolute/path/to/pinned-sdk \
  -DSACCADE_DETECTION_ENGINE=/absolute/path/to/model.engine \
  -DSACCADE_DETECTION_REFERENCES=/absolute/path/to/frozen-refs
cmake --build build-detection --target detection_session_tests detection_session_cuda_tests --parallel 3
ctest --test-dir build-detection -R '^detection_session' --output-on-failure
```

`detection_session_tests ENGINE REFS OUTPUT_DIR [--sanitizer]` 比對獨立串行 F01–F04／CPU F04 reference、session serial／queued 完整候選，並測同步、錯誤及生命週期。test-only friend seam 用於 producer failure／候選 identity 錯誤、D-only tail 及延後 consumer，不是正常 caller 設定。`detection_session_cuda_tests` 只執行 session control／finalize kernels，四種 sanitizer 的範圍不擴張到完整 TensorRT engine。

可重跑 capture 入口為 [`run_detection_session_acceptance.sh`](../scripts/run_detection_session_acceptance.sh)。它記錄 source archive／diff／manifest、engine／references／binary 前後 hashes、命令／退出碼／時間及 CMake 身分。dirty source 必須明確 `--allow-dirty`，只可記為開發證據。完整 nvJPEG／模型 initcheck 預設不執行，可明確設定 `SACCADE_DETECTION_INITCHECK_DIAGNOSTIC=1` 另跑診斷；session control kernels 的四種 sanitizer 仍全部執行。先前完整 initcheck 發生 instrumented launch timeout，原始失敗與尚未釐清的執行錯誤保留，不改為 PASS。原 nvJPEG initcheck 診斷沿用 [使用政策](PREPROCESS_USAGE_POLICY.md)，不改為正常路徑 0 errors；新診斷須逐份判讀，API／應用程式失敗不得豁免或歸為成功。實跑結果及未執行項目見 [session 證據報告](../evidence/detection-session/2026-10-31/REPORT.md)，不能從 test 名稱推導 PASS。
