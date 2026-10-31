# Saccade v1 後續工作

日期：2026-10-31。T01 實驗與有界 F01／F02 使用結案見 [closeout](history/2026-10-31-t01-closeout.md)。以下項目各自保留 OPEN，不把訓練完成或單階段 PASS 當作整條 runtime 已完成。

## 1. F04：先固定觀測政策，再實作

輸入是 F03 的六個 raw cls／ltrb outputs 與 `ImageMapping`；輸出為保留穩定 candidate ID、原始 score／class、原圖 xyxy、kept／suppressed、device count／capacity／status 的 `DetectionCandidates`。依 [DATAFLOW](DATAFLOW.md#4-跨階段資料契約)，suppressed 候選須留給 F04b，suppression 只執行一次。

已依版本化 [`f04-postprocess-fp32/1`](POSTPROCESS_POLICY.md) 實作 [CPU scalar reference 與 CUDA API](POSTPROCESS.md)，沿用 T01 已固定的語義：P3→P5 row-major、stride 8／16／32、anchor `(x+0.5,y+0.5)`、raw ltrb、80 類 sigmoid 後 max／argmax、conf≥0.001、**class 篩選前 top-300**、只留 class 0、greedy NMS IoU>0.5，再映射原圖。保留完整 kept／suppressed pool，不能將此契約縮成 person-only top-k 或只交付 kept boxes。

政策已固定 class／top-k／NMS ties、FP32 計算與閾值邊界、不 clipping、退化／非有限框、capacity／padding／overflow，以及被 NMS suppressed 與低分／非 person／top-k 淘汰的區別。T01 evaluator 的 invalid-topk 只有歷史統計；新增 native 處置不回寫其實驗結果。候選 ID 由篩選前模型序號建立並在 compact 後保留。

完成條件：依 GPU 前固定的 reference／cases／政策，驗證完整輸出與穩定順序、重跑與資源重用、同分與閾值邊界、空候選及明確失敗；保留 F04b 可用的 suppressed pool。正常 runtime 的中途 count／status 留 device，不增加 per-frame postprocess D2H。**實作已完成，本機開發驗證通過；clean-source 正式驗收未執行**。31 個固定 cases、六份依正式 file schema 解析且獨立核對歷史 kept counts 的 F03 raw dumps、完整輸出／重跑／graph／正常 pending gates、四種 sanitizer、正控制與 10 個 mutations 的實際結果見 [F04 證據](../evidence/f04/2026-10-31/REPORT.md)。早期誤讀交錯 file layout 的結果另存錯誤 attempt，不作真實 replay 證據。驗證執行時 source 尚未提交、`source_dirty=true`，sanitizer 下的 callback gates 為 UNEXECUTED，不稱 clean-source formal F04 PASS。F03 plugin／finite statuses 的 producer 聚合器及 native 前處理／模型接續尚未整合，留於完整 session 任務，不因 raw replay 而視為完成。

## 2. 完整 async session：串起已完成階段

依 [ASYNC_CONTROL](ASYNC_CONTROL.md)、[DATAFLOW](DATAFLOW.md#6-gpu-所有權與同步依賴) 與 [ResultPacket](RESULT_PACKET.md)，將 F01→F05／F02／F03／F04、main 的 F06→F08a→F04b→F07／F08b、最後 F09 接成完整 session。

須明列 JPEG staging、RAW／model input、MotionImage、推論 outputs、候選 slots 與 ResultPacket 的 owner／ready／last-use／consumed。候選在 F04b 消費完前不得被下一次 inference 覆寫；原圖只在 detect stream 借用。F04b 使用本幀 GMC＋predict 的 prior，continuation-only 候選維持 `can_birth=false`。正常資料面每幀只交付一次固定容量 ResultPacket D2H。

完成條件：真實模型與影像的 serial／queued 對照、跨幀與跨 session 隔離、pending consumer 下的重用、frame／epoch／版本一致性、錯誤／overflow／缺幀／drain／checked close 全部有證據。現有 F01／F02 的 208 次 usage 測試只驗獨立前處理與 caller-owned consumer，尚未完成本項；雙 host staging／完整 overlap 等規劃依各自實作證據判定。

## 3. P01：依原 readiness 缺件整體判定

以 [MODEL_ARTIFACT_READINESS](MODEL_ARTIFACT_READINESS.md)、[MODEL_MANIFEST](MODEL_MANIFEST.md) 與各階段已驗界線為準，逐項補齊 F04／容量、正常 native 輸入至完整模型的接續整合及其餘原列缺件。指定 checkpoint／ONNX／bundle／engine 與實際 runtime 的身分須一致，未執行項目維持明列。

P01 保留 **OPEN**。Head／F03 racecheck、synccheck 目前 UNEXECUTED；模型品質、tracker／GMC／MOT writer 與完整 tracking 指標依各自驗收，不能由 detection AP 或有限數值對照推導。T01 不因本項未完成而重開已完成的訓練實驗。

## nvJPEG／NVJPG 診斷：目前不主動續查

原 initcheck FAIL、Bug 7021871、逆向／trace／annotation／poison 與所有失敗紀錄保留；依 [使用政策](PREPROCESS_USAGE_POLICY.md)，正常使用輸出等價才是目前驗收目標，不要求 NVIDIA 分類或消除既知報告。本診斷不作上述使用結案的前置工作。

只有新增輸出不符、重跑／重用漂移、API／同步失敗、新環境無法通過輸出驗證，或 NVIDIA 提供可行動的新資訊時，才按具體訊號建立新的診斷項。既有 annotation 不進正常 runtime，也不把被排除的檢查改稱原路徑 PASS。
