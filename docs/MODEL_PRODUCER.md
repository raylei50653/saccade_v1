# F03 producer status 聚合器（`f03-producer/1`）

[`model_producer_cuda`](../include/saccade/model_producer_cuda.hpp) 在 caller 的 detect stream 上，將完整 F03 engine 的九個 plugin statuses 與 input／output finite checks 轉為 F04 已有的八位元組 `PostprocessProducerStatus`。不改 engine、raw tensors、F04 數值政策或候選 ABI。

## 固定輸入與判定

輸入為 FP32 batch-one `images [1,3,640,640]`、六個 raw outputs（`cls_p3, cls_p4, cls_p5, reg_p3, reg_p4, reg_p5`），以及 [`model_io_spec()`](../src/model_trt_network.cpp) 的九個 caller-owned `ScanStatus*`，順序固定為 P3／P4／P5 各 expand→scan→merge。caller 綁定 engine 時仍須核對名稱、mode、dtype、shape；指標排列本身無法辨識全成功 statuses 被交換。

同一 API 固定排入三個 kernels：initialize → finite → aggregate。initialize 清除 aggregate 與兩個 scratch statuses；finite 掃描完整模型輸入及全部六個輸出，涵蓋低分與後續被淘汰的位置。任一 NaN／±Inf 設定既有 `kFrameNonFiniteInput`／`kFrameNonFiniteOutput`。aggregate 完成兩個 checks，讀全部十一個 statuses 並依下表累積錯誤，最後將十一個 source statuses 全部退休為 `{0,0}`。

| 來源 | 允許的 flags |
| --- | --- |
| expand／merge | `kCrossScanNonFiniteInput` |
| scan | `kScanInputFailures \| kScanRangeFailures` |
| input check | `kFrameNonFiniteInput` |
| output check | `kFrameNonFiniteOutput` |

任何 source flags 非零 → aggregate `kPostprocessNumeric`；任何 source `done != kScanDone` 或不屬於其角色的 bits → aggregate `kPostprocessProtocol`；兩者可同時成立。不因第一個錯誤而略過其他來源或退休。這是 **PostprocessFailure** namespace，不使用 ResultPacket `Error` 的數值。角色 mask 比原 frame harness 的共同 known-bits mask 更嚴格。

aggregate 先寫 flags，最後寫 `done = kPostprocessProducerDone`，失敗亦有完成 marker。依既有 [F04 契約](POSTPROCESS.md)，aggregate 任何 flags 非零均轉為 F04 `kPostprocessUpstream` 並交付完整零候選失敗輸出；來源 protocol 的細節留在 aggregate flags，不改 F04 的既有錯誤映射。aggregate 完成後再次呼叫、卻未重新執行 plugins，會因退休後的 done 為零而失敗。

## 所有權與 enqueue 契約

caller 先在 submitter thread 呼叫 `cudaSetDevice`，建立 explicit non-default stream、配置所有資源，將 fresh slot 的九個 plugin statuses 初始化為 `{0,0}`，並在每個 context 的 frame 工作前呼叫 `model_producer_cuda_load_kernels()`。初次漏寫因此也是明確的 protocol failure；後續由退休保護。初始化後 enqueue 無 allocation、transfer、synchronization、event 或 default-stream 工作；固定指標可 capture／replay。CUDA API／launch error 拋出 `runtime_error`，不承諾完成 marker。

所有 accessed ranges 必須位於 current device、對齊、容量足夠且互不重疊。支援 `cudaMalloc` 與 arena subranges；以 `cuMemGetAddressRange` 無法查得 extent 的 allocation 類型拒絕。input count 必為 1,228,800；raw counts 依 F04 固定 profile。scratch 為兩個 `ScanStatus`（目前 16 bytes），需 `alignof(ScanStatus)` 對齊；以 `model_producer_workspace_bytes()` 查本版大小。host argument 錯誤於第一個 enqueue 前全部拒絕。

正常次序為 `engine enqueueV3 → model_producer_cuda → postprocess_cuda`，三者在同 stream；跨 stream 時由 caller 先建立 ready event 邊。只有聚合器退休這組 plugin statuses，不能再把同一組交給 `scan_frame_gate`。images／raw／plugins／scratch／aggregate 須活到各自最後存取；F04 讀完前不得覆寫 raw／aggregate，F04b 讀完前不得覆寫候選。pending consumer 的 consumed 邊由 session 管理。

八位元組 aggregate 沒有 epoch／frame，marker 不是同步機制；退休也不能證明 engine 本幀確實執行。caller 仍須建立生產、ready、最後消費與 consumed 的完整順序，維護 identity 與首次失敗鎖定。

## 建置與驗證

primitive 只需 CUDA，不依賴 TensorRT／nvJPEG；預設 CPU build 不變。獨立 CUDA／F04 測試由以下選項建立（`120` 為本機 sm_120，其他裝置改用自己的架構）：

```sh
cmake -S . -B build-f03-producer -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_MODEL_PRODUCER=ON \
  -DSACCADE_ENABLE_POSTPROCESS=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-f03-producer --target model_producer_cuda_tests --parallel 2
ctest --test-dir build-f03-producer -R '^model_producer_cuda' --output-on-failure
```

額外開啟既有 `SACCADE_ENABLE_MODEL_TRT`／`SACCADE_ENABLE_TENSORRT` 並提供 pinned SDK root 後，可建立 `model_producer_trt_tests`。設定 `SACCADE_MODEL_PRODUCER_ENGINE` 與 `SACCADE_MODEL_PRODUCER_INPUTS` 可將 actual F03 engine 接續測試加入 CTest。資源、固定 artifacts、實際執行與尚未執行項目以本次證據報告為準。

本項僅完成 producer 聚合與有界 engine→F04 接續；native JPEG→F03、完整 async session、候選 ring／F04b、session failure latch、GMC／tracker／ResultPacket／MOT writer 與 P01 仍依 [BACKLOG](BACKLOG.md) 接續。歷史 head／F03 racecheck／synccheck 與模型品質判定不因本 primitive 測試而改寫。
