# 完整路徑 initcheck 嘗試：FAIL，保留未釐清的 instrumented execution error

這是第一次 frozen-source capture，原始 [capture status](capture/capture-status.txt) 為 exit 1，不列為完整 capture PASS。該輪 CPU 13/13、GPU CTest 10/10、正常 session 4,293 checks、完整路徑 memcheck 2,037 checks／0 errors，以及 session control kernels 四種 sanitizer／正控制先行通過；之後的 initcheck 停止整個 capture，未產生最終 binary／artifact 前後核對或 bundle。最終成功 capture 另存於相鄰 `../capture/`，不能用前項結果改寫本次失敗。

[完整 initcheck log](capture/initcheck-diagnostic.log) 有 100 份已列印的三位元組未初始化 global read，指向 `nvjpeg::rgba2rgb<...>+0x170`，host stacks 經 `nvjpegDecodeBatched`／`NativePreprocessor::enqueue`／獨立 serial `independent()`。另有 **12,604 份報告未列印**（共 12,704），無法將這些未列印報告全部歸因既知 nvJPEG 診斷。

其後有 Myelin internal CHECK(false)、CUDA 702 launch timeout、資源卸載／釋放錯誤，應用程式以 `detection_session=FAIL checks=67` 結束。沒有任何完成的 `independent_input=`，尚未進入 `DetectionSession` runtime 測試。根因未定，不推斷這是 nvJPEG read 報告所致、false positive、正常 session 的 API 缺陷或 OOM。已知 nvJPEG read 的使用政策不豁免這些 API／應用程式錯誤。

為完成正常使用驗證，最後 capture 從新目錄重新建置／重跑正常及 memcheck；完整路徑 initcheck 明確 **UNEXECUTED**，新 session control kernels 四種 sanitizer 仍全部執行。原 initcheck FAIL 與本次 instrumented execution error 保留為 OPEN 診斷，不宣稱修復。最後只改 capture script 的 optional-diagnostic 流程及相關說明；runtime／test build inputs 未因這次 failure 改動，前後 source manifest 可核對。
