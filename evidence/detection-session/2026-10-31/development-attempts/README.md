# 開發嘗試：不取代最終 frozen-source capture

保留原始 logs，不將失敗測試重分類為 PASS。這些嘗試沒有各自完整 source archive；本次可核對的最終 build inputs 以相鄰 `capture/source-build-inputs.tar.gz` 為準。

- `run-1.log`：數值／功能案例通過，最後的首次 consumer gate 因 watchdog 釋放而失敗；當時尚無完整 TensorRT context warmup／real JPEG priming。
- `run-2.log`：加入 TensorRT 零輸入初始化 warmup 後，首次 nvJPEG 使用的 D gate 仍等到 watchdog 釋放，測試失敗。
- `run-3.log`：同一 session 先完成一個真實 frame／receive，再建立 D 或 M gate；四種 D-only terminal 與 slots=1／2 pending consumer 均在 watchdog 前通過，正常 suite 4,293 checks PASS。

最終測試保留「gate 未釋放時 enqueue 返回」、「consumer 仍 NotReady」及「terminal 返回前 callback 已完成」斷言。這些是穩態時序擾動的有限證據，不承諾 vendor 首次使用無隱式同步。正式判定只使用從凍結 build inputs 重新建置的最終 capture；以上 logs 僅保存開發過程與失敗原因。
