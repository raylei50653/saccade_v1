# 展示台播放負擔修正

原展示台同時由影片影格回呼與螢幕動畫回呼完整繪圖。暖機後的診斷模式每秒約 90 次 `drawImage(video)`，每次平均約 6–7 ms；關閉框插值或切原圖也有相同複製。尾線每次回掃歷史，ID 側欄每來源幀重新建立按鈕。

一般／原圖模式已改為原生 HTML video 與透明框層，省去影片像素複製。並排模式仍複製相同解碼影格至兩個面板，但快取當前來源幀。尾線快取歷史點，插值只更新末端；ID 控制項沿用節點；相同來源幀的側欄不重寫。影片回呼立即提交當前 packet，螢幕回呼只補畫座標插值；暫停／seek／逐幀仍使用原始座標。

`profile-before.json`、`profile-after.json` 保存量測；對應原稿為 `app-before.js`、`app-after.js`。最終五組各約 3.5 秒、1440×1000、DPR 2 的 headless Chromium 156 量測，共回報 527 個影片影格、0 個 dropped frames；一般模式的影片複製次數為 0，繪圖 JS wall-time p95 為 1.0–1.6 ms。完整細節見 `summary.json`，不是 tracker 或 GPU 推論 benchmark。

環境的 Chromium 使用軟體繪圖，並有其他未停止的工作；前期診斷曾出現冷啟動及排程長尾，因此不把這一輪結果宣稱為普遍 FPS 保證，也不將 cold-start baseline 的 140 ms p95 當作穩定加速比。可確定的變更是一般模式移除了像素複製，以及瀏覽器操作測試驗證此行為。背景仍是來源 30 FPS，框插值不會產生影片中間影像。

38 項互動檢查通過，包含零複製、並排快取、ID 節點沿用、三序列切換及原始座標規則；收據為 `browser-validation.json`。`source-assets-sha256.json` 的 27 個展示來源檔均未變更，未執行新推論、GT 評分或修改 runtime。

重跑 profiler（需已安裝 Playwright／Chromium）可使用：

```sh
node docs/demos/report1/methods/profile_playback.cjs http://127.0.0.1:8765/ docs/demos/report1/app.js build/demo-profile.json
```

這是本機交付；未 commit、push 或發布。
