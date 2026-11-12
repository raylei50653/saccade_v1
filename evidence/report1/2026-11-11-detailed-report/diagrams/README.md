# 第一輪報告的 SVG 圖稿

三張圖採固定座標配置、共用色盤與中文字體，輸出為可編輯文字與向量圖形；不使用 raster image、`foreignObject` 或外部網路資源。圖中基準固定為 `3cbaa518f0dc`，時序圖表示依賴，不表示量測時間比例。

從 repository root 重建（只需 Python 標準函式庫）：

```sh
python3 evidence/report1/2026-11-11-detailed-report/diagrams/draw_system.py
python3 evidence/report1/2026-11-11-detailed-report/diagrams/draw_model.py
python3 evidence/report1/2026-11-11-detailed-report/diagrams/draw_schedule.py
```

輸出位於 `docs/reports/assets/report1-3cbaa518f0dc/`。`svg_helpers.py` 定義共用繪圖元件；三個產生器分別控制系統圖、模型圖與交付時序圖的文字和座標。SVG 的字體依序為 Noto Sans CJK TC、Noto Sans、sans-serif，使用系統字體，沒有嵌入字型。

若需 PNG 預覽，可使用已安裝的 `rsvg-convert`：

```sh
rsvg-convert -w 1600 -o /tmp/report1-system.png docs/reports/assets/report1-3cbaa518f0dc/system.svg
```

本輪已渲染並目視檢查三張圖；報告內保留的 Mermaid 程式碼用來對照依賴，SVG 由上述產生器獨立繪製。
