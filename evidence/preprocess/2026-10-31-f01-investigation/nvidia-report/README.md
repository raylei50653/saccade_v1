# F01 nvJPEG initcheck：NVIDIA 回報（Bug 7021871）

**已向 NVIDIA 回報 [Bug 7021871](https://developer.nvidia.com/bugs/7021871)，附件已寄至 NVSDKIssues@nvidia.com；目前狀態 OPEN，尚無 NVIDIA 回覆。** 回報只請 NVIDIA 分類，沒有宣稱已確認誤報；F01 initcheck FAIL／原因 OPEN 與 P01 OPEN 均不變。

## 送出內容

| 項目 | 紀錄 |
| --- | --- |
| Bug form | 2026-10-31（+08:00）以使用者的 NVIDIA Developer 帳號送出；Area 為 CUDA，OS 選 Linux 與 Windows 11。頁面顯示 `Last Updated: October 30, 2026`，推測為 NVIDIA 端時區，未另行確認 |
| 表單紀錄 | [bug-7021871-page.txt](bug-7021871-page.txt)：送出後 bug 頁面 `<main>` 的文字擷取，含完整 Description 與欄位 |
| 附件 email | 2026-10-31（+08:00）由使用者 Gmail 寄至 NVSDKIssues@nvidia.com，署名 GitHub 帳號 `raylei50653`；Gmail 顯示已送出。主旨、內文與附件名見 [email-sent.txt](email-sent.txt) |
| 附件 | [nvidia-nvjpeg-initcheck.zip](nvidia-nvjpeg-initcheck.zip)，即寄出的同一個檔案 |

`email-sent.txt` 是寄送時使用的草稿檔，Gmail 內文與其 `Hello,` 以下相同；沒有從寄件備份匯出 `.eml`，不宣稱為 byte-exact 的 MIME 原件。

## 附件內容

| zip 內檔案 | SHA-256 | 來源 |
| --- | --- | --- |
| `repro.cu` | `c3b0bbf242218a22d86568b527df363978588f9ee1d56f7c65891e4e0cdbc977` | [`../repro.cu`](../repro.cu) 原樣 |
| `baseline.jpg` | `3dedf93555bdc2aff5aab2a608a99f69cd8d6cc5c0fd64d8d3404ebb2f78da56` | repo `fixtures/preprocess/baseline.jpg` |
| `baseline.rgb` | `761809bdcad84f79c1b01f0d2369bb4cf52f280cb2dff92be87c3ef9f11c7162` | [`../baseline.rgb`](../baseline.rgb) 原樣 |
| `initcheck-full-945.log` | `d786967c3eca7278118ecc75d3decc2ee931b3f42cc160af202d9d5cd8ea0be0` | [`../capture/new-rgb-once-full-initcheck.log`](../capture/new-rgb-once-full-initcheck.log) 原樣 |
| `windows-initcheck-1890.log` | `821fa72452eeda6aaf50d05141ec01a357d26f2cd527ff6d56b2039604cdedb1` | [`../windows-native/capture/windows-rgb-initcheck.log`](../windows-native/capture/windows-rgb-initcheck.log) 原樣 |
| `identity.log` | `2cab874b980b8c1171a10e5e9f76b27aa2b3c2e7360ff6fdef9a2d4fd1d67c28` | [`../capture/identity.log`](../capture/identity.log)，本機工作目錄路徑改為 `<workdir>`，其餘不變 |
| `README.txt` | `7c0a1084bfc51c4013cc68d587edef1bae9a41c1c8629b5b2ee99ac91f440554` | 為 NVIDIA 撰寫的建置／執行說明 |

寄出前已檢查 zip 內檔案不含 `/home` 路徑；Windows log 亦不含使用者目錄。GPU UUID 沒有寫入表單或附件說明。

## 未包含與界線

- 只提供獨立 standalone repro；沒有送出 Saccade、F02、session acceptance 或其 8,151 errors。session 那 8,151 筆仍只印出 100 筆，其餘尚未歸類。
- Poison／allocation identity／nsys 只在表單文字中摘要；原始 [surface-poison](../surface-poison/README.md) 資料沒有附上。
- 回報說明原生 Linux driver 尚未測試，亦未區分 nvJPEG、Compute Sanitizer 或共用 Windows driver 路徑的責任。

NVIDIA 回覆後，另以新紀錄保存原文與分類，不改寫本目錄已送出的內容。
