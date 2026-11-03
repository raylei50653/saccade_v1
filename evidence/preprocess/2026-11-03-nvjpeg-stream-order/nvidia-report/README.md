# nvJPEG HARDWARE stream 順序：NVIDIA 回報（Bug 7038860）

**已向 NVIDIA 回報 [Bug 7038860](https://developer.nvidia.com/bugs/7038860)，附件已寄至 NVSDKIssues@nvidia.com；狀態 OPEN，尚無回覆。** 回報只請 NVIDIA 確認 HARDWARE backend 是否應遵守 stream 順序；若不應遵守，請文件寫明呼叫者必須先同步。Saccade 的處理（TrackingSession 1c03e22、PREPROCESS RGB 借用規則）不因回報而改變。

## 送出內容

| 項目 | 紀錄 |
| --- | --- |
| Bug form | 2026-11-03（+08:00）以使用者的 NVIDIA Developer 帳號送出；Area 為 CUDA，OS 選 Linux 與 Windows 11 |
| 表單紀錄 | [bug-7038860-page.txt](bug-7038860-page.txt)：送出後 bug 頁面 `<main>` 的文字擷取，含完整 Description 與欄位 |
| 附件 email | 2026-11-03 20:31（+08:00）由使用者 Gmail 寄至 NVSDKIssues@nvidia.com，署名 GitHub 帳號 `raylei50653`；Gmail 寄件備份可見，附件為下列 zip。主旨、內文與附件名見 [email-sent.txt](email-sent.txt) |
| 附件 | [nvidia-nvjpeg-stream-order.zip](nvidia-nvjpeg-stream-order.zip)，即寄出的同一個檔案 |

`email-sent.txt` 是寄送時使用的草稿檔，Gmail 內文與其 `Hello,` 以下相同；沒有從寄件備份匯出 `.eml`，不宣稱為 byte-exact 的 MIME 原件。

## 附件內容

zip 內所有檔案都在 `nvidia-nvjpeg-stream-order/` 之下。

| zip 內檔案 | 來源 |
| --- | --- |
| `repro_order.cu`、`generate.py`、`a.jpg`、`b.jpg` | 上層目錄原樣，SHA-256 見 [`../SHA256SUMS`](../SHA256SUMS) |
| `README.txt` | [README.txt](README.txt) 原樣，為 NVIDIA 撰寫的建置／執行說明 |
| `wsl2/*.log`（6 個） | [`../capture/`](../capture/) 的六個 log 原樣 |
| `wsl2/identity.log` | [`../capture/wsl-identity.log`](../capture/wsl-identity.log) 原樣：nvidia-smi、nvcc、g++、ldd、輸入與 libnvjpeg 的 hash |
| `windows/win-*.log`（5 個）、`windows/windows-env.log` | [`../windows-native/capture/`](../windows-native/capture/) 原樣 |

寄出前已檢查 zip 內檔案不含 `/home`、`Users`、使用者名稱、email 或 GPU UUID。Windows 的 `build.log` 含本機路徑，沒有放進 zip。

## 未包含與界線

- 只提供獨立 standalone repro；沒有送出 Saccade、TrackingSession 或 M3 驗收的程式與資料。
- 回報說明原生 Linux driver 尚未測試，也說明第一個 trial 都遵守順序、原因未查。
- 回報沒有宣稱這是 nvJPEG 的 defect，只指出與文件敘述不符並提問。

NVIDIA 回覆後，另以新紀錄保存原文，不改寫本目錄已送出的內容。
