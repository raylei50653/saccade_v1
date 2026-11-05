# NVOF cuArrayGetPlane：NVIDIA 回報（Bug 7048273）

**已向 NVIDIA 回報 [Bug 7048273](https://developer.nvidia.com/bugs/7048273)，附件已寄至 NVSDKIssues@nvidia.com；狀態 OPEN，尚無回覆。** 回報沒有宣稱這是 defect，只問是否刻意、是否可安全忽略，以及 explicit 模式的行為是否符合文件。Memcheck 的 exit 99 在 NVIDIA 說明前仍是 OPEN，不算 PASS。

## 送出內容

| 項目 | 紀錄 |
| --- | --- |
| Bug form | 2026-11-05（+08:00）以使用者的 NVIDIA Developer 帳號送出；Area 為 CUDA，OS 選 Linux 與 Windows 11，內容依 [bug-form.txt](bug-form.txt) |
| 表單紀錄 | [bug-7048273-page.txt](bug-7048273-page.txt)：送出後 bug 頁面 `<main>` 的文字擷取，含完整 Description 與欄位 |
| 附件 email | 2026-11-05 11:39（+08:00）由使用者 Gmail 寄至 NVSDKIssues@nvidia.com，署名 GitHub 帳號 `raylei50653`；Gmail 寄件備份可見。主旨、內文與附件名見 [email-sent.txt](email-sent.txt) |
| 附件 | [nvidia-nvof-getplane.zip](nvidia-nvof-getplane.zip)，20 個檔案 |

- **表單欄位：** Operating system details 不在 `bug-form.txt` 內，是送出時依同一份環境資料填寫，原文見頁面紀錄。頁面顯示的 `Last Updated: November 04, 2026` 是網站的日期，紀錄照原樣保留。
- **附件來源：** 自動上傳被 session 的檔案權限擋下，改由使用者手動附加。Gmail 撰寫視窗顯示 `nvidia-nvof-getplane.zip (30 KB)`，與本目錄檔案的名稱和大小相符；沒有比對寄出附件的雜湊。
- **Email 內文：** `email-sent.txt` 是寄送時使用的文字，Gmail 內文與其 `Hello,` 以下相同；沒有從寄件備份匯出 `.eml`，不宣稱為 byte-exact 的 MIME 原件。

## 附件內容

zip 內所有檔案都在 `nvidia-nvof-getplane/` 之下。

| zip 內檔案 | 來源 |
| --- | --- |
| `repro_getplane.cpp` | [上層目錄](../repro_getplane.cpp)原樣 |
| `nvOpticalFlowCommon.h`、`nvOpticalFlowCuda.h` | `third_party/nvof/` 原樣 |
| `README.txt` | [README.txt](README.txt) 原樣 |
| `wsl2/*.log` | [`../capture/`](../capture/) 原樣，不含空的 `build.log` |
| `windows/*.log` | [`../windows-native/capture/`](../windows-native/capture/)，只去掉換行的 CR |

打包後已檢查 zip 內不含 `/home`、`Users`、使用者名稱、email、主機名稱或 GPU UUID。Windows 的 `build.log` 沒有放進 zip。

## 刻意不包含

- **反組譯結果。** `libnvcuvid` 內部有備用路徑的判讀只留在內部紀錄。回報用不到它，sanitizer 的堆疊位址和 CUPTI 結果已足夠；閉源函式庫的反組譯也可能碰到授權條款。
- **Saccade、TrackingSession 或驗收的程式與資料。**
- **較大的 CUPTI 探測程式。** 表單寫明可應要求提供，來源是 `build/nvof-memcheck/investigation-20261105/`。

## 界線

原生 Linux driver、其他 GPU 和其他 driver 版本都未測試，表單也寫明了。

NVIDIA 回覆後，另以新紀錄保存原文，不改寫本目錄已送出的內容。
