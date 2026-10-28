# CUDA synthetic protocol evidence — 2026-10-28

**狀態：程式與實機協定矩陣通過；compute-sanitizer 驗收受環境阻擋，尚未通過。** 不把下列有限實測升格為任意 N／shape／裝置皆正確，也不宣稱所有驗收條件已無保留完成。

開始時 `git fetch origin` 後 `origin/main` = 已審查基線 `a895ba27bf2b74c0552e7f9941b6af81015474be`，沒有新增變更。實測 build 的 source commit 為 **`c453cec5f634e714a302d7897ce4e0b4151430e3`**，重新 configure 時 worktree 乾淨，所有交付案例 `identity.txt` 的 `build_dirty=false`。本 evidence commit 只增加證據，不改程式。開發中獨立封包答案曾抓到 commit 對已有階段錯誤額外 OR PROTOCOL 的差異；source commit 已修正為與 CPU FailureModel 相同的規則，以下為修正後乾淨 build 的結果。

## 裝置與工具身分

| 項目 | 實測值 |
| --- | --- |
| OS | WSL2，Linux 6.18.33.1-microsoft-standard-WSL2，x86_64 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU，SM 12.0，PCI 0000:01:00.0 |
| GPU UUID | GPU-825ae551-ba16-539a-e444-22726b59ee9a |
| driver | `nvidia-smi --query-gpu` 回報 617.42；NVIDIA-SMI 工具版本 615.78.08；UMD CUDA 13.4 |
| CUDA | nvcc 13.4.92；runtime／driver API 13040；編譯 architecture=120 |
| cuFFT | cufftGetVersion=12400；libcufft.so.12.4.0.43；8×8 R2C plan workspace 需求 0 bytes |
| host compiler／CMake | GNU C++ 16.2.1 20260810／CMake 4.4.4 |
| build | Release、C++20／CUDA20、CUDA `-lineinfo`，source commit 如上 |
| compute-sanitizer | 2026.3.0.0 build 38855100，四種工具皆嘗試，exit 99 |

完整命令輸出見 [environment.txt](environment.txt)，可執行檔 hash 見 [binaries.sha256](binaries.sha256)。每一 session 都另存 build／裝置 identity、完整 resolved 設定及 hash，見 archive。

## 結果與覆蓋

| 驗收項目 | 實際結果與界限 |
| --- | --- |
| CPU 回歸 | CUDA OFF：4/4 CTests；216/216 happens-before configurations、15類同步邊變異；狀態／封包630 checks；設定88 checks；既有 packet_compare CLI 通過 |
| GPU 回歸 | CUDA ON：5/5 CTests；GPU suite **40618 checks、72 sessions 通過** |
| 正常輸出 | 4組 rings × 3個固定 delay seeds（11、29、97），每次16幀；每組先 serial兩次逐bytes相同，再與pipelined比較全部原始bytes；各自也與獨立解析答案比較並decode合法性檢查 |
| ring 重用 | 全單slot、全雙slot、(1,2,3,4)、(2,1,3,4)；每個ring至少初次使用＋3輪重用；帳本獨立核對明確consumed wait，非靠其他ring的背壓 |
| 失敗提交 | before_spectrum=8、after_spectrum=4、after_predict=16、after_update=1；失敗frame3，固定0..7全範圍，各自serial兩次＋pipeline全bytes比較；另有4例首幀失敗 |
| 尾端與隔離 | failure測試N_H=N_RP=8，全部0..7先enqueue後觀察FAILED；frames4..7共4個後續FFT都完成；後續診斷bits=10不能覆寫首次失敗；完整committed TS bytes與SPEC float bits等於只成功0..2的baseline |
| 普通失敗停止 | frame3失敗，N_RP=4；只提交0..6，讀到frame3 FAILED後不再新增frame7；全部已排入packet保留，不以裁切配平 |
| 取消／收尾 | frame7在8個切點各serial／pipeline，16例；包括3種D-only尾幀；只等待成功record世代，兩條terminal完成才釋放 |
| 形狀 | 主矩陣8×8；額外16×8及32×32各8幀，seed=4294967295 |
| CLI | 正常serial與pipeline各16packets／197888bytes；既有packet_compare回報identical。預期失敗CLI為8packets／98944bytes，1 FAILED、4 SKIPPED，exit1 |
| sanitizer | **BLOCKED／未取得有效檢查結果**；四工具都無法初始化WDDM debugger interface，並回報Device not supported，不能宣告無記憶體／同步錯誤 |

參考：[cpu-ctest.log](cpu-ctest.log)、[cpu-tests.log](cpu-tests.log)、[gpu-ctest.log](gpu-ctest.log)、[gpu-tests.log](gpu-tests.log)、[cli-compare.log](cli-compare.log)。[cases.tsv](cases.tsv) 明列72個案例的設定／行為hash與原始封包長度。

每幀writer完整初始化12368bytes，只作一次固定封包D2H；中途count/status沒有額外D2H。terminal_snapshot是測試專用額外讀回，在雙stream終端drain之後，與正常幀資料面分開。

可直接核對的設定身分：

- CLI serial full hash：`7e8b5b398e58fb09d296594d4ddc0c7392f04c06df29929fbcca4205488c9660`。
- CLI pipeline full hash：`5ebc3f4f836d75a27069a32fd348deb42772ee40746fee9d35a67ab91ba6a762`。
- 上述兩者 behavior hash：`b4fb7243b074facdf1842937008091d6709be4ebcdbb49e12c36edf91bfcb423`。
- CLI固定失敗 full hash：`6944a556983f536294e778921d32339543c6653696708dbc16dc2a4445ef6842`；behavior hash：`ad64b4e441249637b54633b78eaa8c609a91e7d18ed9d66cb476269c74b6b03f`。

## Sanitizer 阻擋

實際逐個執行以下命令，tool=memcheck、initcheck、synccheck、racecheck；四者都記錄 exit=99：

```sh
/opt/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  build-cuda/cuda_synthetic_tests --sanitizer build-cuda/final-sanitizer-memcheck
```

完整重跑loop及所有CPU/GPU建置命令在 [CUDA_SYNTHETIC.md](../../../docs/CUDA_SYNTHETIC.md)。原始 [memcheck.log](memcheck.log)、[initcheck.log](initcheck.log)、[synccheck.log](synccheck.log)、[racecheck.log](racecheck.log) 保留工具訊息及11 sessions應用層結果：

```text
Error: Failed to initialize WDDM debugger interface. Please run EnableDebuggerInterface.bat as an administrator
Error: Device not supported. Please refer to the "Supported Devices" section of the sanitizer documentation
```

四次應用程式均自行印出5137 checks／11 GPU sessions passed，但工具沒有取得有效檢查結果。racecheck另印出process未成功終止、可能host Unified Memory存取錯誤的診斷，以及2 hazards；此build使用pinned host allocations、沒有managed memory API，僅憑初始化失敗後的工具訊息無法判定成因，保留原樣並列為未解的工具執行失敗。沒有把這些訊息當作已完成racecheck或當作已排除程式錯誤。

[NVIDIA文件](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)記載WDDM debugging interface需啟用；此處未改Windows管理員設定。後續需在介面可用的環境重跑四工具，才可補齊sanitizer驗收。即使racecheck成功，其shared-memory檢查也不涵蓋跨stream global-memory順序，仍需協定核對、時序擾動及完整bytes比較。

## 證據取回與重跑

[raw-runs.tar.gz](raw-runs.tar.gz) 是原始輸出及metadata的無損封存，含72個完整矩陣sessions、四工具各11個sessions及3個CLI輸出。每個case都有 `packets.bin`（GPU原始D2H bytes）、`resolved.cfg`、`behavior.cfg`、兩個hash、`identity.txt`、event `audit.txt`及完整terminal snapshot。**沒有用host重新編碼的bytes替換GPU輸出。** `raw-evidence.sha256`對archive內所有payload檔案逐一列hash；外層[SHA256SUMS](SHA256SUMS)保護交付檔案。

從repo root驗證封存，無需GPU：

```sh
sha256sum -c evidence/cuda-synthetic/2026-10-28/SHA256SUMS
mkdir -p build-evidence
tar -xzf evidence/cuda-synthetic/2026-10-28/raw-runs.tar.gz -C build-evidence
(cd build-evidence && sha256sum -c raw-evidence.sha256)
build-cpu/packet_compare build-evidence/cli-serial/packets.bin build-evidence/cli-pipelined/packets.bin
```

完整[CPU／GPU重跑、schema與資源說明](../../../docs/CUDA_SYNTHETIC.md)。要重現交付build身分可在乾淨checkout `c453cec5f634e714a302d7897ce4e0b4151430e3`執行該文件的CMake命令；在後續evidence commit重跑時build_commit會如實不同，程式碼相同也不冒充原build。

未驗證：sanitizer有效檢查、任意N／shape／seed／超長序列、跨GPU／跨build逐位元一致性、GPU fatal／OOM／API錯誤注入、header損毀負測試與效能加速比。YOLO／Mamba推論、完整GMC、tracker、MOT writer與模型artifact P01仍由各自任務接續。
