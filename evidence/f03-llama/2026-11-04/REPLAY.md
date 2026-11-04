# F03 llama.cpp 試驗重跑

本機完整capture為`build/f03-llama-20261104/`。以原SDK、engine與四張輸入重跑時，建立新的`build/NAME`，把本evidence的`replay/*.py`複製至該目錄；將`source/`複製至其`source/`，並在該目錄解開`source/baseline-source.tar.gz`，得到`source/full-baseline/`。新目錄只接受尚不存在的量測輸出子目錄。

1. 依`capture/configure-baseline.command`配置新路徑的`source/full-baseline`與`full-baseline`，再依`capture/build-baseline-retry.command`建置`model_benchmark`。兩個source及build路徑都要換成新的`build/NAME`。
2. 執行新目錄的`build_primitive.py`，重建原N16／4states對照harness。`build_variant.py candidate2|candidate3|block32|block64|block128`產生其他candidate與完整模型binary，baseline CUDA object共用。
3. 在`~/.cache/saccade-v1/gpu.lock`下執行`test_variant.py VARIANT`；此腳本依序跑精確比較與三輪ABBA。`run_model_variant.py VARIANT`跑完整F03外層ABBA並比對四張raw outputs／statuses；`run_model_confirmation.py`跑加長的block128確認。`build_suite.py`建置並連結採用版scan suite；在GPU lock下執行`run_scan_checks.py`重跑suite、lazy streams及四種sanitizer。
4. 要比較共用讀取，先執行`build_memory_baseline.py`，再用`build_memory_variant.py read-shuffle|read-shared-bc`。這兩組使用採用版block128為baseline，不使用較早N16。`test_variant.py`及`run_memory_model.py VARIANT`分別跑scan及完整模型。
5. `profile_memory.py baseline|shuffle|shared`使用NCU做隔離P3 counter。每個application replay由`profile_probe.py`建立新輸出目錄；確認CSV確實包含9個metric rows，不能只用exit0判斷有profile到kernel。

原命令、工具輸出與退出碼皆保存於`capture/`。`run.py NAME COMMAND...`將命令、log、退出碼及時間寫入其自身所在目錄；外層GPU工作應沿用`flock`與`timeout`。所有結果仍需標為development evidence，不能直接生成clean-source正式證書。

本evidence中的`run_ncu.py`是最初失敗嘗試，保留供追查；counter重跑使用`profile_memory.py`。採用版scan suite／四種sanitizer的完整參數見`capture/block128-*.command`；原suite以CMake建置後只替換CUDA static library重新連結，其連結命令亦已保存。
