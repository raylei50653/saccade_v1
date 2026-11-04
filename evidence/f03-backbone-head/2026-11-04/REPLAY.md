# Backbone 與 head 分析重跑

完整本機capture在`build/f03-backbone-head-20261104/`。所有重跑需新的`build/NAME`，沿用pinned TensorRT 10.16.1.11、CUDA 13.4.92及相同engine／inputs，並在`~/.cache/saccade-v1/gpu.lock`下串行GPU工作。

## 從來源重新建置

1. 在新目錄解開`source/baseline-source.tar.gz`，得到`source-baseline/`。將本evidence的最終`source/model_benchmark.cpp`覆蓋到該snapshot的`tools/model_benchmark.cpp`；archive保留的是試驗開始時工具版本，含已保留的早期profiler錯誤。
2. 依repository建置方式配置CUDA／TensorRT／MODEL_TRT、Release、sm_120，建立`model_benchmark`及`model_trt`。現行scan來源已包含前輪shape特化／block128。
3. 對每個builder候選另複製source tree，僅以`source/source-variants/VARIANT/src/head_trt_network.cpp`替換相應檔案，再建`model_trt`。FP32原路徑未改，FP16候選各改單一項目。

本輪為避免重編無關objects，原始`build_tools.py`／`compile_benchmark.py`連結前輪固定CMake libraries；精確命令均在`capture/`。若重用這些scripts，先將其`old`及`scan`路徑改為自己的固定build，不能把本機絕對路徑當成可攜設定。

## 量測與 trace

```sh
model_benchmark ENGINE IMAGES NEW_OUT --profile hr-1280x768 \
  --iterations 160 --warmup 40 --rounds 1

# 逐層插樁診斷，獨立於上面的原計時
model_benchmark ENGINE IMAGES NEW_OUT --profile hr-1280x768 \
  --iterations 80 --warmup 30 --rounds 1 --layer-iterations 60

# 只擷取原Graph的20個重播，不啟用逐層profiler
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output NEW_TRACE \
  model_benchmark ENGINE IMAGES NEW_OUT --profile hr-1280x768 \
  --iterations 20 --warmup 30 --rounds 1 --trace-iterations 20
```

外層需baseline→candidate→candidate→baseline，不能用單一程序內ordinary／Graph的ABBA代替設定對照。`replay/context_abba.py`、`context_confirm.py`、`engine_abba.py`及`causal_current_abba.py`保存本輪程序；路徑改為新目錄後執行。各engine輸出都保存四張輸入，計時只固定input0。

分析器固定讀取`ROOT/f03-graph.sqlite`、`ROOT/graph-layer-profile/engine-layers.json`及`ROOT/nsys-model/identity.txt`。以`nsys export --type sqlite --output ROOT/f03-graph.sqlite NEW_TRACE.nsys-rep`匯出後，將新engine inspector與identity放在上述位置，執行`python3 replay/analyze_trace.py ROOT`。它必須核對完整layers→kernel序列、每node calls及unknown0，再使用分類百分比。20次trace span÷20是processing interval；IProfiler插樁總時間不能作未插樁GPU busy。

`capture/*.command`保存每次完整參數、`*.exit`保存結果。新build可能重新選tactics；engine plan hashes及raw bits不保證與本次相同。比較時使用每個engine實際dump出的六raw tensors及完整九status words，保留所有數值差異，不能只確認build成功或flags一致。
