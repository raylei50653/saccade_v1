Compute Sanitizer memcheck reports failing cuArrayGetPlane calls from nvcuvid during valid NVOF API use

Files
- repro_getplane.cpp      standalone repro: CUDA Driver API and the NVOF library only, no kernels, no CUDA arrays of its own (106 lines)
- nvOpticalFlowCommon.h,
  nvOpticalFlowCuda.h     unmodified NVOF API 2.0 headers, NVIDIAOpticalFlowSDK commit edb50da3cf849840d680249aa6dbef248ebce2ca
- wsl2/                   logs from WSL2, plus identity.log (driver, tools, library hashes)
- windows/                logs from the same source built and run as a native Windows executable, plus windows-env.log

SHA-256
  486cea5b138d3425df36f31f073b183dafb0115bb16ca5a40b5f023fcb2d19ca  repro_getplane.cpp
  a83f6045e5c470b35a6c50672f92e082b78d55752fe51cc20eb1a2738be05b9d  nvOpticalFlowCommon.h
  07defc79637fb9893f2a06204972eaa6bcf9e5bea400c88440439ebaaa39f115  nvOpticalFlowCuda.h

What the repro does
  1. cuInit, retain the primary context, create one non-blocking stream
  2. NvOFAPICreateInstanceCuda, nvCreateOpticalFlowCuda, nvOFSetIOCudaStreams
  3. nvOFInit: 1920x1080, grid 4, NV_OF_MODE_OPTICALFLOW, NV_OF_PERF_LEVEL_SLOW
  4. EXECUTES > 0: create two GRAYSCALE8 inputs and one SHORT2 output (CUDEVICEPTR buffers), zero them,
     then call nvOFExecute EXECUTES times, synchronizing the stream after each call
  5. Destroy everything
  It stops at the first CUDA or NVOF call that does not succeed. It never calls cuArrayGetPlane or creates a CUDA array.

Build
  WSL2:     g++ -std=c++17 -O2 -I. -I/opt/cuda/include repro_getplane.cpp -L/usr/lib/wsl/lib -Wl,-rpath,/usr/lib/wsl/lib -lcuda -ldl -o repro_getplane
  Windows:  cl /std:c++17 /O2 /EHsc /I. /I"%CUDA%\include" repro_getplane.cpp /link /LIBPATH:"%CUDA%\lib\x64" cuda.lib
            (from a VS 2022 x64 developer prompt, CUDA = the CUDA 13.4 toolkit directory)

Run
  usage: repro_getplane EXECUTES    (0 stops after nvOFInit)
  exit code: 0 = every CUDA and NVOF call made by the repro succeeded, 1 = one of them failed

  compute-sanitizer --tool memcheck --print-limit 0 --error-exitcode 99 --report-api-errors MODE ./repro_getplane EXECUTES

Results (identical on WSL2 and native Windows)
  no sanitizer          EXECUTES=2    exit 0   every call succeeds
  MODE=explicit         EXECUTES=0    exit 99  74 errors
  MODE=explicit         EXECUTES=1    exit 99  86 errors (74 in nvOFInit, 12 in the first nvOFExecute)
  MODE=explicit         EXECUTES=2    exit 99  86 errors
  MODE=explicit         EXECUTES=30   exit 99  86 errors
  MODE=extended         EXECUTES=2    exit 99  172 errors (the 86 plus 86 driver check messages)
  MODE=no               EXECUTES=2    exit 0   0 errors

  Every report is:
    Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument" on CUDA API call to cuArrayGetPlane.
  and every backtrace runs through nvcuvid under the NVOF library (libnvcuvid.so.1 / libnvidia-opticalflow.so.1 on
  WSL2, nvcuvid.dll / nvofapi64.dll on Windows). In extended mode the driver also logs, once per call:
    CUDA API Error: Check failed: Cannot use this API for non-planar arrays
  All NVOF calls return NV_OF_SUCCESS.

Environment
- GeForce RTX 5070 Ti Laptop GPU (SM 12.0), driver 617.42 (KMD)
- WSL2: guest kernel Linux 6.18.33.1-microsoft-standard-WSL2 x86_64, NVIDIA-SMI 615.78.08, CUDA UMD 13.4,
  Compute Sanitizer 2026.3.0.0 (build 38855100), g++ 16.2.1
- Native Windows: Windows 11 Pro 10.0.26300 x64, WDDM, CUDA toolkit 13.4,
  Compute Sanitizer 2026.3.0.0 (build 38637409), MSVC 19.44
- Same GPU and Windows KMD in both cases. Not tested on a native Linux driver, another GPU or another driver version.
