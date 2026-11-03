nvJPEG HARDWARE backend: nvjpegDecodeBatched writes the output before earlier work in the same stream has completed

Files
- repro_order.cu   standalone repro (CUDA runtime + nvJPEG only, ~140 lines)
- generate.py      script that produced a.jpg and b.jpg (numpy + Pillow 12.2.0)
- a.jpg, b.jpg     inputs: two 1920x1080 baseline 8-bit 4:2:0 JPEGs (quality 95) with different content
- wsl2/            logs from WSL2, plus identity.log (nvidia-smi, nvcc, ldd, hashes)
- windows/         logs from the same source built and run as a native Windows executable

SHA-256
  86f42ef040588642beee0e9ef562a76a7670b34c20671d744141432c44f539cf  repro_order.cu
  690d3813a0db124098b0ff4df5fec1109c4b60208aa17fea3388559798f9d81b  a.jpg
  91cdc26ac1e6036572415abe22652c0c487cd325bffe0f4bb70e512c0221ac49  b.jpg

What the repro does (per trial, all on one non-blocking stream S)
  1. nvjpegDecodeBatched(A -> DST, S); record an event; host waits for it
  2. enqueue on S: spin kernel (SPIN_MS, %globaltimer busy-wait), then copy kernel DST -> SNAP
  3. with no synchronization, nvjpegDecodeBatched(B -> DST, S)
  4. cudaStreamSynchronize(S); download SNAP; compare with decode(A) and decode(B)
  Under stream order the copy in step 2 runs before any GPU write of step 3, so SNAP must
  equal decode(A). "control" mode adds cudaStreamSynchronize(S) before step 3.

Build
  Linux/WSL2: nvcc -O2 -std=c++17 -arch=sm_120 repro_order.cu -lnvjpeg -o repro_order
  Windows:    nvcc -O2 -std=c++17 -arch=sm_120 repro_order.cu -lnvjpeg -o repro_order.exe
              (from a VS 2022 x64 developer prompt, MSVC 14.44)

Run
  usage: repro_order hw|hybrid default|no-pipeline order|control SPIN_MS TRIALS A.jpg B.jpg
  exit code: 0 = every trial ordered, 3 = order violation observed, 1 = API error

  ./repro_order hw     default     order   20 20 a.jpg b.jpg  -> 19/20 ORDER_VIOLATED, exit 3
  ./repro_order hw     no-pipeline order   20 20 a.jpg b.jpg  -> 19/20 ORDER_VIOLATED, exit 3
  ./repro_order hw     default     control 20 20 a.jpg b.jpg  -> 0/20, exit 0
  ./repro_order hybrid default     order   20 20 a.jpg b.jpg  -> 0/20, exit 0
  ./repro_order hw     default     order    1 20 a.jpg b.jpg  -> 0/20, exit 0

  In every violating trial all differing bytes of SNAP equal decode(B), i.e. B was fully
  written to DST before the copy kernel queued ahead of the decode ran. Trial 0 of each run
  was ordered in every run we made; we did not investigate why.
  The HYBRID call blocks the host for 20.4-20.7 ms (until the spin finishes); the HARDWARE
  call returns in 0.40-0.88 ms on WSL2 and 0.04-0.08 ms on native Windows.

Environment
- GeForce RTX 5070 Ti Laptop GPU (SM 12.0), driver 617.42 (KMD)
- WSL2: guest kernel Linux 6.18.33.1-microsoft-standard-WSL2 x86_64, CUDA 13.4.92, nvJPEG 13.2.3
- Native Windows: Windows 11 Pro 10.0.26300 x64, WDDM, nvcc 13.4.92, nvJPEG 13.2.3.58 redist DLL, MSVC 14.44
- Same GPU and Windows KMD in both cases. Not tested on a native Linux driver.
