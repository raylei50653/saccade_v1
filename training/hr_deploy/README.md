# HR Full25 e10 deployment

Fixed student: HR01-MS-FULL25-20261103 mixed e10, checkpoint SHA-256
`53bccffe9bdbf37bdcfc81d03e95ccbd2d142d67a557ea015b76d356e4fa5f7d`.
All outputs belong in `~/.cache/saccade-v1/hr-deploy/`. The research checkout,
its caches, and the pinned T01 source are read only. No training or MOT17-02
evaluation is performed.

Run from this worktree:

```sh
PY=$HOME/.cache/saccade-v1/t01/venv/bin/python
PYX=$HOME/.cache/saccade-v1/t01/venv-export/bin/python
OUT=$HOME/.cache/saccade-v1/hr-deploy
DEV=/home/ray/developer/ai/saccade_v1_dev
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu --parallel 8
ctest --test-dir build/cpu --output-on-failure --parallel 8
CUDA_VISIBLE_DEVICES='' "$PY" training/hr_deploy/test_eval_scdr.py

build/cpu/head_weights convert \
  "$OUT/sources/epoch_0010.ckpt" \
  53bccffe9bdbf37bdcfc81d03e95ccbd2d142d67a557ea015b76d356e4fa5f7d \
  "$OUT/hr-head.swb" "$OUT/hr-head-mapping.tsv"
build/cpu/head_weights convert \
  "$HOME/.cache/saccade-v1/s04/best.ckpt" \
  a5a0e7091cd854ad68210ae4bbd43af73c7d3d098ea119f02ea65fab59c0676c \
  "$OUT/s04-regression.swb" "$OUT/s04-regression-mapping.tsv"
build/cpu/head_weights convert \
  "$HOME/.cache/saccade-v1/t01/runs/t1/attempt-01/run/epoch_0001.ckpt" \
  070580f9f5755e57acd9ee7ed9b4307672f4d9996332b40e6b61ee7cc6e6962b \
  "$OUT/t01-regression.swb" "$OUT/t01-regression-mapping.tsv"

CUDA_VISIBLE_DEVICES='' "$PYX" training/hr_deploy/export.py backbone
flock "$HOME/.cache/saccade-v1/gpu.lock" "$PYX" training/hr_deploy/export.py verify-backbone
flock "$HOME/.cache/saccade-v1/gpu.lock" "$PY" training/hr_deploy/export.py references
CUDA_VISIBLE_DEVICES='' "$PY" training/hr_deploy/eval_scdr.py \
  /home/ray/developer/ai/saccade_v1-worktrees/m1-data/MOT17-10-SDP.scdr \
  /home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP \
  > "$OUT/t01-scdr-validation.json"
CUDA_VISIBLE_DEVICES='' "$PY" training/hr_deploy/export.py validation-replay
CUDA_VISIBLE_DEVICES='' "$PY" training/hr_deploy/export.py manifest
```

The training venv lacks ONNX/ONNX Runtime. The existing export venv has
ONNX 1.21.0 / ONNX Runtime 1.24.4; both venvs have Torch 2.11.0+cu130 and
Ultralytics 8.4.37. No packages are installed by these commands.

The backbone uses the teacher checkpoint's `model` (primary EMA).
`hr_teacher.py` saves raw weights in `train_model`;
`hr_cache.build_teacher` and `OnlineFeatures` both reach the pinned
`build_gated_yolo_detector`, which loads `model`. Verification compares
both states against four train-cache frames and four val-cache frames,
including the cache's CPU FP16 rounding: all 51,118,080 elements match,
while raw weights differ in over 50 million elements. Worst ONNX relative
L2 is 4.11e-6 versus PyTorch and 2.13e-4 versus cache. ONNX verification applies the
BACKBONE_POLICY secondary relative-L2 limits (1e-4 and 2^-11 + 1e-4);
it does not perform the native binary64 16-sigma acceptance.

`refs/cases.tsv` has exactly the eight columns consumed by
`preprocess run REFS OUTPUT_DIR` / `SACCADE_PREPROCESS_REFERENCES`.
The four MOT17-10 frames are 1, 218, 437 and 654. RGB files are planar
uint8; inputs are FP32 NCHW [1,3,768,1280], nvJPEG RGB /255 followed by
antialiased bilinear stretch through the research `hr-input/1` function.

F03 dumps are little-endian FP32 with no header:

- `refs/backbone/input-I.f32`: P3 [1,256,96,160], P4 [1,512,48,80],
  P5 [1,512,24,40], concatenated.
- `refs/model/input-I.f32`: cls P3/P4/P5 (80 channels), then reg P3/P4/P5
  (4 channels), with spatial sizes 96x160, 48x80, 24x40.
- `refs/model-images.f32` and `refs/head-features.f32`: all four inputs
  or feature frames concatenated in the same order.

These match `model_trt run --dump` / `backbone_trt run --dump` and the
tensor order used by `t01_export.py head-compare`; comparisons must use
the HR shapes recorded in `refs/manifest.json`. F03 references use the
live FP32 backbone features and pinned PyTorch stepwise head, with TF32
off and deterministic cuDNN. They do not round features through FP16.

The SCDR tool uses original-pixel boxes, kept rows only, frame+1 GT lookup,
and T01's Accumulator, marked-person / visibility >=0.1 GT rules and
101-point AP50. Native T01 gives AP50 **67.58502167790465%**, compared
with recorded cached-feature validation **67.58440686451651%**
(+0.00061481338814 percentage points). Both have 6,933 TP / 1,459 FP at
0.25 and 7,327 TP / 2,216 FP at 0.1. At 0.001 both have 8,642 TP;
native has 10,493 FP versus 10,497 recorded. The input detection streams
differ: live TensorRT FP32 features and FP32 coordinate mapping versus
FP16-rounded training-cache features, PyTorch head and binary64 metric
mapping. AP equality is therefore not claimed. The exact contribution
of each numerical stage to the small ranking difference is unmeasured.

The saved Full25 canonical score/TP events reproduce
**86.00928859904053%** exactly with T01's AP implementation, with
9,291 TP / 1,958 FP at 0.25. Replay checks the selected checkpoint and
validation hashes and equality of the full recorded metrics.
`manifest.json` records all artifact and executed-source hashes;
`refs/manifest.json` separately hashes each reference file.
