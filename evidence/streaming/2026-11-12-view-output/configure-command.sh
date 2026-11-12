#!/bin/sh
# Run 2: exact requested configure; corrected reference config passed preflight.
cmake -S /home/ray/developer/ai/saccade_v1-worktrees/stream-view -B /home/ray/developer/ai/saccade_v1-worktrees/stream-view/build/stream-view -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_PREPROCESS=ON -DSACCADE_ENABLE_POSTPROCESS=ON \
  -DSACCADE_ENABLE_MODEL_PRODUCER=ON -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_ENABLE_MODEL_TRT=ON \
  -DSACCADE_ENABLE_DETECTION_SESSION=ON -DSACCADE_ENABLE_FLOW=ON \
  -DSACCADE_ENABLE_VIDEO=ON -DSACCADE_ENABLE_NVDEC=ON -DSACCADE_ENABLE_STREAMING=ON \
  -DSACCADE_TENSORRT_ROOT=/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSACCADE_STREAMING_TEST_ENGINE=/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine \
  -DSACCADE_STREAMING_TEST_SEQUENCE=/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
