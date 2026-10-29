cmake -S build-s02-evidence/source -B build-s02-evidence/cuda -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 
