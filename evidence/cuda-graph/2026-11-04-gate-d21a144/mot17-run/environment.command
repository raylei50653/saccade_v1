bash -c date\ -Is\;\ uname\ -srmo\;\ \"\$1\"\ --version\;\ nvidia-smi\ --query-gpu=name\,uuid\,driver_version\,memory.free\,clocks.max.sm\ --format=csv _ nsys 
