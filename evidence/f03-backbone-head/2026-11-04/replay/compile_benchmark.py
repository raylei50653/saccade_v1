from pathlib import Path
import subprocess,shutil,shlex,sys
p=Path(__file__).resolve().parent;label=sys.argv[1]
old=p.parent/'f03-llama-20261104/full-baseline';scan=p.parent/'f03-llama-20261104/libscan_block128.a'
src=p/('source-'+label)/'tools/model_benchmark.cpp';src.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p.parents[1]/'tools/model_benchmark.cpp',src)
flags=['/usr/sbin/c++','-O3','-DNDEBUG','-std=c++20','-Wall','-Wextra','-Wpedantic','-Werror','-I'+str(p/'source-baseline/include'),'-isystem','/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk/usr/include/x86_64-linux-gnu','-isystem','/opt/cuda/targets/x86_64-linux/include','-isystem','/opt/cuda/targets/x86_64-linux/include/cccl']
obj=p/('model_benchmark_'+label+'.o')
subprocess.run(['python3',str(p/'run.py'),'compile-benchmark-'+label,*flags,'-c',str(src),'-o',str(obj)],check=True)
cmd=shlex.split((old/'CMakeFiles/model_benchmark.dir/link.txt').read_text());cmd[cmd.index('-o')+1]=str(p/('model_benchmark_'+label))
cmd=[str(obj) if x=='CMakeFiles/model_benchmark.dir/tools/model_benchmark.cpp.o' else str(scan) if x=='libsaccade_selective_scan_cuda.a' else x for x in cmd]
subprocess.run(['python3',str(p/'run.py'),'link-benchmark-'+label,*cmd],cwd=old,check=True)
