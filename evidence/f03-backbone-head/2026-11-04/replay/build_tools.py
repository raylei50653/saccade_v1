from pathlib import Path
import subprocess,shlex
p=Path(__file__).resolve().parent
old=p.parent/'f03-llama-20261104/full-baseline'
scan=p.parent/'f03-llama-20261104/libscan_block128.a'
flags=['/usr/sbin/c++','-O3','-DNDEBUG','-std=c++20','-Wall','-Wextra','-Wpedantic','-Werror','-I'+str(p/'source-baseline/include'),'-isystem','/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk/usr/include/x86_64-linux-gnu','-isystem','/opt/cuda/targets/x86_64-linux/include','-isystem','/opt/cuda/targets/x86_64-linux/include/cccl']
def run(name,cmd,cwd=None):
 call=['python3',str(p/'run.py'),name,*map(str,cmd)]
 subprocess.run(call,cwd=cwd,check=True)
obj=p/'model_benchmark.o'
run('compile-model-benchmark',flags+['-c',p/'source-baseline/tools/model_benchmark.cpp','-o',obj])
cmd=shlex.split((old/'CMakeFiles/model_benchmark.dir/link.txt').read_text())
cmd[cmd.index('-o')+1]=str(p/'model_benchmark')
cmd=[str(obj) if x=='CMakeFiles/model_benchmark.dir/tools/model_benchmark.cpp.o' else str(scan) if x=='libsaccade_selective_scan_cuda.a' else x for x in cmd]
run('link-model-benchmark',cmd,cwd=old)
for kind in ['baseline','opt5','workspace1g','tiling-fast','causal-padding']:
 src=p/'source-baseline/src/head_trt_network.cpp' if kind=='baseline' else p/'source-variants'/kind/'src/head_trt_network.cpp'
 obj=p/('head-'+kind+'.o');lib=p/('libhead-'+kind+'.a')
 run('compile-head-'+kind,flags+['-c',src,'-o',obj])
 run('archive-head-'+kind,['ar','rcs',lib,obj])
 cmd=shlex.split((old/'CMakeFiles/model_trt.dir/link.txt').read_text())
 cmd[cmd.index('-o')+1]=str(p/('model_trt_'+kind))
 cmd=[str(scan) if x=='libsaccade_selective_scan_cuda.a' else str(lib) if x=='libsaccade_head_trt.a' else x for x in cmd]
 run('link-model-trt-'+kind,cmd,cwd=old)
