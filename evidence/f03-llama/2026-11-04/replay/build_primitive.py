from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
obj=p/'primitive-objects'; obj.mkdir(exist_ok=True)
flags=['/opt/cuda/bin/nvcc','-O3','-DNDEBUG','-std=c++20','-arch=sm_120','--ftz=false','--prec-div=true','--prec-sqrt=true','--fmad=true','-lineinfo','--ptxas-options=-v']
renames=['-D'+n+'='+n+'_baseline' for n in ['selective_scan_cuda','selective_scan_cuda_load_kernels','selective_scan_cuda_identity','cross_scan_expand_cuda','cross_scan_merge_cuda']]
for name,source,extra in [('baseline',p/'source/full-baseline',renames),('candidate',p/'source/candidate',[])]:
 cmd=flags+['-I'+str(source/'include'),'-DSACCADE_BUILD_COMMIT="f03-llama-prototype"','-DSACCADE_BUILD_DIRTY="true"']+extra+['-c',str(source/'cuda/selective_scan.cu'),'-o',str(obj/(name+'.o'))]
 subprocess.run(['python3',str(p/'run.py'),'primitive-compile-'+name,*cmd],check=True)
cmd=flags+['-I'+str(p/'source/full-baseline/include'),'-c',str(p/'source/full-baseline/tools/scan_benchmark.cpp'),'-o',str(obj/'harness.o')]
subprocess.run(['python3',str(p/'run.py'),'primitive-compile-harness',*cmd],check=True)
subprocess.run(['python3',str(p/'run.py'),'primitive-link',*flags,str(obj/'harness.o'),str(obj/'baseline.o'),str(obj/'candidate.o'),'-o',str(p/'scan_benchmark')],check=True)
