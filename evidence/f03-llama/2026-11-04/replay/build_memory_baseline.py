from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
flags=['/opt/cuda/bin/nvcc','-O3','-DNDEBUG','-std=c++20','-arch=sm_120','--ftz=false','--prec-div=true','--prec-sqrt=true','--fmad=true','-lineinfo','--ptxas-options=-v']
renames=['-D'+n+'='+n+'_baseline' for n in ['selective_scan_cuda','selective_scan_cuda_load_kernels','selective_scan_cuda_identity','cross_scan_expand_cuda','cross_scan_merge_cuda']]
source=p/'source/block128'
subprocess.run(['python3',str(p/'run.py'),'memory-baseline-compile',*flags,'-I'+str(source/'include'),*renames,'-c',str(source/'cuda/selective_scan.cu'),'-o',str(p/'memory-baseline.o')],check=True)
