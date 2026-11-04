from pathlib import Path
import subprocess,shlex,sys
p=Path(__file__).resolve().parent
kind=sys.argv[1]; source=p/'source'/kind
obj=p/(kind+'.o')
flags=['/opt/cuda/bin/nvcc','-O3','-DNDEBUG','-std=c++20','-arch=sm_120','--ftz=false','--prec-div=true','--prec-sqrt=true','--fmad=true','-lineinfo','--ptxas-options=-v']
def run(name,cmd): subprocess.run(['python3',str(p/'run.py'),kind+'-'+name,*map(str,cmd)],check=True)
run('compile',flags+['-I'+str(source/'include'),'-DSACCADE_BUILD_COMMIT="f03-llama-'+kind+'"','-DSACCADE_BUILD_DIRTY="true"','-c',str(source/'cuda/selective_scan.cu'),'-o',str(obj)])
run('primitive-link',flags+[str(p/'primitive-objects/harness.o'),str(p/'primitive-objects/baseline.o'),str(obj),'-o',str(p/('scan_benchmark_'+kind))])
lib=p/('libscan_'+kind+'.a'); subprocess.run(['ar','rcs',str(lib),str(obj)],check=True)
work=p/'full-baseline'; cmd=shlex.split((work/'CMakeFiles/model_benchmark.dir/link.txt').read_text())
cmd[cmd.index('-o')+1]=str(p/('model_benchmark_'+kind))
cmd=[str(lib) if x=='libsaccade_selective_scan_cuda.a' else x for x in cmd]
(p/(kind+'-link-model.command')).write_text('cd '+shlex.quote(str(work))+'\n'+shlex.join(cmd)+'\n')
with (p/(kind+'-link-model.log')).open('w') as f: result=subprocess.run(cmd,cwd=work,stdout=f,stderr=subprocess.STDOUT)
(p/(kind+'-link-model.exit')).write_text(str(result.returncode)+'\n'); result.check_returncode()
