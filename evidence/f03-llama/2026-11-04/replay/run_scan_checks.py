from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
suite=p/'selective_scan_cuda_tests_block128'
def run(name,args): subprocess.run([sys.executable,str(p/'run.py'),name,*map(str,args)],check=True)
run('block128-scan-suite',[suite,p/'block128-scan-suite.tsv'])
run('block128-lazy-streams',['env','CUDA_MODULE_LOADING=LAZY',suite,'--group','streams',p/'block128-lazy-streams.tsv'])
for tool in ['memcheck','initcheck','racecheck','synccheck']:
 args=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--report-api-errors','explicit',suite]
 if tool=='initcheck': args+=['--no-held-stream']
 args+=[p/('block128-'+tool+'.tsv')]
 run('block128-'+tool,args)
