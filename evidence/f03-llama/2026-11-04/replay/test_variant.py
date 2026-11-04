from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
kind=sys.argv[1]
for mode,args in [('verify',['--verify-only']),('benchmark',['--benchmark-only','--rounds','3','--iterations','100','--warmup','20'])]:
 name=kind+'-'+mode
 subprocess.run([sys.executable,str(p/'run.py'),name,str(p/('scan_benchmark_'+kind)),str(p/name),*args],check=True)
