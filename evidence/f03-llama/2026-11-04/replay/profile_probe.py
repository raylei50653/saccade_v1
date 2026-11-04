from pathlib import Path
import tempfile,os,sys
p=Path(__file__).resolve().parent
kind=sys.argv[1]
out=Path(tempfile.mkdtemp(prefix='ncu-probe-',dir=p))/'run'
binary=p/('scan_benchmark_'+kind)
args=[str(binary),str(out),'--benchmark-only','--rounds','1','--iterations','1','--warmup','20']
os.execv(binary,args)
