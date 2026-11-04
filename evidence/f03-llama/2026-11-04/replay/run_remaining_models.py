from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
for kind in ['candidate2','candidate3']:
 subprocess.run([sys.executable,str(p/'run.py'),kind+'-model-abba',sys.executable,str(p/'run_model_variant.py'),kind],check=True)
