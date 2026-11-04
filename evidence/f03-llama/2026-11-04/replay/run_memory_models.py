from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
for kind in ['read-shuffle','read-shared-bc']:
 subprocess.run([sys.executable,str(p/'run.py'),kind+'-model-abba',sys.executable,str(p/'run_memory_model.py'),kind],check=True)
