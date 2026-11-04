from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
for kind in ['block32','block64','block128']:
 subprocess.run([sys.executable,str(p/'test_variant.py'),kind],check=True)
