from pathlib import Path
import subprocess,shlex
p=Path(__file__).resolve().parent
work=p/'full-baseline'
subprocess.run(['cmake','--build',str(work),'--target','selective_scan_cuda_tests','--parallel','4'],check=True)
cmd=shlex.split((work/'CMakeFiles/selective_scan_cuda_tests.dir/link.txt').read_text())
cmd[cmd.index('-o')+1]=str(p/'selective_scan_cuda_tests_block128')
cmd=[str(p/'libscan_block128.a') if x=='libsaccade_selective_scan_cuda.a' else x for x in cmd]
subprocess.run(cmd,cwd=work,check=True)
