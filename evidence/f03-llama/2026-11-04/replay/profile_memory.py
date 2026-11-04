from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
kind=sys.argv[1]
metrics='gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum'
filters={'baseline':('read-shuffle','regex:.*scan_kernel.*'),'shuffle':('read-shuffle','regex:.*scan_kernel<.bool.1, .bool.1, .bool.1>.*'),'shared':('read-shared-bc','regex:.*scan_kernel_deploy_shared_bc.*')}
binary,pattern=filters[kind]
cmd=['ncu','--target-processes','all','--csv','--log-file',str(p/('memory-'+kind+'-ncu.csv')),'--replay-mode','application','--cache-control','none','--clock-control','none','--kernel-name-base','demangled','--kernel-name',pattern,'--launch-skip','20','--launch-count','1','--metrics',metrics,sys.executable,str(p/'profile_probe.py'),binary]
subprocess.run([sys.executable,str(p/'run.py'),'memory-'+kind+'-ncu-final','flock','-w','300','/home/ray/.cache/saccade-v1/gpu.lock','timeout','--signal=TERM','--kill-after=10s','120',*cmd],check=True)
