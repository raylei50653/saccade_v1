from pathlib import Path
import subprocess,json
root=Path('build/f02-opt').resolve()
out=root/'counters';out.mkdir()
metrics='gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct'
for backend,kernels,skip,count,exe in [('baseline','horizontal|vertical',40,2,'baseline-resize-probe'),('candidate','tiled_hr',20,1,'final-resize-probe')]:
    cmd=['ncu','--kernel-name-base','demangled','--kernel-name',f'regex:saccade::.*({kernels})','--launch-skip',str(skip),'--launch-count',str(count),'--replay-mode','application','--cache-control','none','--clock-control','none','--metrics',metrics,'--csv','--log-file',str(out/f'{backend}.csv'),str(root/exe)]
    (out/f'{backend}.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    print('START',backend,flush=True)
    with (out/f'{backend}.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    (out/f'{backend}.exit').write_text(str(result.returncode)+'\n');result.check_returncode()
    print('DONE',backend,flush=True)
