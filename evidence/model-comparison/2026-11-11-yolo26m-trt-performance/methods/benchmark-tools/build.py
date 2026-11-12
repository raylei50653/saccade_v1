from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent
REPO=R.parents[2]
SDK=Path('/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk')
LIB=SDK/'usr/lib/x86_64-linux-gnu'
B=REPO/'build/report1-live-20261111/cuda'
for name in ['build.log','build.json']:
    p=R/name
    if p.exists():
        i=1
        while (R/f'build-attempt-{i:02d}{p.suffix}').exists():i+=1
        p.rename(R/f'build-attempt-{i:02d}{p.suffix}')
cmd=['c++','-O3','-DNDEBUG','-std=c++20','-ffp-contract=off','-fno-fast-math',
     '-I'+str(REPO/'include'),'-I'+str(SDK/'usr/include/x86_64-linux-gnu'),'-I/opt/cuda/include',
     str(R/'model_compare.cpp'),'-o',str(R/'model_compare'),'-Wl,--disable-new-dtags',
     '-Wl,-rpath,'+str(LIB)+':/opt/cuda/targets/x86_64-linux/lib','-Wl,--start-group',
     *[str(B/('libsaccade_'+x+'.a')) for x in ['scan_trt','selective_scan_cuda','selective_scan','file_io','sha256']],
     '-Wl,--end-group',str(LIB/'libnvinfer.so.10.16.1'),'/opt/cuda/targets/x86_64-linux/lib/libcudart.so',
     '/opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so','-L/opt/cuda/targets/x86_64-linux/lib',
     '-lcudadevrt','-lcudart_static','-lrt','-lpthread','-ldl']
p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(R/'build.log').write_text(p.stdout)
(R/'build.json').write_text(json.dumps({'command':cmd,'exit_code':p.returncode,
    'source_sha256':hashlib.sha256((R/'model_compare.cpp').read_bytes()).hexdigest(),
    'compiler':subprocess.check_output(['c++','--version'],text=True)},indent=2)+'\n')
print(p.stdout[-2500:]);print('exit',p.returncode)
if p.returncode==0:print(subprocess.check_output([str(R/'model_compare'),'--help'],text=True))
raise SystemExit(p.returncode)
