from pathlib import Path
import json
import shlex
import subprocess

ROOT = Path('/home/ray/developer/ai/saccade_v1')
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'build/report1-live-20261111'
SRC, LIB = BASE / 'source', BASE / 'cuda'
INCLUDES = [str(SRC/'include'), str(OUT)]
SYSTEM = ['/opt/cuda/targets/x86_64-linux/include', '/opt/cuda/targets/x86_64-linux/include/cccl',
          '/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk/usr/include/x86_64-linux-gnu']
COMMANDS = []

def run(command):
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    COMMANDS.append({'command': command, 'exit': result.returncode, 'output': result.stdout})
    (OUT/'build.json').write_text(json.dumps(COMMANDS, indent=2)+'\n')
    print('build', Path(command[-1]).name, result.returncode, flush=True)
    assert result.returncode == 0, result.stdout

CC = ['/usr/bin/c++', '-O3', '-DNDEBUG', '-std=c++20', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
      '-ffp-contract=off', '-DSACCADE_BUILD_COMMIT="3cbaa518f0dc37d691404e95ed85c0109c81130d"',
      '-DSACCADE_BUILD_DIRTY="true"']
CC += [value for item in INCLUDES for value in ['-I', item]]
CC += [value for item in SYSTEM for value in ['-isystem', item]]

for name in ['tracking_session', 'detection_session']:
    run(CC+['-c', str(OUT/(name+'.cpp')), '-o', str(OUT/(name+'.o'))])
NV = ['/opt/cuda/bin/nvcc', '-O3', '-DNDEBUG', '-std=c++20',
      '--generate-code=arch=compute_120,code=[compute_120,sm_120]',
      '--ftz=false', '--prec-div=true', '--prec-sqrt=true', '--fmad=false', '-lineinfo', '-Xcompiler=-ffp-contract=off']
NV += [value for item in INCLUDES+SYSTEM[:2] for value in ['-I', item]]
run(NV+['-c', str(OUT/'yolo_producer.cu'), '-o', str(OUT/'yolo_producer.o')])

def native_link(name, tool_object=None, output=None, both=False):
    command = shlex.split((LIB/'CMakeFiles'/f'{name}.dir/link.txt').read_text())
    command = [str(LIB/value) if value.startswith('CMakeFiles/') or
               (value.startswith('libsaccade_') and value.endswith('.a')) else value for value in command]
    command[command.index('-o')+1] = str(output or OUT/name)
    command = [value for value in command if not value.startswith('-Wl,--dependency-file=')]
    position = next(index for index, value in enumerate(command) if value.endswith('.o'))
    if tool_object is not None:
        command[position] = str(tool_object)
    copies = [OUT/'detection_session.o'] if name=='detect_sequence' else [OUT/'tracking_session.o']
    if both:
        copies += [OUT/'detection_session.o']
    command[position+1:position+1] = [str(value) for value in copies+[OUT/'yolo_producer.o']]
    if both:
        command.insert(command.index(str(LIB/'libsaccade_tracking_session.a'))+1,
                       str(LIB/'libsaccade_detection_session.a'))
    run(command)

for name in ['detect_sequence', 'track_sequence', 'tracking_benchmark']:
    # Recompile the unchanged CLI with research source-dirty identity, rather
    # than inheriting the clean build label embedded in the original object.
    run(CC+['-c', str(SRC/'tools'/(name+'.cpp')), '-o', str(OUT/(name+'.o'))])
    native_link(name, OUT/(name+'.o'))
probe = CC+[str(OUT/'producer_probe.cpp'), str(OUT/'yolo_producer.o'), str(LIB/'libsaccade_postprocess.a'),
            '-L/opt/cuda/targets/x86_64-linux/lib', '-L/opt/cuda/targets/x86_64-linux/lib/stubs',
            '-Wl,-rpath,/opt/cuda/targets/x86_64-linux/lib', '-lcudart', '-lcuda', '-o', str(OUT/'producer_probe')]
run(probe)
if (OUT/'pipeline_benchmark.cpp').exists():
    run(CC+['-c', str(OUT/'pipeline_benchmark.cpp'), '-o', str(OUT/'pipeline_benchmark.o')])
    native_link('tracking_benchmark', OUT/'pipeline_benchmark.o', OUT/'pipeline_benchmark', both=True)
print('research binaries built; no GPU execution', flush=True)
