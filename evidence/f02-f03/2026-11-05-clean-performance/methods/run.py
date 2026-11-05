import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / 'capture'
SOURCES = {
    'baseline': Path('/home/ray/developer/ai/saccade_v1-worktrees/f02-f03-perf-baseline-f4865cd-20261105'),
    'candidate': Path('/home/ray/developer/ai/saccade_v1-worktrees/f02-f03-clean-20261105'),
}
REVISIONS = {
    'baseline': 'f4865cdb27f0e624b8cc26219e78f4aa938690df',
    'candidate': '23ade536906d07ccf09859f1b580b044dafe95dc',
}
ENGINE = Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine')
IMAGES = Path('/home/ray/.cache/saccade-v1/hr-deploy/refs/model-images.f32')
SEQ = Path('/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP')
SDK = Path('/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk')
BUILDS = {name: ROOT / (name + '-build') for name in SOURCES}
records = []
started = time.time()
last = 'preflight'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def clean():
    for name, source in SOURCES.items():
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == REVISIONS[name]
        assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=source) == b'', f'dirty {source}'


def run(name, args, cwd=None, timeout=600):
    global last
    clean()
    last = name
    args = list(map(str, args))
    dump(CAP / (name + '.command.json'), {'argv': args, 'cwd': str(cwd or ROOT)})
    (CAP / (name + '.started')).write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z') + '\n')
    print('START', name, flush=True)
    before = time.monotonic()
    with (CAP / (name + '.log')).open('wb') as log:
        result = subprocess.run(args, cwd=cwd or ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    (CAP / (name + '.exit')).write_text(str(result.returncode) + '\n')
    dump(CAP / (name + '.time.json'), {'wall_seconds': time.monotonic() - before})
    print('DONE', name, 'exit=' + str(result.returncode), flush=True)
    if result.returncode:
        print((CAP / (name + '.log')).read_text()[-5000:], flush=True)
        result.check_returncode()
    clean()


def source_manifest():
    result = {}
    for name, source in SOURCES.items():
        names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=source).split(b'\0')
        result[name] = {os.fsdecode(p): sha(source / os.fsdecode(p)) for p in names if p}
    return result


def input_manifest():
    paths = [ENGINE, IMAGES, SEQ / 'seqinfo.ini']
    paths += [SEQ / 'img1' / f'{i:06}.jpg' for i in range(1, 141)]
    return {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths}


def binaries_manifest():
    paths = []
    for build in BUILDS.values():
        paths += [p for p in build.iterdir() if p.is_file() and (os.access(p, os.X_OK) or p.suffix == '.a')]
    paths += list((ROOT / 'primitive-build').glob('*.o')) + [ROOT / 'primitive-build/resize_benchmark']
    return {str(p): sha(p) for p in sorted(paths)}


telemetry = None
telemetry_file = None
try:
    clean()
    assert sha(ENGINE) == '8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876'
    # Fix scope and sample counts before any timings are observed.
    contract = {
        'schema': 'clean-source-performance-contract/1',
        'sources': {name: {'path': str(source), 'revision': REVISIONS[name], 'dirty': False} for name, source in SOURCES.items()},
        'motion_profile': 'v1-motion/1',
        'engine': str(ENGINE), 'engine_rebuilt': False,
        'engine_sha256': sha(ENGINE), 'model_inputs': str(IMAGES),
        'sequence': str(SEQ), 'outer_order': ['baseline', 'candidate', 'candidate', 'baseline'],
        'outer_rounds': 2, 'model_iterations_per_entry': 200, 'model_warmup_per_entry': 40,
        'model_inner_order': ['ordinary', 'graph', 'graph', 'ordinary'],
        'model_persistent_cache_bytes': 0, 'model_nvtx': 'engine',
        'tracking_measured_frames': 120, 'tracking_warmup_frames': 20,
        'tracking_graph': True, 'tracking_result_slots': 4,
        'tracking_jpegs_preloaded': True, 'tracking_excludes_disk_initialization_warmup_output': True,
        'clock_lock': False, 'outlier_removal': False, 'gpu_lock': '/home/ray/.cache/saccade-v1/gpu.lock',
        'shared_model_harness_revision': REVISIONS['candidate'],
        'baseline_model_binary': 'candidate clean model_benchmark object linked against baseline clean runtime libraries; component provenance explicit',
        'raw_harness_labels': 'development-optimization/formal_acceptance=false are fixed harness labels and will remain byte-identical',
        'verdict_scope': 'bounded clean-source performance measurement, not a new whole-runtime sanitizer or model-quality certificate',
    }
    dump(ROOT / 'contract.json', contract)
    initial_source = source_manifest(); dump(CAP / 'source-before.json', initial_source)
    initial_inputs = input_manifest(); dump(CAP / 'inputs-before.json', initial_inputs)
    run('environment', ['bash', '-c', 'date -Is\nuname -srmo\n/opt/cuda/bin/nvcc --version\n/usr/bin/c++ --version\ncmake --version\nnvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,utilization.gpu,memory.used --format=csv'])
    telemetry_file = (CAP / 'gpu-telemetry.csv').open('wb')
    telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used', '--format=csv', '-l', '1'], stdout=telemetry_file, stderr=subprocess.STDOUT)
    # Both production builds come from unmodified detached worktrees.
    for name, source in SOURCES.items():
        run(name + '-configure', ['cmake', '-S', source, '-B', BUILDS[name], '-DCMAKE_BUILD_TYPE=Release',
            '-DSACCADE_ENABLE_CUDA=ON', '-DSACCADE_ENABLE_PREPROCESS=ON', '-DSACCADE_ENABLE_POSTPROCESS=ON',
            '-DSACCADE_ENABLE_MODEL_PRODUCER=ON', '-DSACCADE_ENABLE_TENSORRT=ON', '-DSACCADE_ENABLE_MODEL_TRT=ON',
            '-DSACCADE_ENABLE_DETECTION_SESSION=ON', '-DSACCADE_TENSORRT_ROOT=' + str(SDK),
            '-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc', '-DCMAKE_CUDA_ARCHITECTURES=120', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
        targets = ['tracking_benchmark', 'selective_scan_cuda_tests', 'resize_cuda_tests']
        if name == 'candidate': targets += ['model_benchmark']
        run(name + '-build', ['cmake', '--build', BUILDS[name], '--parallel', '8', '--target', *targets])
        commands = json.loads((BUILDS[name] / 'compile_commands.json').read_text())
        for filename in ('cuda/resize.cu', 'cuda/selective_scan.cu', 'src/tracking_session.cpp'):
            row, = [row for row in commands if Path(row['file']) == source / filename]
            args = shlex.split(row['command'])
            assert '-DSACCADE_BUILD_DIRTY="false"' in args
            assert '-DSACCADE_BUILD_COMMIT="' + REVISIONS[name] + '"' in args
    # Compile the model driver once. Baseline runtime and kernels remain pure baseline sources.
    args = shlex.split((BUILDS['candidate'] / 'CMakeFiles/model_benchmark.dir/link.txt').read_text())
    args = [a for a in args if not a.startswith('-Wl,--dependency-file=')]
    obj = args.index('CMakeFiles/model_benchmark.dir/tools/model_benchmark.cpp.o')
    args[obj] = str(BUILDS['candidate'] / args[obj])
    args[args.index('-o') + 1] = str(BUILDS['baseline'] / 'model_benchmark')
    run('baseline-link-shared-model-harness', args, cwd=BUILDS['baseline'])
    primitive = ROOT / 'primitive-build'; primitive.mkdir()
    flags = ['-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_120', '-ccbin', '/usr/bin/c++',
             '--ftz=false', '--prec-div=true', '--prec-sqrt=true', '--fmad=true', '-lineinfo']
    symbols = ['resize_input_bytes', 'resize_output_elements', 'resize_workspace_elements',
               'resize_rgb_chw_cuda', 'resize_cuda_load_kernels', 'resize_cuda_identity']
    rename = [f'-D{s}={s}_baseline' for s in symbols]
    run('primitive-compile-baseline', ['/opt/cuda/bin/nvcc', *flags, '-I' + str(SOURCES['baseline'] / 'include'),
        '-DSACCADE_BUILD_COMMIT="' + REVISIONS['baseline'] + '"', '-DSACCADE_BUILD_DIRTY="false"', *rename,
        '-c', SOURCES['baseline'] / 'cuda/resize.cu', '-o', primitive / 'baseline.o'])
    run('primitive-compile-harness', ['/opt/cuda/bin/nvcc', '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_120',
        '-ccbin', '/usr/bin/c++', '-I' + str(SOURCES['candidate'] / 'include'), '-c',
        SOURCES['candidate'] / 'tools/resize_benchmark.cpp', '-o', primitive / 'harness.o'])
    run('primitive-link', ['/opt/cuda/bin/nvcc', '-O3', '-std=c++20', '-arch=sm_120', '-ccbin', '/usr/bin/c++',
        primitive / 'harness.o', primitive / 'baseline.o', BUILDS['candidate'] / 'libsaccade_resize_cuda.a',
        '-o', primitive / 'resize_benchmark'])
    initial_binaries = binaries_manifest(); dump(CAP / 'binaries-before.json', initial_binaries)
    for name, build in BUILDS.items():
        run(name + '-scan-suite', [build / 'selective_scan_cuda_tests'])
        run(name + '-resize-suite', [build / 'resize_cuda_tests'])
    run('primitive-verify', [primitive / 'resize_benchmark', ROOT / 'primitive-verify', '--verify-only'])
    run('primitive-benchmark', [primitive / 'resize_benchmark', ROOT / 'primitive-benchmark', '--benchmark-only',
        '--rounds', '4', '--iterations', '200', '--warmup', '40'])
    for workload in ('model', 'tracking'):
        (ROOT / workload).mkdir()
        for round_id in range(2):
            for position, backend in enumerate(('baseline', 'candidate', 'candidate', 'baseline')):
                dest = ROOT / workload / f'r{round_id}-{position}-{backend}'
                if workload == 'model':
                    args = [BUILDS[backend] / 'model_benchmark', ENGINE, IMAGES, dest, '--profile', 'hr-1280x768',
                            '--iterations', '200', '--warmup', '40', '--rounds', '1']
                else:
                    args = [BUILDS[backend] / 'tracking_benchmark', ENGINE, SEQ, dest, '--profile', 'hr-1280x768',
                            '--frames', '120', '--warmup', '20', '--rounds', '1', '--only', 'graph']
                run(f'{workload}-r{round_id}-{position}-{backend}', args)
                records.append({'workload': workload, 'round': round_id, 'position': position, 'backend': backend, 'directory': str(dest)})
                dump(ROOT / 'records.json', records)
    final_source = source_manifest(); dump(CAP / 'source-after.json', final_source)
    final_inputs = input_manifest(); dump(CAP / 'inputs-after.json', final_inputs)
    final_binaries = binaries_manifest(); dump(CAP / 'binaries-after.json', final_binaries)
    assert initial_source == final_source, 'source drift'
    assert initial_inputs == final_inputs, 'input drift'
    assert initial_binaries == final_binaries, 'binary drift'
    run('analyze', [sys.executable, ROOT / 'methods/analyze.py', ROOT])
    dump(ROOT / 'status.json', {'status': 'PASS', 'source_dirty': False, 'elapsed_seconds': time.time() - started, 'last': last})
    print('CAMPAIGN PASS', ROOT, flush=True)
except BaseException as error:
    dump(ROOT / 'status.json', {'status': 'FAIL', 'last': last, 'error': str(error), 'elapsed_seconds': time.time() - started})
    raise
finally:
    if telemetry:
        telemetry.terminate()
        try: telemetry.wait(timeout=10)
        except subprocess.TimeoutExpired: telemetry.kill(); telemetry.wait()
    if telemetry_file: telemetry_file.close()
