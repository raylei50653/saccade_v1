#!/usr/bin/env python3
"""Bounded report measurement; never changes runtime sources or historical evidence."""
import csv
import difflib
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
WORK = ROOT / 'build/report1-live-20261111'
SOURCE = WORK / 'source'
BUILD = WORK / 'cuda'
BASE = '3cbaa518f0dc37d691404e95ed85c0109c81130d'
ENGINE = Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine')
SEQ = Path('/home/ray/developer/ai/dataset/MOT17/train/MOT17-10-SDP')
SDK = Path('/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk')
EXPECTED_ENGINE = '8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876'

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def manifest(paths):
    result = {}
    for path in paths:
        if path.is_symlink():
            result[str(path)] = {'symlink': os.readlink(path)}
        else:
            result[str(path)] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return result

def run(name, argv, cwd=ROOT, timeout=1200):
    argv = list(map(str, argv))
    dump(EVIDENCE / (name + '.command.json'), {'argv': argv, 'cwd': str(cwd)})
    start = time.monotonic()
    print('START', name, flush=True)
    with (EVIDENCE / (name + '.log')).open('wb') as output:
        proc = subprocess.run(argv, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    dump(EVIDENCE / (name + '.result.json'), {'returncode': proc.returncode, 'wall_seconds': time.monotonic() - start})
    print('DONE', name, proc.returncode, flush=True)
    if proc.returncode:
        print((EVIDENCE / (name + '.log')).read_text()[-5000:], flush=True)
        proc.check_returncode()

def source_manifest():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=SOURCE).split(b'\0')
    return manifest([SOURCE / os.fsdecode(name) for name in files if name])

def clean():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() == BASE
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE)

def stats(values):
    values = sorted(values)
    return {'n': len(values), 'mean': statistics.fmean(values), 'median': statistics.median(values),
            'p50': values[math.ceil(len(values) * .50) - 1], 'p95': values[math.ceil(len(values) * .95) - 1],
            'p99': values[math.ceil(len(values) * .99) - 1], 'min': values[0], 'max': values[-1]}

def instrument():
    original = (SOURCE / 'tools/tracking_benchmark.cpp').read_text()
    revised = original.replace('    Clock::time_point first_frame_start;', '''    Clock::time_point first_frame_start;
    std::vector<Clock::time_point> enqueue_times(images.size()), receive_times(images.size());''')
    revised = revised.replace('        const auto packet = session.receive();', '''        const auto packet = session.receive();
        receive_times[received] = Clock::now();''')
    revised = revised.replace('        session.enqueue({reinterpret_cast<const std::uint8_t*>(image.data()), image.size()}, frame);', '''        enqueue_times[frame] = Clock::now();
        session.enqueue({reinterpret_cast<const std::uint8_t*>(image.data()), image.size()}, frame);''')
    revised = revised.replace('    std::ostringstream samples;', '''    std::ostringstream delivery;
    delivery << "frame,phase,enqueue_us_since_constructor_start,receive_us_since_constructor_start,enqueue_to_receive_us\\n" << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < images.size(); ++i)
        delivery << i << ',' << (i < warmup ? "warmup" : "measured") << ','
                 << microseconds(enqueue_times[i] - before) << ',' << microseconds(receive_times[i] - before) << ','
                 << microseconds(receive_times[i] - enqueue_times[i]) << '\\n';
    saccade::write_text_file(out / (id + ".delivery.csv"), delivery.str());
    std::ostringstream samples;''')
    assert revised != original
    methods = EVIDENCE / 'methods'
    methods.mkdir(exist_ok=True)
    path = methods / 'tracking_benchmark_delivery.cpp'
    path.write_text(revised)
    (methods / 'benchmark-instrumentation.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), revised.splitlines(True), fromfile='tools/tracking_benchmark.cpp@' + BASE,
        tofile='methods/tracking_benchmark_delivery.cpp')))
    commands = json.loads((BUILD / 'compile_commands.json').read_text())
    row, = [r for r in commands if Path(r['file']) == SOURCE / 'tools/tracking_benchmark.cpp']
    args = shlex.split(row['command'])
    args[args.index('-o') + 1] = str(WORK / 'delivery.o')
    args[args.index('-c') + 1] = str(path)
    run('compile-instrumented-harness', args, BUILD)
    args = shlex.split((BUILD / 'CMakeFiles/tracking_benchmark.dir/link.txt').read_text())
    args = [arg for arg in args if not arg.startswith('-Wl,--dependency-file=')]
    args[args.index('CMakeFiles/tracking_benchmark.dir/tools/tracking_benchmark.cpp.o')] = str(WORK / 'delivery.o')
    args[args.index('-o') + 1] = str(WORK / 'tracking_benchmark_delivery')
    run('link-instrumented-harness', args, BUILD)
    return WORK / 'tracking_benchmark_delivery'

def main():
    WORK.mkdir(exist_ok=False)
    contract = {'schema': 'report1-current-g1-latency-contract/1', 'base_commit': BASE,
        'runtime_clean_source': True, 'instrumented_harness': 'adds host enqueue/receive timestamps only; runtime unchanged',
        'engine_sha256': EXPECTED_ENGINE, 'engine_rebuilt': False, 'sequence': str(SEQ),
        'profile': 'hr-1280x768', 'motion': 'v1-motion/2 G1 NVOFA', 'graph': True,
        'nsa_enabled': False, 'measurement_box': False, 'result_slots': 4, 'warmup_frames': 20,
        'measured_frames_per_session': 120, 'prefix_frames': 140, 'fresh_sessions_per_style': 4,
        'order': ['single', 'queued'] * 4, 'input_io': 'JPEGs preloaded; disk I/O excluded',
        'excluded': ['constructor', 'warmup', 'final file output'], 'outlier_removal': False,
        'quantile': 'nearest-rank; median averages middle pair', 'clock_lock': False,
        'gpu_lock': '/home/ray/.cache/saccade-v1/gpu.lock', 'telemetry': 'nvidia-smi 1 second during measured process',
        'verdict_scope': 'bounded current-source latency and output parity; no model quality or sanitizer acceptance'}
    dump(EVIDENCE / 'contract.json', contract)
    run('create-clean-worktree', ['git', 'worktree', 'add', '--detach', SOURCE, BASE])
    clean()
    before_source = source_manifest()
    dump(EVIDENCE / 'source-before.json', before_source)
    inputs = [ENGINE, SEQ / 'seqinfo.ini'] + [SEQ / 'img1' / f'{i:06}.jpg' for i in range(1, 141)]
    before_inputs = manifest(inputs)
    dump(EVIDENCE / 'inputs-before.json', before_inputs)
    assert sha(ENGINE) == EXPECTED_ENGINE
    run('environment', ['bash', '-c', 'date -Is\nuname -srmo\n/opt/cuda/bin/nvcc --version\n/usr/bin/c++ --version\ncmake --version\nnvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,utilization.gpu,memory.used --format=csv'])
    run('configure', ['cmake', '-S', SOURCE, '-B', BUILD, '-DCMAKE_BUILD_TYPE=Release',
        '-DSACCADE_ENABLE_CUDA=ON', '-DSACCADE_ENABLE_PREPROCESS=ON', '-DSACCADE_ENABLE_POSTPROCESS=ON',
        '-DSACCADE_ENABLE_MODEL_PRODUCER=ON', '-DSACCADE_ENABLE_TENSORRT=ON', '-DSACCADE_ENABLE_MODEL_TRT=ON',
        '-DSACCADE_ENABLE_DETECTION_SESSION=ON', '-DSACCADE_ENABLE_FLOW=ON',
        '-DSACCADE_TENSORRT_ROOT=' + str(SDK), '-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc',
        '-DCMAKE_CUDA_ARCHITECTURES=120', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
    run('build', ['cmake', '--build', BUILD, '--parallel', '8', '--target', 'tracking_benchmark'])
    clean()
    binary = instrument()
    binary_inputs = [p for p in BUILD.glob('*.a')] + [BUILD / 'tracking_benchmark', binary, WORK / 'delivery.o']
    before_binary = manifest(binary_inputs)
    dump(EVIDENCE / 'binaries-before.json', before_binary)
    with (EVIDENCE / 'gpu-telemetry.csv').open('wb') as telemetry_output:
        telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used', '--format=csv', '-l', '1'], stdout=telemetry_output, stderr=subprocess.STDOUT)
        try:
            run('measure', [binary, ENGINE, SEQ, WORK / 'runs', '--profile', 'hr-1280x768',
                '--frames', '120', '--warmup', '20', '--rounds', '2', '--only', 'graph'], timeout=300)
        finally:
            telemetry.terminate()
            telemetry.wait(timeout=10)
    clean()
    after_source, after_inputs, after_binary = source_manifest(), manifest(inputs), manifest(binary_inputs)
    dump(EVIDENCE / 'source-after.json', after_source)
    dump(EVIDENCE / 'inputs-after.json', after_inputs)
    dump(EVIDENCE / 'binaries-after.json', after_binary)
    assert before_source == after_source and before_inputs == after_inputs and before_binary == after_binary
    raw = EVIDENCE / 'raw'
    raw.mkdir()
    for path in (WORK / 'runs').iterdir():
        if path.suffix == '.bin':
            with path.open('rb') as src, gzip.GzipFile(filename=str(raw / (path.name + '.gz')), mode='wb', mtime=0) as dest:
                dest.write(src.read())
        else:
            (raw / path.name).write_bytes(path.read_bytes())
    runs = list(csv.DictReader((WORK / 'runs/runs.csv').open()))
    packets = list((WORK / 'runs').glob('*.bin'))
    expected = packets[0].read_bytes()
    assert all(p.read_bytes() == expected for p in packets)
    delivery = {}
    for style in ('single', 'queued'):
        values = []
        for path in (WORK / 'runs').glob('*-' + style + '.delivery.csv'):
            values.extend(float(r['enqueue_to_receive_us']) / 1000 for r in csv.DictReader(path.open()) if r['phase'] == 'measured')
        delivery[style] = stats(values)
    queued_intervals = [float(r['wall_us_per_frame']) / 1000 for r in runs if r['style'] == 'queued']
    summary = {'schema': 'report1-current-g1-latency-summary/1', 'status': 'PASS_BOUNDED_MEASUREMENT_AND_PACKET_PARITY',
        'contract': contract, 'driver_version': '621.14', 'source_inputs_binaries_unchanged': True,
        'sessions': len(runs), 'measured_samples_per_style': 480, 'single_latency_ms': delivery['single'],
        'queued_delivery_latency_ms': delivery['queued'], 'queued_wall_interval_ms_per_frame': stats(queued_intervals),
        'queued_throughput_fps': 1000 / statistics.fmean(queued_intervals),
        'constructor_ms': stats([float(r['init_us']) / 1000 for r in runs]),
        'first_frame_ms': stats([float(r['first_frame_us']) / 1000 for r in runs]),
        'packet_comparison': 'all 8 full 140-frame payloads compared directly, including warmup and unused bytes',
        'packet_sha256': hashlib.sha256(expected).hexdigest(), 'packet_bytes_each': len(expected),
        'retained_full_packets': str(WORK / 'runs'), 'limitations': ['one sequence, 140-frame prefix',
            '4 sessions per style', 'new driver differs from historical 617.42',
            'host timestamp instrumentation adds small overhead', 'preloaded input, no camera/network/final output',
            'no clock lock', 'single always precedes queued; order not counterbalanced',
            'no quality rescoring, sanitizer, or external real-time end-to-end measurement']}
    dump(EVIDENCE / 'summary.json', summary)
    checksums = {str(p.relative_to(EVIDENCE)): sha(p) for p in EVIDENCE.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
    (EVIDENCE / 'SHA256SUMS').write_text(''.join(value + '  ' + path + '\n' for path, value in sorted(checksums.items())))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with open('/home/ray/.cache/saccade-v1/gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
