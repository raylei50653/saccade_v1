"""Freeze this completed performance campaign without rebuilding its binaries."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/home/ray/developer/ai/saccade_v1')
DEST = REPO / 'evidence/f02-f03/2026-11-05-clean-performance'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


assert read(ROOT / 'status.json')['status'] == 'PASS'
a = read(ROOT / 'analysis.json')
assert a['status'] == 'PASS'
contract = read(ROOT / 'contract.json')
for prefix in ('source', 'inputs', 'binaries'):
    assert read(ROOT / f'capture/{prefix}-before.json') == read(ROOT / f'capture/{prefix}-after.json')
    if prefix == 'source':
        for backend, items in read(ROOT / 'capture/source-before.json').items():
            source = Path(contract['sources'][backend]['path'])
            assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == contract['sources'][backend]['revision']
            assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=source) == b''
            for name, expected in items.items(): assert digest(source / name) == expected
    else:
        for name, expected in read(ROOT / f'capture/{prefix}-before.json').items():
            assert digest(Path(name)) == (expected['sha256'] if isinstance(expected, dict) else expected)
for path in (ROOT / 'capture').glob('*.exit'):
    assert path.read_text().strip() == '0', path
verify = read(ROOT / 'primitive-verify/summary.json')
assert verify == {'verified_cases': 128, 'verified_output_elements': 65864364, 'bitwise_equivalence': True}
primitive = read(ROOT / 'primitive-benchmark/summary.json')
assert not DEST.exists()

archive = ROOT / 'accepted-binaries.tar.gz'
with tarfile.open(archive, 'x:gz') as tar:
    for name in read(ROOT / 'capture/binaries-before.json'):
        tar.add(name, arcname=str(Path(name).relative_to(ROOT)), recursive=False)
with tarfile.open(archive) as tar:
    for name, expected in read(ROOT / 'capture/binaries-before.json').items():
        with tar.extractfile(str(Path(name).relative_to(ROOT))) as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected

DEST.mkdir()
copies = []


def copy(source, relative=None):
    relative = relative or source.relative_to(ROOT)
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    expected = digest(source)
    assert digest(target) == expected
    copies.append({'source': str(source), 'archive': str(relative), 'sha256': expected})


for name in ('contract.json', 'analysis.json', 'records.json', 'status.json', 'model-samples.csv', 'tracking-runs.csv', 'tracking-latencies.csv'):
    copy(ROOT / name)
for source in (ROOT / 'methods').glob('*.py'): copy(source)
for source in (ROOT / 'capture').iterdir():
    if source.name not in ('source-before.json', 'source-after.json') and source.is_file(): copy(source)
for group in ('primitive-verify', 'primitive-benchmark', 'model', 'tracking'):
    for source in sorted((ROOT / group).rglob('*')):
        if source.is_file() and source.suffix in ('.csv', '.json', '.txt'):
            copy(source)
for backend in ('baseline', 'candidate'):
    for name in ('CMakeCache.txt', 'compile_commands.json'):
        copy(ROOT / f'{backend}-build/{name}', Path('build-metadata') / backend / name)
    copy(ROOT / f'{backend}-build/CMakeFiles/tracking_benchmark.dir/link.txt', Path('build-metadata') / backend / 'tracking-link.txt')
    source = Path(contract['sources'][backend]['path'])
    for name in ('cuda/resize.cu', 'cuda/selective_scan.cu', 'include/saccade/resize_cuda.hpp', 'include/saccade/selective_scan_cuda.hpp',
                 'src/tracking_session.cpp', 'src/scan_trt_plugin.cpp', 'tools/tracking_benchmark.cpp', 'cmake/Preprocess.cmake', 'CMakeLists.txt'):
        copy(source / name, Path('source') / backend / name)
for name in ('tools/model_benchmark.cpp', 'tools/resize_benchmark.cpp'):
    copy(Path(contract['sources']['candidate']['path']) / name, Path('source/shared-harness') / name)

payloads = {}
for record in read(ROOT / 'records.json'):
    directory = Path(record['directory'])
    for path in directory.iterdir():
        if path.suffix in ('.bin', '.f32', '.status'):
            payloads[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
local = {
    'campaign_root': str(ROOT),
    'binary_archive': {'path': str(archive), 'sha256': digest(archive), 'bytes': archive.stat().st_size},
    'source_manifests': {name: {'path': str(ROOT / 'capture' / name), 'sha256': digest(ROOT / 'capture' / name)}
                         for name in ('source-before.json', 'source-after.json')},
    'source_file_counts': {backend: len(items) for backend, items in read(ROOT / 'capture/source-before.json').items()},
    'payloads': payloads,
    'project_binaries_and_libraries': read(ROOT / 'capture/binaries-before.json'),
}
write(DEST / 'local-artifacts.json', local)
write(DEST / 'archive-provenance.json', {'files': copies, 'method_sha256': digest(Path(__file__))})

model = a['model_ms']['graph']; single = a['tracking_us']['single']; queued = a['tracking_us']['queued']


def row(label, old, new, unit):
    return f'| {label} | {old:.6f} {unit} | {new:.6f} {unit} | {100*(1-new/old):.2f}% |'


table = '\n'.join([
    '| 指標 | baseline | candidate | 時間降低 |', '| --- | ---: | ---: | ---: |',
    row('隔離 F02 ordinary interval', primitive['ordinary_baseline_event_us_per_operation'], primitive['ordinary_candidate_event_us_per_operation'], 'µs'),
    row('隔離 F02 Graph interval', primitive['graph_baseline_event_us_per_operation'], primitive['graph_candidate_event_us_per_operation'], 'µs'),
    row('完整 F03 ordinary', a['model_ms']['ordinary']['baseline']['mean'], a['model_ms']['ordinary']['candidate']['mean'], 'ms'),
    row('完整 F03 Graph', model['baseline']['mean'], model['candidate']['mean'], 'ms'),
    row('tracking single mean', single['baseline']['mean']/1000, single['candidate']['mean']/1000, 'ms'),
    row('tracking single p50', single['baseline']['p50']/1000, single['candidate']['p50']/1000, 'ms'),
    row('tracking single p95', single['baseline']['p95']/1000, single['candidate']['p95']/1000, 'ms'),
    row('tracking single p99', single['baseline']['p99']/1000, single['candidate']['p99']/1000, 'ms'),
    row('queued wall interval', queued['baseline']['mean']/1000, queued['candidate']['mean']/1000, 'ms/frame'),
])
rounds = '\n'.join(f'| {r["workload"]}／{r["mode"]}／第 {r["round"]+1} 輪 | {r["mean_reduction_percent"]:.3f}% |'
                   for r in a['per_round'])
report = f'''# F02／F03 clean-source ABBA 效能量測

2026-11-05。**量測與等價守門 PASS，`source_dirty=false`。**
固定加速前 `f4865cdb27f0e624b8cc26219e78f4aa938690df` 與加速後
`23ade536906d07ccf09859f1b580b044dafe95dc`，使用原 `v1-motion/1`（cuFFT motion）。
這是 clean-source 效能補驗；原 [整合正確性證書](../2026-11-05-clean-23ade53/REPORT.md) 與開發量測保持原 bytes／來源身分。

## 結果

{table}

Queued throughput 為 **{queued['baseline']['fps']:.3f} → {queued['candidate']['fps']:.3f} FPS**，
吞吐增加 **{100*(queued['throughput_speedup']-1):.2f}%**；wall interval 不是單幀交付延遲。
各 backend 的八次 queued interval 範圍為
{queued['baseline']['min']/1000:.6f}–{queued['baseline']['max']/1000:.6f} 與
{queued['candidate']['min']/1000:.6f}–{queued['candidate']['max']/1000:.6f} ms/frame。

兩輪外層 ABBA 分開計算如下；所有長尾保留，沒有刪除 outliers。

| workload／mode／round | 平均時間降低 |
| --- | ---: |
{rounds}

完整 samples、每個程序與每輪統計、最大的 20 筆單幀 latency 見 [analysis.json](analysis.json)、
[model-samples.csv](model-samples.csv)、[tracking-runs.csv](tracking-runs.csv) 與 [tracking-latencies.csv](tracking-latencies.csv)。

## 固定條件與量測範圍

- GPU：RTX 5070 Ti Laptop，UUID `GPU-825ae551-ba16-539a-e444-22726b59ee9a`；WSL2、driver 617.42、CUDA 13.4.92、TensorRT 10.16.1.11。未鎖 clocks；全程持有 GPU lock，保留 [telemetry](capture/gpu-telemetry.csv)。
- Engine 未重建，固定 SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`；HR 1280×768 FP16，input／raw tensors 與 scan plugins 為 FP32。
- F02 是 deterministic xorshift RGB 的 1920×1080→1280×768 隔離 interval；四輪 ABBA，每 entry 暖機 40 次、batch 200 次，每 backend／mode 有八個 batch means。F02 Graph 只屬 primitive，TrackingSession 的 F02 仍在 Graph 外。
- 完整 F03 含 backbone／neck、head、scan plugins；固定 input 0 計時，排除 H2D／D2H、初始化、暖機、外部驗證及 profiler。兩輪外層 A→B→B→A；每程序內 ordinary→Graph→Graph→ordinary，每 entry 暖機 40、測 200，各 backend／mode 共 1,600 samples；cache=0，NVTX requested=engine／actual=detailed（兩版相同），沒有 layer profiler。
- Tracking 使用 MOT17-10 前 140 幀：20 幀暖機、120 幀量測，Graph 啟用、result_slots=4；兩輪 ABBA 共 32 fresh sessions。每 backend 有 960 個 single latency samples、八個 queued intervals。JPEG 預載，排除磁碟、constructor、暖機與輸出寫檔；首幀／初始化原值另存，不與 single latency 混算。這不是全七序列含 I/O 的 FPS。

## 正確性與來源保管

- F02 共 128 cases／65,864,364 元素，ordinary／repeat／Graph 全 bits 相同，guards／poison 通過；benchmark-only 的 equivalence=null 保持原樣，等價由獨立 [verify](primitive-verify/summary.json) 決定。
- 兩份 clean build 的 scan／resize suites 通過。四張完整模型的六 raw outputs＋九 statuses，在八個程序間直接逐 byte 比對相同；各程序內 ordinary／repeat／Graph 亦一致。
- 32 份 140 幀完整 packets（每份 1,731,520 B，包含暖機／unused bytes）直接逐 byte 比對相同，SHA-256 `{a['tracking_packet_comparison']['sha256']}`；非僅比較摘要 hash。
- 兩個 detached checkout 的 HEAD／clean status 在每步前後檢查，完整 tracked source manifests、engine／inputs 與接受 binaries 前後一致。來源、編譯與連結命令、退出碼均保留；大型 payload 與已核驗 binary tar 的位置／hash 見 [local-artifacts.json](local-artifacts.json)。
- Baseline 原 commit 沒有 model benchmark target：共用 driver object 由乾淨 `23ade53` 的 `tools/model_benchmark.cpp` 編譯，baseline model binary 只連乾淨 `f4865cd` 的 runtime libraries。這是明列 component provenance 的共用 harness 對照，不宣稱 baseline executable 全部出自單一 commit。Tracking binaries 各由其完整 clean commit 建置。
- 原 model harness 固定寫 `capture_kind=development-optimization`／`formal_acceptance=false`；raw outputs 與 labels 全保留。此字串不代表 dirty build，實際 compile／runtime identities 均核對 `dirty=false`；外層 contract 與本報告定義本輪 clean-source 效能補驗的範圍。

## 保留界線與重跑

本輪沒有讀 GT、重新計分、重建 engine 或重跑 sanitizer／semantic mutations。
效能限本機、固定 engine／input 與此 prefix，不延伸到任意 workload／跨 GPU／SDK。
現行 G1／`v1-motion/2` 完整 session memcheck 仍 OPEN，完整 HR session racecheck 與原 nvJPEG initcheck 診斷亦保留。
Tracking 結果同時涵蓋 F02＋F03 兩項改動，不能把合併收益分別歸給其中一項，也不能相加 primitive 百分比。

方法：[contract.json](contract.json)、[run.py](methods/run.py)、[analyze.py](methods/analyze.py)、[archive.py](methods/archive.py)。
重跑須把 methods 放入新的獨立 campaign 目錄、確認 source 路徑與乾淨 commit，並在
`flock ~/.cache/saccade-v1/gpu.lock` 下執行；所有輸出目錄必須全新，不能覆用本次 capture。
`run.py` 的 GPU campaign 與 `archive.py` 的目的地是本次固定路徑，重跑前須改成新的路徑；不可直接在 evidence 目錄執行。
Archive helper 不執行 GPU／build，僅在 campaign 與 hashes 通過後封存。所有封存檔由 `SHA256SUMS` 核驗。
'''
(DEST / 'REPORT.md').write_text(report)
with (DEST / 'SHA256SUMS').open('w') as stream:
    for path in sorted(DEST.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS':
            stream.write(f'{digest(path)}  {path.relative_to(DEST)}\n')
print('ARCHIVED', DEST)
