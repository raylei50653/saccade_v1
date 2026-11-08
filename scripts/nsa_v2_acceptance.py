#!/usr/bin/env python3
"""Fixed TRACKING §22 acceptance driver; large artifacts stay in build/."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

RESEARCH = Path('/home/ray/developer/ai/saccade_research')
WORKTREES = Path('/home/ray/developer/ai/saccade_v1-worktrees')
SELECTION = RESEARCH / 'experiments/bridge-pair-fullpool-replay/selection.json'
STOCK = RESEARCH / 'runs/bridge-pair-fullpool-replay/20261108-002'
G2 = RESEARCH / 'runs/nsa-gmc-gate/20261108-001'
ENGINE = Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine')
SDK = Path('/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk')
DATA = Path('/home/ray/developer/ai/dataset')
G1 = WORKTREES / 'gmc-flow/build/gmc-flow/arms/G1'
PP_G1 = WORKTREES / 'gmc-flow/build/gmc-flow2/pp22/arms/G1'
PP_LIST = PP_G1.parent.parent / 'included.txt'
PINNED = {
    SELECTION: '6462c4f2b050c715ff177762cc0207fb3f6b9a0cefe7667301d131ac9e3a42cb',
    STOCK / 'run.json': '1de6b730d2baa9bc337751d4327fb0b8cf72b790ce524d802549ce1f053c13c8',
    G2 / 'run.json': '3ccda77db32725cdf38b188a7a03f99ee8af630a21374a525c317e4872de3e6c',
    # The manifest is outside run.json; pin its bytes as well as its G2 files.
    G2 / 'artifact-manifest.json': 'f1723dca40618fba362018c879a873637513355448de721b2dc2542bc02d55a3',
    PP_LIST: '9f0eaad3aa8b4e3ce4d254d176d991f152a5fa4a1fb7ca3a6e45be01d60f6f60',
    ENGINE: '8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876',
    RESEARCH / 'experiments/nsa-gmc-gate/nsa.patch': '345570229e4e4b76a93f6b500df54f4f8ce8b4e2dcba208a61014af30c87bfef',
}
SPEC_SHA = '1e7b734c6aaf0fef174cd8fe78ec963fe058f03efe86d655d44bf934a32757ad'
MOT17 = [f'MOT17-{s}-SDP' for s in ('02','04','05','09','10','11','13')]


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def source_hashes():
    paths = subprocess.check_output(['git','ls-files','-z','--','CMakeLists.txt','cmake','include',
                                    'src','cuda','tests','tools','fixtures','third_party','scripts',
                                    'training','docs/TRACKING.md']).decode().split('\0')
    return {p: sha(p) for p in sorted(set(paths)) if p}


class Capture:
    def __init__(self, out, dry):
        self.out = out.resolve()
        self.cap = self.out / 'capture'
        self.cap.mkdir(parents=True)
        self.dry = dry
        self.revision = git('rev-parse','HEAD')
        self.source = source_hashes()
        self.step = 0
        self.last = 'preflight'
        self.inputs = {}
        self.steps = {}
        self.started = time.time()
        self.timings = []
        write_json(self.cap/'source.json', {'revision': self.revision, 'files': self.source,
                                           'dry_run': dry, 'initial_status': git('status','--porcelain')})

    def check_source(self):
        require(git('rev-parse','HEAD') == self.revision, 'HEAD changed')
        require(self.dry or not git('status','--porcelain'), 'source is dirty')
        require(source_hashes() == self.source, 'build inputs changed')

    def run(self, name, *command, limit=1800):
        self.check_source()
        self.last = name
        command = [str(x) for x in command]
        write_json(self.cap/f'{name}.command.json', command)
        start = time.time()
        print(f'step {self.step}: {name}: started', flush=True)
        with (self.cap/f'{name}.log').open('w') as log:
            result = subprocess.run(['/usr/bin/time','-v','-o',str(self.cap/f'{name}.time'),
                                     'timeout','--kill-after=30',str(limit),*command], stdout=log, stderr=subprocess.STDOUT)
        write_json(self.cap/f'{name}.exit.json', {'exit': result.returncode,'started_unix':start,'seconds':time.time()-start})
        self.check_source()
        print(f'step {self.step}: {name}: exit={result.returncode}', flush=True)
        require(result.returncode == 0, f'{name} exit={result.returncode}; see {self.cap/name}.log')
        return (self.cap/f'{name}.log').read_text()

    def track_input(self, path, expected=None):
        path = Path(path)
        actual = sha(path)
        require(expected is None or actual == expected, f'input hash mismatch: {path}: {actual} != {expected}')
        self.inputs[str(path)] = actual

    def compare(self, name, actual, expected):
        self.run(name, 'cmp', actual, expected)

    def pass_step(self):
        self.steps[str(self.step)] = 'DRY PASS' if self.dry else 'PASS'
        write_json(self.cap/'steps.json',self.steps)

    def snapshot_binaries(self):
        return {str(p):sha(p) for d in ('cpu','cuda') for p in sorted((self.out/d).glob('*'))
                if p.is_file() and (os.access(p,os.X_OK) or p.suffix == '.a')}

    def preflight(self):
        self.check_source()
        doc = Path('docs/TRACKING.md').read_bytes()
        # Permit only the appended acceptance result, leaving the fixed spec exact.
        fixed = doc.split('\n**結果（NSA v2'.encode(),1)[0]
        require(hashlib.sha256(fixed).hexdigest() == SPEC_SHA, 'frozen TRACKING specification hash mismatch')
        self.track_input('docs/TRACKING.md')
        require(os.statvfs(self.out).f_bavail * os.statvfs(self.out).f_frsize >= 40*1024**3, 'disk <40 GiB')
        for path, expected in PINNED.items():
            self.track_input(path,expected)
        require(not any(k.startswith('SACCADE_CFG_') for k in os.environ), 'config environment overrides present')
        self.run('gpu-idle','nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader')
        require(not (self.cap/'gpu-idle.log').read_text().strip(), 'GPU is busy')
        self.run('environment','bash','-c',
                 'uname -a; df -BG .; nvidia-smi; /opt/cuda/bin/nvcc --version; '
                 '/opt/cuda/bin/compute-sanitizer --version; cmake --version; c++ --version')
        stock = json.loads((STOCK/'run.json').read_text())['sequence_results']
        manifest = json.loads((G2/'artifact-manifest.json').read_text())['artifacts']
        selection = json.loads(SELECTION.read_text())['sequences']
        require(len(selection) == 80, 'selection count !=80')
        pp = PP_LIST.read_text().splitlines()
        require(len(pp) == 69, 'PP22 count !=69')
        for s in selection:
            for key in ('detections','motion'):
                self.track_input(s[key]['path'],s[key]['sha256'])
            s['stock_hash'] = (stock[s['id']]['arms']['clean_stock']['artifacts']['mot.txt']
                               if s['benchmark'] == 'MOT20' else s['expected_baseline_mot_sha256'])
            s['stock_path'] = str(STOCK/'arms/clean_stock'/s['id']/'mot.txt')
            self.track_input(s['stock_path'],s['stock_hash'])
            s['g2_path'] = str(G2/'arms/G2'/s['id']/'mot.txt')
            self.track_input(s['g2_path'],manifest[f"arms/G2/{s['id']}/mot.txt"])
            if s['id'] in MOT17 or s['id'] in pp:
                s['native'] = str(DATA/('MOT17/train' if s['benchmark']=='MOT17' else 'PersonPath22/mot_test_kf')/s['id'])
                s['g1_path'] = str((G1 if s['benchmark']=='MOT17' else PP_G1)/(s['id']+'.txt'))
                self.track_input(s['g1_path'],s['stock_hash'])
                directory = Path(s['native'])
                self.track_input(directory/'seqinfo.ini')
                frames = sorted((directory/'img1').glob('*'))
                require(len(frames) == s['frames'], f"native frame count: {s['id']}")
                for p in frames:
                    self.track_input(p)
        # Record toolchain/SDK, NVOF, T01 test corpus and test engines, too.
        for p in [Path('/opt/cuda/bin/nvcc'),Path('/opt/cuda/bin/compute-sanitizer'),
                  Path('/usr/lib/wsl/lib/libnvidia-opticalflow.so.1'),
                  SDK/'include/NvInferVersion.h', SDK/'lib/libnvinfer.so',
                  Path('/home/ray/developer/ai/saccade_v1/build/t01-f03-evidence/stage/model.engine')]:
            self.track_input(p)
        for d in [WORKTREES/'m1-data-v2', Path('/home/ray/developer/ai/saccade_v1/build/preprocess-evidence/refs')]:
            for p in sorted(d.rglob('*')):
                if p.is_file(): self.track_input(p)
        for p in sorted(Path('third_party').rglob('*')):
            if p.is_file() and 'NvOF' in p.name: self.track_input(p)
        write_json(self.cap/'inputs.json',self.inputs)
        write_json(self.cap/'selection.json',selection)
        self.selection = selection
        self.make_lists(selection)

    def make_lists(self, selection):
        self.enabled = self.out/'enabled.config'
        self.enabled.write_text('schema=saccade-tracking-config/1\ntracker.kalman.nsa_enabled=true\n'
                                'tracker.kalman.nsa_s0=0.9305\ntracker.rows.measurement_box=true\n')
        self.records = self.out/'records.tsv'
        def line(s): return f"{s['id']}\t{s['detections']['path']}\t{s['motion']['path']}\n"
        self.records.write_text(''.join(map(line,selection)))
        self.sanitizer_records = self.out/'sanitizer-records.tsv'
        self.sanitizer_records.write_text(line(next(s for s in selection if s['id']=='MOT17-10-SDP')))
        self.timing_records = self.out/'timing-records.tsv'
        self.timing_records.write_text(line(next(s for s in selection if s['id']=='MOT17-04-SDP')))

    def build(self):
        self.step = 1
        self.run('cpu-configure','cmake','-S','.','-B',self.out/'cpu','-G','Ninja','-DCMAKE_BUILD_TYPE=Release')
        self.run('cpu-build','cmake','--build',self.out/'cpu','--parallel','8')
        self.run('cpu-ctest','ctest','--test-dir',self.out/'cpu','--output-on-failure','--stop-on-failure','--parallel','1')
        receipt = (self.out/'cpu/Testing/Temporary/LastTest.log').read_text()
        hashes = re.search(r'default tracker config sha256 old=([a-f0-9]{64}) new=([a-f0-9]{64})',receipt)
        require(hashes is not None,'default tracker config hash receipt missing')
        write_json(self.cap/'default-config-hashes.json',{'old':hashes[1],'new':hashes[2]})
        engine = '/home/ray/developer/ai/saccade_v1/build/t01-f03-evidence/stage/model.engine'
        flags = ['-DSACCADE_ENABLE_'+x+'=ON' for x in ('CUDA','PREPROCESS','POSTPROCESS','MODEL_PRODUCER',
                 'TENSORRT','MODEL_TRT','DETECTION_SESSION','FLOW')]
        self.run('cuda-configure','cmake','-S','.','-B',self.out/'cuda','-G','Ninja','-DCMAKE_BUILD_TYPE=Release',*flags,
                 f'-DSACCADE_TENSORRT_ROOT={SDK}','-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc',
                 '-DCMAKE_CUDA_ARCHITECTURES=120','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                 f'-DSACCADE_DETECTION_ENGINE={engine}',
                 '-DSACCADE_DETECTION_REFERENCES=/home/ray/developer/ai/saccade_v1/build/preprocess-evidence/refs',
                 f'-DSACCADE_TRACKING_ENGINE={engine}', f'-DSACCADE_TRACKING_SEQUENCES={DATA}/MOT17/train',
                 f'-DSACCADE_TRACKING_RECORDS={WORKTREES}/m1-data-v2')
        self.run('cuda-build','cmake','--build',self.out/'cuda','--parallel','8')
        self.run('cuda-ctest','ctest','--test-dir',self.out/'cuda','--output-on-failure','--stop-on-failure','--parallel','1',limit=7200)
        self.binaries = self.snapshot_binaries()
        write_json(self.cap/'binaries.json',self.binaries)
        self.pass_step()

    def offline(self, arm, selection, short=False):
        out = self.out/arm/'offline'
        out.mkdir(parents=True)
        for s in selection:
            self.run(f"{arm}-offline-{s['id']}",self.out/'cuda/track_offline',s['detections']['path'],
                     s['motion']['path'],out/(s['id']+'.txt'),*(['--config',self.enabled] if arm=='enabled' else []))
            if not short:
                self.compare(f"{arm}-offline-compare-{s['id']}",out/(s['id']+'.txt'),s['g2_path'] if arm=='enabled' else s['stock_path'])

    def native(self, arm, selection, serial=False):
        kind = 'serial' if serial else 'pipelined'
        out = self.out/arm/kind
        out.mkdir(parents=True)
        for s in selection:
            if 'native' not in s or (serial and s['benchmark']!='MOT17'): continue
            self.run(f"{arm}-{kind}-{s['id']}",self.out/'cuda/track_sequence',ENGINE,s['native'],out/(s['id']+'.txt'),
                     '--profile','hr-1280x768',*(['--config',self.enabled] if arm=='enabled' else []),
                     *(['--serial'] if serial else []), '--resolved-config',out/(s['id']+'.config'))
            # Native epoch/source metadata may differ; MOT bytes must be identical.
            expected = self.out/arm/'offline'/(s['id']+'.txt') if arm=='enabled' or self.dry else s['g1_path']
            self.compare(f"{arm}-{kind}-compare-{s['id']}",out/(s['id']+'.txt'),expected)

    def sanitizer(self):
        self.step = 4
        for tool in ('memcheck','racecheck','synccheck','initcheck'):
            args = ['--tool',tool,'--error-exitcode','99']
            if tool == 'initcheck': args += ['--check-api-memory-access','no']
            text = self.run('sanitizer-'+tool,'/opt/cuda/bin/compute-sanitizer',*args,self.out/'cuda/tracking_cuda_tests',
                            '--config',self.enabled,'--record-list',self.sanitizer_records,'--max-frames','60',limit=3600)
            summary = ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck'
                       else '========= ERROR SUMMARY: 0 errors')
            require(text.splitlines().count(summary)==1, f'{tool} missing clean summary')
            require('checks passed' in text and 'MOT17-10-SDP: 60 frames, zero mismatches' in text, f'{tool} suite incomplete')
        self.pass_step()

    def performance(self, selection):
        self.step = 5
        for arm in ('default','enabled'):
            self.run('timing-records-'+arm,self.out/'cuda/tracking_cuda_tests','--records-only',
                     '--record-list',self.timing_records,*(['--config',self.enabled] if arm=='enabled' else []),
                     *(['--max-frames','60'] if self.dry else []))
            for s in selection:
                if s['benchmark']!='MOT17': continue
                for trial in range(1,4):
                    name=f"timing-{arm}-{s['id']}-{trial}"
                    output=self.out/'timing'/f'{name}.txt'; output.parent.mkdir(exist_ok=True)
                    text=self.run(name,self.out/'cuda/track_sequence',ENGINE,s['native'],output,'--profile','hr-1280x768',
                                  *(['--config',self.enabled] if arm=='enabled' else []))
                    match=re.search(r'frames=(\d+).*?ms_per_frame=([0-9.]+)',text)
                    require(match is not None,f'timing output unrecognized: {name}')
                    self.timings.append({'arm':arm,'sequence':s['id'],'trial':trial,'frames':int(match[1]),'ms_per_frame':float(match[2])})
                    self.compare(name+'-compare',output,self.out/arm/'pipelined'/(s['id']+'.txt'))
        write_json(self.cap/'timings.json',self.timings)
        self.pass_step()

    def dry_selection(self):
        selected=[]
        for s in [next(s for s in self.selection if s['id']=='MOT17-10-SDP'),
                  next(s for s in self.selection if s['benchmark']=='PP22')]:
            s=dict(s)
            dest=self.out/'short'/s['id']; (dest/'img1').mkdir(parents=True)
            count=min(60,s['frames'])
            text=Path(s['native'],'seqinfo.ini').read_text()
            (dest/'seqinfo.ini').write_text(re.sub(r'seqLength=\d+',f'seqLength={count}',text))
            for p in sorted(Path(s['native'],'img1').glob('*'))[:count]:
                (dest/'img1'/p.name).symlink_to(p)
            # Short native comparison uses freshly detected records instead of
            # editing wire payloads or changing the production tools.
            self.run('dry-detect-'+s['id'],self.out/'cuda/detect_sequence',ENGINE,dest,dest/'detections.scdr','--profile','hr-1280x768')
            self.run('dry-motion-'+s['id'],self.out/'cuda/track_sequence',ENGINE,dest,dest/'motion-pass.txt',
                     '--profile','hr-1280x768','--motion',dest/'motion.scmo')
            s['detections']={'path':str(dest/'detections.scdr')};s['motion']={'path':str(dest/'motion.scmo')}
            s['native']=str(dest);s['frames']=count
            selected.append(s)
        return selected

    def execute(self):
        self.preflight()
        if self.dry:
            # Development uses existing builds; formal acceptance always builds fresh.
            require((self.out/'cuda').is_dir(),'dry-run requires symlink to development CUDA build')
            self.binaries=self.snapshot_binaries()
            selected=self.dry_selection()
            self.step=1
            self.run('dry-math-and-synthetic',self.out/'cuda/tracking_cuda_tests','--config',self.enabled)
            self.pass_step()
        else:
            self.build();selected=self.selection
        self.step=2
        self.offline('default',selected,short=self.dry);self.native('default',selected)
        self.pass_step()
        self.step=3
        self.offline('enabled',selected,short=self.dry)
        self.run('enabled-recorded-parity',self.out/'cuda/tracking_cuda_tests','--config',self.enabled,
                 '--record-list',self.records,*(['--max-frames','60'] if self.dry else []),limit=7200)
        self.native('enabled',selected);self.native('enabled',selected,serial=True)
        self.pass_step()
        self.sanitizer();self.performance(selected)
        self.check_source()
        require(self.snapshot_binaries()==self.binaries,'binary hashes changed')
        # Re-hash every input, including images and references, before a full PASS.
        for p,h in self.inputs.items(): require(sha(p)==h,f'input changed during capture: {p}')
        write_json(self.cap/'inputs-after.json',{p:sha(p) for p in self.inputs})
        outputs={str(p.relative_to(self.out)):sha(p) for arm in ('default','enabled','timing')
                 for p in sorted((self.out/arm).rglob('*')) if p.is_file()}
        write_json(self.cap/'outputs.json',outputs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    require(args.dry_run or not git('status','--porcelain'),'clean committed source required')
    require(git('branch','--show-current')=='feat/nsa-v2-measurement-box','wrong acceptance branch')
    require(args.output.parent==Path('build') and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.output.name),
            'output must be build/NAME')
    if not args.dry_run: args.output.mkdir(exist_ok=False)
    cap=Capture(args.output,args.dry_run)
    status=1
    def terminate(signum, _frame): raise RuntimeError(f'terminated by signal {signum}')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        cap.execute();status=0
    except Exception as e:
        print(f'FAIL step={cap.step} command={cap.last}: {e}',file=sys.stderr,flush=True)
    finally:
        # A killed compute-sanitizer can leave its child on the GPU.
        subprocess.run(['pkill','-KILL','-f','^'+re.escape(str(cap.out/'cuda'))+'/'],check=False)
        write_json(cap.cap/'status.json',{'exit':status,'step':cap.step,'last_command':cap.last,'steps':cap.steps,
                   'revision':cap.revision,'seconds':time.time()-cap.started,'dry_run':args.dry_run})
        write_json(cap.cap/'binaries-after.json',cap.snapshot_binaries())
        if cap.inputs:
            write_json(cap.cap/'inputs-after.json',{p:sha(p) for p in cap.inputs})
        for d in ('cpu','cuda'):
            p=cap.out/d/'Testing/Temporary/LastTest.log'
            if p.is_file(): (cap.cap/f'{d}-LastTest.log').write_bytes(p.read_bytes())
    return status


if __name__=='__main__': sys.exit(main())
