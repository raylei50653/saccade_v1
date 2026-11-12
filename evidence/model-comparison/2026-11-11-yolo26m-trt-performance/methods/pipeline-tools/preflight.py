"""Run only under the shared GPU flock; correctness gates, not timing results."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = RUN.parent.parent
OUT = HERE / 'preflight'
RECORDS = REPO.parent / 'saccade_v1-worktrees/hr-records-v2'
FORMAL = REPO.parent / 'saccade_v1-worktrees/g1-ofa/build/v1-motion-2-acceptance-3/hr'
DATA = REPO.parent / 'dataset/MOT17/train/MOT17-10-SDP'
MAIN = Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir()
    receipt = {'scope': 'targeted producer and native-adapter correctness; not performance acceptance', 'steps': []}
    def execute(name, command):
        command = [str(item) for item in command]
        result = subprocess.run(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (OUT/(name+'.log')).write_text(result.stdout)
        receipt['steps'].append({'name': name, 'command': command, 'exit': result.returncode})
        (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        assert result.returncode == 0, (name, result.stdout[-4000:])
        print(name, 'PASS', flush=True)
    binaries = [HERE/name for name in ['producer_probe', 'detect_sequence', 'track_sequence']]
    receipt['binary_hashes_before'] = {str(p):sha(p) for p in binaries}
    execute('producer-negative-controls', [HERE/'producer_probe'])
    execute('main-detection', [HERE/'detect_sequence',MAIN,DATA,OUT/'main.scdr','--profile','hr-1280x768'])
    assert (OUT/'main.scdr').read_bytes() == (RECORDS/'MOT17-10-SDP.scdr').read_bytes()
    execute('main-tracking', [HERE/'track_sequence',MAIN,DATA,OUT/'main.txt','--profile','hr-1280x768',
            '--motion',OUT/'main.scmo','--packets',OUT/'main.packets.bin','--resolved-config',OUT/'main.config',
            '--set','execution.candidate_slots=4','--set','execution.motion_slots=4','--set','execution.result_slots=4'])
    assert (OUT/'main.txt').read_bytes() == (FORMAL/'mot/MOT17-10-SDP.txt').read_bytes()
    assert (OUT/'main.packets.bin').read_bytes() == (FORMAL/'MOT17-10-SDP-pipelined.bin').read_bytes()
    assert (OUT/'main.scmo').read_bytes() == (RECORDS/'MOT17-10-SDP.scmo').read_bytes()
    execute('official-detection', [HERE/'detect_sequence',RUN/'artifacts/official/model.engine',DATA,
                                 OUT/'official.scdr','--profile','hr-1280x768'])
    receipt['main_SCDR_equal_frozen'] = True
    receipt['main_MOT_packets_motion_equal_formal_G1'] = True
    receipt['official_detection_frames'] = 654
    receipt['binary_hashes_after'] = {str(p):sha(p) for p in binaries}
    assert receipt['binary_hashes_before'] == receipt['binary_hashes_after']
    receipt['result'] = 'PASS'
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':
    main()
