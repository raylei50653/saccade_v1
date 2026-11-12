"""Replay seven frozen main-detector streams against G1; verify formal output parity."""
from pathlib import Path
import json
import subprocess
import time
from replay_arm import ROOT, RUN, TOOLS, RECORDS, EXE, validate_scdr, sha

FORMAL = Path('/home/ray/developer/ai/saccade_v1-worktrees/g1-ofa/build/v1-motion-2-acceptance-3/hr')
SEQUENCES = [('MOT17-04-SDP', 1050), ('MOT17-05-SDP', 837), ('MOT17-09-SDP', 525),
             ('MOT17-11-SDP', 900), ('MOT17-13-SDP', 750), ('MOT17-10-SDP', 654), ('MOT17-02-SDP', 600)]

def main():
    assert sha(EXE) == '746cf86479ec787ec943eff2efe266d34f272fca181426cc5547f37af0c43c3b'
    dest = RUN / 'main/mot'
    dest.mkdir(parents=True, exist_ok=True)
    receipt = {'arm': 'main', 'scope': 'current CPU fixed-records baseline; seven-sequence formal G1 parity; not new inference',
               'track_offline_sha256': sha(EXE), 'sequences': {}}
    for seq, count in SEQUENCES:
        det, motion = RECORDS / (seq + '.scdr'), RECORDS / (seq + '.scmo')
        baseline_mot, baseline_packets = FORMAL / 'mot' / (seq + '.txt'), FORMAL / (seq + '-pipelined.bin')
        assert sha(motion) == sha(FORMAL / (seq + '.scmo'))
        validation = validate_scdr(det, count)
        mot, config, packets = dest / (seq + '.txt'), dest / (seq + '.config'), dest / (seq + '.packets.bin')
        command = [str(EXE), str(det), str(motion), str(mot), '--resolved-config', str(config), '--packets', str(packets)]
        start = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (dest / (seq + '.replay.log')).write_text(result.stdout)
        assert result.returncode == 0, result.stdout
        packet_command = [str(TOOLS / 'validate_packets'), str(packets), str(count)]
        checked = subprocess.run(packet_command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (dest / (seq + '.packet-validation.log')).write_text(checked.stdout)
        assert checked.returncode == 0, checked.stdout
        rec = {'input_validation': validation, 'detection_path': str(det), 'detection_sha256': sha(det),
               'motion_path': str(motion), 'motion_sha256': sha(motion), 'command': command,
               'elapsed_s': time.monotonic() - start, 'exit': result.returncode,
               'mot_sha256': sha(mot), 'formal_mot_path': str(baseline_mot), 'formal_mot_sha256': sha(baseline_mot),
               'mot_equal': mot.read_bytes() == baseline_mot.read_bytes(), 'packets_sha256': sha(packets),
               'formal_packets_path': str(baseline_packets), 'formal_packets_sha256': sha(baseline_packets),
               'packets_equal': packets.read_bytes() == baseline_packets.read_bytes(),
               'resolved_config_sha256': sha(config), 'packet_validation_command': packet_command,
               'packet_validation_exit': checked.returncode, 'packet_validation': checked.stdout.strip()}
        assert rec['mot_equal'] and rec['packets_equal']
        receipt['sequences'][seq] = rec
        (RUN / 'main/replay-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(seq, 'MOT_EQUAL PACKETS_EQUAL', checked.stdout.strip(), flush=True)
    assert sha(EXE) == receipt['track_offline_sha256']
    assert sum(r['input_validation']['frames'] for r in receipt['sequences'].values()) == 5316

if __name__ == '__main__':
    main()
