"""Independent read-only audit of all7 score extraction and CLEAR group MOTA."""
from pathlib import Path
import csv
import hashlib
import json

REPO = Path('/home/ray/developer/ai/saccade_v1')
RUN = REPO / 'build/yolo26m-threeway-all7-20261111'
EVIDENCE = REPO / 'evidence/model-comparison/2026-11-11-yolo26m-tracking-all7'
ARMS = ['official', 'finetuned', 'main']
SEQUENCES = {'MOT17-02-SDP': 600, 'MOT17-04-SDP': 1050, 'MOT17-05-SDP': 837,
             'MOT17-09-SDP': 525, 'MOT17-10-SDP': 654, 'MOT17-11-SDP': 900, 'MOT17-13-SDP': 750}
GROUPS = {'training': ['MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-11-SDP', 'MOT17-13-SDP'],
          'validation': ['MOT17-10-SDP'], 'post_M4_eval': ['MOT17-02-SDP']}
MAPPING = {**{key: key+'___AUC' for key in ['HOTA', 'DetA', 'AssA']},
           'IDF1': 'IDF1', 'MOTA': 'MOTA', 'IDSW': 'IDSW', 'Frag': 'Frag',
           'CLR_FP': 'CLR_FP', 'CLR_FN': 'CLR_FN', 'GT_Dets': 'GT_Dets', 'GT_IDs': 'GT_IDs', 'Frames': 'CLR_Frames'}
PERCENT = {'HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA'}
COUNTS = ['IDSW', 'Frag', 'CLR_FP', 'CLR_FN', 'GT_Dets', 'GT_IDs', 'Frames']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    return list(csv.DictReader(path.open()))

def check_close(left, right, tolerance=2e-12):
    assert abs(left-right) <= tolerance, (left, right)

def main():
    source = EVIDENCE / 'scores.csv'
    rows = read_csv(source)
    assert len(rows) == 24
    lookup = {(row['arm'], row['sequence']): row for row in rows}
    assert len(lookup) == 24
    row_checks = []
    details = {}
    for arm in ARMS:
        path = RUN / arm / 'mot/trackeval/results/v1/pedestrian_detailed.csv'
        detailed = {row['seq']: row for row in read_csv(path)}
        details[arm] = detailed
        assert set(detailed) == {*SEQUENCES, 'COMBINED'}
        for sequence, native in detailed.items():
            row = lookup[arm, sequence]
            for outkey, inkey in MAPPING.items():
                expected = float(native[inkey]) * (100 if outkey in PERCENT else 1)
                check_close(float(row[outkey]), expected, 2e-12 if outkey in PERCENT else 0)
            mota = 100 * (1-(int(row['CLR_FP']) + int(row['CLR_FN']) + int(row['IDSW'])) / int(row['GT_Dets']))
            check_close(mota, float(row['MOTA']))
            assert int(row['Frames']) == SEQUENCES.get(sequence, 5316)
            if sequence != 'COMBINED':
                ap50 = json.loads((RUN / arm / (sequence+'.ap50.json')).read_text())['ap50_percent']
                assert float(row['AP50']) == ap50
            row_checks.append({'arm': arm, 'sequence': sequence, 'mapped_fields': len(MAPPING),
                               'frames': int(row['Frames']), 'mota_formula_percent': mota,
                               'mota_recorded_percent': float(row['MOTA']), 'all_checks_pass': True})
        for key in COUNTS:
            assert int(lookup[arm, 'COMBINED'][key]) == sum(int(lookup[arm, seq][key]) for seq in SEQUENCES)
    for seq in [*SEQUENCES, 'COMBINED']:
        assert len({tuple(lookup[arm, seq][key] for key in ['GT_Dets', 'GT_IDs', 'Frames']) for arm in ARMS}) == 1

    prior = json.loads((EVIDENCE.parent / '2026-11-11-yolo26m-tracking/scores.json').read_text())
    scores = json.loads((EVIDENCE / 'scores.json').read_text())
    for arm in ARMS:
        for seq in ['MOT17-02-SDP', 'MOT17-10-SDP']:
            assert scores[arm][seq] == prior[arm][seq]

    frozen_base = REPO / 'evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2/capture/scoring'
    frozen_checks = []
    for seq in SEQUENCES:
        folder = 'val' if seq == 'MOT17-10-SDP' else ('MOT17-02' if seq == 'MOT17-02-SDP' else 'in-domain')
        path = frozen_base / folder / 'pedestrian_detailed.csv'
        frozen = {row['seq']: row for row in read_csv(path)}[seq]
        current = details['main'][seq]
        assert set(current) == set(frozen)
        for key in frozen:
            if key != 'seq':
                assert float(current[key]) == float(frozen[key]), (seq, key)
        frozen_checks.append({'sequence': seq, 'reference': str(path), 'sha256': sha(path),
                              'numeric_fields_exact_equal': len(frozen)-1})

    groups_path = EVIDENCE / 'group-MOTA.json'
    groups = json.loads(groups_path.read_text())
    group_defs = {**GROUPS, '02+10': ['MOT17-02-SDP', 'MOT17-10-SDP'], 'all7': list(SEQUENCES)}
    assert set(groups) == set(group_defs)
    group_checks = []
    for name, seqs in group_defs.items():
        for arm in ARMS:
            row = groups[name][arm]
            sums = {key: sum(int(lookup[arm, seq][key]) for seq in seqs)
                    for key in ['Frames', 'GT_Dets', 'CLR_FP', 'CLR_FN', 'IDSW']}
            for key in sums:
                assert row[key] == sums[key], (name, arm, key)
            mota = 100 * (1-(sums['CLR_FP']+sums['CLR_FN']+sums['IDSW'])/sums['GT_Dets'])
            check_close(row['MOTA'], mota)
            if name == 'all7':
                check_close(mota, float(lookup[arm, 'COMBINED']['MOTA']))
            group_checks.append({'group': name, 'arm': arm, 'frames': sums['Frames'],
                                 'GT_Dets': sums['GT_Dets'], 'MOTA': mota, 'all_checks_pass': True})
    plan = json.loads((EVIDENCE / 'PLAN.json').read_text())
    assert plan['groups'] == GROUPS
    assert sum(SEQUENCES.values()) == 5316
    review = {'result': 'PASS', 'scope': 'independent read-only audit; no rescoring',
              'scores_csv_sha256': sha(source), 'group_MOTA_json_sha256': sha(groups_path),
              'rows_verified': 24, 'row_checks': row_checks, 'combined_counts_additive': True,
              'cross_arm_GT_and_frame_coverage_identical': True, 'main_published_reference_comparison': frozen_checks,
              'prior_02_10_scores_exact_equal': True, 'group_MOTA_checks': group_checks,
              'data_roles': GROUPS, 'all7_includes_training_sequences': True,
              'all7_is_independent_generalization_result': False, 'failures': []}
    output = RUN / 'replay-tools/scoring-review.json'
    output.write_text(json.dumps(review, indent=2)+'\n')
    print(json.dumps({'result': 'PASS', 'rows': 24, 'group_rows': len(group_checks),
                      'main_numeric_fields_exact_equal': sum(row['numeric_fields_exact_equal'] for row in frozen_checks),
                      'frames_per_arm': 5316, 'output': str(output)}, indent=2))

if __name__ == '__main__':
    main()
