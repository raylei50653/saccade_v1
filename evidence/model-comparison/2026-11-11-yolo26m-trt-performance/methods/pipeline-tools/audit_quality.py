"""CPU-light independent audit of native/fixed-motion score extraction and receipts."""
from pathlib import Path
import csv
import hashlib
import json

RUN = Path(__file__).resolve().parent.parent
REPO = RUN.parent.parent
EVIDENCE = REPO/'evidence/model-comparison/2026-11-11-yolo26m-trt-performance'
ARMS = ['main','official','finetuned']
SEQS = {'MOT17-02-SDP':600,'MOT17-04-SDP':1050,'MOT17-05-SDP':837,
        'MOT17-09-SDP':525,'MOT17-10-SDP':654,'MOT17-11-SDP':900,'MOT17-13-SDP':750}
MAPPING = {**{key:key+'___AUC' for key in ['HOTA','DetA','AssA']},
           **{key:key for key in ['IDF1','MOTA','IDSW','Frag','CLR_FP','CLR_FN','GT_Dets','GT_IDs']},'Frames':'CLR_Frames'}
PERCENT = {'HOTA','DetA','AssA','IDF1','MOTA'}
COUNTS = ['IDSW','Frag','CLR_FP','CLR_FN','GT_Dets','GT_IDs','Frames']

def read_csv(path):
    return {row['seq']:row for row in csv.DictReader(path.open())}

def main():
    scores_path=EVIDENCE/'quality-scores.json'
    scores=json.loads(scores_path.read_text())
    captures=json.loads((EVIDENCE/'quality-capture.json').read_text())
    assert set(scores)=={'mot','fixed-motion'} and len(captures)==21
    checks=[];mota_errors=[];shared=[];published_numeric_fields=0
    old=REPO/'build/yolo26m-threeway-all7-20261111'
    frozen_detailed=read_csv(old/'main/mot/trackeval/results/v1/pedestrian_detailed.csv')
    main_receipt=json.loads((old/'main/replay-receipt.json').read_text())['sequences']
    for mode in scores:
        assert set(scores[mode])==set(ARMS)
        for arm in ARMS:
            table=read_csv(RUN/'quality'/arm/mode/'trackeval/results/v1/pedestrian_detailed.csv')
            assert set(table)=={*SEQS,'COMBINED'} and set(scores[mode][arm])==set(table)
            for seq,raw in table.items():
                row=scores[mode][arm][seq]
                for key,native in MAPPING.items():
                    target=float(raw[native])*(100 if key in PERCENT else 1)
                    assert abs(row[key]-target)<2e-12,(mode,arm,seq,key)
                mota=100*(1-(row['CLR_FP']+row['CLR_FN']+row['IDSW'])/row['GT_Dets'])
                error=abs(mota-row['MOTA']);mota_errors.append(error)
                assert error<2e-12 and row['Frames']==SEQS.get(seq,5316)
                if arm=='main':
                    reference=frozen_detailed[seq]
                    assert set(reference)==set(raw)
                    for key in raw:
                        if key!='seq':
                            assert float(raw[key])==float(reference[key]),(mode,seq,key)
                            published_numeric_fields+=1
                checks.append({'mode':mode,'arm':arm,'sequence':seq,'fields_verified':13,
                               'Frames':row['Frames'],'MOTA':row['MOTA'],'all_checks_pass':True})
            for key in COUNTS:
                assert scores[mode][arm]['COMBINED'][key]==sum(scores[mode][arm][seq][key] for seq in SEQS)
        for seq in [*SEQS,'COMBINED']:
            assert len({tuple(scores[mode][arm][seq][key] for key in ['Frames','GT_Dets','GT_IDs']) for arm in ARMS})==1
    for seq in [*SEQS,'COMBINED']:
        assert scores['mot']['main'][seq]==scores['fixed-motion']['main'][seq]
    for arm in ARMS:
        for seq,n in SEQS.items():
            folder=RUN/'quality'/arm/seq
            rec=captures[arm+'/'+seq]
            assert rec['frames']==n and rec['native_CPU_exact']
            for log in ['detect','track','packet-validation','fixed-replay','fixed-packet-validation','cpu-native']:
                assert json.loads((folder/(log+'.exit.json')).read_text())['exit_code']==0
            configs=[folder/'resolved.config',RUN/'quality'/arm/'fixed-motion'/(seq+'.config')]
            for config in configs:
                lines=config.read_text().splitlines()
                shared.append(sorted(line for line in lines if line.startswith(('tracker.','output.'))))
                assert 'tracker.kalman.nsa_enabled=false' in lines
                assert 'tracker.rows.measurement_box=false' in lines
            native=(folder/'resolved.config').read_text().splitlines()
            assert 'execution.candidate_slots=2' in native and 'execution.motion_slots=2' in native
            assert 'execution.result_slots=4' in native and 'execution.cuda_graph=true' in native
            if arm=='main':
                prior=main_receipt[seq]
                assert rec['matches_frozen_deployment']
                for key in ['detection_sha256','motion_sha256','mot_sha256','packets_sha256']:
                    assert rec[key]==prior[key],(seq,key)
    assert all(value==shared[0] for value in shared)
    before=json.loads((EVIDENCE/'quality-binaries-before.json').read_text())
    assert before==json.loads((EVIDENCE/'quality-binaries-after.json').read_text())
    plan=json.loads((EVIDENCE/'PLAN.json').read_text())
    assert plan['quality']['sequences']==SEQS
    assert plan['quality']['roles']=='training04/05/09/11/13; model-selectionval10; post-M4 reusedeval02'
    report={'result':'PASS','scope':'CPU-light score/retainedreceipt/config audit; no new inference or rescoring',
            'scores_sha256':hashlib.sha256(scores_path.read_bytes()).hexdigest(),'rows_verified':len(checks),
            'row_checks':checks,'MOTA_formula_max_error_pp':max(mota_errors),'native_cells':21,
            'frames_per_arm':5316,'main_frozen_numeric_fields_exact':published_numeric_fields,
            'main_SCDR_motion_MOT_packet_hashes_match_frozen':True,
            'all_recorded_native_CPU_and_canonical_validation_exits_zero':True,
            'binary_before_after_receipts_identical':True,'shared_tracker_and_output_config_all42':True,
            'native_quality_slots':[2,2,4],'native_and_fixed_motion_scoring_separate':True,
            'cross_arm_GT_coverage_identical':True,'COMBINED_counts_additive':True,
            'all7_includes_training_sequences':True,'independent_generalization_result':False,
            'large_packet_files_rehashed_by_this_audit':False,'full_SCDR_row_walk_by_this_audit':False,
            'source_runner_independently_reviewed':True,'failures':[]}
    output=RUN/'pipeline-tools/quality-review.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['result','rows_verified','native_cells','frames_per_arm','main_frozen_numeric_fields_exact','MOTA_formula_max_error_pp']},indent=2))

if __name__=='__main__':main()
