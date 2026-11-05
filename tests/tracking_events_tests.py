"""Cross-check event extraction against local TrackEval, including empty-frame behavior."""
import sys
import os
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
LEGACY=Path(os.environ.get('SACCADE_LEGACY_ROOT','/home/ray/developer/ai/saccade'))
sys.path.insert(0,str(LEGACY/'third_party/TrackEval'))
from tracking_events import extract
import trackeval


def fixture(gs,ts,sims,n=2):
    return {'gt_ids':[np.array(x,dtype=int) for x in gs],
            'tracker_ids':[np.array(x,dtype=int) for x in ts],
            'similarity_scores':[np.array(x,dtype=float).reshape(len(g),len(t)) for x,g,t in zip(sims,gs,ts)],
            'num_gt_ids':n,'num_tracker_ids':3,'num_gt_dets':sum(map(len,gs)),
            'num_tracker_dets':sum(map(len,ts)),'num_timesteps':len(gs)}


class Events(unittest.TestCase):
    def check_official(self,d):
        es,ms,c=extract(d,np,linear_sum_assignment)
        official=trackeval.metrics.CLEAR({'PRINT_CONFIG':False}).eval_sequence(d)
        for k in ('IDSW','Frag','CLR_TP','CLR_FN','CLR_FP'):
            self.assertEqual(c.get(k,0),official[k],k)
        return es,ms,c

    def test_switch_and_gap(self):
        d=fixture([[0],[0],[0],[0]],[[0],[1],[2],[2]],[[.9],[.9],[.1],[.9]],1)
        es,_,c=self.check_official(d)
        self.assertEqual(c['IDSW'],2)
        self.assertEqual(c['Frag'],1)
        self.assertEqual(es[-1]['gap_frames'],1)
        self.assertTrue(es[-1]['gap_recovery'])

    def test_empty_frame_preserves_official_previous_step(self):
        # CLEAR early-continues on a frame with no tracker detections.
        d=fixture([[0],[0],[],[0]],[[0],[],[2],[1]],[[.8],[],[],[.8]],1)
        es,_,c=self.check_official(d)
        self.assertEqual(c['IDSW'],1)
        self.assertEqual(c.get('Frag',0),0)
        self.assertTrue(es[-1]['gap_recovery'])

    def test_random_assignment_and_threshold_edges(self):
        rng=np.random.default_rng(80)
        for _ in range(30):
            gs=[];ts=[];sims=[]
            for t in range(20):
                g=sorted(rng.choice(2,size=int(rng.integers(3)),replace=False).tolist())
                tr=sorted(rng.choice(3,size=int(rng.integers(4)),replace=False).tolist())
                gs.append(g);ts.append(tr)
                sims.append(rng.choice([0.,.499,.5,.501,.9],size=(len(g),len(tr))))
            self.check_official(fixture(gs,ts,sims))


if __name__=='__main__': unittest.main()
