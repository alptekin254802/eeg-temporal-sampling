"""Reconstruct registered schedules or check time-domain QC against raw primary EEG."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_preregistered_analysis import (validate_registered_inputs,_schedule_from_mask,
    _load_required_support,time_domain_qc_mask,file_digest,true_value)
from sampling_design import candidates_c,candidates_f5,draw_pair

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule',action='store_true',help='Regenerate every primary schedule from the registered masks and seeds')
    parser.add_argument('--primary-root',type=Path,help='Check all 536 structurally eligible recordings against the registered QC masks')
    args=parser.parse_args();inputs=validate_registered_inputs();done={}
    if args.schedule:
        for i,(participant,mask) in enumerate(zip(inputs['participant_ids'],inputs['eligible_masks'])):
            regenerated=np.empty((3,200,2,10),dtype=np.uint8)
            for r,name in enumerate(['C','F5','F10']):
                prepared=candidates_c(mask) if name=='C' else candidates_f5(mask) if name=='F5' else None
                for b in range(200):
                    regenerated[r,b]=draw_pair(mask,'ds005385',participant,name,b,prepared)
            if not np.array_equal(regenerated,inputs['schedule'][i]): raise ValueError('Schedule differs for '+participant)
            if (i+1)%50==0: print('Schedules',i+1,'/531',flush=True)
        done['primary_schedules_reproduced']=531
    if args.primary_root:
        rows=[r for r in inputs['census'] if true_value(r['structurally_eligible'])]
        for i,row in enumerate(rows):
            path=args.primary_root/row['target_recording']
            if path.stat().st_size!=int(row['annex_bytes']) or file_digest(path)!=row['annex_digest']:
                raise ValueError('Input identity differs for '+row['participant_id'])
            data,fs=_load_required_support(path,'edf')
            mask=time_domain_qc_mask(data*1e6,fs)
            expected=np.array([x=='1' for x in row['qc_eligible_mask']])
            if not np.array_equal(mask,expected): raise ValueError('QC differs for '+row['participant_id'])
            if (i+1)%50==0: print('QC',i+1,'/536',flush=True)
        done['primary_QC_masks_reproduced']=len(rows)
    if not done: parser.error('Choose --schedule and/or --primary-root')
    print(json.dumps(done,indent=2))
if __name__=='__main__':main()
