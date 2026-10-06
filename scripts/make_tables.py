"""Export numerical content of Tables 1-2 and S1-S4 from released inputs."""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/tables')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((ROOT/'results/preregistered/summary_results.json').read_text('utf-8'))
    exploratory=json.loads((ROOT/'results/exploratory_2026-09-20/summary_results.json').read_text('utf-8'))
    with np.load(ROOT/'osf_upload_package/structural/ds005385_sampling_schedule.npz',allow_pickle=False) as data:
        schedule=data['schedule'].astype(float)
    geometry=[]
    for r,name in enumerate(['C','F5','F10']):
        s=schedule[:,r];a=s[:,:,0];b=s[:,:,1]
        distances=np.abs(a[...,None]-b[...,None,:])*4
        nearest=np.concatenate([distances.min(axis=-1),distances.min(axis=-2)],axis=-1).mean()
        geometry.append(dict(strategy=name,participants=len(s),paired_draws_per_participant=s.shape[1],arms=len(s)*s.shape[1]*2,
            retained_seconds=40,periodograms_per_arm=10,periodogram_seconds=4,
            mean_span_seconds=float(((s.max(axis=-1)-s.min(axis=-1)+1)*4).mean()),
            mean_AB_centroid_difference_seconds=float((np.abs(a.mean(axis=-1)-b.mean(axis=-1))*4).mean()),
            mean_symmetric_nearest_separation_seconds=float(nearest)))
    write(args.output/'table1_geometry.csv',geometry)
    rows=[];full=[]
    for name,key in [('Primary','PRIMARY'),('Complete support','PRESPECIFIED_SENSITIVITY'),('Replication','DIRECTIONAL_REPLICATION')]:
        r=summary[key];rho=dict(r['participant_rank_reproducibility']);z=dict(r['mean_fisher_z'])
        if name=='Primary':
            rho['F5']=summary['PRESPECIFIED_DESCRIPTIVE_SECONDARY']['F5']['participant_rank_reproducibility']
            z['F5']=summary['PRESPECIFIED_DESCRIPTIVE_SECONDARY']['F5']['mean_fisher_z']
        n=r.get('participant_n',r.get('participant_n_after_time_domain_qc'))
        rows.append(dict(cohort=name,N=n,rho_C=f"{rho['C']:.4f}",rho_F5=f"{rho['F5']:.4f}" if 'F5' in rho else '',rho_F10=f"{rho['F10']:.4f}",
            delta_z=f"{r['delta_z_F10_minus_C']:.4f}",CI_lower=f"{r['percentile_95_interval'][0]:.4f}",CI_upper=f"{r['percentile_95_interval'][1]:.4f}",MCSE=f"{r['monte_carlo_standard_error']:.4f}"))
        for quantity,value in [('N',n),*[(f'rho_{k}',v) for k,v in rho.items()],*[(f'mean_Fisher_z_{k}',v) for k,v in z.items()],
                ('delta_z',r['delta_z_F10_minus_C']),('CI_lower',r['percentile_95_interval'][0]),('CI_upper',r['percentile_95_interval'][1]),('MCSE',r['monte_carlo_standard_error'])]:
            full.append(dict(cohort=name,quantity=quantity,value=value))
    secondary=summary['PRESPECIFIED_DESCRIPTIVE_SECONDARY']['absolute_displacement']
    for key,val in [('mean',secondary['cohort_mean']),('CI_lower',secondary['percentile_95_interval'][0]),('CI_upper',secondary['percentile_95_interval'][1])]:
        full.append(dict(cohort='Primary signed logit displacement',quantity=key,value=val))
    write(args.output/'table2_results.csv',rows);write(args.output/'tableS2_full_precision.csv',full)
    demographics=[]
    with (ROOT/'osf_upload_package/DS005385_PARTICIPANT_CENSUS.csv').open(encoding='utf-8',newline='') as f:
        primary_acquisition={r['participant_id']:r for r in csv.DictReader(f)}
    with (ROOT/'structural/ds004148_session1_census.csv').open(encoding='utf-8',newline='') as f:
        replication_acquisition={r['participant_id']:r for r in csv.DictReader(f)}
    with np.load(ROOT/'results/preregistered/scientific_results.npz',allow_pickle=False) as z:
        for cohort,dataset in [('primary','ds005385'),('sensitivity','ds005385'),('replication','ds004148')]:
            with (ROOT/f'data/{dataset}/participants.tsv').open(encoding='utf-8',newline='') as f:
                lookup={r['participant_id']:r for r in csv.DictReader(f,delimiter='\t')}
            ids=z[cohort+'_participant_ids'].astype(str);ages=np.array([float(lookup[i]['age']) for i in ids]);sex=Counter(lookup[i]['sex'].upper() for i in ids)
            assert np.isfinite(ages).all() and len(ids)==len(set(ids))
            acq=replication_acquisition if cohort=='replication' else primary_acquisition
            rate_key='sampling_hz' if cohort=='replication' else 'edf_eeg_sampling_hz'
            channels_key='vhdr_channel_count' if cohort=='replication' else 'bids_eeg_channel_count'
            rates={float(acq[i][rate_key]) for i in ids};channels={int(acq[i][channels_key]) for i in ids}
            assert len(rates)==len(channels)==1
            demographics.append(dict(cohort=cohort,N=len(ids),age_mean=f'{ages.mean():.2f}',age_SD=f'{ages.std(ddof=1):.2f}',
                age_min=int(ages.min()),age_max=int(ages.max()),female=sex['F'],male=sex['M'],
                native_sampling_Hz=int(next(iter(rates))),EEG_channels=next(iter(channels))))
    write(args.output/'tableS1_sample.csv',demographics)
    rows=[]
    for r in exploratory['simulation_scenarios']:
        rows.append(dict(scenario=r['scenario_id'],tau=r['between_sd'],phi=r['phi'],gamma=r['slope_sd'],repetitions=r['repetitions'],
            rho_C=f"{r['mean_rho_C']:.4f}",rho_F10=f"{r['mean_rho_F10']:.4f}",delta_z=f"{r['mean_delta_z']:.4f}",SD=f"{r['sample_sd_delta_z']:.4f}",
            central_range_lower=f"{r['central_95_range'][0]:.4f}",central_range_upper=f"{r['central_95_range'][1]:.4f}",MCSE_of_mean=f"{r['mcse_mean_delta_z']:.6f}"))
    write(args.output/'tableS3_simulations.csv',rows)
    sensitivity=json.loads((ROOT/'results/post_audit_2026-10-05/boundary_sensitivity_summary.json').read_text('utf-8'))
    rows=[]
    for name,record,n in [
        ('Primary',summary['PRIMARY'],summary['PRIMARY']['participant_n']),
        ('Prespecified complete support',summary['PRESPECIFIED_SENSITIVITY'],summary['PRESPECIFIED_SENSITIVITY']['participant_n']),
        ('Exploratory annotation exclusion',sensitivity,sensitivity['n']),
    ]:
        rho=record['participant_rank_reproducibility']
        rows.append(dict(analysis=name,N=n,
            delta_z=f"{record['delta_z_F10_minus_C']:.4f}",
            interval_lower=f"{record['percentile_95_interval'][0]:.4f}",
            interval_upper=f"{record['percentile_95_interval'][1]:.4f}",
            MCSE=f"{record['monte_carlo_standard_error']:.4f}",
            rho_C=f"{rho['C']:.4f}",rho_F10=f"{rho['F10']:.4f}"))
    write(args.output/'tableS4_boundary_sensitivity.csv',rows)
    print('Wrote six numerical tables to',args.output)
if __name__=='__main__':main()
