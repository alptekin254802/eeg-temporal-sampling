"""Check released numerical results; optionally recompute all participant bootstraps."""
import argparse
import json
from pathlib import Path
import numpy as np
from preregistered_analysis import (point_statistics, primary_contrast_statistics,
    primary_bootstrap_distribution, generate_bootstrap_indices, absolute_displacement_summary,
    percentile_interval, bootstrap_index_sha256)
from exploratory_sampling import (arm_markers, rank_summary, permuted_powers,
    simulation_powers, scenarios, summarize_repetitions)
from run_preregistered_analysis import validate_registered_inputs

ROOT=Path(__file__).resolve().parents[1]
CHECKS=[]
def close(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape: raise ValueError(f'{label}: shapes {a.shape} and {b.shape}')
    np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,err_msg=label)
    CHECKS.append(dict(check=label,max_absolute_difference=float(np.max(np.abs(a-b)))))
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def summary_close(actual,expected,label):
    for key in ['repetitions','mean_delta_z','sample_sd_delta_z','mcse_mean_delta_z','central_95_range','mean_rho_C','mean_rho_F10']:
        close(actual[key],expected[key],label+' '+key)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-bootstrap',action='store_true',help='Recompute all 5000 resamples in each empirical cohort')
    parser.add_argument('--output',type=Path,help='Optional JSON verification report')
    args=parser.parse_args()
    registered=validate_registered_inputs()
    arrays=load(ROOT/'results/preregistered/scientific_results.npz')
    summary=json.loads((ROOT/'results/preregistered/summary_results.json').read_text('utf-8'))
    assert arrays['primary_participant_ids'].astype(str).tolist()==registered['participant_ids']
    selection=np.array([registered['participant_ids'].index(i) for i in arrays['sensitivity_participant_ids'].astype(str)])
    assert np.array_equal(arrays['sensitivity_schedule'],registered['schedule'][selection])
    close(arrays['sensitivity_markers_C_F10'],arrays['primary_markers'][selection][:,[0,2]],'sensitivity marker subset')
    for cohort,label,section,suffix in [('primary','primary_ds005385','PRIMARY',''),('sensitivity','complete_support_ds005385','PRESPECIFIED_SENSITIVITY','_C_F10'),('replication','replication_ds004148','DIRECTIONAL_REPLICATION','')]:
        markers=arrays[cohort+'_markers'+suffix]
        point=(primary_contrast_statistics if cohort=='sensitivity' else point_statistics)(markers)
        for result_key,point_key in [('draw_rho'+suffix,'draw_rho'),('draw_fisher_z'+suffix,'draw_z'),('draw_delta_z','draw_differences')]:
            close(arrays[cohort+'_'+result_key],point[point_key],cohort+' '+result_key)
        record=summary[section]
        for j,strategy in enumerate(['C','F10'] if cohort=='sensitivity' else ['C','F5','F10']):
            value=record['participant_rank_reproducibility'].get(strategy)
            zvalue=record['mean_fisher_z'].get(strategy)
            if value is None:
                value=summary['PRESPECIFIED_DESCRIPTIVE_SECONDARY']['F5']['participant_rank_reproducibility']
                zvalue=summary['PRESPECIFIED_DESCRIPTIVE_SECONDARY']['F5']['mean_fisher_z']
            close(point['back_transformed_rho'][j],value,cohort+' '+strategy+' rho')
            close(point['mean_z'][j],zvalue,cohort+' '+strategy+' Fisher z')
        close(point['delta_z'],record['delta_z_F10_minus_C'],cohort+' contrast')
        close(point['mcse'],record['monte_carlo_standard_error'],cohort+' MCSE')
        stored_bootstrap=arrays[cohort+'_bootstrap_delta_z']
        close(percentile_interval(stored_bootstrap),record['percentile_95_interval'],cohort+' saved bootstrap interval')
        indices=generate_bootstrap_indices(len(markers),label)
        assert bootstrap_index_sha256(indices)==record['bootstrap_index_sha256']
        selected=np.arange(5000) if args.full_bootstrap else np.array([0,2499,4999])
        pair=markers if cohort=='sensitivity' else markers[:,[0,2]]
        regenerated=primary_bootstrap_distribution(pair,indices[selected])
        close(regenerated,stored_bootstrap[selected],cohort+' recomputed bootstrap')
        print('Verified',cohort,'including',len(selected),'recomputed bootstrap draws',flush=True)
    secondary=absolute_displacement_summary(arrays['primary_markers'],generate_bootstrap_indices(531,'primary_ds005385'))
    close(secondary['participant_displacement'],arrays['secondary_participant_displacement'],'signed displacement per participant')
    close(secondary['bootstrap_distribution'],arrays['secondary_bootstrap_displacement'],'all secondary bootstrap means')
    expected=summary['PRESPECIFIED_DESCRIPTIVE_SECONDARY']['absolute_displacement']
    close(secondary['estimate'],expected['cohort_mean'],'secondary mean')
    close(secondary['interval'],expected['percentile_95_interval'],'secondary interval')

    base=ROOT/'results/exploratory_2026-09-20'
    plan=json.loads((ROOT/'docs/exploratory/analysis_plan_2026-09-20.json').read_text('utf-8'))
    e=json.loads((base/'summary_results.json').read_text('utf-8'))
    cache=load(base/'cell_band_powers.npz');order=load(base/'order_permutations.npz');sim=load(base/'simulation_results.npz')
    assert np.array_equal(cache['participant_ids'],arrays['sensitivity_participant_ids'])
    assert np.array_equal(cache['schedule'],arrays['sensitivity_schedule'][:,[0,2]])
    close(arm_markers(cache['cell_band_powers'],cache['schedule']),arrays['sensitivity_markers_C_F10'],'all cached original arm markers')
    for name,data,count in [('order',order,1000),('simulation',sim,500)]:
        z=np.arctanh(data['draw_rho'])
        delta=(z[...,1,:]-z[...,0,:]).mean(axis=-1)
        rho=np.tanh(z.mean(axis=-1))
        close(delta,data['delta_z'],name+' contrasts from all draw correlations')
        close(rho,data['rho'],name+' Fisher-averaged strategy correlations')
        if name=='order':
            assert delta.shape==(count,)
            summary_close(summarize_repetitions(delta,rho),e['order_control'],'order summary')
        else:
            assert delta.shape==(12,count)
            for i,case in enumerate(scenarios(plan)):
                assert all(e['simulation_scenarios'][i][k]==v for k,v in case.items())
                summary_close(summarize_repetitions(delta[i],rho[i]),e['simulation_scenarios'][i],f'scenario {i+1}')
    for rep in [0,999]:
        actual=rank_summary(arm_markers(permuted_powers(cache['cell_band_powers'],rep)[0],cache['schedule']))
        close(actual['draw_rho'],order['draw_rho'][rep],f'recomputed permutation {rep}')
    for rep in [0,499]:
        for i,powers in enumerate(simulation_powers(plan,rep,516)):
            close(rank_summary(arm_markers(powers,cache['schedule']))['draw_rho'],sim['draw_rho'][i,rep],f'recomputed simulation {i+1}, repetition {rep}')
    result=dict(passed=True,checks=CHECKS,comparison=dict(atol=1e-12,rtol=0),
        empirical_bootstrap_recomputed_per_cohort=5000 if args.full_bootstrap else 3,
        secondary_bootstrap_means_recomputed=5000,exploratory_summary_repetitions=dict(order=1000,simulation_per_scenario=500),
        exploratory_seed_replay=dict(order=[0,999],simulation=[0,499]),raw_EEG_recomputed=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if __name__=='__main__': main()
