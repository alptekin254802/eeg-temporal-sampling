"""Reproduce all reported exploratory repetitions from cell powers or local raw EEG."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from exploratory_sampling import (arm_markers,cell_band_powers,permuted_powers,rank_summary,
    scenarios,simulation_powers,summarize_repetitions)
from preregistered_analysis import compute_cell_periodograms
from run_preregistered_analysis import _load_required_support,file_digest,validate_registered_inputs

ROOT=Path(__file__).resolve().parents[1]
def worker_init(plan_path,cache_path):
    global PLAN,SCHEDULE,POWERS
    PLAN=json.loads(Path(plan_path).read_text('utf-8'))
    with np.load(cache_path,allow_pickle=False) as z:
        SCHEDULE=z['schedule'];POWERS=z['cell_band_powers']
def order_worker(rep):
    return rank_summary(arm_markers(permuted_powers(POWERS,rep)[0],SCHEDULE))['draw_rho']
def simulation_worker(rep):
    return np.array([rank_summary(arm_markers(p,SCHEDULE))['draw_rho'] for p in simulation_powers(PLAN,rep,len(POWERS))])
def summarize_draws(rho):
    z=np.arctanh(rho)
    return (z[...,1,:]-z[...,0,:]).mean(axis=-1),np.tanh(z.mean(axis=-1))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--primary-root',type=Path,help='Re-extract cell powers from ds005385 v1.0.3 when provided')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/exploratory_reproduction')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    plan_path=ROOT/'docs/exploratory/analysis_plan_2026-09-20.json'
    plan=json.loads(plan_path.read_text('utf-8'))
    expected='2d365a36e5d49ec06a9ae87f8fcea20f736b43c90e7d6b931e9dd88f32eeb3f4'
    if file_digest(plan_path)!=expected: raise RuntimeError('The reported exploratory plan has changed')
    registered=validate_registered_inputs()
    if not args.run:
        print(json.dumps(dict(mode='validation_only',plan=plan),indent=2));return
    if args.workers<1: raise ValueError('workers must be positive')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a new empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'results/exploratory_2026-09-20/cell_band_powers.npz'
    with np.load(cache,allow_pickle=False) as z:
        ids=z['participant_ids'];schedule=z['schedule'];powers=z['cell_band_powers']
    if args.primary_root:
        rows={r['participant_id']:r for r in registered['sensitivity_rows']}
        new=[]
        for i,participant in enumerate(ids.astype(str)):
            row=rows[participant];path=args.primary_root/row['target_recording']
            if path.stat().st_size!=int(row['annex_bytes']) or file_digest(path)!=row['annex_digest']:
                raise RuntimeError('Raw input identity mismatch: '+participant)
            data,fs=_load_required_support(path,'edf')
            frequency,psd=compute_cell_periodograms(data,fs)
            new.append(cell_band_powers(psd,frequency))
            if (i+1)%50==0: print('Cell powers',i+1,'/516',flush=True)
        np.testing.assert_allclose(new,powers,atol=1e-12,rtol=0)
        powers=np.asarray(new)
        cache=args.output/'cell_band_powers.npz'
        np.savez_compressed(cache,participant_ids=ids,schedule=schedule,cell_band_powers=powers)
    with np.load(ROOT/'results/preregistered/scientific_results.npz',allow_pickle=False) as z:
        assert np.array_equal(ids,z['sensitivity_participant_ids'])
        assert np.array_equal(schedule,z['sensitivity_schedule'][:,[0,2]])
        np.testing.assert_allclose(arm_markers(powers,schedule),z['sensitivity_markers_C_F10'],atol=1e-12,rtol=0)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(str(plan_path),str(cache))) as pool:
        orders=[]
        for i,rho in enumerate(pool.map(order_worker,range(plan['order_permutations']))):
            orders.append(rho)
            if (i+1)%100==0: print('Order permutations',i+1,'/1000',flush=True)
        simulations=[]
        for i,rho in enumerate(pool.map(simulation_worker,range(plan['simulation_repetitions_per_scenario']))):
            simulations.append(rho)
            if (i+1)%50==0: print('Simulation repetitions',i+1,'/500 for all 12 scenarios',flush=True)
    order_rho=np.array(orders);sim_rho=np.array(simulations).transpose(1,0,2,3)
    od,orr=summarize_draws(order_rho);sd,sr=summarize_draws(sim_rho)
    np.savez_compressed(args.output/'order_permutations.npz',draw_rho=order_rho,delta_z=od,rho=orr)
    np.savez_compressed(args.output/'simulation_results.npz',draw_rho=sim_rho,delta_z=sd,rho=sr)
    result=dict(status=plan['status'],order_control=summarize_repetitions(od,orr),
        simulation_scenarios=[dict(scenario_id=i+1,**case,**summarize_repetitions(sd[i],sr[i])) for i,case in enumerate(scenarios(plan))])
    (args.output/'summary_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    # Numerical equality is required; NPZ archive metadata may differ.
    for name in ['order_permutations.npz','simulation_results.npz']:
        with np.load(args.output/name,allow_pickle=False) as actual, np.load(ROOT/'results/exploratory_2026-09-20'/name,allow_pickle=False) as expected:
            assert set(actual.files)==set(expected.files)
            for key in actual.files: np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=0)
    print('All reported exploratory repetitions reproduced within absolute tolerance 1e-12.')
if __name__=='__main__': main()
