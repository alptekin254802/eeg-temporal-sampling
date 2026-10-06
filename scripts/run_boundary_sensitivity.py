"""Repeat the analysis after excluding recordings with extra boundary annotations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from preregistered_analysis import primary_contrast_statistics, primary_bootstrap_distribution, percentile_interval
from verification_tolerances import assert_summary_close

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/post_audit/boundary_sensitivity_plan_2026-10-05.json'
PLAN_SHA256 = 'c7eac5aa4cf5106499e56e0f932fb05083be546f9f73a420a36de287171f0d47'
ACCEPTED = ROOT / 'results/post_audit_2026-10-05'

def inputs():
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != PLAN_SHA256:
        raise ValueError('The pre-calculation sensitivity plan differs')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    source = ROOT / plan['source_results']
    if hashlib.sha256(source.read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('Original accepted result identity differs')
    with np.load(source, allow_pickle=False) as z:
        ids = z['primary_participant_ids'].astype(str)
        if len(ids) != plan['original_n'] or len(set(ids)) != len(ids):
            raise ValueError('Unexpected original participant roster')
        if not set(plan['exclude_ids']).issubset(set(ids)):
            raise ValueError('An excluded participant is absent from the original roster')
        rows = np.flatnonzero(~np.isin(ids, plan['exclude_ids']))
        values = z['primary_markers'][rows][:, plan['regime_indices']]
        if len(rows) != plan['retained_n'] or values.shape != (526, 2, 200, 2):
            raise ValueError('Unexpected sensitivity dimensions')
    indices = np.random.default_rng(plan['bootstrap_seed']).integers(0, len(rows), size=(plan['resamples'], len(rows)), dtype=np.int64)
    return plan, ids[rows], rows, values, indices

def record(plan, ids, point, bootstrap, indices):
    return {
        'status': plan['status'], 'plan_sha256': PLAN_SHA256,
        'source_sha256': plan['source_sha256'], 'n': len(ids),
        'excluded_ids': plan['exclude_ids'], 'draws': plan['draws'],
        'delta_z_F10_minus_C': point['delta_z'],
        'percentile_95_interval': list(percentile_interval(bootstrap)),
        'monte_carlo_standard_error': point['mcse'],
        'participant_rank_reproducibility': dict(zip(['C', 'F10'], point['back_transformed_rho'].tolist())),
        'bootstrap_resamples': plan['resamples'],
        'bootstrap_seed': plan['bootstrap_seed'],
        'bootstrap_indices_sha256': hashlib.sha256(indices.tobytes(order='C')).hexdigest(),
        'original_results_changed': False, 'raw_EEG_recomputed': False,
        'physical_continuity_established': False,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true', help='Recompute this additional 5000-resample sensitivity only')
    mode.add_argument('--check', action='store_true', help='Check stored summaries and three selected bootstrap draws')
    parser.add_argument('--output', type=Path, help='New empty output directory, required with --run')
    args = parser.parse_args()
    if args.run and args.output is None:
        parser.error('--run requires --output')
    if args.output and args.output.resolve().is_relative_to((ROOT / 'results/preregistered').resolve()):
        parser.error('Original results cannot be overwritten')
    plan, ids, rows, values, indices = inputs()
    point = primary_contrast_statistics(values)
    if args.check:
        expected = json.loads((ACCEPTED / 'boundary_sensitivity_summary.json').read_text(encoding='utf-8'))
        with np.load(ACCEPTED / 'boundary_sensitivity_results.npz', allow_pickle=False) as z:
            if not np.array_equal(z['participant_ids'].astype(str), ids) or not np.array_equal(z['original_row_indices'], rows):
                raise ValueError('Sensitivity roster differs')
            for key in ['draw_rho', 'draw_z', 'draw_differences']:
                np.testing.assert_allclose(point[key], z[key], atol=1e-12, rtol=0)
            selected = np.array([0, 2499, 4999])
            np.testing.assert_allclose(primary_bootstrap_distribution(values, indices[selected]), z['bootstrap_delta_z'][selected], atol=1e-12, rtol=0)
            actual = record(plan, ids, point, z['bootstrap_delta_z'], indices)
        assert_summary_close(actual, expected)
        print(json.dumps({'status': 'PASS', 'scope': 'annotation-exclusion sensitivity summary and selected bootstrap replay', 'selected_resamples': selected.tolist(), 'plan_sha256': PLAN_SHA256}, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose a new empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    blocks = []
    for start in range(0, len(indices), 250):
        blocks.append(primary_bootstrap_distribution(values, indices[start:start+250]))
        print(f'Additional sensitivity bootstrap: {min(start+250,len(indices))}/{len(indices)}', flush=True)
    bootstrap = np.concatenate(blocks)
    np.savez_compressed(args.output / 'boundary_sensitivity_results.npz', participant_ids=ids, original_row_indices=rows, draw_rho=point['draw_rho'], draw_z=point['draw_z'], draw_differences=point['draw_differences'], bootstrap_delta_z=bootstrap)
    result = record(plan, ids, point, bootstrap, indices)
    (args.output / 'boundary_sensitivity_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
