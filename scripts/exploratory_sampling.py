"""Cell-order controls and idealized cell-band-power simulations.

These functions implement the explicitly post hoc 2026-09-20 addendum.
No data are read or outcomes calculated at import time.
"""
import hashlib
from itertools import product
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

NAMESPACE='temporal-sampling-exploratory-2026-09-20-v1|{analysis}|{repetition}'

def rng_for(analysis, repetition):
    seed=int.from_bytes(hashlib.sha256(NAMESPACE.format(analysis=analysis,repetition=repetition).encode()).digest()[:8],'big')
    return np.random.Generator(np.random.PCG64(seed))

def cell_band_powers(psd, frequency):
    """Return cell x (alpha,total); linear reduction preserves PSD aggregation."""
    psd=np.asarray(psd,dtype=float)
    frequency=np.asarray(frequency)
    if psd.ndim!=3 or psd.shape[:2]!=(3,42) or psd.shape[-1]!=len(frequency):
        raise ValueError('Expected 3 x 42 x frequency PSD array')
    roi=psd.mean(axis=0)
    alpha=(frequency>=8)&(frequency<=13)
    total=(frequency>=1)&(frequency<=30)
    return np.stack([np.trapezoid(roi[:,alpha],frequency[alpha],axis=-1),np.trapezoid(roi[:,total],frequency[total],axis=-1)],axis=-1)

def arm_markers(powers, schedule):
    powers=np.asarray(powers,dtype=float)
    schedule=np.asarray(schedule)
    if powers.ndim!=3 or powers.shape[1:]!=(42,2):
        raise ValueError('Expected N x 42 x (alpha,total)')
    if schedule.shape!=(len(powers),2,200,2,10):
        raise ValueError('Expected N x 2 x 200 x 2 x 10 schedule')
    if not np.isfinite(powers).all() or np.any(powers[...,0]<=0) or np.any(powers[...,1]<=powers[...,0]):
        raise ValueError('Band powers must be finite with 0 < alpha < total')
    rows=np.arange(len(powers))[:,None,None,None,None]
    numerator=powers[...,0][rows,schedule].mean(axis=-1)
    denominator=powers[...,1][rows,schedule].mean(axis=-1)
    values=numerator/denominator
    if not np.isfinite(values).all() or not ((values>0)&(values<1)).all():
        raise ValueError('Invalid relative power')
    return values

def rank_summary(markers):
    markers=np.asarray(markers,dtype=float)
    if markers.ndim!=4 or markers.shape[1:]!=(2,200,2) or len(markers)<3 or not np.isfinite(markers).all():
        raise ValueError('Invalid marker array')
    ranks=rankdata(markers,method='average',axis=0)
    ranks-=ranks.mean(axis=0,keepdims=True)
    numerator=(ranks[...,0]*ranks[...,1]).sum(axis=0)
    denominator=np.sqrt((ranks[...,0]**2).sum(axis=0)*(ranks[...,1]**2).sum(axis=0))
    with np.errstate(invalid='ignore',divide='ignore'):
        rho=numerator/denominator
        z=np.arctanh(rho)
    if not np.isfinite(z).all():
        raise ValueError('Unestimable correlation or Fisher transform; no repair')
    mean_z=z.mean(axis=1)
    return dict(draw_rho=rho,mean_z=mean_z,rho=np.tanh(mean_z),delta_z=float((z[1]-z[0]).mean()))

def permuted_powers(powers, repetition):
    rng=rng_for('order',repetition)
    permutations=np.stack([rng.permutation(42) for _ in range(len(powers))])
    return powers[np.arange(len(powers))[:,None],permutations,:], permutations

def ar1_path(initial, innovations, phi):
    if not 0<=phi<1:
        raise ValueError('AR coefficient must be in [0,1)')
    initial=np.asarray(initial,dtype=float)
    innovations=np.asarray(innovations,dtype=float)
    path=np.empty((len(initial),innovations.shape[1]+1))
    path[:,0]=initial
    scale=np.sqrt(1-phi**2)
    for t in range(1,path.shape[1]):
        path[:,t]=phi*path[:,t-1]+scale*innovations[:,t-1]
    return path

def scenarios(plan):
    model=plan['simulation_model']
    return [dict(between_sd=tau,phi=phi,slope_sd=gamma) for tau,phi,gamma in product(model['between_participant_sd'],model['ar_coefficients'],model['slope_sd'])]

def simulation_powers(plan, repetition, n):
    model=plan['simulation_model']
    rng=rng_for('simulation',repetition)
    between=rng.standard_normal(n)
    slopes=rng.standard_normal(n)
    initial=rng.standard_normal(n)
    innovations=rng.standard_normal((n,41))
    total=np.exp(model['total_power_log_sd']*rng.standard_normal((n,42)))
    paths={phi:ar1_path(initial,innovations,phi) for phi in model['ar_coefficients']}
    time=np.linspace(-0.5,0.5,42)
    for case in scenarios(plan):
        latent=(model['mean_logit']+case['between_sd']*between[:,None]
                +model['within_participant_stationary_sd']*paths[case['phi']]
                +case['slope_sd']*slopes[:,None]*time)
        yield np.stack([total*expit(latent),total],axis=-1)

def summarize_repetitions(delta, rho):
    delta=np.asarray(delta,dtype=float)
    rho=np.asarray(rho,dtype=float)
    if not np.isfinite(delta).all() or not np.isfinite(rho).all():
        raise ValueError('Nonfinite repetitions; cannot discard')
    sd=float(delta.std(ddof=1))
    return dict(repetitions=len(delta),mean_delta_z=float(delta.mean()),sample_sd_delta_z=sd,
                mcse_mean_delta_z=sd/np.sqrt(len(delta)),central_95_range=np.percentile(delta,[2.5,97.5],method='linear').tolist(),
                mean_rho_C=float(rho[:,0].mean()),mean_rho_F10=float(rho[:,1].mean()))
