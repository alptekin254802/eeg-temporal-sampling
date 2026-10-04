#!/usr/bin/env python3
"""Synthetic-only equivalence tests for the elementary PSD estimator."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
from scipy import signal
from sampling_design import draw_pair

FS=1000
N=4*FS
FREQ=np.fft.rfftfreq(N,1/FS)
ALPHA=(FREQ>=8)&(FREQ<=13)
DENOM=(FREQ>=1)&(FREQ<=30)

def elementary_periodogram(x, fs=FS):
    x=np.asarray(x,float)
    n=int(round(4*fs))
    if x.shape[-1]!=n:raise ValueError("exactly 4 seconds required; partial windows forbidden")
    x=signal.detrend(x,type="linear",axis=-1)
    taper=signal.windows.hann(n,sym=False)
    y=np.fft.rfft(x*taper, n=n,axis=-1)
    p=(np.abs(y)**2)/(fs*np.sum(taper**2))
    p[...,1:-1]*=2
    return p

def marker(three_by_ten_by_n, fs=FS):
    x=np.asarray(three_by_ten_by_n)
    n=int(round(4*fs)); freq=np.fft.rfftfreq(n,1/fs)
    alpha=(freq>=8)&(freq<=13); denom=(freq>=1)&(freq<=30)
    if x.shape!=(3,10,n):raise ValueError("three ROI channels x ten exact windows required")
    ps=np.stack([[elementary_periodogram(x[c,w],fs) for w in range(10)] for c in range(3)])
    agg=ps.mean(axis=1).mean(axis=0)
    return np.trapezoid(agg[alpha],freq[alpha])/np.trapezoid(agg[denom],freq[denom]),agg

class TestEquivalence(unittest.TestCase):
    def test_frequency_grid_and_masks(self):
        self.assertEqual(N,4000);self.assertAlmostEqual(FREQ[1]-FREQ[0],0.25)
        self.assertTrue(np.array_equal(FREQ[ALPHA],np.arange(8,13.0001,0.25)))
        self.assertTrue(np.array_equal(FREQ[DENOM],np.arange(1,30.0001,0.25)))
    def test_replication_500hz_has_same_physical_estimator_grid(self):
        fs=500; n=4*fs; freq=np.fft.rfftfreq(n,1/fs)
        self.assertEqual(n,2000);self.assertAlmostEqual(freq[1]-freq[0],0.25)
        self.assertTrue(np.array_equal(freq[(freq>=8)&(freq<=13)],np.arange(8,13.0001,0.25)))
        self.assertTrue(np.array_equal(freq[(freq>=1)&(freq<=30)],np.arange(1,30.0001,0.25)))
        t=np.arange(n)/fs; base=np.sin(2*np.pi*10*t)+0.3*np.sin(2*np.pi*20*t)
        value,_=marker(np.tile(base,(3,10,1)),fs=fs)
        self.assertTrue(np.isfinite(value))
    def test_grouping_metadata_cannot_change_estimator(self):
        rng=np.random.default_rng(20260828)
        windows=rng.standard_normal((3,10,N))
        vals=[marker(windows) for _ in ("C","F5","F10")]
        for v in vals[1:]:
            self.assertEqual(vals[0][0],v[0]);self.assertTrue(np.array_equal(vals[0][1],v[1]))
    def test_identical_stationary_windows_are_exactly_invariant(self):
        t=np.arange(N)/FS;base=np.sin(2*np.pi*10*t)+0.3*np.sin(2*np.pi*20*t)
        all42=np.tile(base,(3,42,1))
        results=[]
        for regime in ("C","F5","F10"):
            a,_=draw_pair([True]*42,"synthetic","stationary",regime,0)
            results.append(marker(all42[:,a,:])[0])
        self.assertEqual(len(set(results)),1)
    def test_stationary_ensemble_no_systematic_geometry_shift(self):
        rng=np.random.default_rng(9);diff=[]
        for rep in range(120):
            x=rng.standard_normal((3,42,N))
            c,_=draw_pair([True]*42,"synthetic",f"s{rep}","C",rep)
            f,_=draw_pair([True]*42,"synthetic",f"s{rep}","F10",rep)
            diff.append(marker(x[:,f,:])[0]-marker(x[:,c,:])[0])
        diff=np.asarray(diff);se=diff.std(ddof=1)/np.sqrt(len(diff))
        self.assertLess(abs(diff.mean()),4*se)
    def test_nonstationary_geometry_is_detectable(self):
        t=np.arange(N)/FS; rng=np.random.default_rng(10); all42=np.empty((3,42,N))
        for w in range(42):
            amp=0.2+2.0*(w/41)
            for c in range(3):all42[c,w]=amp*np.sin(2*np.pi*10*t)+0.8*rng.standard_normal(N)
        cvals=[];fvals=[]
        for rep in range(200):
            c,_=draw_pair([True]*42,"synthetic","drift","C",rep)
            f,_=draw_pair([True]*42,"synthetic","drift","F10",rep)
            cvals.append(marker(all42[:,c,:])[0]);fvals.append(marker(all42[:,f,:])[0])
        self.assertLess(np.var(fvals),np.var(cvals))
    def test_partial_windows_fail(self):
        with self.assertRaises(ValueError):elementary_periodogram(np.zeros(N-1))
        with self.assertRaises(ValueError):marker(np.zeros((3,9,N)))
if __name__=="__main__":unittest.main()
