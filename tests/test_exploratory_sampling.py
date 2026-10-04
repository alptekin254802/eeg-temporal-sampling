import json
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from exploratory_sampling import arm_markers,rank_summary,permuted_powers,ar1_path,rng_for,simulation_powers,scenarios,cell_band_powers
from preregistered_analysis import relative_alpha_from_cell_psds

def fixture_schedule(n):
    result=np.empty((n,2,200,2,10),dtype=np.uint8)
    result[:,0,:,0]=np.arange(10)
    result[:,0,:,1]=np.arange(10,20)
    result[:,1,:,0]=np.arange(0,40,4)
    result[:,1,:,1]=np.arange(1,41,4)
    return result

class ExploratoryChecks(unittest.TestCase):
    def test_ratio_of_powers_not_mean_ratios(self):
        powers=np.ones((3,42,2));powers[...,0]=0.2
        powers[:,0]=[9,10]
        actual=arm_markers(powers,fixture_schedule(3))[0,0,0,0]
        self.assertAlmostEqual(actual,(9+9*0.2)/19)
        self.assertNotAlmostEqual(actual,(0.9+9*0.2)/10)

    def test_linear_power_cache_matches_direct_psd_aggregation(self):
        frequency=np.arange(0,50.25,0.25)
        rng=np.random.default_rng(82)
        psd=rng.uniform(0.01,4,(3,42,len(frequency)))
        powers=cell_band_powers(psd,frequency)
        schedule=fixture_schedule(3)
        actual=arm_markers(np.repeat(powers[None],3,axis=0),schedule)
        for regime in range(2):
            for arm in range(2):
                expected=relative_alpha_from_cell_psds(psd,frequency,schedule[0,regime,0,arm])
                self.assertAlmostEqual(actual[0,regime,0,arm],expected,places=14)

    def test_permutation_is_joint_and_bijective(self):
        powers=np.empty((5,42,2))
        powers[...,0]=np.arange(42)[None,:]+np.arange(5)[:,None]*1000
        powers[...,1]=powers[...,0]+100
        shuffled,index=permuted_powers(powers,8)
        for i in range(5):
            np.testing.assert_array_equal(np.sort(index[i]),np.arange(42))
            np.testing.assert_array_equal(shuffled[i],powers[i,index[i]])
        np.testing.assert_array_equal(shuffled[...,1]-shuffled[...,0],100)
        self.assertFalse(np.array_equal(index[0],index[1]))
        np.testing.assert_array_equal(permuted_powers(powers,8)[0],shuffled)

    def test_rank_summary_matches_scipy_with_ties(self):
        rng=np.random.default_rng(53)
        values=rng.integers(1,9,(31,2,200,2)).astype(float)
        result=rank_summary(values)
        expected=np.array([[spearmanr(values[:,r,b,0],values[:,r,b,1]).statistic for b in range(200)] for r in range(2)])
        np.testing.assert_allclose(result['draw_rho'],expected,atol=1e-12,rtol=0)
        self.assertAlmostEqual(result['delta_z'],float((np.arctanh(expected[1])-np.arctanh(expected[0])).mean()),places=14)

    def test_degenerate_correlation_fails_without_clipping(self):
        with self.assertRaises(ValueError): rank_summary(np.ones((20,2,200,2)))

    def test_invalid_bandpowers_fail(self):
        with self.assertRaises(ValueError):arm_markers(np.ones((3,42,2)),fixture_schedule(3))

    def test_ar_stationary_moments_and_initialization(self):
        rng=np.random.default_rng(34)
        initial=rng.standard_normal(30000)
        innovations=rng.standard_normal((30000,41))
        path=ar1_path(initial,innovations,0.7)
        np.testing.assert_array_equal(path[:,0],initial)
        self.assertLess(np.max(np.abs(path.var(axis=0)-1)),0.04)
        self.assertAlmostEqual(np.corrcoef(path[:,0],path[:,1])[0,1],0.7,delta=0.02)
        np.testing.assert_array_equal(ar1_path(initial,innovations,0)[:,1:],innovations)

    def test_rng_stream_separation(self):
        np.testing.assert_array_equal(rng_for('order',0).random(10),rng_for('order',0).random(10))
        self.assertFalse(np.array_equal(rng_for('order',0).random(10),rng_for('simulation',0).random(10)))

    def test_simulation_complete_grid_and_shared_total_power(self):
        plan=json.loads((ROOT/'docs/exploratory/analysis_plan_2026-09-20.json').read_text())
        generated=list(simulation_powers(plan,0,7))
        self.assertEqual(len(generated),len(scenarios(plan)))
        self.assertEqual(len(generated),12)
        for powers in generated:
            self.assertEqual(powers.shape,(7,42,2))
            self.assertTrue(((powers[...,0]>0)&(powers[...,0]<powers[...,1])).all())
            np.testing.assert_array_equal(powers[...,1],generated[0][...,1])

if __name__=='__main__':unittest.main()
