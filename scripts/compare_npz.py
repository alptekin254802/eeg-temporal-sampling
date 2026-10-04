"""Compare scientific NPZ contents independently of ZIP timestamps and compression."""
import argparse
import json
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actual');parser.add_argument('expected')
    args=parser.parse_args();checks=[]
    with np.load(args.actual,allow_pickle=False) as a,np.load(args.expected,allow_pickle=False) as b:
        if set(a.files)!=set(b.files): raise ValueError('Array-key sets differ')
        for key in sorted(a.files):
            x,y=a[key],b[key]
            if x.shape!=y.shape or x.dtype.kind!=y.dtype.kind: raise ValueError('Shape or type differs: '+key)
            if x.dtype.kind in 'fc':
                if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite value: '+key)
                np.testing.assert_allclose(x,y,atol=1e-12,rtol=0,err_msg=key)
                checks.append(dict(array=key,max_absolute_difference=float(np.max(np.abs(x-y)))))
            elif not np.array_equal(x,y): raise ValueError('Exact array mismatch: '+key)
            else: checks.append(dict(array=key,exact=True))
    print(json.dumps(dict(passed=True,checks=checks),indent=2))
if __name__=='__main__':main()
