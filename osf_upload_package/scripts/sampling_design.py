#!/usr/bin/env python3
"""Deterministic temporal-sampling construction and legality checks."""
from __future__ import annotations
import csv, hashlib, itertools, json
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
NWIN=42
NDRAWS=200

def split_strata(n,k):
    q,r=divmod(n,k); out=[]; start=0
    for i in range(k):
        size=q+(1 if i<r else 0)
        out.append(tuple(range(start,start+size))); start+=size
    return out

STRATA5=split_strata(NWIN,5)
STRATA10=split_strata(NWIN,10)

def seed64(dataset,participant,regime,replicate,arm):
    # AUX004 is a historical deterministic namespace. It is retained verbatim
    # because changing it would change the prespecified sampling schedule.
    s=f"AUX004|{dataset}|{participant}|{regime}|{replicate}|{arm}"
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],"big",signed=False)

def candidates_c(mask):
    return [tuple(range(s,s+10)) for s in range(NWIN-9) if all(mask[s:s+10])]

def candidates_f5(mask):
    starts=[]
    for st in STRATA5:
        starts.append([s for s in st[:-1] if s+1 in st and mask[s] and mask[s+1]])
    out=[]
    for combo in itertools.product(*starts):
        blocks=[(s,s+1) for s in combo]
        if all(blocks[i+1][0]-blocks[i][1]>=2 for i in range(4)):
            out.append(tuple(x for b in blocks for x in b))
    return out

def choose(seq,seed):
    if not seq: raise ValueError("empty candidates")
    return seq[int(np.random.default_rng(seed).integers(0,len(seq)))]

def draw_pair(mask,dataset,participant,regime,replicate,prepared=None):
    if regime=="C":
        c=prepared if prepared is not None else candidates_c(mask)
        a=choose(c,seed64(dataset,participant,regime,replicate,"A"))
        bpool=[x for x in c if set(x).isdisjoint(a)]
        b=choose(bpool,seed64(dataset,participant,regime,replicate,"B"))
    elif regime=="F5":
        c=prepared if prepared is not None else candidates_f5(mask)
        a=choose(c,seed64(dataset,participant,regime,replicate,"A"))
        aset=set(a); rng=np.random.default_rng(seed64(dataset,participant,regime,replicate,"B"))
        b=None
        for _ in range(100000):
            candidate=c[int(rng.integers(0,len(c)))]
            if aset.isdisjoint(candidate):
                b=candidate;break
        if b is None:raise ValueError("no disjoint F5 B candidate found")
    elif regime=="F10":
        a=[];b=[]
        for i,st in enumerate(STRATA10):
            ea=[x for x in st if mask[x]]
            ai=choose(ea,seed64(dataset,participant,regime,replicate,f"A{i}"))
            bi=choose([x for x in ea if x!=ai],seed64(dataset,participant,regime,replicate,f"B{i}"))
            a.append(ai);b.append(bi)
        a=tuple(a);b=tuple(b)
    else: raise ValueError(regime)
    assert len(a)==len(b)==10 and set(a).isdisjoint(b)
    return a,b

def legal_summary(mask,dataset,participant):
    out={}
    for regime in ("C","F5","F10"):
        prepared=candidates_c(mask) if regime=="C" else candidates_f5(mask) if regime=="F5" else None
        pairs=[]; errors=[]
        for rep in range(NDRAWS):
            try:pairs.append(draw_pair(mask,dataset,participant,regime,rep,prepared))
            except Exception as e:errors.append(f"{type(e).__name__}:{e}")
        out[regime]={"valid_draws":len(pairs),"unique_realized_pairs":len(set(pairs)),
          "errors":sorted(set(errors)),"feasible":len(pairs)==NDRAWS}
        if regime=="C":
            c=prepared; out[regime]["arm_candidates"]=len(c)
            out[regime]["ordered_disjoint_candidate_pairs"]=sum(set(a).isdisjoint(b) for a in c for b in c)
        elif regime=="F5":
            c=prepared; out[regime]["arm_candidates"]=len(c)
            # The exact cross-product can be very large. Realized uniqueness and
            # successful construction are the legality checks; draws may reuse.
            out[regime]["ordered_disjoint_candidate_pairs"]="NOT-ENUMERATED"
        else:
            sizes=[sum(mask[x] for x in st) for st in STRATA10]
            out[regime]["eligible_per_stratum"]=sizes
            out[regime]["ordered_disjoint_candidate_pairs"]=int(np.prod([s*(s-1) for s in sizes]))
    return out

def main():
    toy=legal_summary([True]*NWIN,"ds005385","TOY-FULL-MASK")
    out=REPO/"structural";out.mkdir(exist_ok=True)
    (out/"full_mask_legality.json").write_text(json.dumps(toy,indent=2),encoding="utf-8")
    print(json.dumps(toy,indent=2))
if __name__=="__main__":main()
