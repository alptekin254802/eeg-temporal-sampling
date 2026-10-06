#!/usr/bin/env python3
"""Time-domain QC for structurally eligible ds005385 files.

Each raw EDF is downloaded, hash/size verified, read only for prespecified time-domain
support flags, then deleted. No frequency-domain quantity is computed.
"""
from __future__ import annotations
import csv, hashlib, json, os, threading, time, urllib.request
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import mne
import numpy as np
from sampling_design import legal_summary

REPO=Path(__file__).resolve().parents[1]
CENSUS=REPO/"DS005385_PARTICIPANT_CENSUS.csv"
CACHE=REPO/"raw_cache"/"ds005385"
OUT=REPO/"structural"
TRANSFER_LEDGER=OUT/"raw_download_ledger.jsonl"
TRANSFER_LOCK=threading.Lock()
URL="https://s3.amazonaws.com/openneuro.org/ds005385/"
REQ=("O1","Oz","O2","TP9","TP10")
# Prespecified physical/time-domain thresholds.
ABS_UV=500.0
P2P_UV=1000.0
STEP_UV=200.0
FLAT_SD_UV=0.1
IDENTICAL_RUN_MS=100.0

def download(row):
    CACHE.mkdir(parents=True,exist_ok=True)
    target=CACHE/(row["participant_id"]+".edf")
    expected=int(row["annex_bytes"])
    transferred=False
    if not target.exists() or target.stat().st_size!=expected:
        tmp=target.with_suffix(".part")
        last=None
        for attempt in range(5):
            try:
                req=urllib.request.Request(URL+row["target_recording"],headers={"User-Agent":"resting-eeg-temporal-sampling-qc/1.0"})
                with urllib.request.urlopen(req,timeout=180) as src,tmp.open("wb") as dst:
                    while True:
                        b=src.read(1024*1024)
                        if not b:break
                        dst.write(b)
                last=None;break
            except Exception as e:
                last=e
                if tmp.exists():tmp.unlink()
                time.sleep(2**attempt)
        if last is not None:raise last
        if tmp.stat().st_size!=expected: raise RuntimeError(f"size {tmp.stat().st_size}!={expected}")
        tmp.replace(target)
        transferred=True
    h=hashlib.sha256()
    with target.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    if h.hexdigest()!=row["annex_digest"]:raise RuntimeError("sha256 mismatch")
    if transferred:
        record={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
          "participant_id":row["participant_id"],"bytes":expected,
          "sha256":row["annex_digest"],"source":"network_transfer_verified"}
        with TRANSFER_LOCK:
            with TRANSFER_LEDGER.open("a",encoding="utf-8") as f:
                f.write(json.dumps(record,sort_keys=True)+"\n")
    return target

def longest_equal_run(x):
    if x.size<2:return x.size
    d=np.diff(x); idx=np.flatnonzero(d!=0)
    if idx.size==0:return x.size
    bounds=np.concatenate(([-1],idx,[x.size-1]))
    return int(np.diff(bounds).max())

def process(row):
    p=None
    try:
        p=download(row)
        raw=mne.io.read_raw_edf(p,preload=False,verbose="ERROR")
        missing=[x for x in REQ if x not in raw.ch_names]
        if missing:raise RuntimeError("missing:"+",".join(missing))
        sf=float(raw.info["sfreq"]); start=int(round(8*sf)); n=int(round(168*sf))
        data=raw.get_data(picks=list(REQ),start=start,stop=start+n)*1e6
        if data.shape!=(5,n):raise RuntimeError(f"support_shape:{data.shape}")
        mast=(data[3]+data[4])/2.0
        post=data[:3]-mast
        mask=[]; reasons=[]
        run_n=int(round(IDENTICAL_RUN_MS*sf/1000))
        for wi in range(42):
            a=int(wi*4*sf);b=int((wi+1)*4*sf)
            raww=data[:,a:b]; postw=post[:,a:b]; why=[]
            # EDF physical extrema are source-documented as invalid and can
            # induce large constant offsets after scaling. Gross amplitude is
            # therefore assessed after per-window median removal;
            # peak-to-peak and first differences are offset-invariant.
            centered=postw-np.median(postw,axis=1,keepdims=True)
            if not np.isfinite(raww).all():why.append("nonfinite")
            if np.nanmax(np.abs(centered))>ABS_UV:why.append("centered_abs_gt_500uV")
            if np.nanmax(np.ptp(postw,axis=1))>P2P_UV:why.append("p2p_gt_1000uV")
            if np.nanmax(np.abs(np.diff(postw,axis=1)))>STEP_UV:why.append("step_gt_200uV")
            if np.nanmin(np.std(raww,axis=1))<FLAT_SD_UV:why.append("flat_sd_lt_0.1uV")
            if any(longest_equal_run(ch)>=run_n for ch in raww):why.append("identical_run_ge_100ms")
            mask.append(not why);reasons.append("|".join(why))
        legal=legal_summary(mask,"ds005385",row["participant_id"])
        qc=all(legal[r]["feasible"] for r in ("C","F5","F10"))
        return {"participant_id":row["participant_id"],"downloaded_bytes":p.stat().st_size,
          "sfreq":sf,"eligible_window_count":sum(mask),"eligible_mask":"".join("1" if x else "0" for x in mask),
          "window_reasons":";".join(f"{i}:{x}" for i,x in enumerate(reasons) if x),
          "C_valid_draws":legal["C"]["valid_draws"],"C_unique_pairs":legal["C"]["unique_realized_pairs"],
          "F5_valid_draws":legal["F5"]["valid_draws"],"F5_unique_pairs":legal["F5"]["unique_realized_pairs"],
          "F10_valid_draws":legal["F10"]["valid_draws"],"F10_unique_pairs":legal["F10"]["unique_realized_pairs"],
          "qc_eligible":qc,"qc_exclusion_reason":"" if qc else "insufficient_legal_draw_support",
          "error":""}
    except Exception as e:
        return {"participant_id":row["participant_id"],"downloaded_bytes":p.stat().st_size if p and p.exists() else 0,
          "sfreq":"","eligible_window_count":0,"eligible_mask":"","window_reasons":"",
          "C_valid_draws":0,"C_unique_pairs":0,"F5_valid_draws":0,"F5_unique_pairs":0,
          "F10_valid_draws":0,"F10_unique_pairs":0,"qc_eligible":False,
          "qc_exclusion_reason":"qc_read_or_processing_error","error":f"{type(e).__name__}:{e}"}
    finally:
        if p and p.exists():p.unlink()

def main():
    with CENSUS.open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
    targets=[r for r in rows if r["structurally_eligible"]=="True"]
    final=OUT/"ds005385_qc_results.csv"
    if final.exists():
        with final.open(encoding="utf-8",newline="") as f:completed=list(csv.DictReader(f))
        if len(completed)!=len(targets) or {r["participant_id"] for r in completed}!={r["participant_id"] for r in targets}:
            raise RuntimeError("Existing final QC roster does not match structural targets")
        summary={"structural_n":len(targets),
          "qc_n":sum(r["qc_eligible"]=="True" for r in completed),
          "unique_raw_bytes_accessed":sum(int(r["downloaded_bytes"]) for r in completed),
          "errors":sum(bool(r["error"]) for r in completed)}
        print(json.dumps(summary,indent=2))
        return
    done={}
    partial=OUT/"ds005385_qc_results.partial.csv"
    if partial.exists():
        with partial.open(encoding="utf-8",newline="") as f:done={r["participant_id"]:r for r in csv.DictReader(f)}
    todo=[r for r in targets if r["participant_id"] not in done]
    fields=None
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(process,r):r for r in todo}
        for i,fut in enumerate(as_completed(futures),1):
            z=fut.result();done[z["participant_id"]]=z;fields=list(z)
            ordered=[done[k] for k in sorted(done)]
            with partial.open("w",encoding="utf-8",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(ordered)
            print(f"{len(done)}/{len(targets)} {z['participant_id']} eligible={z['qc_eligible']}",flush=True)
    partial.replace(final)
    total=sum(int(r["downloaded_bytes"]) for r in done.values())
    summary={"structural_n":len(targets),"qc_n":sum(str(r["qc_eligible"])=="True" or r["qc_eligible"] is True for r in done.values()),
      "raw_bytes_downloaded":total,"errors":sum(bool(r["error"]) for r in done.values())}
    (OUT/"ds005385_qc_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
