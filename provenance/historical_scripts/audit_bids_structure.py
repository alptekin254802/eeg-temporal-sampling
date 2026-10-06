#!/usr/bin/env python3
"""Structural BIDS audit for the temporal-sampling study.

Downloads EDF/BrainVision header/marker bytes only. It never reads signal samples
and never computes a spectral feature.
"""
from __future__ import annotations
import csv, json, re, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ON = REPO / "evidence" / "openneuro"
D1 = next((ON / "ds005385_snapshot").iterdir())
D2 = next((ON / "ds004148_snapshot").iterdir())
OUT = REPO / "structural"
HDR = ON / "ds005385_edf_headers"
BV = ON / "ds004148_brainvision_headers"
D1_COMMIT = "6a558cd5852503e66df9ac7fdac2f3a7f4ed5f12"
D2_COMMIT = "0c740d838a2c33feeec0b6e514aea323d0e65ca4"

def pointer_info(path: Path):
    if not path.exists():
        return None, None
    s = path.read_text(encoding="utf-8").strip()
    m = re.search(r"(?:SHA256|MD5)E-s(\d+)--([0-9a-f]+)", s)
    return (int(m.group(1)), m.group(2)) if m else (path.stat().st_size, None)

def get_range(url: str, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{end}", "User-Agent": "resting-eeg-temporal-sampling-structural-audit/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
        if len(data) > end + 1:
            raise RuntimeError(f"Server ignored Range for {url}: {len(data)} bytes")
        return data

def cache_edf_header(participant: str, rel: str) -> Path:
    HDR.mkdir(parents=True, exist_ok=True)
    out = HDR / f"{participant}.edf.header"
    if out.exists() and out.stat().st_size >= 256:
        return out
    url = "https://s3.amazonaws.com/openneuro.org/ds005385/" + rel.replace("\\", "/")
    first = get_range(url, 255)
    if len(first) != 256:
        raise RuntimeError(f"Short EDF fixed header: {participant} {len(first)}")
    hbytes = int(first[184:192].decode("ascii").strip())
    full = get_range(url, hbytes - 1)
    if len(full) != hbytes:
        raise RuntimeError(f"Short EDF header: {participant} {len(full)} != {hbytes}")
    out.write_bytes(full)
    return out

def parse_edf_header(path: Path):
    b = path.read_bytes()
    hbytes = int(b[184:192].decode("ascii").strip())
    reserved = b[192:236].decode("latin1").strip()
    nrec = int(b[236:244].decode("ascii").strip())
    recdur = float(b[244:252].decode("ascii").strip())
    nsig = int(b[252:256].decode("ascii").strip())
    off = 256
    labels = [b[off+i*16:off+(i+1)*16].decode("latin1").strip() for i in range(nsig)]
    off += 16 * nsig
    off += 80 * nsig + 8 * nsig + 8 * nsig + 8 * nsig + 8 * nsig + 8 * nsig + 80 * nsig
    samples = [int(b[off+i*8:off+(i+1)*8].decode("ascii").strip()) for i in range(nsig)]
    return {
        "header_bytes": hbytes, "reserved": reserved, "data_records": nrec,
        "record_duration_s": recdur, "actual_duration_s": nrec * recdur,
        "signal_count": nsig, "labels": labels, "samples_per_record": samples,
        "sample_rates": [x / recdur for x in samples],
    }

def read_tsv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def audit_ds005385():
    targets = sorted(D1.glob("sub-*/ses-1/eeg/*task-EyesClosed_acq-pre_eeg.json"))
    def prefetch(js):
        pid = js.parts[-4]
        eeg = js.with_name(js.name.replace("_eeg.json", "_eeg.edf"))
        return cache_edf_header(pid, eeg.relative_to(D1).as_posix())
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(prefetch, targets))
    rows = []
    for js in targets:
        pid = js.parts[-4]
        base = js.name.replace("_eeg.json", "")
        eeg = js.with_name(base + "_eeg.edf")
        ch = js.with_name(base + "_channels.tsv")
        ev = js.with_name(base + "_events.tsv")
        side = json.loads(js.read_text(encoding="utf-8"))
        channels = [r["name"] for r in read_tsv(ch)]
        events = read_tsv(ev)
        types = [r.get("type", "") for r in events]
        if set(types) == {"boundary"}:
            event_class = "ONLY-INITIAL-BOUNDARY"
        elif set(types) <= {"boundary", "no USB Connection to actiCAP"}:
            event_class = "USB-DISCONNECT-ANNOTATION"
        else:
            event_class = "OTHER-TASK-MARKERS"
        size, digest = pointer_info(eeg)
        rel = eeg.relative_to(D1).as_posix()
        hp = cache_edf_header(pid, rel)
        eh = parse_edf_header(hp)
        continuous = "EDF+D" not in eh["reserved"] and side.get("RecordingType") == "continuous"
        actual = eh["actual_duration_s"]
        roi = all(x in channels for x in ("O1", "Oz", "O2"))
        eligible = bool(size and roi and continuous and actual >= 180 and event_class == "ONLY-INITIAL-BOUNDARY")
        rows.append({
            "participant_id": pid, "target_recording": rel, "snapshot_commit": D1_COMMIT,
            "snapshot_version": "1.0.3", "annex_bytes": size, "annex_digest": digest,
            "bids_duration_s": side.get("RecordingDuration"), "edf_actual_duration_s": actual,
            "edf_data_records": eh["data_records"], "edf_record_duration_s": eh["record_duration_s"],
            "edf_header_bytes": eh["header_bytes"], "edf_reserved": eh["reserved"],
            "edf_continuous": continuous, "edf_signal_count": eh["signal_count"],
            "bids_sampling_hz": side.get("SamplingFrequency"), "edf_eeg_sampling_hz": eh["sample_rates"][0],
            "reference": side.get("EEGReference"), "bids_eeg_channel_count": side.get("EEGChannelCount"),
            "channels_tsv_count": len(channels), "has_O1": "O1" in channels,
            "has_Oz": "Oz" in channels, "has_O2": "O2" in channels,
            "required_posterior_channels": roi, "event_count": len(events),
            "event_class": event_class, "initial_boundary_at_1s": any(r.get("type")=="boundary" and float(r.get("onset", -1))==1 for r in events),
            "usb_disconnect_annotation": "no USB Connection to actiCAP" in types,
            "other_task_markers": event_class == "OTHER-TASK-MARKERS",
            "duration_ge_180s": actual >= 180, "structurally_eligible": eligible,
            "structural_exclusion_reason": "" if eligible else ";".join(
                x for x, yes in [
                    ("missing_raw_pointer", not size), ("missing_posterior_roi", not roi),
                    ("discontinuous_edf", not continuous), ("duration_lt_180s", actual < 180),
                    ("usb_disconnect_annotation", event_class=="USB-DISCONNECT-ANNOTATION"),
                    ("other_task_markers", event_class=="OTHER-TASK-MARKERS")
                ] if yes),
            "qc_status": "NOT-COMPUTED", "qc_eligible": "",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    path = REPO / "DS005385_PARTICIPANT_CENSUS.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = {
        "n_rows": len(rows), "n_structurally_eligible": sum(r["structurally_eligible"] for r in rows),
        "n_required_posterior": sum(r["required_posterior_channels"] for r in rows),
        "n_duration_ge_180": sum(r["duration_ge_180s"] for r in rows),
        "n_event_compatible": sum(r["event_class"]=="ONLY-INITIAL-BOUNDARY" for r in rows),
        "event_classes": Counter(r["event_class"] for r in rows),
        "duration_min": min(r["edf_actual_duration_s"] for r in rows),
        "duration_max": max(r["edf_actual_duration_s"] for r in rows),
        "header_bytes_downloaded": sum(p.stat().st_size for p in HDR.glob("*.header")),
    }
    (OUT / "ds005385_summary.json").write_text(json.dumps(summary, indent=2, default=dict), encoding="utf-8")
    return summary

def cache_bv_file(rel: str, out: Path):
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = "https://s3.amazonaws.com/openneuro.org/ds004148/" + rel
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"resting-eeg-temporal-sampling-structural-audit/1.0"}), timeout=60) as r:
        data = r.read()
    if len(data) > 2_000_000:
        raise RuntimeError(f"Unexpectedly large BrainVision header/marker {rel}: {len(data)}")
    out.write_bytes(data)
    return out

def audit_ds004148():
    targets=sorted(D2.glob("sub-*/ses-session1/eeg/*task-eyesclosed_eeg.json"))
    def prefetch(js):
        pid=js.parts[-4]; base=js.name.replace("_eeg.json","")
        vh=js.with_name(base+"_eeg.vhdr"); vm=js.with_name(base+"_eeg.vmrk")
        cache_bv_file(vh.relative_to(D2).as_posix(), BV/pid/vh.name)
        cache_bv_file(vm.relative_to(D2).as_posix(), BV/pid/vm.name)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(prefetch, targets))
    rows=[]
    for js in targets:
        pid=js.parts[-4]; base=js.name.replace("_eeg.json","")
        side=json.loads(js.read_text(encoding="utf-8"))
        channels=[r["name"] for r in read_tsv(js.with_name(base+"_channels.tsv"))]
        events=read_tsv(js.with_name(base+"_events.tsv"))
        eeg=js.with_name(base+"_eeg.eeg"); vh=js.with_name(base+"_eeg.vhdr"); vm=js.with_name(base+"_eeg.vmrk")
        esize,edigest=pointer_info(eeg); vsize,_=pointer_info(vh); msize,_=pointer_info(vm)
        relvh=vh.relative_to(D2).as_posix(); relvm=vm.relative_to(D2).as_posix()
        vhlocal=cache_bv_file(relvh, BV/pid/vh.name); vmlocal=cache_bv_file(relvm, BV/pid/vm.name)
        vht=vhlocal.read_text(encoding="latin1")
        datapoints=re.search(r"DataPoints=(\d+)",vht)
        interval=re.search(r"SamplingInterval=([0-9.]+)",vht)
        nchan=re.search(r"NumberOfChannels=(\d+)",vht)
        hz=1_000_000/float(interval.group(1)) if interval else side.get("SamplingFrequency")
        duration=(int(datapoints.group(1))/hz) if datapoints else side.get("RecordingDuration")
        vmt=vmlocal.read_text(encoding="latin1")
        new_segments=len(re.findall(r"New Segment",vmt,re.I))
        roi=all(x in channels for x in ("O1","Oz","O2"))
        compatible=bool(esize and duration>=180 and roi and new_segments<=1)
        rows.append({"participant_id":pid,"snapshot_commit":D2_COMMIT,"snapshot_version":"1.0.0",
          "annex_signal_bytes":esize,"annex_digest":edigest,"vhdr_bytes":vsize,"vmrk_bytes":msize,
          "bids_duration_s":side.get("RecordingDuration"),"brainvision_duration_s":duration,
          "sampling_hz":hz,"reference":side.get("EEGReference"),"json_eeg_channels":side.get("EEGChannelCount"),
          "channels_tsv_count":len(channels),"vhdr_channel_count":int(nchan.group(1)) if nchan else None,
          "has_O1":"O1" in channels,"has_Oz":"Oz" in channels,"has_O2":"O2" in channels,
          "required_posterior_channels":roi,"event_count":len(events),
          "sync_status_event_count":sum(r.get("type")=="SyncStatus" for r in events),
          "brainvision_new_segment_markers":new_segments,"structurally_compatible":compatible})
    path=OUT/"ds004148_session1_census.csv"; OUT.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"n":len(rows),"compatible":sum(r["structurally_compatible"] for r in rows),
      "durations":Counter(r["brainvision_duration_s"] for r in rows),
      "channel_counts":Counter(r["channels_tsv_count"] for r in rows),
      "header_marker_bytes_downloaded":sum(p.stat().st_size for p in BV.rglob("*") if p.is_file())}
    (OUT/"ds004148_summary.json").write_text(json.dumps(summary,indent=2,default=dict),encoding="utf-8")
    return summary

if __name__=="__main__":
    print(json.dumps({"ds005385":audit_ds005385(),"ds004148":audit_ds004148()},indent=2,default=dict))
