"""Reconstruct the original primary census and check its event annotations.

Reads only a metadata checkout with annex pointers and separately supplied EDF
headers. It does not download data, read signal samples, or revise the census.
"""
import argparse
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CENSUS = ROOT / "osf_upload_package/DS005385_PARTICIPANT_CENSUS.csv"


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def historical_event_class(events):
    """Apply the original event-type rule, which does not check counts or onsets."""
    types = {row.get("type", "") for row in events}
    if types == {"boundary"}:
        return "ONLY-INITIAL-BOUNDARY"
    if types <= {"boundary", "no USB Connection to actiCAP"}:
        return "USB-DISCONNECT-ANNOTATION"
    return "OTHER-TASK-MARKERS"


def event_diagnostics(events, metadata, sampling_hz):
    """Check boundary counts and onsets separately from the original inclusion rule."""
    unit = str(metadata.get("onset", {}).get("Units", "")).strip()
    scale = {"s": 1., "sec": 1., "seconds": 1., "ms": .001,
             "millisecond": .001, "milliseconds": .001}.get(unit.lower())
    boundaries = [row for row in events if row.get("type") == "boundary"]
    raw = [row.get("onset", "") for row in boundaries]
    seconds = None
    if scale is not None:
        try:
            seconds = [float(value) * scale for value in raw]
            if not all(math.isfinite(value) for value in seconds):
                seconds = None
        except (TypeError, ValueError):
            seconds = None
    initial_limit = 1. / float(sampling_hz)
    strict = (len(events) == 1 and len(boundaries) == 1 and seconds is not None
              and 0 <= seconds[0] <= initial_limit)
    return dict(
        onset_unit=unit or "UNDECLARED", boundary_count=len(boundaries),
        boundary_onsets_declared=raw, boundary_onsets_seconds=seconds,
        single_initial_boundary_diagnostic=bool(strict) if seconds is not None else None,
        diagnostic_initial_interval_seconds=[0., initial_limit],
        boundary_onsets_in_support_seconds=None if seconds is None else [t for t in seconds if 8 <= t < 176],
    )


def pointer_info(path):
    # Never follow a payload symlink or decode a signal file as text.
    if path.is_symlink():
        text = os.readlink(path)
    elif path.is_file() and path.stat().st_size <= 8192:
        text = path.read_text(encoding="utf-8").strip()
    else:
        raise ValueError(f"Expected a small annex pointer in the metadata checkout: {path}")
    match = re.search(r"SHA256E-s(\d+)--([0-9a-f]{64})", text)
    if not match:
        raise ValueError(f"No SHA256 annex pointer identity: {path}")
    return int(match.group(1)), match.group(2), text


def parse_edf_header(path):
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Short EDF fixed header: {path}")
        hbytes = int(fixed[184:192])
        nsig = int(fixed[252:256])
        if hbytes != 256 + 256 * nsig or nsig < 1:
            raise ValueError(f"Invalid EDF header dimensions: {path}")
        content = fixed + stream.read(hbytes - 256)
    if len(content) != hbytes or path.stat().st_size != hbytes:
        raise ValueError(f"Supply header bytes only, exactly {hbytes} bytes: {path}")
    nrec = int(fixed[236:244]); recdur = float(fixed[244:252])
    if nrec < 0 or recdur <= 0:
        raise ValueError(f"Unsupported EDF record count/duration: {path}")
    offset = 256 + nsig * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    samples = [int(content[offset+i*8:offset+(i+1)*8]) for i in range(nsig)]
    return dict(header_bytes=hbytes, reserved=fixed[192:236].decode("latin1").strip(),
                data_records=nrec, record_duration_s=recdur, actual_duration_s=nrec*recdur,
                signal_count=nsig, sample_rates=[n/recdur for n in samples])


def same_value(actual, expected):
    if str(actual) == expected:
        return True
    try:
        return Decimal(str(actual)) == Decimal(expected)
    except InvalidOperation:
        return False


def audit_record(sidecar, snapshot_root, header_root):
    pid = sidecar.parts[-4]
    base = sidecar.name.removesuffix("_eeg.json")
    paths = {kind: sidecar.with_name(base+suffix) for kind, suffix in (
        ("pointer", "_eeg.edf"), ("channels", "_channels.tsv"),
        ("events", "_events.tsv"), ("event_units", "_events.json"))}
    paths["sidecar"] = sidecar
    paths["header"] = header_root / f"{pid}.edf.header"
    side = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    channels = [row["name"] for row in read_tsv(paths["channels"])]
    events = read_tsv(paths["events"])
    metadata = json.loads(paths["event_units"].read_text(encoding="utf-8-sig")) if paths["event_units"].is_file() else {}
    size, digest, pointer_text = pointer_info(paths["pointer"])
    header = parse_edf_header(paths["header"])
    event_class = historical_event_class(events)
    continuous = "EDF+D" not in header["reserved"] and side.get("RecordingType") == "continuous"
    roi = all(name in channels for name in ("O1", "Oz", "O2"))
    duration = header["actual_duration_s"]
    eligible = bool(size and roi and continuous and duration >= 180 and event_class == "ONLY-INITIAL-BOUNDARY")
    reasons = [("missing_raw_pointer", not size), ("missing_posterior_roi", not roi),
               ("discontinuous_edf", not continuous), ("duration_lt_180s", duration < 180),
               ("usb_disconnect_annotation", event_class == "USB-DISCONNECT-ANNOTATION"),
               ("other_task_markers", event_class == "OTHER-TASK-MARKERS")]
    record = dict(participant_id=pid, target_recording=paths["pointer"].relative_to(snapshot_root).as_posix(),
        annex_bytes=size, annex_digest=digest, bids_duration_s=side.get("RecordingDuration"),
        edf_actual_duration_s=duration, edf_data_records=header["data_records"],
        edf_record_duration_s=header["record_duration_s"], edf_header_bytes=header["header_bytes"],
        edf_reserved=header["reserved"], edf_continuous=continuous, edf_signal_count=header["signal_count"],
        bids_sampling_hz=side.get("SamplingFrequency"), edf_eeg_sampling_hz=header["sample_rates"][0],
        reference=side.get("EEGReference"), bids_eeg_channel_count=side.get("EEGChannelCount"),
        channels_tsv_count=len(channels), has_O1="O1" in channels, has_Oz="Oz" in channels, has_O2="O2" in channels,
        required_posterior_channels=roi, event_count=len(events), event_class=event_class,
        initial_boundary_at_1s=any(row.get("type") == "boundary" and float(row.get("onset", -1)) == 1 for row in events),
        usb_disconnect_annotation=any(row.get("type") == "no USB Connection to actiCAP" for row in events),
        other_task_markers=event_class == "OTHER-TASK-MARKERS", duration_ge_180s=duration >= 180,
        structurally_eligible=eligible, structural_exclusion_reason="" if eligible else ";".join(name for name, bad in reasons if bad))
    diagnostic = dict(participant_id=pid, historical_structurally_eligible=eligible,
                      **event_diagnostics(events, metadata, header["sample_rates"][0]))
    identities = []
    for role, path in paths.items():
        if role == "pointer":
            payload = pointer_text.encode("utf-8")
        elif path.is_file():
            payload = path.read_bytes()
        else:
            continue
        identities.append(dict(participant_id=pid, role=role,
                               path=path.name if role == "header" else path.relative_to(snapshot_root).as_posix(),
                               sha256=hashlib.sha256(payload).hexdigest(),
                               identity_kind="pointer_text_utf8" if role == "pointer" else "file_bytes"))
    return record, diagnostic, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True, help="ds005385 1.0.3 metadata checkout containing annex pointers")
    parser.add_argument("--edf-header-root", type=Path, required=True, help="Read-only directory of sub-XXX.edf.header files, fixed+signal headers only")
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--participants", nargs="+", help="Optional bounded diagnostic; omitted means the full 608-record inventory")
    parser.add_argument("--output", type=Path, required=True, help="New empty directory for diagnostic reports, never the accepted census")
    args = parser.parse_args()
    with args.census.open(encoding="utf-8", newline="") as stream:
        expected = list(csv.DictReader(stream))
    ids = [row["participant_id"] for row in expected]
    if len(ids) != 608 or len(set(ids)) != len(ids):
        raise ValueError("Expected 608 unique accepted census records")
    wanted = set(args.participants) if args.participants else set(ids)
    if not wanted <= set(ids):
        raise ValueError(f"Unknown participant IDs: {sorted(wanted-set(ids))}")
    targets = sorted(args.snapshot_root.glob("sub-*/ses-1/eeg/*task-EyesClosed_acq-pre_eeg.json"))
    found = [path.parts[-4] for path in targets]
    if len(found) != len(set(found)) or set(found) != set(ids):
        raise ValueError("Metadata participant inventory differs from the accepted 608-record census")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    accepted = {row["participant_id"]: row for row in expected}
    records = []; diagnostics = []; identities = []; mismatches = []
    for sidecar in targets:
        if sidecar.parts[-4] not in wanted:
            continue
        record, diagnostic, evidence = audit_record(sidecar, args.snapshot_root, args.edf_header_root)
        diagnostic["accepted_final_eligible"] = accepted[record["participant_id"]]["final_eligible"].lower() == "true"
        records.append(record); diagnostics.append(diagnostic); identities.extend(evidence)
        for field, actual in record.items():
            if not same_value(actual, accepted[record["participant_id"]][field]):
                mismatches.append(dict(participant_id=record["participant_id"], field=field,
                                       actual=actual, expected=accepted[record["participant_id"]][field]))
    summary = dict(scope="full_inventory" if args.participants is None else "selected_participants",
        records_checked=len(records), historical_structurally_eligible=sum(row["structurally_eligible"] for row in records),
        historical_event_classes=dict(Counter(row["event_class"] for row in records)),
        historical_census_matches=not mismatches, mismatches=mismatches,
        eligible_event_diagnostic_alerts=[row for row in diagnostics if row["historical_structurally_eligible"]
                                         and row["single_initial_boundary_diagnostic"] is not True],
        accepted_primary_event_diagnostic_alerts=[row for row in diagnostics if row["accepted_final_eligible"]
                                                and row["single_initial_boundary_diagnostic"] is not True],
        interpretation="Historical replay only. Event diagnostics do not establish physical discontinuity or change the accepted sample.",
        accepted_census_sha256=hashlib.sha256(args.census.read_bytes()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [("summary.json", summary), ("event_diagnostics.json", diagnostics), ("input_identities.json", identities)]:
        (args.output/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    with (args.output/"historical_structural_replay.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(json.dumps(summary, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
