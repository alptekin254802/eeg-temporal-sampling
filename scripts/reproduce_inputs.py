"""Rebuild sampling schedules and check quality-control masks against raw EEG."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from run_preregistered_analysis import (
    REPO, REPLICATION_CENSUS, validate_registered_inputs, _load_required_support,
    time_domain_qc_mask, file_digest, true_value, read_csv,
    _brainvision_references, _validate_brainvision_continuity,
)
from sampling_design import candidates_c, candidates_f5, draw_pair

REPLICATION_ROSTER = REPO / "results/preregistered/replication_qc_roster.csv"
ACCEPTED_RESULTS = REPO / "results/preregistered/scientific_results.npz"
REGIMES = ("C", "F5", "F10")


def require_files(paths, scope):
    """List all missing files before attempting to read any signals."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{scope}: {len(missing)} required file(s) missing. No files were skipped.\n"
            + "\n".join(missing)
        )


def parse_mask(text):
    if len(text) != 42 or set(text) - {"0", "1"}:
        raise ValueError("QC mask must contain exactly 42 binary cells")
    return np.asarray([value == "1" for value in text], dtype=bool)


@lru_cache(maxsize=128)
def prepared_candidates(mask_tuple):
    return candidates_c(mask_tuple), candidates_f5(mask_tuple)


def rebuild_schedule(mask, dataset, participant):
    """Rebuild a schedule with the analysis sampling function and exclusion rules."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (42,):
        raise ValueError("Expected a 42-cell mask")
    contiguous, five_bouts = prepared_candidates(tuple(mask.tolist()))
    schedule = np.zeros((3, 200, 2, 10), dtype=np.uint8)
    valid = []
    for regime_index, regime in enumerate(REGIMES):
        prepared = contiguous if regime == "C" else five_bouts if regime == "F5" else None
        count = 0
        for draw in range(200):
            try:
                pair = draw_pair(mask, dataset, participant, regime, draw, prepared)
            except (AssertionError, IndexError, ValueError):
                continue
            schedule[regime_index, draw] = pair
            count += 1
        valid.append(count)
    return (schedule if valid == [200, 200, 200] else None), valid


def load_replication_roster():
    roster = read_csv(REPLICATION_ROSTER)
    census = read_csv(REPLICATION_CENSUS)
    ids = [row["participant_id"] for row in roster]
    if len(ids) != 60 or len(set(ids)) != 60 or ids != [row["participant_id"] for row in census]:
        raise ValueError("Replication QC roster must match all 60 structural-census IDs and their order")
    if not all(true_value(row["structurally_compatible"]) for row in census):
        raise ValueError("Replication structural census contains an incompatible recording")
    for row in roster:
        mask = parse_mask(row["eligible_mask"])
        if int(row["eligible_cell_count"]) != int(mask.sum()):
            raise ValueError("Replication eligible-cell count differs: " + row["participant_id"])
    return roster, census


def verify_replication_schedule(roster, accepted_ids, accepted_schedule):
    ids = []; schedules = []
    for row in roster:
        participant = row["participant_id"]
        schedule, counts = rebuild_schedule(parse_mask(row["eligible_mask"]), "ds004148", participant)
        expected_counts = [int(row[name+"_valid_draws"]) for name in REGIMES]
        if counts != expected_counts:
            raise ValueError(f"Replication valid-draw counts differ for {participant}: {counts} != {expected_counts}")
        eligible = schedule is not None
        if eligible != true_value(row["qc_schedule_eligible"]):
            raise ValueError("Replication schedule eligibility differs: " + participant)
        expected_reason = "" if eligible else "insufficient_legal_draw_support"
        if row["exclusion_reason"] != expected_reason:
            raise ValueError("Replication exclusion reason differs: " + participant)
        if eligible:
            ids.append(participant); schedules.append(schedule)
    observed = np.asarray(schedules, dtype=np.uint8)
    if ids != list(np.asarray(accepted_ids).astype(str)):
        raise ValueError("Replication accepted participant IDs/order differ from roster reconstruction")
    if observed.shape != accepted_schedule.shape or accepted_schedule.dtype != np.uint8:
        raise ValueError("Replication accepted schedule dimensions/dtype differ")
    if not np.array_equal(observed, accepted_schedule):
        raise ValueError("Replication reconstructed schedule differs from accepted NPZ")
    return dict(replication_roster_decisions_reproduced=len(roster),
                replication_schedules_reproduced=len(ids),
                replication_schedule_comparison="exact integer equality; all strategies, 200 draws, both arms")


def verify_replication_raw(root, roster, census, participants=None):
    all_ids = {row["participant_id"] for row in roster}
    selected = set(participants) if participants else all_ids
    if not selected <= all_ids or (participants and len(participants) != len(selected)):
        raise ValueError("Requested replication IDs are unknown or duplicated")
    rows = [row for row in census if row["participant_id"] in selected]
    expected = {row["participant_id"]: row for row in roster}
    headers = [root/row["participant_id"]/"ses-session1/eeg"/
               f"{row['participant_id']}_ses-session1_task-eyesclosed_eeg.vhdr" for row in rows]
    require_files(headers, "Replication QC header preflight")
    references = [_brainvision_references(path) for path in headers]
    require_files([path for pair in references for path in pair], "Replication QC companion preflight")
    checked = []
    for row, header, (data_path, marker_path) in zip(rows, headers, references):
        participant = row["participant_id"]
        if data_path.stat().st_size != int(row["annex_signal_bytes"]) or file_digest(data_path, "md5") != row["annex_digest"]:
            raise ValueError("Replication raw identity differs: " + participant)
        _validate_brainvision_continuity(marker_path)
        data, sampling_rate = _load_required_support(header, "brainvision")
        if sampling_rate != float(row["sampling_hz"]) or sampling_rate != float(expected[participant]["sampling_hz"]):
            raise ValueError("Replication sampling rate differs: " + participant)
        mask = time_domain_qc_mask(data * 1e6, sampling_rate)
        if not np.array_equal(mask, parse_mask(expected[participant]["eligible_mask"])):
            raise ValueError("Replication raw QC mask differs: " + participant)
        checked.append(participant)
    return dict(replication_raw_QC_scope="all_60_recordings" if participants is None else "selected_participants_only",
                replication_raw_QC_masks_reproduced=len(checked), replication_raw_QC_participants=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", action="store_true", help="Regenerate every primary schedule from published masks and seeds")
    parser.add_argument("--primary-root", type=Path, help="Check all 536 structurally eligible recordings against registered QC masks (531 analysis files are insufficient)")
    parser.add_argument("--replication-schedule", action="store_true", help="Rebuild all 60 roster decisions and all 59 accepted schedules from released QC masks")
    parser.add_argument("--replication-root", type=Path, help="Also derive replication QC from raw EEG and compare with all 60 released masks")
    parser.add_argument("--participants", nargs="+", help="Restrict only replication raw QC to these IDs; explicitly reports partial coverage")
    args = parser.parse_args()
    if args.participants and not args.replication_root:
        parser.error("--participants requires --replication-root; schedule reconstruction always covers the full roster")
    if not any((args.schedule, args.primary_root, args.replication_schedule, args.replication_root)):
        parser.error("Choose --schedule, --primary-root, --replication-schedule and/or --replication-root")
    inputs = validate_registered_inputs(); done = {}
    # Fail before any signal processing or schedule work if primary files are absent.
    primary_qc = [row for row in inputs["census"] if true_value(row["structurally_eligible"])]
    if args.primary_root:
        require_files([args.primary_root/row["target_recording"] for row in primary_qc],
                      "Primary QC requires all 536 structurally eligible recordings; the 531-record analysis subset is insufficient")
    if args.schedule:
        for participant, mask, accepted in zip(inputs["participant_ids"], inputs["eligible_masks"], inputs["schedule"]):
            regenerated, _ = rebuild_schedule(mask, "ds005385", participant)
            if regenerated is None or not np.array_equal(regenerated, accepted):
                raise ValueError("Primary schedule differs: " + participant)
        done["primary_schedules_reproduced"] = len(inputs["participant_ids"])
    if args.primary_root:
        for row in primary_qc:
            path = args.primary_root/row["target_recording"]
            if path.stat().st_size != int(row["annex_bytes"]) or file_digest(path) != row["annex_digest"]:
                raise ValueError("Input identity differs: " + row["participant_id"])
            data, sampling_rate = _load_required_support(path, "edf")
            mask = time_domain_qc_mask(data*1e6, sampling_rate)
            if not np.array_equal(mask, parse_mask(row["qc_eligible_mask"])):
                raise ValueError("Primary QC differs: " + row["participant_id"])
        done["primary_QC_masks_reproduced"] = len(primary_qc)
    if args.replication_schedule or args.replication_root:
        roster, census = load_replication_roster()
        with np.load(ACCEPTED_RESULTS, allow_pickle=False) as archive:
            done.update(verify_replication_schedule(roster, archive["replication_participant_ids"], archive["replication_schedule"]))
        if args.replication_root:
            done.update(verify_replication_raw(args.replication_root, roster, census, args.participants))
    print(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
