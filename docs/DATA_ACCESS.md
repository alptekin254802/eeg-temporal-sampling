# External data and input layout

Use the versioned OpenNeuro releases:

| Role | Release | DOI |
|---|---|---|
| Primary | [ds005385, 1.0.3](https://openneuro.org/datasets/ds005385/versions/1.0.3) | [10.18112/openneuro.ds005385.v1.0.3](https://doi.org/10.18112/openneuro.ds005385.v1.0.3) |
| Replication | [ds004148, 1.0.0](https://openneuro.org/datasets/ds004148/versions/1.0.0) | [10.18112/openneuro.ds004148.v1.0.0](https://doi.org/10.18112/openneuro.ds004148.v1.0.0) |

Download the required recordings through the corresponding version page. Keep the BIDS paths relative to each dataset root. A dataset Git checkout may contain annex pointers; the analysis requires the actual recording payloads. File byte counts and content digests in the supplied censuses distinguish pointers or wrong versions from the expected recordings.

For example, a primary dataset root contains:

```text
sub-001/ses-1/eeg/sub-001_ses-1_task-EyesClosed_acq-pre_eeg.edf
```

`osf_upload_package/DS005385_PARTICIPANT_CENSUS.csv` supplies `target_recording`, `annex_bytes`, and SHA-256 `annex_digest` for every candidate. The primary analysis requires the 531 rows with `final_eligible=True`; checking the original time-domain masks requires all 536 rows with `structurally_eligible=True`. The structural inventory covers 608 nominal participants. Primary schedules and cohort selection are supplied with the supporting pre-analysis materials; the raw-data analysis command does not repeat the metadata-only structural census. Inclusion of those materials in the OSF registration snapshot has not been verified.

A replication root contains, for every `sub-01` through `sub-60`:

```text
sub-01/ses-session1/eeg/sub-01_ses-session1_task-eyesclosed_eeg.vhdr
```

Retain the data and marker companions referenced by each header's `DataFile` and `MarkerFile` fields in the same directory. Their filenames must match the header. `structural/ds004148_session1_census.csv` supplies signal byte counts and MD5 digests. The analysis script reads all 60 session-1 recordings and reconstructs the 59-participant eligible sample. The released per-recording QC masks and decisions are in `results/preregistered/replication_qc_roster.csv`.

The computation reads [8,176) s from O1, Oz, O2, TP9 and TP10 in recordings at least 180 s long. Native rates are 1000 Hz (primary) and 500 Hz (replication). The released estimator applies the common physical spectral grid without filtering or resampling.

The small derived arrays support verification and figure regeneration immediately. Raw EEG is obtained from the versioned data providers. The selected `data/*/participants.tsv` files contain only participant ID, age and source-recorded sex for the analyzed samples. Their source file hashes and selection rules are recorded in `data/metadata_provenance.json`; field meanings and dataset license/attribution are in the adjacent JSON files.

## Further input checks

The [computational conventions](COMPUTATIONAL_CONVENTIONS.md) describe the random
seeds, quality-control thresholds, sampling strategies and verification commands.
`reproduce_inputs.py --replication-schedule` reconstructs the eligibility decisions
for all 60 recordings and the schedules for the 59 included participants.
`--replication-root` also checks quality-control masks against raw EEG; add
`--participants` to check raw EEG for selected participants only. Primary raw
quality control requires all 536 structurally eligible recordings. If any files
are missing, the command lists them and stops before processing the signals.

`verify_structural_census.py` uses metadata and EDF headers to reconstruct the
original selection of 536 recordings from 608 participants. It also checks event
counts and declared onset units. The original `ONLY-INITIAL-BOUNDARY` label was
based on event types alone, so it does not establish that a recording contains
only an initial event. These checks leave the original census unchanged; the
physical meaning of the boundary annotations remains unresolved.

The original scripts are preserved in `provenance/historical_scripts/`. They
require the original directory layout; use the commands above for this repository.
