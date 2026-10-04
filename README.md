# Temporal Sampling Geometry and Participant-Level Reproducibility in Fixed-Duration Resting EEG

Computational companion for the study of how temporal placement of equal-duration resting EEG changes the reproducibility of participant ordering. Each sample retains ten 4-s periodograms (40 s) from a common 42-cell support. The measured feature is posterior relative alpha at O1/Oz/O2 after a mean TP9/TP10 reference. Paired samples come from the same recording.

Study registration: [OSF nh4d8](https://osf.io/nh4d8/). Supporting pre-analysis materials: [OSF g2apc](https://osf.io/g2apc/). The project materials are cited as a separate supporting record; their inclusion in the registration snapshot has not been verified. The registered primary, secondary, complete-support and separate replication analyses are distinguished from the exploratory temporal-order control and cell-band-power simulations. The exploratory settings were recorded on 20 September 2026 after the original results were known.

## Install and check

Run commands from this directory. The reference environment uses CPython 3.14.3 and the package versions in `requirements.txt`. Creating a virtual environment is recommended:

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell, then run:

```text
python -m pip install -r requirements.txt
python -B scripts/verify_release.py
python -B -m unittest discover -s tests -v
python -B scripts/run_preregistered_analysis.py
python -B scripts/verify_results.py --output outputs/result_checks.json
```

The first verifier checks release-file identity. The analysis command, without `--run`, checks supplied input hashes, roster ordering and schedule legality. The result verifier recalculates all stored draw correlations, point estimates, saved-distribution summaries and the 5000 secondary bootstrap means. Its default also regenerates bootstrap draws 0, 2499 and 4999 per empirical cohort, cell-order repetitions 0 and 999, and simulation repetitions 0 and 499 for all twelve scenarios. These are selected replay checks. To recompute all 5000 empirical bootstrap draws per cohort:

```text
python -B scripts/verify_results.py --full-bootstrap --output outputs/full_bootstrap_checks.json
```

The numerical comparison tolerance is absolute `1e-12`, with zero relative tolerance. Array shapes, identifier order, schedule indices and declared input hashes are checked exactly. `release_manifest.json` records snapshot identity; it is separate from scientific reproduction. These commands use the released derived arrays and do not require raw EEG.

## Regenerate figures and numerical tables

```text
python -B scripts/make_figures.py --output outputs/figures
python -B scripts/make_exploratory_figures.py --output outputs/exploratory
python -B scripts/make_tables.py --output outputs/tables
```

Outputs are Figures 1-3 and S1-S3 as vector PDF and 600-dpi PNG, a LaTeX version of Table S3, and CSV numerical content for Tables 1-2 and S1-S3. Table 3 contains reporting considerations in the article rather than a computed result. CSV tables expose the numerical values; typography remains part of the article. Regenerated PDF metadata can differ while the plotted content is identical.

Main correlations, contrasts, intervals and MCSE use four decimal places; Table S1 ages use two; exploratory MCSE of a mean uses six. The full-precision table retains machine-readable values for traceability. Participant-bootstrap intervals and descriptive ranges across randomizations/simulated cohorts have different meanings.

## Reproduce the exploratory repetitions

```text
python -B scripts/run_exploratory_analysis.py --run --workers 2 --output outputs/exploratory_reproduction
```

This regenerates all 1000 cell-order randomizations and all 500 cohorts for each of twelve simulation scenarios, using the released cell powers and original schedules. It compares every generated correlation and summary array with the released reference arrays. Use a new output directory for each run. Re-extracting cell powers from raw data is available through `--primary-root` as described in [data access](docs/DATA_ACCESS.md).

## Reproduce from raw EEG

Obtain the exact versioned datasets and preserve their BIDS directory layout as described in [data access](docs/DATA_ACCESS.md). Replace the example paths below with the dataset roots on your machine:

```text
python -B scripts/run_preregistered_analysis.py --run --primary-root "raw_data/ds005385" --replication-root "raw_data/ds004148" --output outputs/raw_reproduction/summary_results.json
python -B scripts/compare_npz.py outputs/raw_reproduction/scientific_results.npz results/preregistered/scientific_results.npz
python -B scripts/run_exploratory_analysis.py --run --primary-root "raw_data/ds005385" --workers 2 --output outputs/raw_exploratory
```

The primary computation starts from the hash-identified primary census and sampling schedule. The replication recomputes time-domain QC and schedule construction. These additional commands reconstruct the primary schedules and the prespecified primary QC masks, respectively:

```text
python -B scripts/reproduce_inputs.py --schedule
python -B scripts/reproduce_inputs.py --primary-root "raw_data/ds005385"
```

Full raw reproduction reads the dataset files and recomputes the empirical bootstrap distributions. The exploratory simulations generate 6000 cohorts in total. Allow substantially more time than for the short checks; computation time depends on CPU, memory and storage. Each analysis command refuses to overwrite existing result files/directories.

## Repository contents

| Path | Purpose |
|---|---|
| `scripts/` | Registered estimator/inference, exploratory models, execution, verification and figure/table generation |
| `tests/` | Synthetic checks of the released computations |
| `osf_upload_package/` | Hash-identified subset of supporting pre-analysis inputs used by the implementation |
| `structural/ds004148_session1_census.csv` | Versioned replication recording inventory and file digests |
| `results/preregistered/` | Reported results, underlying arrays and replication QC roster |
| `results/exploratory_2026-09-20/` | Cell-power cache and all reported exploratory repetitions/summaries |
| `data/` | Public age/sex fields for the analyzed participants, field definitions and source provenance |
| `docs/exploratory/analysis_plan_2026-09-20.json` | Executed exploratory model, grid, counts and RNG specification |
| `release_manifest.json` | File identities and sizes for this computational snapshot |

Dataset participant identifiers are the source repositories' pseudonymous IDs. `data/metadata_provenance.json` identifies the source participant files and selection used for Table S1. The literal `AUX004` in the sampling seed namespace is required to reproduce the primary sampling schedule. The identifier `absolute_displacement` refers to the **signed** difference between ten-stratum distributed sampling and contiguous sampling on the logit scale; no absolute-value transformation is applied.

## Data attribution and citation

The primary data are [Wascher and colleagues, OpenNeuro ds005385 v1.0.3](https://doi.org/10.18112/openneuro.ds005385.v1.0.3); replication data are [Wang and colleagues, OpenNeuro ds004148 v1.0.0](https://doi.org/10.18112/openneuro.ds004148.v1.0.0). Their versioned dataset descriptions specify CC0 and are included with the demographic metadata. Cite these datasets when using their data. Study title and registration above identify the accompanying research.

## License and citation

Analysis software and documentation use the [MIT License](LICENSE). Dataset attribution and scope are described in [DATA_LICENSE.md](DATA_LICENSE.md). For software citation, see [CITATION.cff](CITATION.cff) or [CITATION.md](CITATION.md).
