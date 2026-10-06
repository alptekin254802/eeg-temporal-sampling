# Temporal Sampling Geometry and Participant-Level Reproducibility in Fixed-Duration Resting EEG

Computational companion for the study of how temporal placement of equal-duration resting EEG changes the reproducibility of participant ordering. Each sample retains ten 4-s periodograms (40 s) from a common 42-cell support. The measured feature is posterior relative alpha at O1/Oz/O2 after a mean TP9/TP10 reference. Paired samples come from the same recording.

Study registration: [OSF nh4d8](https://osf.io/nh4d8/). Supporting pre-analysis materials: [OSF g2apc](https://osf.io/g2apc/). We cite the supporting materials separately because we have not verified whether they were included in the archived registration. The primary, secondary, complete-support and replication analyses were preregistered. The temporal-order control, cell-band-power simulations and annotation-exclusion sensitivity are exploratory: we specified the first two on 20 September 2026 and the sensitivity analysis on 5 October 2026, after the primary results were known.

The software is archived under the all-versions DOI [10.5281/zenodo.23145601](https://doi.org/10.5281/zenodo.23145601), which resolves to the latest archived version. See [CITATION.md](CITATION.md) for citation details, including the DOI for version 1.0.1. `release_manifest.json` lists the files and checksums for this copy of the software.

This is version **1.0.2** of the accompanying software.

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

Dimensionless markers and reported contrast arrays use absolute tolerance `1e-12`, with zero relative tolerance. Raw cell powers in V² use relative tolerance `1e-10` and zero absolute tolerance; expected zeros must remain zero. The [computational conventions](docs/COMPUTATIONAL_CONVENTIONS.md) explain the quality-control thresholds, random seeds and checks. Array shapes, identifier order, schedule indices and input hashes must match exactly. The manifest checks file integrity; the analysis commands check numerical agreement. These commands use the released derived arrays and do not require raw EEG.

## Regenerate figures and numerical tables

```text
python -B scripts/make_figures.py --output outputs/figures
python -B scripts/make_exploratory_figures.py --output outputs/exploratory
python -B scripts/make_tables.py --output outputs/tables
```

Outputs are Figures 1-3 and S1-S3 as vector PDF and 600-dpi PNG, a LaTeX version of Table S3, and CSV numerical content for Tables 1-2 and S1-S4. Table 3 contains reporting considerations in the article rather than a computed result. CSV tables expose the numerical values; typography remains part of the article. Table S4 includes the annotation-exclusion sensitivity alongside the primary and complete-support results. Its source is `results/post_audit_2026-10-05/boundary_sensitivity_summary.json`; the command below checks that result. Regenerated PDF metadata can differ while the plotted content is identical.

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
python -B scripts/reproduce_inputs.py --replication-schedule
python -B scripts/reproduce_inputs.py --replication-root "raw_data/ds004148" --participants sub-03 sub-43
```

The last command checks raw EEG for the two named participants. Omit `--participants` to check all 60 replication recordings. Primary raw quality control requires all 536 structurally eligible files, including the five recordings excluded later. The command lists any missing files before reading the signals. To reconstruct the selection of 536 recordings from the 608-person primary census using metadata and headers, see the [computational conventions](docs/COMPUTATIONAL_CONVENTIONS.md). That check reports agreement with the original census and examines event annotations separately.

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
| `results/post_audit_2026-10-05/` | Results of the exploratory annotation-exclusion sensitivity |
| `docs/post_audit/` | Plan recorded before calculating that sensitivity analysis |
| `docs/COMPUTATIONAL_CONVENTIONS.md` | Sampling, QC, random seeds and comparison rules |
| `provenance/historical_scripts/` | Original input-preparation scripts and their checksums |
| `CHANGELOG.md` | Notes on the software versions |
| `release_manifest.json` | File identities and sizes for this computational snapshot |

Dataset participant identifiers are the source repositories' pseudonymous IDs. `data/metadata_provenance.json` identifies the source participant files and selection used for Table S1. The literal `AUX004` in the sampling seed namespace is required to reproduce the primary sampling schedule. The identifier `absolute_displacement` refers to the **signed** difference between ten-stratum distributed sampling and contiguous sampling on the logit scale; no absolute-value transformation is applied.

## Data attribution and citation

The primary data are [Wascher and colleagues, OpenNeuro ds005385 v1.0.3](https://doi.org/10.18112/openneuro.ds005385.v1.0.3); replication data are [Wang and colleagues, OpenNeuro ds004148 v1.0.0](https://doi.org/10.18112/openneuro.ds004148.v1.0.0). Their versioned dataset descriptions specify CC0 and are included with the demographic metadata. Cite these datasets when using their data. Study title and registration above identify the accompanying research.

## License and citation

Analysis software and documentation use the [MIT License](LICENSE). Dataset attribution and scope are described in [DATA_LICENSE.md](DATA_LICENSE.md). For software citation, see [CITATION.cff](CITATION.cff) or [CITATION.md](CITATION.md).

## Exploratory annotation-exclusion sensitivity

We repeated the analysis after excluding the five primary participants whose recordings had additional boundary annotations, leaving 526 participants. This exploratory check uses the same saved measurements and schedules. We recorded the exclusion list and random seed in the [analysis plan](docs/post_audit/boundary_sensitivity_plan_2026-10-05.json) before running this calculation, but after seeing the primary results. To check the supplied result, run:

```text
python -B scripts/run_boundary_sensitivity.py --check
```

To recompute the sensitivity analysis from saved measurements, with 526 participants and 5000 bootstrap resamples:

```text
python -B scripts/run_boundary_sensitivity.py --run --output outputs/boundary_sensitivity
```

The command runs only this sensitivity analysis from the saved measurements. It does not read raw EEG or rerun the original bootstrap, temporal-order control or simulations. The supplied results are in `results/post_audit_2026-10-05/`. Whether the boundary annotations mark actual interruptions in recording still needs clarification from the data source.
