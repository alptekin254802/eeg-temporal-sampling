# Computational conventions and verification

This guide describes the implementation details needed to reproduce the supplied
results, including random seeds, quality-control thresholds and comparison
tolerances. The estimators, participant selection and pre-analysis specification
are unchanged. Use the dependency versions in the main README and run commands
from the repository root.

## Sampling strategies and random seeds

We compare **contiguous sampling**, **five-stratum distributed sampling** and
**ten-stratum distributed sampling**. Commands and arrays use the identifiers
`C`, `F5` and `F10`, respectively. Each draw contains two disjoint arms of ten
four-second cells. Cells are numbered 0 through 41 in the support [8,176) seconds.
Draws are numbered 0 through 199. Participant IDs keep their leading zeros:
for example `sub-001` for the primary dataset and `sub-01` for replication.

The sampling seed is the unsigned big-endian integer encoded by the first eight
bytes of the SHA-256 digest of this UTF-8 string, with no trailing newline:

```text
AUX004|{dataset}|{participant}|{regime}|{replicate}|{arm}
```

Keep this string exactly as shown to reproduce the original seeds. Dataset
identifiers are `ds005385` and `ds004148`. Contiguous sampling and five-stratum distributed sampling
use arm tokens `A` and `B`; ten-stratum distributed sampling uses `A0`/`B0`
through `A9`/`B9`. NumPy `default_rng(seed)` uses PCG64 in the pinned environment.

Candidate blocks are enumerated in ascending start order; Cartesian products use
the order returned by `itertools.product`. A contiguous first arm is drawn from
all eligible ten-cell blocks, then the second from the disjoint subset. A first
arm with no disjoint second arm is a failed draw. The five-stratum distributed sampling second arm uses
a seeded rejection loop with at most 100000 attempts from the same candidate
list. The algorithm does not modify candidates or replace failed draws. A
participant is eligible only if all 200 paired draws succeed for every strategy.
A recording can therefore contain some valid pairs and still be excluded, as
happened for `sub-552`.

Strata split 42 cells using quotient and remainder, allocating the extra cells to
the first strata. Five strata have sizes 9,9,8,8,8; ten strata have sizes
5,5,4,4,4,4,4,4,4,4. Five-stratum distributed sampling blocks in successive strata must leave at
least one unselected intervening cell. Ten-stratum distributed sampling selections may be
adjacent across a stratum boundary.

Participant bootstraps use the same hash-to-integer rule for:

```text
OSF-nh4d8|participant_bootstrap|{analysis_label}
```

| Literal analysis label | Seed |
|---|---:|
| `primary_ds005385` | 15066513651140358874 |
| `complete_support_ds005385` | 6846413471351983530 |
| `replication_ds004148` | 7728558626112585725 |

One `default_rng(seed).integers(0, N, size=(5000, N), dtype=np.int64)` call
generates the participant index matrix in the order of the included participants. Each row is
used jointly for all strategies, draws, and arms. Percentiles use NumPy's `linear`
method. Bootstrap identity hashes serialize the index matrix as contiguous
little-endian signed 64-bit integers in C order.

Exploratory repetitions use `Generator(PCG64(seed))` with seeds derived from:

```text
temporal-sampling-exploratory-2026-09-20-v1|{analysis}|{repetition}
```

The analysis tokens are `order` and `simulation`; repetitions start at zero.
For order controls the generator produces one 42-cell permutation per participant
in the order of the included participants. Simulation draws occur in this order: participant offsets,
slopes, initial autoregressive states, `(N,41)` innovations, and `(N,42)` total
power draws. All scenarios within a repetition share these draws. Scenario order
is the Cartesian product of between-participant SD, autoregressive coefficient,
and slope SD in the order recorded in the exploratory plan.

## Time-domain QC and numerical comparison

The QC function reads O1, Oz, O2, TP9 and TP10 in microvolts over the 42 complete
cells. Posterior channels are referenced to the arithmetic mean of TP9 and TP10.
A cell is rejected if it contains any nonfinite raw sample, any posterior
median-centered absolute amplitude **>500 microvolts**, posterior peak-to-peak
amplitude **>1000 microvolts**, adjacent posterior sample step **>200 microvolts**,
any raw-channel population SD **<0.1 microvolts** (`np.std`, `ddof=0`), or a raw
identical-value run **>=100 ms**. Values exactly at the amplitude or SD thresholds pass; an identical-value run
exactly at the duration threshold fails. The run
threshold is `round(100 * sampling_rate / 1000)` samples. No filtering or
resampling is used. These rules are implemented in `time_domain_qc_mask`.

Raw regenerated cell band powers are in **V²**, checked with relative tolerance
`1e-10` and absolute tolerance **zero**. Both arrays must be finite, nonnegative,
and have identical dimensions. An expected zero must remain exactly zero;
positive values use the same relative tolerance regardless of their size.
Checking powers directly matters because a common scale error in alpha and total
power could cancel in their ratio. Dimensionless markers and reported
contrast arrays retain their separately specified absolute tolerance `1e-12`.
Integer schedules, masks and participant ordering require exact equality.
File hashes check for identical bytes; numerical results and rendered figures
need separate comparisons.

## Sampling geometry summaries

A selected cell with zero-based index i has center `8 + 4*(i+0.5)` seconds. The
absolute centroid difference is the absolute difference between the two arms'
mean cell centers, summarized over participant-by-draw pairs. An arm's temporal
span is `4*(max_index-min_index+1)`, including gaps, summarized over both arms of
every pair. Symmetric nearest separation averages the ten nearest-center
distances from the first arm to the second and the ten reverse distances, then
summarizes the resulting value per pair. Means, medians, and 2.5th/97.5th
percentiles pool these observations; quantiles use linear interpolation.
Cross-stratum adjacency records consecutive selected indices belonging to
different strata. The paired-draw fraction counts either arm; the arm fraction
uses twice as many observations.

## Running the input checks

These commands leave the supplied results unchanged. They read raw data from
external directories arranged as described in [DATA_ACCESS.md](DATA_ACCESS.md).

```bash
# Reconstruct all 60 eligibility decisions and 59 replication schedules.
python scripts/reproduce_inputs.py --replication-schedule

# Full raw replication QC: all 60 recordings, including the excluded recording.
python scripts/reproduce_inputs.py --replication-root /data/ds004148

# Check raw EEG for two participants.
python scripts/reproduce_inputs.py --replication-root /data/ds004148 --participants sub-03 sub-43

# Check primary schedules and raw quality-control masks separately.
python scripts/reproduce_inputs.py --schedule
python scripts/reproduce_inputs.py --primary-root /data/ds005385
```

The replication checks use the same sampling function as the analysis. They
compare valid-draw counts and eligibility decisions with the supplied roster,
and participant order and every cell index with
`results/preregistered/scientific_results.npz`. Supplying a raw root additionally
checks raw signal identity, continuity markers, sampling rate and each regenerated
QC mask against the 60-row roster. These checks do not calculate spectral
outcomes or bootstrap estimates. When `--participants` is supplied, raw EEG is
checked only for those participants.

Primary raw QC needs all **536 structurally eligible recordings**, including the
five later excluded by QC/schedule eligibility; the **531 analysis recordings**
alone are insufficient. If any required files are missing, the command lists
them and stops before reading the signals.

For the historical 608-to-536 structural census, provide a metadata checkout of
the pinned primary release and a directory containing only each EDF's complete
header, named `sub-001.edf.header` through `sub-608.edf.header`. The checkout must
retain the small SHA-256 annex pointers, sidecar JSON, channels TSV and events
TSV/JSON files. Obtain headers from the corresponding versioned recordings:
the first 256 bytes declare total header length at byte offsets 184:192; preserve
exactly that many leading bytes. Prepare these header files before running the command; the header directory
should contain only the extracted headers, not complete signal files.

```bash
python scripts/verify_structural_census.py --snapshot-root /data/ds005385-metadata --edf-header-root /data/ds005385-headers --output /tmp/new-census-check
```

The command checks all 608 participant IDs, reconstructs the original structural
fields and compares them with the supplied census. It leaves the inputs unchanged
and writes the reconstructed census, input hashes and event checks to a new
empty output directory. `--participants sub-114 sub-206 sub-319
sub-493 sub-554` checks only those records and identifies them in the report.

The original event classifier used event **types only**. Its
`ONLY-INITIAL-BOUNDARY` label did not check event count or onset; we retain it to
reconstruct the original census. A separate check reads the declared onset units
and flags any recording without exactly one boundary event between zero and one
sampling period. Unrecognized units are reported as unknown. The check also
reports boundary onsets within [8,176) seconds. It leaves the analysis sample
unchanged and cannot determine what a boundary annotation means physically.
Similarly, the original `initial_boundary_at_1s` field tests `onset == 1` without
converting units, even when the declared unit is milliseconds.

The four original census, quality-control and sampling-geometry scripts are
preserved in `provenance/historical_scripts/`; see its README and checksum file.
They require the original directory layout, and some download files or replace
outputs. Use the commands above to check this version of the software.
