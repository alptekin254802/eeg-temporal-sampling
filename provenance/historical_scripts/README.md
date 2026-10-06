# Original analysis scripts

These four scripts are unchanged copies from the supporting pre-analysis
package. Their SHA-256 hashes match those in
`osf_upload_package/analysis_specification.yaml`; see `SHA256.json`.

We retain them to document the original implementation, including its known
limitations. They require the original directory layout, and some download raw
recordings, delete temporary recordings or replace census and schedule outputs.
Use the verification commands below instead of running these archived scripts.
We have not verified whether these files were included in the archived registration.

`scripts/verify_structural_census.py` reconstructs the census from metadata and
headers and checks event annotations. `scripts/reproduce_inputs.py` checks
sampling schedules and raw quality-control masks. See
`docs/COMPUTATIONAL_CONVENTIONS.md` for examples and the inputs each check requires.
The original census script classifies events by type alone: its
`ONLY-INITIAL-BOUNDARY` label does not check event count or onset.
