#!/usr/bin/env python3
"""Merge prespecified time-domain QC fields into the 608-row structural census."""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CENSUS = REPO / "DS005385_PARTICIPANT_CENSUS.csv"
QC = REPO / "structural" / "ds005385_qc_results.csv"
SUMMARY = REPO / "structural" / "ds005385_final_eligibility_summary.json"


def main():
    with CENSUS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        original_fields = list(rows[0])
    with QC.open(encoding="utf-8", newline="") as f:
        qc_rows = {r["participant_id"]: r for r in csv.DictReader(f)}

    if len(rows) != 608 or len({r["participant_id"] for r in rows}) != 608:
        raise RuntimeError("Census must contain 608 unique participants")
    structural_ids = {r["participant_id"] for r in rows if r["structurally_eligible"] == "True"}
    if structural_ids != set(qc_rows):
        raise RuntimeError("QC roster does not exactly match structural roster")

    added = [
        "qc_assessed", "qc_eligible_window_count", "qc_eligible_mask",
        "qc_window_reasons", "C_valid_draws", "F5_valid_draws",
        "F10_valid_draws", "complete_support_sensitivity", "final_eligible",
        "final_exclusion_reason",
    ]
    fields = [x for x in original_fields if x not in ("qc_status", "qc_eligible")] + [
        "qc_status", "qc_eligible", *added
    ]
    for row in rows:
        q = qc_rows.get(row["participant_id"])
        if q is None:
            row.update({
                "qc_status": "NOT-ASSESSED-STRUCTURAL-FAIL",
                "qc_eligible": "False", "qc_assessed": "False",
                "qc_eligible_window_count": "", "qc_eligible_mask": "",
                "qc_window_reasons": "", "C_valid_draws": "",
                "F5_valid_draws": "", "F10_valid_draws": "",
                "complete_support_sensitivity": "False", "final_eligible": "False",
                "final_exclusion_reason": row["structural_exclusion_reason"],
            })
        else:
            eligible = q["qc_eligible"] == "True"
            row.update({
                "qc_status": "ASSESSED-TIME-DOMAIN-ONLY",
                "qc_eligible": str(eligible), "qc_assessed": "True",
                "qc_eligible_window_count": q["eligible_window_count"],
                "qc_eligible_mask": q["eligible_mask"],
                "qc_window_reasons": q["window_reasons"],
                "C_valid_draws": q["C_valid_draws"],
                "F5_valid_draws": q["F5_valid_draws"],
                "F10_valid_draws": q["F10_valid_draws"],
                "complete_support_sensitivity": str(eligible and int(q["eligible_window_count"]) == 42),
                "final_eligible": str(eligible),
                "final_exclusion_reason": "" if eligible else q["qc_exclusion_reason"],
            })

    with CENSUS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "nominal_n": len(rows),
        "required_posterior_channels_n": sum(r["required_posterior_channels"] == "True" for r in rows),
        "duration_ge_180s_n": sum(r["duration_ge_180s"] == "True" for r in rows),
        "event_compatible_n": sum(r["event_class"] == "ONLY-INITIAL-BOUNDARY" for r in rows),
        "structurally_eligible_n": sum(r["structurally_eligible"] == "True" for r in rows),
        "qc_eligible_n": sum(r["final_eligible"] == "True" for r in rows),
        "complete_support_sensitivity_n": sum(r["complete_support_sensitivity"] == "True" for r in rows),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
