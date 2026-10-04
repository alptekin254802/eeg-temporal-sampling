"""Create manuscript figures from registered schedules and accepted summaries only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results" / "preregistered" / "summary_results.json"
SCHEDULE = REPO / "osf_upload_package" / "structural" / "ds005385_sampling_schedule.npz"
RESULTS = REPO / "results" / "preregistered" / "scientific_results.npz"
COLORS = {"C": "#3B5B92", "F5": "#D18F2F", "F10": "#298C8C"}


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure1(output: Path) -> None:
    with np.load(SCHEDULE, allow_pickle=False) as archive:
        example = archive["schedule"][0, :, 0]
    fig = plt.figure(figsize=(7.0, 4.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    rows = [(0, 0, "Contiguous sampling (A)"), (0, 1, "Contiguous sampling (B)"),
            (1, 0, "Five-stratum distributed\nsampling (A)"), (1, 1, "Five-stratum distributed\nsampling (B)"),
            (2, 0, "Ten-stratum distributed\nsampling (A)"), (2, 1, "Ten-stratum distributed\nsampling (B)")]
    y = np.arange(len(rows))[::-1]
    for yi, (ri, ai, label) in zip(y, rows):
        ax.broken_barh([(8, 168)], (yi - 0.35, 0.7), facecolors="#ECEFF3")
        for cell in example[ri, ai]:
            ax.broken_barh([(8 + 4 * int(cell), 4)], (yi - 0.35, 0.7),
                           facecolors=COLORS[["C", "F5", "F10"][ri]],
                           edgecolors="white", linewidth=0.35)
    ax.set_yticks(y, [r[2] for r in rows])
    ax.set_xlim(8, 176)
    ax.set_xticks([8, 40, 80, 120, 176])
    ax.set_xlabel("Time from recording onset (s)")
    ax.set_title("Illustrative paired draw", loc="left")
    ax.text(-0.10, 1.075, "(a)", transform=ax.transAxes, weight="bold", fontsize=10)
    ax.text(0.0, -0.23, "Every strategy retains ten 4-s cells (40 s); A and B do not overlap.",
            transform=ax.transAxes, fontsize=8)

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.axis("off")
    participants = ["P1", "P2", "P3", "...", "PN"]
    ranks_a = [1, 2, 3, 4, 5]
    ranks_b = [1, 3, 2, 4, 5]
    for j, (p, ra, rb) in enumerate(zip(participants, ranks_a, ranks_b)):
        yy = 0.80 - j * 0.13
        ax2.text(0.03, yy, p, va="center", ha="right")
        ax2.text(0.31, yy, str(ra), ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.22", fc="#E8EEF7", ec="#3B5B92"))
        ax2.text(0.75, yy, str(rb), ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.22", fc="#E8F4F3", ec="#298C8C"))
        ax2.annotate("", xy=(0.69, yy), xytext=(0.37, yy),
                     arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7))
    ax2.text(0.24, 0.91, "Rank\nin A", ha="center", va="center",
             weight="bold", fontsize=7.5, linespacing=0.9)
    ax2.text(0.84, 0.91, "Rank\nin B", ha="center", va="center",
             weight="bold", fontsize=7.5, linespacing=0.9)
    ax2.text(0.5, 0.07, r"Across-participant Spearman $\rho$",
             ha="center", fontsize=9, weight="bold")
    ax2.text(0.5, 0.00, "computed separately for every paired draw",
             ha="center", fontsize=8)
    ax2.text(-0.08, 1.05, "(b)", transform=ax2.transAxes, weight="bold", fontsize=10)
    save(fig, output, "figure1_sampling_geometry")


def figure2(summary: dict, output: Path) -> None:
    primary = summary["PRIMARY"]
    f5 = summary["PRESPECIFIED_DESCRIPTIVE_SECONDARY"]["F5"]
    sensitivity = summary["PRESPECIFIED_SENSITIVITY"]
    regimes = ["C", "F5", "F10"]
    values = [primary["participant_rank_reproducibility"]["C"],
              f5["participant_rank_reproducibility"],
              primary["participant_rank_reproducibility"]["F10"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    ax = axes[0]
    for x, regime, value in zip(range(3), regimes, values):
        ax.scatter(x, value, s=52, color=COLORS[regime], edgecolor="black", lw=0.5, zorder=2)
        ax.text(x, value + 0.004, f"{value:.4f}",
                ha="left" if x == 0 else "center", va="bottom")
    ax.set_xticks(range(3), ["Contiguous\nsampling", "Five-stratum\ndistributed\nsampling", "Ten-stratum\ndistributed\nsampling"])
    ax.set_ylabel(r"Rank reproducibility ($\rho$)")
    ax.set_ylim(0.90, 0.98)
    ax.text(-0.16, 1.05, "(a)", transform=ax.transAxes, weight="bold", fontsize=10)

    ax = axes[1]
    rows = [("Primary cohort (N=531)", primary),
            ("All-cells-eligible sensitivity (N=516)", sensitivity)]
    for yi, (label, record) in enumerate(rows[::-1]):
        estimate = record["delta_z_F10_minus_C"]
        lo, hi = record["percentile_95_interval"]
        ax.errorbar(estimate, yi, xerr=[[estimate - lo], [hi - estimate]], fmt="o",
                    color="#232323", ecolor="#3B5B92", capsize=3, ms=5)
    ax.axvline(0, color="#777777", ls="--", lw=0.9)
    ax.set_yticks(
        [0, 1],
        ["Complete support\n(N=516)", "Primary\n(N=531)"],
    )
    ax.set_xlabel("Ten-stratum distributed sampling minus\ncontiguous sampling ($\\Delta z$)")
    ax.set_xlim(-0.02, 0.40)
    ax.text(-0.16, 1.05, "(b)", transform=ax.transAxes, weight="bold", fontsize=10)
    save(fig, output, "figure2_primary_results")


def figure3(summary: dict, output: Path) -> None:
    primary = summary["PRIMARY"]
    f5p = summary["PRESPECIFIED_DESCRIPTIVE_SECONDARY"]["F5"]["participant_rank_reproducibility"]
    replication = summary["DIRECTIONAL_REPLICATION"]
    regimes = ["C", "F5", "F10"]
    pvals = [primary["participant_rank_reproducibility"]["C"], f5p,
             primary["participant_rank_reproducibility"]["F10"]]
    rvals = [replication["participant_rank_reproducibility"][r] for r in regimes]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    ax = axes[0]
    offsets = np.array([-0.07, 0.07])
    ax.scatter(np.arange(3) + offsets[0], pvals, marker="o", color="#3B5B92",
               label="Primary cohort (N=531)")
    ax.scatter(np.arange(3) + offsets[1], rvals, marker="s", color="#A65E2E",
               label="Replication cohort (N=59)")
    ax.set_xticks(range(3), ["Contiguous\nsampling", "Five-stratum\ndistributed\nsampling", "Ten-stratum\ndistributed\nsampling"])
    ax.set_ylabel(r"Rank reproducibility ($\rho$)")
    ax.set_ylim(0.78, 0.99)
    ax.legend(frameon=False, loc="lower right")
    ax.text(-0.16, 1.05, "(a)", transform=ax.transAxes, weight="bold", fontsize=10)

    ax = axes[1]
    rows = [("Primary cohort", primary), ("Replication cohort", replication)]
    for yi, (label, record) in enumerate(rows[::-1]):
        estimate = record["delta_z_F10_minus_C"]
        lo, hi = record["percentile_95_interval"]
        color = "#3B5B92" if label == "Primary cohort" else "#A65E2E"
        ax.errorbar(estimate, yi, xerr=[[estimate - lo], [hi - estimate]], fmt="o",
                    color=color, ecolor=color, capsize=3, ms=5)
    ax.axvline(0, color="#777777", ls="--", lw=0.9)
    ax.set_yticks(
        [0, 1],
        ["Replication\n(N=59)", "Primary\n(N=531)"],
    )
    ax.set_xlabel("Ten-stratum distributed sampling minus\ncontiguous sampling ($\\Delta z$)")
    ax.set_xlim(-0.03, 0.65)
    ax.text(-0.16, 1.05, "(b)", transform=ax.transAxes, weight="bold", fontsize=10)
    save(fig, output, "figure3_directional_replication")


def figure_s1(summary: dict, output: Path) -> None:
    """Plot the prespecified participant-level displacement already in the result artifact."""
    with np.load(RESULTS, allow_pickle=False) as archive:
        displacement = archive["secondary_participant_displacement"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    ax.hist(displacement, bins=32, color="#6C8EBF", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.0, label="Zero")
    cohort_mean = summary["PRESPECIFIED_DESCRIPTIVE_SECONDARY"]["absolute_displacement"]["cohort_mean"]
    ax.axvline(cohort_mean, color="#A65E2E", linewidth=1.3,
               label="Cohort mean")
    ax.set_xlabel("Participant-level logit relative-alpha displacement\n"
                  "(ten-stratum distributed sampling minus contiguous sampling)")
    ax.set_ylabel("Participants")
    ax.legend(frameon=False)
    save(fig, output, "figure_s1_participant_displacement")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "outputs" / "figures")
    args = parser.parse_args()
    setup()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    figure1(args.output)
    figure2(summary, args.output)
    figure3(summary, args.output)
    figure_s1(summary, args.output)


if __name__ == "__main__":
    main()
