"""Small human-readable monitoring reports and a two-panel aggregate figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def write_monitoring(report, output):
    output = Path(output)
    (output / "monitoring.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Batch monitoring",
        "",
        f"Status: **{report['status']}**. Customer snapshots: {report['rows']:,}.",
        f"Cutoff: {report['cutoff']}; reference: {report['reference_cutoff']} "
        f"({report['reference_rows']:,} customers).",
        "",
        "Warnings are review heuristics. Drift does not establish reduced model quality.",
        "",
        f"Contact rate: {report['current_contact_rate']:.1%} "
        f"(reference {report['reference_contact_rate']:.1%}).",
        "",
        "| Feature | Reference median | Current median | Missing now | Histogram TV |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in report["features"].items():
        if values["distribution_tv"] is None:
            continue
        ref = "—" if values["reference_median"] is None else f"{values['reference_median']:.2f}"
        current = "—" if values["current_median"] is None else f"{values['current_median']:.2f}"
        lines.append(
            f"| {name} | {ref} | {current} | {values['current_missing_rate']:.1%} "
            f"| {values['distribution_tv']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Warnings: " + (", ".join(report["warnings"]) or "none"),
            "",
            "No future labels were used. Sparse-history customers remain included.",
            "Review source coverage and customer mix before changing the model or reference.",
        ]
    )
    (output / "monitoring.md").write_text("\n".join(lines) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    # Profiles reserve explicit underflow, overflow and missing bins.
    x = np.arange(10)
    for key, label in (
        ("reference_probability_mass", "Reference"),
        ("current_probability_mass", "Current"),
    ):
        axes[0].plot(x, report[key][1:11], marker="o", label=label)
    axes[0].set(
        xticks=x,
        xticklabels=[f"{v / 10:.1f}" for v in x],
        xlabel="Probability bin lower edge",
        ylabel="Snapshot share",
        title="Purchase probabilities",
    )
    axes[0].legend()
    x = np.arange(3)
    axes[1].bar(x - 0.2, report["reference_history_group_mass"], width=0.4, label="Reference")
    axes[1].bar(x + 0.2, report["current_history_group_mass"], width=0.4, label="Current")
    axes[1].set(
        xticks=x,
        xticklabels=report["history_group_labels"],
        xlabel="Observed invoices",
        ylabel="Snapshot share",
        title="Customer history mix",
    )
    axes[1].legend()
    figure.savefig(output / "monitoring.png", dpi=120)
    plt.close(figure)
