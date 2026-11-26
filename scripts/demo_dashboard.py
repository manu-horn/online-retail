"""Export a self-contained dashboard from one completed batch demo."""

import argparse
import base64
import json
from html import escape
from pathlib import Path


def dashboard(demo: Path) -> str:
    batch = json.loads((demo / "batch/monitoring.json").read_text())
    outcome = json.loads((demo / "outcomes/report.json").read_text())
    if any(batch[key] != outcome[key] for key in ("run_id", "cutoff", "model_version")):
        raise ValueError("Monitoring and outcome reports belong to different batches.")
    overall = outcome["overall"]
    chart = base64.b64encode((demo / "batch/monitoring.png").read_bytes()).decode("ascii")
    warnings = escape(", ".join(batch["warnings"]) or "None")
    rows = []
    for group, metrics in outcome["history_groups"].items():
        auc = "—" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}"
        rows.append(
            f"<tr><th scope='row'>{escape(group)}</th><td>{metrics['snapshots']:,}</td>"
            f"<td>{metrics['buyers']:,}</td><td>{auc}</td>"
            f"<td>{metrics['buyer_precision']:.1%}</td><td>{metrics['buyer_recall']:.1%}</td></tr>"
        )
    cards = [
        ("Customers scored", f"{overall['snapshots']:,}"),
        ("Flagged for contact", f"{batch['current_contact_rate']:.1%}"),
        ("Buyer precision", f"{overall['buyer_precision']:.1%}"),
        ("Buyer recall", f"{overall['buyer_recall']:.1%}"),
    ]
    tiles = "".join(
        f"<div><strong>{value}</strong><span>{label}</span></div>" for label, value in cards
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retail repurchase replay</title>
<style>
* {{box-sizing:border-box}}
body {{margin:0;background:#edf3f5;color:#142d39;font:16px/1.5 system-ui,sans-serif}}
main {{max-width:1100px;margin:auto;padding:2.5rem 1.2rem}}
h1 {{font-size:clamp(2rem,4vw,3.4rem);margin:0;letter-spacing:-.04em}}
h2 {{margin-top:0}}
header {{border-top:6px solid #027269;padding:1.2rem 0}}
section,details,.tiles div {{background:white;border:1px solid #cfdde2;
  border-radius:12px;padding:1.2rem}}
section,details {{margin-top:1rem;overflow:auto}}
.tiles {{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.8rem}}
.tiles strong,.tiles span {{display:block}}
.tiles strong {{font-size:2rem;color:#02685f}}
img {{width:100%;height:auto}}
table {{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td {{padding:.65rem;text-align:right;border-bottom:1px solid #dbe6e9}}
th:first-child {{text-align:left}}
summary {{cursor:pointer;font-weight:650}}
</style></head><body><main>
<header><h1>Retail repurchase replay</h1>
<p>Scored {escape(batch["cutoff"][:10])} · 50-day purchase window ·
contact threshold {outcome["threshold"]:.2f}</p></header>
<div class="tiles">{tiles}</div>
<section><h2>Batch monitoring</h2>
<p>Reference cutoff {escape(batch["reference_cutoff"][:10])} · Warnings {warnings}</p>
<img src="data:image/png;base64,{chart}" alt="Reference and replay distributions for purchase
probabilities and customer invoice history">
</section>
<section><h2>Observed outcomes by customer history</h2>
<p>One historical batch with complete follow-up. Groups use invoice counts at scoring time.</p>
<table><thead><tr><th scope="col">Invoices</th><th scope="col">Customers</th>
<th scope="col">Buyers</th><th scope="col">ROC-AUC</th>
<th scope="col">Precision</th><th scope="col">Recall</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></section>
<details><summary>Run provenance</summary>
<p>Scoring run <code>{escape(batch["run_id"])}</code></p>
<p>Model SHA-256 <code>{escape(batch["model_version"])}</code></p></details>
</main></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dashboard(args.demo)
    with args.output.open("x", encoding="utf-8") as file:
        file.write(result)


if __name__ == "__main__":
    main()
