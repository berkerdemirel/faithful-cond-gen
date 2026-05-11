"""Aggregate per-step FPR95 selection CSVs into one delta-KID summary CSV.

Reads ``{dataset}_fpr95_selection_{model}_posthoc_step{k}.csv`` for each k in
--steps and the dinov3 oracle, computes ΔKID% = (kid_rand - kid_acc) / kid_rand
× 100, and writes a single CSV consumed by plot_rxrx1_posthoc_timestep_story.py
(and friends).

Usage:
    PYTHONPATH=src uv run python scripts/aggregate_fpr95_delta_kid.py \\
        --dataset rxrx1 --model vanilla_marginal_ts \\
        --trust-dir outputs/trust_evaluation_rxrx1_vanilla_marginal_ts \\
        --out outputs/trust_evaluation_rxrx1_vanilla_marginal_ts/fpr95_delta_kid_per_step.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

DEFAULT_STEPS = [0, 27, 55, 83, 110, 138, 166, 193, 221, 248]


def _read_row(selection_csv: Path, component: str = "trust"):
    if not selection_csv.exists():
        return None
    df = pd.read_csv(selection_csv)
    if "component" in df.columns:
        rows = df[df["component"] == component]
        if len(rows) == 0:
            rows = df
    else:
        rows = df
    if len(rows) == 0:
        return None
    r = rows.iloc[0]
    kid_acc = float(r["kid_raw_accepted"])
    kid_rand = float(r["kid_raw_random"])
    delta_pct = (kid_rand - kid_acc) / kid_rand * 100.0 if kid_rand > 0 else float("nan")
    return {
        "kid_acc": kid_acc,
        "kid_rand": kid_rand,
        "delta_pct": delta_pct,
        "acc_rate": float(r["acceptance_rate"]),
        "n_accepted": int(r["n_accepted"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True,
                   help="Model alias used in trust-eval CSV filenames (e.g. vanilla_marginal_ts).")
    p.add_argument("--trust-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    args = p.parse_args()

    sel_prefix = f"{args.dataset}_fpr95_selection_{args.model}"

    rows = []
    for k in args.steps:
        sel_csv = args.trust_dir / f"{sel_prefix}_posthoc_step{k}.csv"
        d = _read_row(sel_csv)
        if d is None:
            print(f"[warn] missing: {sel_csv}")
            continue
        rows.append({"k": k, "is_oracle": 0, **d})

    oracle_csv = args.trust_dir / f"{sel_prefix}_dinov3.csv"
    d = _read_row(oracle_csv)
    if d is not None:
        rows.append({"k": -1, "is_oracle": 1, **d})
    else:
        print(f"[warn] oracle CSV missing: {oracle_csv}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["k", "is_oracle", "kid_acc", "kid_rand", "delta_pct", "acc_rate", "n_accepted"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
