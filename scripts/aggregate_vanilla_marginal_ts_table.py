"""Build the posthoc_mapped timestep-ablation table for celeba_vanilla_marginal.

Reads:
  - celeba_full_ranking_vanilla_marginal_ts_posthoc_step{k}.csv  (condition-level trust vs delta_kid)
  - celeba_fpr95_selection_vanilla_marginal_ts_posthoc_step{k}.csv (global pooled FPR95 KID delta)
  - consecutive_image_distance.csv (optional)

Emits a minimal LaTeX table with:
  - Spearman rho(trust_mean, delta_kid) over ALL 16 conditions (ranking correlation)
  - Global Delta-KID improvement from pooled P95-trust acceptance set vs random
    matched-size subset (negative KID_accepted - KID_random => improvement %)
  - Gen.saved per row

The oracle row is the existing vanilla_marginal dinov3 result pointed at by --oracle-dir.

Usage:
    PYTHONPATH=src uv run python scripts/aggregate_vanilla_marginal_ts_table.py \\
        --trust-dir outputs/trust_evaluation_vanilla_marginal_ts \\
        --image-distance-csv outputs/gen/celeba_vanilla_marginal_timesteps/consecutive_image_distance.csv
"""
import argparse
import csv
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

STEPS = [0, 27, 55, 83, 110, 138, 166, 193, 221, 248]


def _rho(ranking_csv: Path):
    if not ranking_csv.exists():
        return None, None
    df = pd.read_csv(ranking_csv).dropna(subset=["trust_mean", "delta_kid"])
    if len(df) < 3:
        return None, None
    rho, _ = spearmanr(df["trust_mean"], df["delta_kid"])
    return float(rho), len(df)


def _global_fpr95(selection_csv: Path, component: str = "trust"):
    """Return (delta_pct, kid_acc, kid_rand, acceptance_rate) for the given component row."""
    if not selection_csv.exists():
        return None
    df = pd.read_csv(selection_csv)
    # fpr95_selection CSV has one row per ... usually scored component; some schemas
    # label it by index not component. Fall back to first row if no component col.
    if "component" in df.columns:
        row = df[df["component"] == component]
        if len(row) == 0:
            row = df
    else:
        row = df
    if len(row) == 0:
        return None
    r = row.iloc[0]
    kid_acc = float(r["kid_raw_accepted"])
    kid_rand = float(r["kid_raw_random"])
    # "Improvement" mirrors the reference table's convention:
    #  positive = accepted subset has LOWER KID than random (better).
    delta_pct = (kid_rand - kid_acc) / kid_rand * 100.0 if kid_rand > 0 else None
    return {
        "kid_acc": kid_acc,
        "kid_rand": kid_rand,
        "acc_rate": float(r["acceptance_rate"]),
        "delta_pct": delta_pct,
        "n_accepted": int(r["n_accepted"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trust-dir", type=Path, required=True)
    p.add_argument("--oracle-dir", type=Path, default=None)
    p.add_argument("--image-distance-csv", type=Path, default=None)
    p.add_argument("--dataset", default="celeba",
                   help="Dataset prefix for trust-eval CSV filenames (celeba|rxrx1).")
    p.add_argument("--model", default="vanilla_marginal_ts",
                   help="Model alias used in trust-eval CSV filenames.")
    args = p.parse_args()

    if args.oracle_dir is None:
        args.oracle_dir = args.trust_dir  # dinov3 row is in the same run

    img_dist = {}
    if args.image_distance_csv and args.image_distance_csv.exists():
        with open(args.image_distance_csv) as f:
            for row in csv.DictReader(f):
                img_dist[(int(row["k_from"]), int(row["k_to"]))] = float(row["mean_l2"])

    prefix = f"{args.dataset}_full_ranking_{args.model}"
    sel_prefix = f"{args.dataset}_fpr95_selection_{args.model}"

    rows = []
    for k in STEPS:
        feat = f"posthoc_step{k}"
        ranking = args.trust_dir / f"{prefix}_{feat}.csv"
        selection = args.trust_dir / f"{sel_prefix}_{feat}.csv"
        rho, n = _rho(ranking)
        sel = _global_fpr95(selection)
        rows.append({
            "label": f"$k{{=}}{k}$\\,($t{{\\approx}}{1.0 - (k / 249.0) * (1.0 - 0.04):.2f}$)",
            "k": k,
            "rho": rho,
            "n_conds": n,
            "sel": sel,
            "gen_saved": 1.0 - (k / 250.0),
        })

    # Oracle row: dinov3 (full decode + DINOv3 encoder on the new gen)
    ranking = args.oracle_dir / f"{prefix}_dinov3.csv"
    selection = args.oracle_dir / f"{sel_prefix}_dinov3.csv"
    rho, n = _rho(ranking)
    sel = _global_fpr95(selection)
    rows.append({
        "label": r"k=249 \,(oracle)",
        "k": "oracle",
        "rho": rho,
        "n_conds": n,
        "sel": sel,
        "gen_saved": None,
    })

    def fmt_rho(v):
        return f"${v:.2f}$" if v is not None else "--"

    def fmt_delta(v):
        return f"{v:+.1f}\\%" if v is not None else "--"

    def fmt_acc(v):
        return f"({v*100:.0f}\\%)" if v is not None else ""

    def fmt_saved(v):
        return f"${{\\approx}}{int(round(v*100))}\\%$" if v is not None else "---"

    # "Image $\Delta$" = mean L2 between decoded x0_hat at previous-k and current-k
    # The first row has no previous step; oracle has no step index.
    prev_k = None
    image_dl2 = {}
    for r in rows:
        if r["k"] == "oracle":
            image_dl2["oracle"] = None
            continue
        if prev_k is None:
            image_dl2[r["k"]] = None
        else:
            image_dl2[r["k"]] = img_dist.get((prev_k, r["k"]))
        prev_k = r["k"]

    def fmt_kid(v):
        return f"{v:.3f}" if v is not None else "--"

    def fmt_img(v):
        return f"{v:.1f}" if v is not None else "--"

    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(r"Scoring step & $\rho_{\text{trust}}$ & KID$_{\text{acc}}$ & KID$_{\text{rand}}$ & $\Delta$KID\% (acc.) & Image $\Delta$ L2 & Gen.\ saved \\")
    print(r"\midrule")
    for r in rows:
        if r["k"] == "oracle":
            print(r"\midrule")
        sel = r["sel"]
        kid_a = fmt_kid(sel.get("kid_acc")) if sel else "--"
        kid_r = fmt_kid(sel.get("kid_rand")) if sel else "--"
        dpct = fmt_delta(sel.get("delta_pct")) if sel else "--"
        acc = fmt_acc(sel.get("acc_rate")) if sel else ""
        img = fmt_img(image_dl2.get(r["k"]))
        print(
            f"{r['label']}  & {fmt_rho(r['rho'])} & {kid_a} & {kid_r} & "
            f"{dpct} {acc} & {img} & {fmt_saved(r['gen_saved'])} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n% Raw values:")
    for r in rows:
        sel = r["sel"] or {}
        print(
            f"%   k={r['k']}: rho={r['rho']}, "
            f"kid_acc={sel.get('kid_acc')}, kid_rand={sel.get('kid_rand')}, "
            f"delta_pct={sel.get('delta_pct')}, acc_rate={sel.get('acc_rate')}, "
            f"gen_saved={r['gen_saved']}"
        )
    if img_dist:
        print("\n% Consecutive decoded-x0_hat image L2 distance:")
        for (a, b), v in img_dist.items():
            print(f"%   k={a}->{b}: {v:.4f}")


if __name__ == "__main__":
    main()
