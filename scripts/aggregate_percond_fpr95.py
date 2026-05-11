"""
Aggregate per-condition FPR@95 CSVs into a paper-style summary table.

Matches tab:fpr95-selection layout but uses per-condition averaging:
  Acc% = mean over conditions of (n_accepted / n_total)
  KID_sel = mean over conditions of kid_accepted
  KID_rand = mean over conditions of kid_random
  Delta% = mean over conditions of improvement_pct
  N_conds = number of conditions with finite data (for context)
  frac_better = fraction of conditions where accepted < random
"""

import pandas as pd

CELEBA_DIR = "outputs/trust_evaluation_celeba_v7"
RXRX1_DIR = "outputs/trust_evaluation_rxrx1_v6"

ROWS = [
    ("CelebA", "Vanilla",       "full",    f"{CELEBA_DIR}/celeba_fpr95_per_condition_vanilla_full_dinov3.csv"),
    ("CelebA", "Vanilla",       "stress",  f"{CELEBA_DIR}/celeba_fpr95_per_condition_vanilla_marginal_dinov3.csv"),
    ("CelebA", "REPA (DINOv3)", "full",    f"{CELEBA_DIR}/celeba_fpr95_per_condition_repa_full_dinov3.csv"),
    ("CelebA", "REPA (DINOv3)", "stress",  f"{CELEBA_DIR}/celeba_fpr95_per_condition_repa_marginal_dinov3.csv"),
    ("CelebA", "REPA (SigLIP)", "full",    f"{CELEBA_DIR}/celeba_fpr95_per_condition_repa_siglip_full_dinov3.csv"),
    ("CelebA", "REPA (SigLIP)", "stress",  f"{CELEBA_DIR}/celeba_fpr95_per_condition_repa_siglip_marginal_dinov3.csv"),
    ("RxRx1",  "Vanilla",       "full",    f"{RXRX1_DIR}/rxrx1_fpr95_per_condition_vanilla_full_dinov3.csv"),
    ("RxRx1",  "Vanilla",       "heldout", f"{RXRX1_DIR}/rxrx1_fpr95_per_condition_vanilla_marginal_dinov3.csv"),
    ("RxRx1",  "REPA (DINOv3)", "full",    f"{RXRX1_DIR}/rxrx1_fpr95_per_condition_repa_full_dinov3.csv"),
    ("RxRx1",  "REPA (DINOv3)", "heldout", f"{RXRX1_DIR}/rxrx1_fpr95_per_condition_repa_marginal_dinov3.csv"),
    ("RxRx1",  "REPA (SigLIP)", "full",    f"{RXRX1_DIR}/rxrx1_fpr95_per_condition_repa_siglip_full_dinov3.csv"),
    ("RxRx1",  "REPA (SigLIP)", "heldout", f"{RXRX1_DIR}/rxrx1_fpr95_per_condition_repa_siglip_marginal_dinov3.csv"),
]


def aggregate(path, component="trust"):
    df = pd.read_csv(path)
    df = df[df["component"] == component].copy()
    df = df[df["kid_accepted"].notna() & df["kid_random"].notna()].copy()
    n_conds = len(df)
    if n_conds == 0:
        return dict(n=0, acc=float("nan"), kid_sel=float("nan"),
                    kid_sel_std=float("nan"), kid_rand=float("nan"),
                    kid_rand_std=float("nan"), delta_pct=float("nan"),
                    frac_better=float("nan"))
    acc = (df["n_accepted"] / df["n_total"]).mean() * 100
    return dict(
        n=n_conds,
        acc=acc,
        kid_sel=df["kid_accepted"].mean(),
        kid_sel_std=df["kid_accepted"].std(),
        kid_rand=df["kid_random"].mean(),
        kid_rand_std=df["kid_random"].std(),
        delta_pct=df["improvement_pct"].mean(),
        frac_better=(df["delta"] > 0).mean() * 100,
    )


def main():
    print("\n=== Per-condition averaged FPR@95 (trust component, DINOv3 scoring) ===\n")
    print(f"{'Dataset':<8} {'Model':<16} {'Setting':<9} "
          f"{'N':>5} {'Acc%':>6} {'KID_sel':>12} {'KID_rand':>12} "
          f"{'Δ%':>8} {'%better':>9}")
    print("-" * 100)
    results = []
    for ds, model, setting, path in ROWS:
        r = aggregate(path)
        results.append((ds, model, setting, r))
        print(f"{ds:<8} {model:<16} {setting:<9} "
              f"{r['n']:>5d} {r['acc']:>6.1f} "
              f"{r['kid_sel']:>7.4f}±{r['kid_sel_std']:.3f} "
              f"{r['kid_rand']:>7.4f}±{r['kid_rand_std']:.3f} "
              f"{r['delta_pct']:>+7.2f} {r['frac_better']:>8.1f}")

    # LaTeX
    print("\n\n% LaTeX table (per-condition averaged)")
    print(r"\begin{tabular}{ll ccccc}")
    print(r"\toprule")
    print(r"Model & Setting & Acc.\% & KID$_{\text{sel}}$ & KID$_{\text{rand}}$ & $\Delta$\% & \%better \\")
    print(r"\midrule")
    cur_ds = None
    for ds, model, setting, r in results:
        if ds != cur_ds:
            print(rf"\multicolumn{{6}}{{l}}{{\emph{{{ds}}}}} \\")
            cur_ds = ds
        print(f"{model:<16} & {setting:<7} & "
              f"{r['acc']:5.1f} & "
              f"{r['kid_sel']:.3f}{{\\tiny$\\pm${r['kid_sel_std']:.3f}}} & "
              f"{r['kid_rand']:.3f}{{\\tiny$\\pm${r['kid_rand_std']:.3f}}} & "
              f"${r['delta_pct']:+.1f}$ & {r['frac_better']:.0f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
