"""
Aggregate trust evaluation across scoring methods into the FPR95 selection table.

Loads `detailed_results_<dataset>.pt` from one or more output dirs (one per
scoring method) and for each method prints:

1. A LaTeX-shaped FPR@95 selection table matching the paper format:
   Model | Setting | Acc.% | KID_sel ± std | KID_rand ± std | Δ% (KID improvement)
2. Per-config Spearman ρ(per-condition trust_mean, ΔKID) — ranking validity.
3. Cross-method Spearman ρ on per-condition trust_mean — how much scorers agree.

Usage:
  PYTHONPATH=src uv run python scripts/aggregate_scorer_ablations.py \\
      --dataset celeba \\
      --runs mahalanobis=outputs/trust_evaluation \\
             linear_probe=outputs/trust_evaluation_linear_probe \\
             knn_per_attr=outputs/trust_evaluation_knn_per_attr
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from scipy.stats import spearmanr


# Display names for the paper-style table.
MODEL_LABELS = {
    "vanilla_full":         ("Vanilla",       "full"),
    "vanilla_marginal":     ("Vanilla",       "stress"),
    "repa_full":            ("REPA (DINOv3)", "full"),
    "repa_marginal":        ("REPA (DINOv3)", "stress"),
    "repa_siglip_full":     ("REPA (SigLIP)", "full"),
    "repa_siglip_marginal": ("REPA (SigLIP)", "stress"),
}
# RxRx1 uses "heldout" instead of "stress" for marginal models.
RXRX1_LABEL_OVERRIDE = {"stress": "heldout"}
MODEL_ORDER = [
    "vanilla_full", "vanilla_marginal",
    "repa_full", "repa_marginal",
    "repa_siglip_full", "repa_siglip_marginal",
]


def load_run(path: Path, dataset: str) -> Dict:
    f = path / f"detailed_results_{dataset}.pt"
    if not f.exists():
        sys.exit(f"missing {f}")
    return torch.load(f, map_location="cpu", weights_only=False)


def parse_runs(items):
    out = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"--runs expects name=path, got {item!r}")
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def _fmt_num_std(mean: float, std: float) -> str:
    if mean != mean:  # NaN
        return "—"
    if std is None or std != std:
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


def _fmt_pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.1f}"


def _fmt_delta(kid_sel: float, kid_rand: float) -> str:
    if kid_sel != kid_sel or kid_rand != kid_rand or kid_rand == 0:
        return "—"
    return f"{100 * (kid_rand - kid_sel) / kid_rand:+.1f}"


def _model_setting(config_key: str, dataset: str) -> Tuple[str, str]:
    model = config_key.split("/")[0]
    label = MODEL_LABELS.get(model)
    if label is None:
        return model, "—"
    name, setting = label
    if dataset == "rxrx1":
        setting = RXRX1_LABEL_OVERRIDE.get(setting, setting)
    return name, setting


def build_fpr_table(d: Dict, dataset: str) -> pd.DataFrame:
    rows = []
    fpr = d.get("fpr95_results", {})
    # Restrict to the 6 core models and sort in preferred order.
    configs = [c for c in fpr.keys() if c.split("/")[0] in MODEL_ORDER]
    configs = sorted(configs, key=lambda c: MODEL_ORDER.index(c.split("/")[0]))
    for cfg in configs:
        r = fpr.get(cfg, {})
        model, setting = _model_setting(cfg, dataset)
        rows.append({
            "Model": model,
            "Setting": setting,
            "Acc.%": _fmt_pct(r.get("acceptance_rate", float("nan"))),
            "KID_sel": _fmt_num_std(r.get("kid_raw_accepted", float("nan")),
                                    r.get("kid_raw_accepted_std", float("nan"))),
            "KID_rand": _fmt_num_std(r.get("kid_raw_random", float("nan")),
                                     r.get("kid_raw_random_std", float("nan"))),
            "Δ%": _fmt_delta(r.get("kid_raw_accepted", float("nan")),
                             r.get("kid_raw_random", float("nan"))),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="celeba")
    p.add_argument("--runs", nargs="+", required=True,
                   help="method=output_dir pairs")
    args = p.parse_args()

    runs = parse_runs(args.runs)
    loaded = {name: load_run(path, args.dataset) for name, path in runs.items()}

    cfg_sets = [set(d["ranking_results"].keys()) for d in loaded.values()]
    configs = sorted(set.intersection(*cfg_sets))
    # Restrict to the 6 core models for the cross-method tables.
    configs = [c for c in configs if c.split("/")[0] in MODEL_ORDER]
    configs = sorted(configs, key=lambda c: MODEL_ORDER.index(c.split("/")[0]))
    if not configs:
        sys.exit("no shared configs across runs")

    # 1. Per-method FPR@95 selection table (paper format).
    for name, d in loaded.items():
        tbl = build_fpr_table(d, args.dataset)
        print("=" * 78)
        print(f"FPR@95 selection table — scorer: {name}  (dataset: {args.dataset})")
        print("=" * 78)
        print(tbl.to_string(index=False))
        print()

    # 2. Ranking validity: Spearman ρ(per-condition trust_mean, ΔKID).
    print("=" * 78)
    print("Layer 1 — Spearman ρ(trust_mean per condition, ΔKID) across scorers")
    print("=" * 78)
    rows = []
    for cfg in configs:
        row = {"config": cfg}
        for name, d in loaded.items():
            row[name] = d["ranking_results"][cfg].get("spearman_rho", float("nan"))
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format="%+.3f"))
    print()

    # 3. Cross-method ranking correlation on per-condition trust_mean.
    print("=" * 78)
    print("Cross-method Spearman ρ on per-condition trust_mean")
    print("=" * 78)
    methods = list(loaded)
    for cfg in configs:
        trust_by_method = {}
        for name in methods:
            df = loaded[name]["ranking_results"][cfg]["cond_stats"]
            trust_by_method[name] = df.set_index("condition")["trust_mean"]
        merged = pd.DataFrame(trust_by_method).dropna()
        if merged.empty:
            continue
        mat = pd.DataFrame(index=methods, columns=methods, dtype=float)
        for i, a in enumerate(methods):
            for j, b in enumerate(methods):
                if i <= j:
                    rho, _ = spearmanr(merged[a].values, merged[b].values)
                    mat.loc[a, b] = rho
                    mat.loc[b, a] = rho
        print(f"\n{cfg} (n={len(merged)} conditions):")
        print(mat.to_string(float_format="%+.3f"))


if __name__ == "__main__":
    main()
