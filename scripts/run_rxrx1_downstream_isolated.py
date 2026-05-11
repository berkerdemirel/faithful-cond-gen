"""Isolated RxRx1 downstream bin-selection evaluation for marginal models.
Always uses dinov3 features for classification, scoring-space features for binning.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from faithful_cond_gen.eval.trust_eval.config import (
    CONDITION_ATTRS, RXRX1_HELDOUT_PAIRS,
)
from faithful_cond_gen.eval.trust_eval.feature_io import load_features_for_dataset
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    compute_trust_results_from_features,
)
from faithful_cond_gen.eval.trust_eval.eval_layers.downstream import (
    evaluate_rxrx1_downstream_bin_selection,
)

ck = CONDITION_ATTRS["rxrx1"]
out = Path("outputs/rxrx1_downstream_isolated_v2")
out.mkdir(parents=True, exist_ok=True)

configs = [
    ("vanilla_marginal", "dinov3"),
    ("repa_marginal", "dinov3"),
    ("repa_marginal", "aligned_mean"),
    ("repa_siglip_marginal", "dinov3"),
    ("repa_siglip_marginal", "aligned_mean"),
]

for model, ft in configs:
    print(f"\n{'='*60}")
    print(f"{model}/{ft}")
    print(f"{'='*60}", flush=True)

    # Load scoring-space features
    real_f, real_m, gen_f, gen_m = load_features_for_dataset("rxrx1", model, ft, normalize_mode="l2")
    if real_f is None or gen_f is None:
        print("  SKIP: features not found")
        continue

    # Always load dinov3 for classifier
    if ft != "dinov3":
        ds_real, ds_real_m, ds_gen, ds_gen_m = load_features_for_dataset("rxrx1", model, "dinov3", normalize_mode="l2")
    else:
        ds_real, ds_real_m, ds_gen, ds_gen_m = real_f, real_m, gen_f, gen_m

    # Compute seen combos for marginal
    all_combos = set()
    for i in range(len(real_f)):
        c = tuple(
            int(real_m[k][i].item() if isinstance(real_m[k][i], torch.Tensor) else real_m[k][i])
            for k in ck
        )
        all_combos.add(c)
    seen_combos = all_combos - RXRX1_HELDOUT_PAIRS
    print(f"  {len(all_combos)} total, {len(seen_combos)} seen", flush=True)

    # Compute trust scores in scoring space
    print("  Computing trust scores...", flush=True)
    trust_res = compute_trust_results_from_features(
        "rxrx1", model, ft, real_f, real_m, gen_f, gen_m, ck,
        filter_by_seen=True, seen_combos=seen_combos,
    )

    # Run downstream: 20 pairs (5 heldout + 15 seen), 5 per cell type
    print("  Running downstream bin-selection (20pairs)...", flush=True)
    evaluate_rxrx1_downstream_bin_selection(
        trust_results=trust_res,
        gen_feats=ds_gen,
        gen_meta=ds_gen_m,
        real_feats=ds_real,
        real_meta=ds_real_m,
        output_dir=out,
        config_key=f"{model}/{ft}",
        mode="100pairs",
        dataset="rxrx1",
        n_pairs=20,
        n_heldout=5,
    )

print("\nDone.", flush=True)
