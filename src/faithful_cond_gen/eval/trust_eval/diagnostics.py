"""
Diagnostic utilities for trust evaluation.

Contains transitivity checks and other debugging utilities.
"""

from typing import Dict, List

import numpy as np
import torch

from faithful_cond_gen.eval.trust_eval.config import FEATURE_CONFIGS
from faithful_cond_gen.eval.trust_eval.feature_io import (
    load_features_for_dataset,
    verify_feature_ordering,
)
from faithful_cond_gen.eval.trust_eval.metrics_kid import calculate_kid_same_m


def print_cosine_kid_transitivity_checks(
    dataset: str,
    model: str,
    condition_keys: List[str],
    ref_trust_results: Dict,
    n_conditions: int = 3,
    k: int = 256,
    seed: int = 0,
):
    """
    Console-only sanity check for cosine-KID scales and "triangulation" across:
      - KID_rr: cosineKID(real_dino_A, real_dino_B)
      - KID_rg_dino: cosineKID(real_dino_A, gen_dino)
      - KID_rg_aligned: cosineKID(real_dino_A, gen_aligned)
      - KID_gg_cross: cosineKID(gen_dino, gen_aligned)

    Also prints:
      - feature L2 norms (pre-normalization)
      - paired cosine similarities for (real-real), (real-gen_dino), (real-gen_aligned), (gen_dino-gen_aligned)
      - a basic filename-based alignment check (if metadata has filenames/paths)
    """
    print("\n" + "=" * 80)
    print(
        f"[TRANSITIVITY CHECK] dataset={dataset} model={model}  (cosine-KID + paired cosines)"
    )
    print("=" * 80)

    # Must have aligned_mean config to run
    if (dataset, model, "aligned_mean") not in FEATURE_CONFIGS or (
        dataset,
        model,
        "dinov3",
    ) not in FEATURE_CONFIGS:
        print(
            "  Skipping: this model/dataset does not have both dinov3 and aligned_mean feature caches."
        )
        return

    # Load features
    real_feats, real_meta, gen_dino_feats, gen_dino_meta = load_features_for_dataset(
        dataset, model, "dinov3"
    )
    _, _, gen_aligned_feats, gen_aligned_meta = load_features_for_dataset(
        dataset, model, "aligned_mean"
    )

    if real_feats is None or gen_dino_feats is None or gen_aligned_feats is None:
        print("  Skipping: failed to load one of real/dino_gen/aligned_gen features.")
        return

    # Verify feature ordering between dinov3 and aligned_mean caches
    # This check happens once at load-time as required
    try:
        verify_feature_ordering(
            gen_dino_meta, gen_aligned_meta, "gen_dinov3", "gen_aligned_mean"
        )
    except ValueError as e:
        print(f"  ERROR: {e}")
        print("  Transitivity check aborted due to ordering mismatch.")
        return

    # Convert to numpy
    real_np = real_feats.numpy()
    gen_dino_np = gen_dino_feats.numpy()
    gen_aligned_np = gen_aligned_feats.numpy()

    # Length check (should pass if verification passed)
    if len(gen_dino_np) != len(gen_aligned_np):
        print(
            f"  ERROR: dinov3 ({len(gen_dino_np)}) and aligned_mean ({len(gen_aligned_np)}) "
            f"gen feature lengths differ. Transitivity check aborted."
        )
        return

    # Build gen indices by condition from trust_results ordering (assumed same ordering as gen feature cache)
    if "true_conditions" not in ref_trust_results:
        print("  Skipping: ref_trust_results missing true_conditions.")
        return

    gen_conditions = ref_trust_results["true_conditions"]
    if len(gen_conditions) != len(gen_dino_np):
        print(
            f"  WARNING: true_conditions length ({len(gen_conditions)}) != gen feats length ({len(gen_dino_np)})."
        )
        print(
            "           Results may be misindexed. Consider aligning conditions by filenames in metadata."
        )
        # still proceed, but clip to min
        n = min(len(gen_conditions), len(gen_dino_np))
        gen_conditions = gen_conditions[:n]
        gen_dino_np = gen_dino_np[:n]
        gen_aligned_np = gen_aligned_np[:n]

    gen_by_cond = {}
    for i, cond in enumerate(gen_conditions):
        gen_by_cond.setdefault(cond, []).append(i)

    # Build real indices by condition from real_meta
    real_by_cond = {}
    for i in range(len(real_np)):
        cond = tuple(
            int(
                real_meta[key][i].item()
                if isinstance(real_meta[key][i], torch.Tensor)
                else real_meta[key][i]
            )
            for key in condition_keys
        )
        real_by_cond.setdefault(cond, []).append(i)

    # Select candidate conditions with enough support
    candidates = []
    for cond, gidx in gen_by_cond.items():
        ridx = real_by_cond.get(cond, [])
        if len(ridx) >= 2 * max(10, min(k, 500)) and len(gidx) >= max(10, min(k, 500)):
            candidates.append(cond)

    if not candidates:
        print("  No conditions have sufficient real/gen support for this check.")
        return

    # Pick a few conditions deterministically (sorted) for reproducibility
    candidates = sorted(candidates)
    selected = candidates[:n_conditions]

    # Helpers
    def _l2norm_stats(X):
        n = np.linalg.norm(X, axis=1)
        return float(np.median(n)), float(np.mean(n)), float(np.std(n))

    def _paired_cos_mean(A, B):
        # paired dot products after L2 normalization
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return float(np.mean(np.sum(A * B, axis=1)))

    rng = np.random.default_rng(seed)

    for cond in selected:
        ridx = np.array(real_by_cond[cond], dtype=int)
        gidx = np.array(gen_by_cond[cond], dtype=int)

        # Choose effective sample size
        k_eff = min(k, len(gidx), len(ridx) // 2, 500)
        if k_eff < 10:
            continue

        # Sample real split A/B and gen indices
        perm_r = rng.permutation(ridx)[: 2 * k_eff]
        real_a = real_np[perm_r[:k_eff]]
        real_b = real_np[perm_r[k_eff:]]
        chosen_g = rng.choice(gidx, size=k_eff, replace=False)
        gen_d = gen_dino_np[chosen_g]
        gen_a = gen_aligned_np[chosen_g]

        # Cosine-KIDs
        kid_rr = calculate_kid_same_m(real_a, real_b, use_cosine=True)
        kid_rg_dino = calculate_kid_same_m(real_a, gen_d, use_cosine=True)
        kid_rg_aligned = calculate_kid_same_m(real_a, gen_a, use_cosine=True)
        kid_gg_cross = calculate_kid_same_m(gen_d, gen_a, use_cosine=True)

        # Paired cosine similarity sanity checks
        cos_rr = _paired_cos_mean(real_a, real_b)
        cos_rg_d = _paired_cos_mean(real_a, gen_d)
        cos_rg_a = _paired_cos_mean(real_a, gen_a)
        cos_gg = _paired_cos_mean(gen_d, gen_a)
        # Permuted comparison: shuffle gen_aligned to break the matching
        perm_idx = rng.permutation(k_eff)
        cos_gg_permuted = _paired_cos_mean(gen_d, gen_a[perm_idx])

        # Norm stats (pre-normalization; should not matter for cosine-kid but helps spot degeneracy)
        real_med, real_mean, real_std = _l2norm_stats(real_a)
        gd_med, gd_mean, gd_std = _l2norm_stats(gen_d)
        ga_med, ga_mean, ga_std = _l2norm_stats(gen_a)

        # Print
        cond_str = ", ".join(f"{key}={v}" for key, v in zip(condition_keys, cond))
        print("\n" + "-" * 80)
        print(
            f"Condition: {cond_str}   (k={k_eff}, n_real={len(ridx)}, n_gen={len(gidx)})"
        )
        print("Cosine-KID (MMD^2, lower better; small negatives are OK):")
        print(f"  KID_rr          = {kid_rr:.6f}   [real_dino vs real_dino]")
        print(f"  KID_rg_dino     = {kid_rg_dino:.6f}   [real_dino vs gen_dino]")
        print(f"  KID_rg_aligned  = {kid_rg_aligned:.6f}   [real_dino vs gen_aligned]")
        print(f"  KID_gg_cross    = {kid_gg_cross:.6f}   [gen_dino  vs gen_aligned]")
        print("Paired mean cosine similarities (higher is more aligned directionally):")
        print(f"  mean cos(real_a, real_b)      = {cos_rr:.4f}")
        print(f"  mean cos(real_a, gen_dino)    = {cos_rg_d:.4f}")
        print(f"  mean cos(real_a, gen_aligned) = {cos_rg_a:.4f}")
        print(
            f"  mean cos(gen_dino, gen_aligned)         = {cos_gg:.4f}  [matched pairs]"
        )
        print(
            f"  mean cos(gen_dino, gen_aligned_permuted)= {cos_gg_permuted:.4f}  [random pairs]"
        )
        print("Median/mean/std of L2 norms (pre-normalization):")
        print(
            f"  real_a  norm: med={real_med:.3f} mean={real_mean:.3f} std={real_std:.3f}"
        )
        print(f"  gen_dino norm: med={gd_med:.3f} mean={gd_mean:.3f} std={gd_std:.3f}")
        print(f"  gen_algn norm: med={ga_med:.3f} mean={ga_mean:.3f} std={ga_std:.3f}")

        # Minimal "triangulation" commentary (console only, not a statistical claim)
        if (
            np.isfinite(kid_rg_dino)
            and np.isfinite(kid_rg_aligned)
            and abs(kid_rg_dino) > 1e-12
        ):
            ratio = kid_rg_aligned / kid_rg_dino
            print(f"Scale check: KID_rg_aligned / KID_rg_dino = {ratio:.2f}x")
            if cos_gg > 0.85 and ratio > 5.0:
                print(
                    "  NOTE: high gen_dino↔gen_aligned cosine but much larger cross-space KID suggests distribution-level mismatch"
                )
                print(
                    "        (not just feature norm), or ordering misalignment if metadata alignment is imperfect."
                )

    print("\n[TRANSITIVITY CHECK DONE]\n")
