"""Check for near-duplicate images using cosine similarity on features."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CONDITION_KEYS = ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]

CONFIGS = {
    "vanilla_full": "celeba_vanilla_full",
    "vanilla_marginal": "celeba_vanilla_marginal",
    "repa_full": "celeba_repa_full",
    "repa_marginal": "celeba_repa_marginal",
}

# Cosine similarity threshold - samples above this are "duplicates"
SIMILARITY_THRESHOLDS = [0.99, 0.995, 0.999]


def count_near_duplicates(feats: torch.Tensor, threshold: float) -> int:
    """Count unique samples after removing near-duplicates above threshold."""
    feats = F.normalize(feats, dim=1)
    n = len(feats)

    # For efficiency, process in chunks
    keep_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep_mask[i]:
            continue
        # Compare with remaining samples
        sims = (feats[i:i+1] @ feats[i+1:].T).squeeze(0)
        duplicates = (sims > threshold).numpy()
        # Mark duplicates for removal (keep first occurrence)
        dup_indices = np.where(duplicates)[0] + i + 1
        keep_mask[dup_indices] = False

    return int(keep_mask.sum())


def main():
    for model_name, model_dir in CONFIGS.items():
        feat_path = Path(f"outputs/gen/{model_dir}/dinov3_meanpatch_features.pt")
        if not feat_path.exists():
            print(f"\n{model_name}: features not found")
            continue

        data = torch.load(feat_path, map_location="cpu", weights_only=False)
        gen_feats = data["features"]
        gen_meta = data["metadata"]
        n = len(gen_feats)

        # Build conditions
        conditions = []
        for i in range(n):
            cond = tuple(
                int(gen_meta[k][i].item() if isinstance(gen_meta[k][i], torch.Tensor) else gen_meta[k][i])
                for k in CONDITION_KEYS
            )
            conditions.append(cond)

        # Group by condition
        cond_to_indices = defaultdict(list)
        for i, cond in enumerate(conditions):
            cond_to_indices[cond].append(i)

        print(f"\n{'='*80}")
        print(f"{model_name}")
        print(f"{'='*80}")

        for thresh in SIMILARITY_THRESHOLDS:
            total_unique = 0
            print(f"\nThreshold: cosine > {thresh}")
            print(f"{'Condition':<45} {'Orig':>6} {'Uniq':>6} {'Dupe':>6}")
            print("-" * 70)

            for cond in sorted(cond_to_indices.keys()):
                indices = cond_to_indices[cond]
                feats = gen_feats[indices]
                n_unique = count_near_duplicates(feats, thresh)
                n_dupes = len(indices) - n_unique
                total_unique += n_unique

                cond_str = ", ".join(f"{k}={v}" for k, v in zip(CONDITION_KEYS, cond))
                print(f"{cond_str:<45} {len(indices):>6} {n_unique:>6} {n_dupes:>6}")

            print(f"\nTotal: {n} -> {total_unique} unique ({n - total_unique} duplicates)")


if __name__ == "__main__":
    main()
