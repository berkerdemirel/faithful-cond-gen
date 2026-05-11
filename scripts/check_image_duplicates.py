"""Check for duplicate images using perceptual hashing."""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CONDITION_KEYS = ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]

CONFIGS = {
    "vanilla_full": "celeba_vanilla_full",
    "vanilla_marginal": "celeba_vanilla_marginal",
    "repa_full": "celeba_repa_full",
    "repa_marginal": "celeba_repa_marginal",
}


def condition_to_signature(condition: tuple) -> str:
    """Same as in trust_eval_extensions.py"""
    pairs = list(zip(CONDITION_KEYS, condition))
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    parts = [f"{k}{v}" for k, v in pairs_sorted]
    return "_".join(parts)


def image_hash(img: Image.Image, size: int = 8) -> str:
    """Simple average hash - resize to small, convert to binary."""
    img = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return hex(int(bits, 2))


def main():
    for model_name, model_dir in CONFIGS.items():
        image_dir = Path(f"outputs/gen/{model_dir}/images")
        if not image_dir.exists():
            print(f"\n{model_name}: image dir not found at {image_dir}")
            continue

        # Load feature metadata to get conditions
        feat_path = Path(f"outputs/gen/{model_dir}/dinov3_meanpatch_features.pt")
        data = torch.load(feat_path, map_location="cpu", weights_only=False)
        gen_meta = data["metadata"]
        n = len(data["features"])

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

        # Check duplicates per condition using image hashing
        total_unique = 0
        total_original = 0
        cond_stats = {}

        for cond in sorted(cond_to_indices.keys()):
            indices = cond_to_indices[cond]
            sig = condition_to_signature(cond)

            hashes = set()
            hash_to_indices = defaultdict(list)

            for local_idx, global_idx in enumerate(indices):
                img_path = image_dir / f"{sig}_{local_idx}.png"
                if img_path.exists():
                    img = Image.open(img_path)
                    h = image_hash(img)
                    hashes.add(h)
                    hash_to_indices[h].append(local_idx)

            n_unique = len(hashes)
            n_orig = len(indices)
            n_dupes = n_orig - n_unique

            total_unique += n_unique
            total_original += n_orig
            cond_stats[cond] = (n_orig, n_unique, n_dupes)

        print(f"\n{'='*70}")
        print(f"{model_name}: {total_original} -> {total_unique} unique ({total_original - total_unique} duplicates)")
        print(f"{'='*70}")
        print(f"{'Condition':<45} {'Orig':>6} {'Uniq':>6} {'Dupe':>6}")
        print("-" * 70)
        for cond in sorted(cond_stats.keys()):
            n_orig, n_unique, n_dupes = cond_stats[cond]
            cond_str = ", ".join(f"{k}={v}" for k, v in zip(CONDITION_KEYS, cond))
            print(f"{cond_str:<45} {n_orig:>6} {n_unique:>6} {n_dupes:>6}")


if __name__ == "__main__":
    main()
