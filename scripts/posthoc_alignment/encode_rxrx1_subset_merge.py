"""Encode per-cond shards from diag_subset and merge with the balanced50
overlap rows already persisted in feature_cache_rxrx1_subset/.

Produces the final 10000-row cache:
    outputs/posthoc_alignment/feature_cache_rxrx1_subset/{model_key}_encoded.pt

Schema: {gen_siglip, gen_dinov3, gen_hidden, gen_meta: {cell_type_id, sirna_id}}
matching the existing balanced50 cache format.

Usage:
    PYTHONPATH=src uv run python \\
        scripts/posthoc_alignment/encode_rxrx1_subset_merge.py \\
        --checkpoint-key rxrx1_vanilla_marginal_v1

    PYTHONPATH=src uv run python \\
        scripts/posthoc_alignment/encode_rxrx1_subset_merge.py \\
        --checkpoint-key rxrx1_repa_siglip_marginal_v1
"""

import argparse
import glob
from pathlib import Path

import torch

from faithful_cond_gen.data.rxrx1 import to_rgb
from faithful_cond_gen.model.repa_encoder import REPAEncoder

REPO = Path(__file__).resolve().parents[2]
CK = ["cell_type_id", "sirna_id"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode_batch(images, encoder_name, bs=64):
    enc = REPAEncoder(encoder_name=encoder_name, resolution=256,
                      in_channels=3, device=str(DEVICE))
    enc.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(images), bs):
            e = min(s + bs, len(images))
            batch = images[s:e].to(DEVICE)
            if batch.shape[1] == 6:
                batch = to_rgb(batch)
            out.append(enc(batch).mean(dim=1).cpu())
    del enc
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out)


def load_shards(shard_dir):
    pts = sorted(glob.glob(str(shard_dir / "cond_*.pt")))
    if not pts:
        return None
    imgs, hids, cts, srs = [], [], [], []
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        imgs.append(d["images"])
        hids.append(d["raw_hidden"])
        n = d["images"].shape[0]
        ct, sr = d["condition"]
        cts.append(torch.full((n,), int(ct), dtype=torch.long))
        srs.append(torch.full((n,), int(sr), dtype=torch.long))
    return {
        "images": torch.cat(imgs),
        "gen_hidden": torch.cat(hids),
        "cell_type_id": torch.cat(cts),
        "sirna_id": torch.cat(srs),
        "n_conds": len(pts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-key", required=True)
    args = parser.parse_args()
    key = args.checkpoint_key

    shard_dir = REPO / f"outputs/posthoc_alignment/diag_subset/{key}/gen_cache"
    partial_path = REPO / f"outputs/posthoc_alignment/feature_cache_rxrx1_subset/{key}_encoded.pt"
    out_path = partial_path

    print(f"[{key}]")
    print(f"  shards:  {shard_dir}")
    print(f"  partial: {partial_path}")

    if partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        partial_pairs = set(
            zip(partial["gen_meta"]["cell_type_id"].tolist(),
                partial["gen_meta"]["sirna_id"].tolist())
        )
        print(f"  partial rows: {partial['gen_hidden'].shape[0]}, unique pairs: {len(partial_pairs)}")
    else:
        print("  no partial cache found -- writing fresh encoded cache from shards only")
        partial = {
            "gen_siglip": torch.empty(0, 1152),
            "gen_dinov3": torch.empty(0, 1024),
            "gen_hidden": torch.empty(0, 768),
            "gen_meta": {
                "cell_type_id": torch.empty(0, dtype=torch.long),
                "sirna_id": torch.empty(0, dtype=torch.long),
            },
        }
        partial_pairs = set()

    shards = load_shards(shard_dir)
    if shards is None:
        print("  no shards found -- nothing to merge")
        return
    n_new = shards["gen_hidden"].shape[0]
    print(f"  new shards: {n_new} rows from {shards['n_conds']} conds")

    new_pairs = set(zip(shards["cell_type_id"].tolist(), shards["sirna_id"].tolist()))
    overlap = partial_pairs & new_pairs
    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping pairs between partial and new shards -- new shards will take precedence")

    print("  encoding SigLIP...")
    new_siglip = encode_batch(shards["images"], "siglip")
    print("  encoding DINOv3...")
    new_dinov3 = encode_batch(shards["images"], "dinov3-vit-l")

    if overlap:
        mask = torch.tensor(
            [(int(a), int(b)) not in new_pairs
             for a, b in zip(partial["gen_meta"]["cell_type_id"].tolist(),
                              partial["gen_meta"]["sirna_id"].tolist())],
            dtype=torch.bool,
        )
        partial = {
            "gen_siglip": partial["gen_siglip"][mask],
            "gen_dinov3": partial["gen_dinov3"][mask],
            "gen_hidden": partial["gen_hidden"][mask],
            "gen_meta": {
                "cell_type_id": partial["gen_meta"]["cell_type_id"][mask],
                "sirna_id": partial["gen_meta"]["sirna_id"][mask],
            },
        }

    merged = {
        "gen_siglip": torch.cat([partial["gen_siglip"], new_siglip]),
        "gen_dinov3": torch.cat([partial["gen_dinov3"], new_dinov3]),
        "gen_hidden": torch.cat([partial["gen_hidden"], shards["gen_hidden"]]),
        "gen_meta": {
            "cell_type_id": torch.cat(
                [partial["gen_meta"]["cell_type_id"], shards["cell_type_id"]]
            ),
            "sirna_id": torch.cat(
                [partial["gen_meta"]["sirna_id"], shards["sirna_id"]]
            ),
        },
    }
    final_pairs = set(zip(merged["gen_meta"]["cell_type_id"].tolist(),
                          merged["gen_meta"]["sirna_id"].tolist()))
    print(f"  final rows: {merged['gen_hidden'].shape[0]}, unique pairs: {len(final_pairs)}")

    torch.save(merged, out_path)
    print(f"  wrote -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
