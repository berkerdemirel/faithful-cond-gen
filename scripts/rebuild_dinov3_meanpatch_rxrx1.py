"""Rebuild dinov3_meanpatch_features.pt for rxrx1 gen models with sharding.

Usage (single shard):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 uv run python scripts/rebuild_dinov3_meanpatch_rxrx1.py \
      --model_dir outputs/gen/rxrx1_vanilla_full --shard_id 0 --n_shards 2

After all shards for a model finish, run with --merge to combine them:
  PYTHONPATH=src uv run python scripts/rebuild_dinov3_meanpatch_rxrx1.py \
      --model_dir outputs/gen/rxrx1_vanilla_full --merge --n_shards 2
"""
import argparse
import glob
import os
import re

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from faithful_cond_gen.data.rxrx1 import to_rgb
from faithful_cond_gen.model.repa_encoder import REPAEncoder


class ShardedRxRx1RawDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        img = torch.load(fpath, map_location="cpu", weights_only=False)
        if img.shape[0] == 6:
            img = to_rgb(img.unsqueeze(0))[0]
        if img.max() > 1.0:
            img = img / 255.0
        basename = os.path.basename(fpath)
        m = re.match(r"cell(\d+)_sirna(\d+)_", basename)
        cell_type_id = torch.tensor(int(m.group(1)), dtype=torch.long)
        sirna_id = torch.tensor(int(m.group(2)), dtype=torch.long)
        return img, cell_type_id, sirna_id, basename


@torch.no_grad()
def extract_shard(model_dir: str, shard_id: int, n_shards: int, batch_size: int = 128):
    images_dir = os.path.join(model_dir, "images")
    all_files = sorted(glob.glob(os.path.join(images_dir, "*.pt")))
    assert len(all_files) > 0, f"No .pt files in {images_dir}"

    shard_files = all_files[shard_id::n_shards]
    print(f"[{model_dir} shard {shard_id}/{n_shards}] {len(shard_files)} / {len(all_files)} files")

    ds = ShardedRxRx1RawDataset(shard_files)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True, persistent_workers=True,
    )

    device = "cuda:0"
    enc = REPAEncoder(
        encoder_name="dinov3-vit-l",
        resolution=256, in_channels=3, target_grid=16, device=device,
    )
    enc.eval()

    feats_list = []
    ct_list = []
    sirna_list = []
    fn_list = []
    for imgs, ct, sirna, fns in tqdm(loader, desc=f"shard {shard_id}"):
        imgs = imgs.to(device, non_blocking=True)
        if imgs.shape[1] == 6:
            imgs = to_rgb(imgs)
        out = enc(imgs).mean(dim=1).cpu()
        feats_list.append(out)
        ct_list.append(ct)
        sirna_list.append(sirna)
        fn_list.extend(list(fns))

    feats = torch.cat(feats_list, dim=0)
    ct_all = torch.cat(ct_list, dim=0)
    sirna_all = torch.cat(sirna_list, dim=0)

    shard_path = os.path.join(model_dir, f".dinov3_meanpatch_shard{shard_id}_of{n_shards}.pt")
    torch.save({
        "features": feats,
        "cell_type_id": ct_all,
        "sirna_id": sirna_all,
        "filenames": fn_list,
    }, shard_path)
    print(f"[{model_dir} shard {shard_id}] saved {feats.shape[0]} -> {shard_path}")


def merge_shards(model_dir: str, n_shards: int, out_name: str = "dinov3_meanpatch_features_rebuilt.pt"):
    shards = []
    for i in range(n_shards):
        p = os.path.join(model_dir, f".dinov3_meanpatch_shard{i}_of{n_shards}.pt")
        shards.append(torch.load(p, map_location="cpu", weights_only=False))

    # Interleave back to global sorted filename order
    # Each shard contains files at indices [shard_id::n_shards] of the global sorted list
    all_files = sorted(glob.glob(os.path.join(model_dir, "images", "*.pt")))
    n = len(all_files)
    feat_dim = shards[0]["features"].shape[1]
    features = torch.empty((n, feat_dim), dtype=shards[0]["features"].dtype)
    cell_type_id = torch.empty((n,), dtype=torch.long)
    sirna_id = torch.empty((n,), dtype=torch.long)

    for i in range(n_shards):
        shard = shards[i]
        global_idx = list(range(i, n, n_shards))
        features[global_idx] = shard["features"]
        cell_type_id[torch.tensor(global_idx, dtype=torch.long)] = shard["cell_type_id"]
        sirna_id[torch.tensor(global_idx, dtype=torch.long)] = shard["sirna_id"]
        # sanity: filenames in shard must match global_idx basenames
        for local, g in enumerate(global_idx):
            assert shard["filenames"][local] == os.path.basename(all_files[g]), (
                f"filename mismatch at shard {i} local {local}: "
                f"{shard['filenames'][local]} vs {os.path.basename(all_files[g])}"
            )

    payload = {
        "features": features,
        "metadata": {"cell_type_id": cell_type_id, "sirna_id": sirna_id},
        "encoder_name": "dinov3-vit-l_meanpatch",
        "feature_dim": feat_dim,
    }
    out_path = os.path.join(model_dir, out_name)
    torch.save(payload, out_path)
    print(f"[{model_dir}] merged {n} samples -> {out_path}")

    for i in range(n_shards):
        os.remove(os.path.join(model_dir, f".dinov3_meanpatch_shard{i}_of{n_shards}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_shards(args.model_dir, args.n_shards)
    else:
        extract_shard(args.model_dir, args.shard_id, args.n_shards, args.batch_size)
