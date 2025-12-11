#!/usr/bin/env python
import argparse
import os
from typing import Dict, Optional, Tuple

import pandas as pd
from faithful_cond_gen.data.celeba import CelebaDataConfig, CelebaDataModule


def test_split_leakage_celeba(cache_dir: str | None = None) -> None:
    """Check that train/val/test splits do not overlap.

    We first report true sample counts, then (optionally) check ID leakage
    if we can find a per-image ID column.
    """
    cfg = CelebaDataConfig(cache_dir=cache_dir)
    dm = CelebaDataModule(cfg)

    n_train = len(dm.ds_train)
    n_val = len(dm.ds_val)
    n_test = len(dm.ds_test)

    print(
        f"[test_split_leakage_celeba] Sample counts: "
        f"train={n_train}, val={n_val}, test={n_test}"
    )

    # Optional: look for an ID column that appears unique per split
    sample = dm.ds_train[0]
    id_key = None
    for cand in ["image_id", "img_id"]:
        if cand in sample:
            id_key = cand
            break

    if id_key is None:
        print(
            "[test_split_leakage_celeba] No per-image ID column found "
            "(image_id/img_id). Skipping ID-based leakage check."
        )
        return

    ids_train = set(dm.ds_train[id_key])
    ids_val = set(dm.ds_val[id_key])
    ids_test = set(dm.ds_test[id_key])

    inter_train_val = ids_train & ids_val
    inter_train_test = ids_train & ids_test
    inter_val_test = ids_val & ids_test

    assert (
        len(inter_train_val) == 0
    ), f"CelebA train/val leakage: {len(inter_train_val)} overlapping IDs."
    assert (
        len(inter_train_test) == 0
    ), f"CelebA train/test leakage: {len(inter_train_test)} overlapping IDs."
    assert (
        len(inter_val_test) == 0
    ), f"CelebA val/test leakage: {len(inter_val_test)} overlapping IDs."

    print(
        "[test_split_leakage_celeba] No ID-based leakage detected "
        f"using id_key='{id_key}'."
    )


def _check_split_consistency(
    md: pd.DataFrame,
    split_name: str,
    selected_attrs,
    train_counts: Dict[Tuple[int, ...], int],
    rare_threshold: int,
) -> None:
    """Check that comp_category matches train combo counts for a split."""
    for idx, row in md.iterrows():
        cat = row["comp_category"]
        if cat == "contradictory":
            # rule-based override; we don't enforce anything else here
            continue

        combo = tuple(int(row[a]) for a in selected_attrs)
        c = train_counts.get(combo, None)

        if c is None:
            # never seen in train -> must be unseen
            assert cat == "unseen", (
                f"{split_name} sample at idx={idx} has combo={combo} which "
                f"never occurs in train, but comp_category='{cat}', "
                f"expected 'unseen'."
            )
        else:
            expected = "rare" if c < rare_threshold else "seen"
            assert cat == expected, (
                f"{split_name} sample at idx={idx} has combo={combo} with "
                f"train count={c}, rare_threshold={rare_threshold}, "
                f"but comp_category='{cat}', expected '{expected}'."
            )


def test_composition_categories_celeba(cache_dir: Optional[str] = None) -> None:
    """Check that CelebA comp_category matches train combo counts and test unseen behavior."""
    cfg = CelebaDataConfig(cache_dir=cache_dir)
    dm = CelebaDataModule(cfg)
    selected_attrs = dm.selected_attrs
    rare_threshold = cfg.rare_threshold

    # These are built inside CelebaDataModule._compute_composition_categories
    md_train = dm._md_train.copy()
    md_val = dm._md_val.copy()
    md_test = dm._md_test.copy()

    # Allowed categories
    allowed = {"seen", "rare", "unseen", "contradictory"}

    for split_name, md in [("train", md_train), ("val", md_val), ("test", md_test)]:
        cats = set(md["comp_category"].unique())
        unknown = cats - allowed
        if unknown:
            raise AssertionError(
                f"Split '{split_name}' has unexpected comp_category values: "
                f"{unknown} (allowed: {allowed})"
            )

    # Build train combo counts
    if "combo" in md_train.columns:
        combo_col = "combo"
    else:
        md_train["combo"] = md_train[selected_attrs].apply(
            lambda row: tuple(int(row[a]) for a in selected_attrs),
            axis=1,
        )
        combo_col = "combo"

    counts = md_train[combo_col].value_counts().to_dict()

    # Base consistency checks
    _check_split_consistency(md_train, "train", selected_attrs, counts, rare_threshold)
    _check_split_consistency(md_val, "val", selected_attrs, counts, rare_threshold)
    _check_split_consistency(md_test, "test", selected_attrs, counts, rare_threshold)

    # Summary per split
    print("[test_composition_categories_celeba] Summary per split and category:")
    for split_name, md in [("train", md_train), ("val", md_val), ("test", md_test)]:
        if len(md) == 0:
            print(f"  {split_name}: 0 samples")
            continue
        counts_cat: Dict[str, int] = md["comp_category"].value_counts().to_dict()
        total = len(md)
        pretty = ", ".join(
            f"{k}={v} ({v/total:.3f})" for k, v in sorted(counts_cat.items())
        )
        print(f"  {split_name}: n={total} -> {pretty}")

    # ---- Extra: test 'unseen' behavior via synthetic held-out combo ----
    md_val_test = pd.concat([md_val, md_test], axis=0, ignore_index=True)
    if md_val_test.empty:
        print(
            "[test_composition_categories_celeba] No val/test samples; "
            "skipping unseen behavior test."
        )
        return

    # pick a combo from val/test to hold out
    candidate_row = md_val_test.iloc[0]
    candidate_combo = tuple(int(candidate_row[a]) for a in selected_attrs)

    print(
        f"[test_composition_categories_celeba] Testing unseen behavior with "
        f"held_out_combos={candidate_combo}"
    )

    cfg_unseen = CelebaDataConfig(
        cache_dir=cache_dir,
        image_size=cfg.image_size,
        augment_train=cfg.augment_train,
        normalize=cfg.normalize,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        rare_threshold=cfg.rare_threshold,
        held_out_combos=[candidate_combo],
    )
    dm_unseen = CelebaDataModule(cfg_unseen)

    md2_train = dm_unseen._md_train.copy()
    md2_val = dm_unseen._md_val.copy()
    md2_test = dm_unseen._md_test.copy()

    # recompute train combo counts in the unseen config
    if "combo" in md2_train.columns:
        combo_col2 = "combo"
    else:
        md2_train["combo"] = md2_train[selected_attrs].apply(
            lambda row: tuple(int(row[a]) for a in selected_attrs),
            axis=1,
        )
        combo_col2 = "combo"

    counts2 = md2_train[combo_col2].value_counts().to_dict()

    # sanity: candidate combo must not be in train counts now
    assert candidate_combo not in counts2, (
        f"Held-out combo {candidate_combo} still appears in train "
        f"composition counts of unseen config."
    )

    # val/test samples with candidate combo should now be 'unseen'
    def _combo_from_row(row: pd.Series) -> Tuple[int, ...]:
        return tuple(int(row[a]) for a in selected_attrs)

    md2_val_test = pd.concat([md2_val, md2_test], axis=0, ignore_index=True)
    mask_candidate = md2_val_test.apply(
        lambda r: _combo_from_row(r) == candidate_combo, axis=1
    )
    subset = md2_val_test[mask_candidate]

    if len(subset) == 0:
        print(
            "[test_composition_categories_celeba] WARNING: held-out combo does not "
            "appear in val/test in unseen config; skipping unseen assertion."
        )
    else:
        unseen_mask = subset["comp_category"] == "unseen"
        unseen_count = int(unseen_mask.sum())
        total_vt = len(subset)

        assert unseen_count == total_vt, (
            f"Expected all val/test samples of held-out combo {candidate_combo} "
            f"to be 'unseen', but only {unseen_count} of {total_vt} are."
        )

        print(
            f"[test_composition_categories_celeba] Unseen behavior OK for combo "
            f"{candidate_combo}: {total_vt} val/test samples, unseen_count={unseen_count}."
        )


def main():
    parser = argparse.ArgumentParser(description="CelebA dataset tests")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache dir (optional, default: HF default cache)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir

    # 1) Split leakage
    test_split_leakage_celeba(cache_dir=cache_dir)

    # 2) Composition categories consistency + summary
    test_composition_categories_celeba(cache_dir=cache_dir)


if __name__ == "__main__":
    main()
