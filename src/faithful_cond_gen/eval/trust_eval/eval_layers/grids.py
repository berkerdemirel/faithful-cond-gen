"""
Task 2: Realism/Faithfulness 2×2 Grids.

Visual examples of quadrants (good/bad realism × good/bad faithfulness).
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from faithful_cond_gen.eval.trust_eval.image_utils import (
    create_image_grid,
    get_image_path,
)


def create_realism_faithfulness_grids(
    trust_results: Dict,
    model_dir: str,
    dataset: str,
    condition_keys: List[str],
    output_dir: Path,
    config_key: str,
    seed: int = 42,
) -> Dict:
    """
    Task 2: Create 2×2 grids per condition (realism × faithfulness).

    Uses robust corner picking to ensure grids are generated for all conditions.
    Fixes indexing bug by mapping global index to within-condition index.
    """
    # Sanity check: verify image directory exists
    image_dir = Path(f"outputs/gen/{model_dir}/images")
    if image_dir.exists():
        sample_files = list(image_dir.glob("*.png"))[:3]
        if sample_files:
            print(
                f"    Image dir check: {image_dir} exists, sample files: {[f.name for f in sample_files]}"
            )
    else:
        print(f"    Warning: Image directory not found: {image_dir}")

    conditions = trust_results["true_conditions"]
    realism_z = trust_results["realism_global_z"]
    faithfulness_z = trust_results["faithfulness_margin_z"]

    # Build global_idx -> within_condition_idx mapping
    global_to_local_idx = {}
    cond_counters = {}
    for global_idx, cond in enumerate(conditions):
        if cond not in cond_counters:
            cond_counters[cond] = 0
        local_idx = cond_counters[cond]
        global_to_local_idx[global_idx] = (cond, local_idx)
        cond_counters[cond] += 1

    # Group by condition with local indices
    samples_by_cond = {}
    for global_idx, cond in enumerate(conditions):
        if cond not in samples_by_cond:
            samples_by_cond[cond] = []
        _, local_idx = global_to_local_idx[global_idx]
        samples_by_cond[cond].append(
            {
                "global_idx": global_idx,
                "local_idx": local_idx,
                "realism_z": realism_z[global_idx],
                "faithfulness_z": faithfulness_z[global_idx],
            }
        )

    # Create grids directory
    grid_dir = output_dir / "grids" / f"{dataset}_{config_key.replace('/', '_')}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    manifest = []
    grids_created = 0

    # Select conditions (all 16 in CelebA, top 16 by count otherwise)
    if dataset == "celeba" and len(samples_by_cond) == 16:
        selected_conditions = list(samples_by_cond.keys())
    else:
        cond_counts = [
            (cond, len(samples)) for cond, samples in samples_by_cond.items()
        ]
        cond_counts.sort(key=lambda x: x[1], reverse=True)
        selected_conditions = [c for c, _ in cond_counts[:16]]

    for cond in selected_conditions:
        samples = samples_by_cond[cond]
        if len(samples) < 4:
            continue  # Need at least 4 samples

        # Robust corner picking: greedy selection without duplicates
        r_values = np.array([s["realism_z"] for s in samples])
        f_values = np.array([s["faithfulness_z"] for s in samples])

        # Track which samples are already used
        used_indices = set()

        def pick_best(score_fn):
            """Pick best sample by score_fn, excluding already used."""
            scores = score_fn(r_values, f_values)
            sorted_idx = np.argsort(scores)
            for idx in sorted_idx:
                if idx not in used_indices:
                    used_indices.add(idx)
                    return samples[idx]
            # Fallback: return best even if used (shouldn't happen with 4+ samples)
            return samples[sorted_idx[0]]

        corners = {
            # lowest r + lowest f (good realism, good faithfulness)
            "good_r_good_f": pick_best(lambda r, f: r + f),
            # lowest r + highest f (good realism, bad faithfulness)
            "good_r_bad_f": pick_best(lambda r, f: r - f),
            # highest r + lowest f (bad realism, good faithfulness)
            "bad_r_good_f": pick_best(lambda r, f: -r + f),
            # highest r + highest f (bad realism, bad faithfulness)
            "bad_r_bad_f": pick_best(lambda r, f: -r - f),
        }

        # Build grid
        image_paths = []
        titles = []
        scores = []
        selected_info = {}

        quad_order = [
            ("good_r_good_f", "Good R + Good F"),
            ("good_r_bad_f", "Good R + Bad F"),
            ("bad_r_good_f", "Bad R + Good F"),
            ("bad_r_bad_f", "Bad R + Bad F"),
        ]

        for quad_name, title in quad_order:
            sample = corners[quad_name]
            local_idx = sample["local_idx"]
            r = sample["realism_z"]
            f = sample["faithfulness_z"]

            # Get image path using local index
            img_path = get_image_path(cond, local_idx, model_dir, condition_keys)
            image_paths.append(img_path)
            titles.append(title)
            scores.append((r, f))

            selected_info[quad_name] = {
                "local_idx": local_idx,
                "realism_z": float(r),
                "faithfulness_z": float(f),
                "filename": img_path.name,
            }

        # Create grid
        try:
            grid_img = create_image_grid(image_paths, titles, scores)
            cond_str = "_".join(f"{k}{v}" for k, v in zip(condition_keys, cond))
            grid_path = grid_dir / f"condition_{cond_str}.png"
            grid_img.save(grid_path)

            manifest.append(
                {
                    "condition": str(cond),
                    "condition_str": cond_str,
                    "grid_path": str(grid_path.relative_to(output_dir)),
                    "good_r_good_f": str(selected_info["good_r_good_f"]),
                    "good_r_bad_f": str(selected_info["good_r_bad_f"]),
                    "bad_r_good_f": str(selected_info["bad_r_good_f"]),
                    "bad_r_bad_f": str(selected_info["bad_r_bad_f"]),
                }
            )
            grids_created += 1
        except Exception as e:
            print(f"  Warning: Failed to create grid for {cond}: {e}")

    # Save manifest
    if manifest:
        manifest_df = pd.DataFrame(manifest)
        manifest_path = grid_dir / "manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
    else:
        manifest_path = None

    return {
        "status": "success" if grids_created > 0 else "no_grids_created",
        "n_grids": grids_created,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "grid_dir": str(grid_dir),
    }
