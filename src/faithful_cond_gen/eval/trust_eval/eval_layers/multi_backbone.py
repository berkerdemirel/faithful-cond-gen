"""
Layer 5: Multi-Backbone Aggregation.

Reports per-feature-type aggregate statistics.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def evaluate_multi_backbone(all_results: List[Dict], model: str) -> Dict:
    """
    Layer 5: Multi-feature-type analysis.

    Reports per-feature-type aggregate statistics only.
    NOTE: Sample-level correlations across feature types removed because there's
    no guarantee samples are in the same order across different feature caches.
    """
    # Filter to single model
    model_results = [r for r in all_results if r["model"] == model]
    if len(model_results) < 1:
        return {"n_feature_types": 0}

    feature_types = [
        r.get("feature_type", r.get("encoder", "unknown")) for r in model_results
    ]

    # Per-feature-type aggregate statistics
    feature_stats = []
    for r in model_results:
        trust = r["trust_updated"]
        realism = r["realism_global_z"]
        faithfulness = r["faithfulness_margin_z"]

        feature_stats.append(
            {
                "feature_type": r.get("feature_type", r.get("encoder", "unknown")),
                "n_samples": r["n_samples"],
                "trust_mean": float(np.mean(trust)),
                "trust_std": float(np.std(trust)),
                "realism_mean": float(np.mean(realism)),
                "realism_std": float(np.std(realism)),
                "faithfulness_mean": float(np.mean(faithfulness)),
                "faithfulness_std": float(np.std(faithfulness)),
                "authenticity_mean": float(np.mean(r["authenticity"])),
            }
        )

    result = {
        "n_feature_types": len(feature_types),
        "feature_types": feature_types,
        "feature_stats": pd.DataFrame(feature_stats),
    }

    return result
