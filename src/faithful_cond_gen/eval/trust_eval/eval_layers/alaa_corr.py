"""
Layer 4: Correlation with Alaa et al. Metrics.

Evaluates whether trust scores align with established sample-level metrics.
"""

from typing import Dict

import numpy as np
from scipy.stats import spearmanr


def evaluate_alaa_correlation(trust_results: Dict) -> Dict:
    """
    Layer 4: Does our score align with established sample-level metrics?

    Compute correlations between trust, realism, faithfulness and Alaa et al. metrics.
    """
    trust = trust_results["trust_updated"]
    realism = trust_results["realism_global_z"]
    faithfulness = trust_results["faithfulness_margin_z"]
    gen_center_dist = trust_results["gen_center_dist_real"]
    auth = trust_results["authenticity"]

    # Filter valid samples
    valid = np.isfinite(trust) & np.isfinite(realism) & np.isfinite(faithfulness)
    trust_v = trust[valid]
    realism_v = realism[valid]
    faithfulness_v = faithfulness[valid]
    gen_center_dist_v = gen_center_dist[valid]
    auth_v = auth[valid]

    results = {}

    # Realism vs gen_center_dist (should correlate - both capture support)
    if len(trust_v) > 10:
        rho, p = spearmanr(realism_v, gen_center_dist_v)
        results["realism_vs_gen_center_dist"] = {"spearman_rho": rho, "p": p}

    # Faithfulness vs gen_center_dist (should be weaker)
    if len(trust_v) > 10:
        rho, p = spearmanr(faithfulness_v, gen_center_dist_v)
        results["faithfulness_vs_gen_center_dist"] = {"spearman_rho": rho, "p": p}

    # Trust vs gen_center_dist
    if len(trust_v) > 10:
        rho, p = spearmanr(trust_v, gen_center_dist_v)
        results["trust_vs_gen_center_dist"] = {"spearman_rho": rho, "p": p}

    # Trust vs authenticity
    if len(trust_v) > 10:
        rho, p = spearmanr(trust_v, auth_v)
        results["trust_vs_authenticity"] = {"spearman_rho": rho, "p": p}

    return results
