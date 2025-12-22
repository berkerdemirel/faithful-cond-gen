import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import ScoreFunction

log = logging.getLogger(__name__)


class MarginalMahalanobisScore(ScoreFunction):
    """
    Computes Mahalanobis distance based on Marginal Distributions.

    Instead of fitting a distribution for every unique COMBINATION (which fails for unseen combos),
    this fits a distribution for every unique VALUE of every ATTRIBUTE.

    Score(x | y={a, b}) = Mean( Normalized_Maha(x | a), Normalized_Maha(x | b) )

    Features:
    - Supports L2 feature normalization (Cosine-aligned).
    - Robust to Unseen Combinations (as long as atomic attributes are known).
    - "Benefit of Doubt" for Unseen Atomic Values (Min dist to known values).
    """

    def __init__(
        self,
        device: str = "cuda",
        regularization: float = 1e-5,
        normalize_feats: bool = True,
    ):
        super().__init__(device)
        self.reg = regularization
        self.normalize_feats = normalize_feats

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_feats:
            return F.normalize(x, p=2, dim=1)
        return x

    def fit(self, features: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        Fits statistics for each attribute key independently.
        Structure:
        self.stats = {
            'attributes': {
                'cell_type': { 0: {mu, cov_inv}, 1: ... },
                'sirna_id': { 10: {mu, cov_inv}, ... }
            },
            'norm_factors': { 'cell_type': 12.5, 'sirna_id': 8.2 }
        }
        """
        features = features.to(self.device)
        features = self._normalize(features)
        N, D = features.shape

        self.stats["attributes"] = {}
        self.stats["norm_factors"] = {}

        # We assume metadata keys are the attribute names
        keys = sorted(list(metadata.keys()))

        log.info("Fitting MARGINAL stats...")

        for attr_name in keys:
            log.info(f"Processing attribute: {attr_name}")
            self.stats["attributes"][attr_name] = {}

            # 1. Group by this specific attribute's values
            values = metadata[attr_name]
            if isinstance(values, torch.Tensor):
                values = values.tolist()

            # Map value -> list of indices
            val_map = {}
            for i, v in enumerate(values):
                # Handle scalar tensors
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if v not in val_map:
                    val_map[v] = []
                val_map[v].append(i)

            all_attr_scores = []

            # 2. Fit for each value of this attribute
            for val, indices in tqdm(val_map.items(), desc=f"Values of {attr_name}"):
                indices_tensor = torch.tensor(indices, device=self.device)
                feats_sub = features[indices_tensor]

                # Mean
                mu = feats_sub.mean(dim=0)

                # Covariance
                centered = feats_sub - mu.unsqueeze(0)
                M_cond = feats_sub.shape[0]

                if M_cond > 1:
                    cov = torch.matmul(centered.T, centered) / (M_cond - 1)
                else:
                    cov = torch.eye(D, device=self.device)

                # Regularization
                cov_reg = cov + self.reg * torch.eye(D, device=self.device)

                try:
                    L = torch.linalg.cholesky(cov_reg)
                    cov_inv = torch.cholesky_inverse(L)
                except RuntimeError:
                    cov_inv = torch.linalg.pinv(cov_reg)

                self.stats["attributes"][attr_name][val] = {
                    "mu": mu.cpu(),
                    "cov_inv": cov_inv.cpu(),
                }

                # Compute ID scores for this attribute value
                term1 = torch.matmul(centered, cov_inv)
                dists = torch.sum(term1 * centered, dim=1)
                all_attr_scores.append(dists)

            # 3. Compute Normalization Factor for this attribute
            # This balances attributes with different variances/difficulties
            if all_attr_scores:
                flat_scores = torch.cat(all_attr_scores)
                norm_val = torch.quantile(flat_scores, 0.999).item()
                self.stats["norm_factors"][attr_name] = norm_val
            else:
                self.stats["norm_factors"][attr_name] = 1.0

        log.info("Marginal Fit Complete.")

    def score(self, features: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Scores samples by Averaging the normalized marginal distance for each attribute.
        Score = Mean( Maha(x|attr_i) / Max_Maha(attr_i) )
        """
        features = features.to(self.device)
        features = self._normalize(features)
        N = features.shape[0]

        # Accumulators
        total_scores = torch.zeros(N, device=self.device)
        valid_attr_counts = torch.zeros(N, device=self.device)

        keys = sorted(list(metadata.keys()))

        # Iterate over attributes (e.g. "cell_type", then "sirna_id")
        for attr_name in keys:
            if attr_name not in self.stats["attributes"]:
                continue

            attr_stats = self.stats["attributes"][attr_name]
            norm_factor = self.stats["norm_factors"].get(attr_name, 1.0)

            # Get values for this attribute for the whole batch
            values = metadata[attr_name]
            if isinstance(values, torch.Tensor):
                values = values.tolist()

            # Group by value for batch processing
            val_map = {}
            for i, v in enumerate(values):
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if v not in val_map:
                    val_map[v] = []
                val_map[v].append(i)

            attr_scores = torch.zeros(N, device=self.device)

            for val, indices in val_map.items():
                indices_tensor = torch.tensor(indices, device=self.device)
                feats_sub = features[indices_tensor]

                if val in attr_stats:
                    # --- CASE A: Known Attribute Value ---
                    stat = attr_stats[val]
                    mu = stat["mu"].to(self.device)
                    cov_inv = stat["cov_inv"].to(self.device)

                    centered = feats_sub - mu.unsqueeze(0)
                    term1 = torch.matmul(centered, cov_inv)
                    dists = torch.sum(term1 * centered, dim=1)

                else:
                    # --- CASE B: Unseen Attribute Value (Benefit of Doubt) ---
                    # Calculate min distance to any known value of this attribute
                    # This does not trigger for compositional generalization.
                    known_stats = list(attr_stats.values())

                    if not known_stats:
                        dists = torch.full(
                            (len(indices),), float("nan"), device=self.device
                        )
                    else:
                        min_dists = torch.full(
                            (len(indices),), float("inf"), device=self.device
                        )

                        for stat in known_stats:
                            mu = stat["mu"].to(self.device)
                            cov_inv = stat["cov_inv"].to(self.device)

                            centered = feats_sub - mu.unsqueeze(0)
                            term1 = torch.matmul(centered, cov_inv)
                            curr_dists = torch.sum(term1 * centered, dim=1)

                            min_dists = torch.minimum(min_dists, curr_dists)

                        dists = min_dists

                # Normalize by that attribute's typical scale
                attr_scores[indices] = dists / (norm_factor + 1e-8)

            # Accumulate
            total_scores += attr_scores
            valid_attr_counts += 1

        # Return Average Score across attributes
        final_scores = total_scores / (valid_attr_counts + 1e-8)

        return final_scores
