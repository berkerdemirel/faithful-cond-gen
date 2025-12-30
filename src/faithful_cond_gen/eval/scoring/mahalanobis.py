import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

from .base import ScoreFunction

log = logging.getLogger(__name__)


class MahalanobisScore(ScoreFunction):
    """
    Computes Mahalanobis distance between feature vectors and
    condition-specific Gaussian distributions (Mean, Covariance).
    """

    def __init__(
        self,
        device: str = "cuda",
        regularization: float = 1e-5,
        normalize_feats: bool = True,
        use_shrinkage: bool = True,
        min_samples_for_shrinkage: int = 2,
    ):
        super().__init__(device)
        self.reg = regularization
        self.normalize_feats = normalize_feats
        self.use_shrinkage = use_shrinkage
        self.min_samples_for_shrinkage = min_samples_for_shrinkage

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_feats:
            return F.normalize(x, p=2, dim=1)
        return x

    def fit(self, features: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        Computes Mean and Precision (Inverse Covariance) for every unique condition
        found in the metadata.
        """
        features = features.to(self.device)
        features = self._normalize(features)
        N, D = features.shape

        # 1. Group indices by condition
        keys = list(metadata.keys())
        cond_map = {}  # Key -> List[Index]

        log.info("Grouping features by condition for Mahalanobis fitting...")
        for i in range(N):
            sample_cond = {k: metadata[k][i] for k in keys}
            key = self._hash_condition(sample_cond)
            if key not in cond_map:
                cond_map[key] = []
            cond_map[key].append(i)

        self.stats["conditions"] = {}
        all_train_scores = []

        log.info(f"Fitting stats for {len(cond_map)} unique conditions...")

        # 2. Iterate conditions and compute stats
        for cond_key, indices in tqdm(cond_map.items(), desc="Fitting Conditions"):
            indices_tensor = torch.tensor(indices, device=self.device)
            feats_sub = features[indices_tensor]  # (M_cond, D)

            # A. Mean
            mu = feats_sub.mean(dim=0)

            # B. Covariance with Ledoit-Wolf Shrinkage
            M_cond = feats_sub.shape[0]

            if M_cond >= self.min_samples_for_shrinkage and self.use_shrinkage:
                # Use Ledoit-Wolf shrinkage estimator (sample efficient)
                # Convert to numpy for sklearn
                feats_np = feats_sub.cpu().numpy()
                lw = LedoitWolf()
                cov_np = lw.fit(feats_np).covariance_
                cov = torch.from_numpy(cov_np).to(device=self.device, dtype=feats_sub.dtype)

                # Store shrinkage coefficient for analysis
                shrinkage_coef = lw.shrinkage_
            elif M_cond > 1:
                # Fallback: Empirical covariance with basic regularization
                centered = feats_sub - mu.unsqueeze(0)
                cov = torch.matmul(centered.T, centered) / (M_cond - 1)
                cov = cov + self.reg * torch.eye(D, device=self.device)
                shrinkage_coef = None
            else:
                # Single sample: Use identity matrix
                cov = torch.eye(D, device=self.device)
                shrinkage_coef = None

            # Ensure numerical stability
            cov_reg = cov + self.reg * torch.eye(D, device=self.device)

            # D. Precision Matrix (Inverse)
            try:
                # Cholesky is faster and numerically stable for PD matrices
                L = torch.linalg.cholesky(cov_reg)
                cov_inv = torch.cholesky_inverse(L)
            except RuntimeError:
                # Fallback if Cholesky fails
                cov_inv = torch.linalg.pinv(cov_reg)

            # Save to stats (move to CPU to save VRAM)
            self.stats["conditions"][cond_key] = {
                "mu": mu.cpu(),
                "cov_inv": cov_inv.cpu(),
                "n_samples": M_cond,
                "shrinkage": shrinkage_coef,
            }

            # E. Compute "Self-Scores" for normalization
            centered = feats_sub - mu.unsqueeze(0)  # Recompute for all cases
            term1 = torch.matmul(centered, cov_inv)
            dists = torch.sum(term1 * centered, dim=1)
            all_train_scores.append(dists)

        # 3. Compute Normalization Factor (Max Observed ID Score)
        if all_train_scores:
            all_scores_flat = torch.cat(all_train_scores)
            # Using 99.9% percentile is safer than strict max() against outliers
            norm_val = torch.quantile(all_scores_flat, 0.999).item()
            self.stats["normalization_factor"] = norm_val
        else:
            self.stats["normalization_factor"] = 1.0

        # Log shrinkage statistics
        shrinkage_values = [
            s["shrinkage"] for s in self.stats["conditions"].values() if s["shrinkage"] is not None
        ]
        if shrinkage_values:
            avg_shrinkage = np.mean(shrinkage_values)
            log.info(
                f"Fit complete. Normalization Factor: {self.stats['normalization_factor']:.4f}, "
                f"Avg Shrinkage: {avg_shrinkage:.4f} ({len(shrinkage_values)}/{len(cond_map)} conditions)"
            )
        else:
            log.info(
                f"Fit complete. Normalization Factor: {self.stats['normalization_factor']:.4f} "
                f"(no shrinkage used)"
            )

    def score(self, features: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Computes Normalized Mahalanobis distance.
        If a condition is UNSEEN, it computes the MINIMUM distance to ANY known condition
        (giving the sample the "benefit of doubt").

        Returns: (M,) tensor. Higher = Less Faithful.
        """
        features = features.to(self.device)
        features = self._normalize(features)
        N = features.shape[0]
        scores = torch.zeros(N, device=self.device)

        keys = list(metadata.keys())
        norm_factor = self.stats.get("normalization_factor", 1.0)

        known_stats = list(self.stats["conditions"].values())

        # 1. Group Test Data (Batch optimization)
        cond_map = {}
        for i in range(N):
            sample_cond = {k: metadata[k][i] for k in keys}
            key = self._hash_condition(sample_cond)
            if key not in cond_map:
                cond_map[key] = []
            cond_map[key].append(i)

        # 2. Score per group
        for cond_key, indices in cond_map.items():
            indices_tensor = torch.tensor(indices, device=self.device)
            feats_sub = features[indices_tensor]  # (B, D)

            if cond_key in self.stats["conditions"]:
                # --- CASE A: Known Condition ---
                stat = self.stats["conditions"][cond_key]
                mu = stat["mu"].to(self.device)
                cov_inv = stat["cov_inv"].to(self.device)

                centered = feats_sub - mu.unsqueeze(0)
                term1 = torch.matmul(centered, cov_inv)
                dists = torch.sum(term1 * centered, dim=1)

            else:
                # --- CASE B: Unseen Condition (Benefit of Doubt) ---
                # The condition requested is not in our stats.
                # We calculate the distance to ALL known conditions and pick the minimum.
                if not known_stats:
                    # No reference stats available at all
                    dists = torch.full(
                        (len(indices),), float("nan"), device=self.device
                    )
                else:
                    # Initialize with infinity
                    min_dists = torch.full(
                        (len(indices),), float("inf"), device=self.device
                    )

                    # Iterate over all known clusters
                    # We do this iteratively to check every possible known condition
                    for stat in known_stats:
                        mu = stat["mu"].to(self.device)
                        cov_inv = stat["cov_inv"].to(self.device)

                        centered = feats_sub - mu.unsqueeze(0)
                        term1 = torch.matmul(centered, cov_inv)
                        curr_dists = torch.sum(term1 * centered, dim=1)

                        min_dists = torch.minimum(min_dists, curr_dists)

                    dists = min_dists

            # Normalize
            scores[indices] = dists / (norm_factor + 1e-8)

        return scores
