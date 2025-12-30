import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf

from .base import ScoreFunction

log = logging.getLogger(__name__)


class RelativeMahalanobisScore(ScoreFunction):
    """
    Computes Relative Mahalanobis Distance (RMD).

    Score(x) = Min_k( Maha(x | Class_k) ) - Maha(x | Background)
    """

    def __init__(
        self,
        device: str = "cuda",
        regularization: float = 1e-5,
        normalize_feats: bool = True,
        use_shrinkage: bool = True,
        min_samples_for_shrinkage: int = 10,
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
        Fits two models:
        1. Class-Conditional Gaussian (Pooled Covariance).
        2. Background Gaussian (Global Mean/Covariance).
        """
        features = features.to(self.device)
        features = self._normalize(features)
        N, D = features.shape

        # ---------------------------------------------------------
        # 1. Parse Metadata & Assign Class IDs
        # ---------------------------------------------------------
        keys = list(metadata.keys())
        cond_to_idx = {}
        y = torch.zeros(N, dtype=torch.long, device=self.device)

        log.info("Mapping conditions to class IDs...")
        for i in range(N):
            sample_cond = {k: metadata[k][i] for k in keys}
            key_hash = self._hash_condition(sample_cond)

            if key_hash not in cond_to_idx:
                cond_to_idx[key_hash] = len(cond_to_idx)
            y[i] = cond_to_idx[key_hash]

        num_classes = len(cond_to_idx)
        log.info(f"Found {num_classes} unique classes.")

        # ---------------------------------------------------------
        # 2. Compute Statistics (Means)
        # ---------------------------------------------------------
        # A. Background Mean
        mu_0 = features.mean(dim=0)

        # B. Conditional Means
        mu_k = torch.zeros((num_classes, D), device=self.device)
        class_counts = torch.zeros(num_classes, device=self.device)

        mu_k.index_add_(0, y, features)
        class_counts.index_add_(0, y, torch.ones(N, device=self.device))

        class_counts = class_counts.clamp(min=1.0)
        mu_k = mu_k / class_counts.unsqueeze(1)

        # ---------------------------------------------------------
        # 3. Compute Covariances with Ledoit-Wolf Shrinkage
        # ---------------------------------------------------------
        log.info("Computing Covariances...")

        # A. Background Covariance
        if N >= self.min_samples_for_shrinkage and self.use_shrinkage:
            feats_np = features.cpu().numpy()
            lw_bg = LedoitWolf()
            cov_0_np = lw_bg.fit(feats_np).covariance_
            cov_0 = torch.from_numpy(cov_0_np).to(device=self.device, dtype=features.dtype)
            shrinkage_bg = lw_bg.shrinkage_
        else:
            centered_0 = features - mu_0.unsqueeze(0)
            cov_0 = torch.matmul(centered_0.T, centered_0) / (N - 1)
            shrinkage_bg = None

        cov_0 = cov_0 + self.reg * torch.eye(D, device=self.device)

        # B. Pooled Conditional Covariance
        # Center each sample by ITS OWN class mean
        centered_cond = features - mu_k[y]

        if N >= self.min_samples_for_shrinkage and self.use_shrinkage:
            centered_np = centered_cond.cpu().numpy()
            lw_cond = LedoitWolf()
            cov_cond_np = lw_cond.fit(centered_np).covariance_
            cov_cond = torch.from_numpy(cov_cond_np).to(device=self.device, dtype=features.dtype)
            shrinkage_cond = lw_cond.shrinkage_
        else:
            cov_cond = torch.matmul(centered_cond.T, centered_cond) / (N - 1)
            shrinkage_cond = None

        cov_cond = cov_cond + self.reg * torch.eye(D, device=self.device)

        bg_str = f"{shrinkage_bg:.4f}" if shrinkage_bg is not None else "None"
        cond_str = f"{shrinkage_cond:.4f}" if shrinkage_cond is not None else "None"
        log.info(f"Shrinkage: background={bg_str}, conditional={cond_str}")

        # ---------------------------------------------------------
        # 4. Invert & Save
        # ---------------------------------------------------------
        def invert(C):
            try:
                L = torch.linalg.cholesky(C)
                return torch.cholesky_inverse(L)
            except RuntimeError:
                return torch.linalg.pinv(C)

        self.stats["mu_0"] = mu_0.cpu()
        self.stats["prec_0"] = invert(cov_0).cpu()
        self.stats["mu_k"] = mu_k.cpu()
        self.stats["prec_cond"] = invert(cov_cond).cpu()

        log.info("RMD Fit Complete. Saved raw means and precisions.")

    def score(
        self, features: torch.Tensor, metadata: Dict[str, Any] = None
    ) -> torch.Tensor:
        """
        Computes RMD using explicit expansion of the Mahalanobis term.
        Score = Min_k( (x-mu_k)P(x-mu_k) ) - (x-mu_0)P0(x-mu_0)
        """
        features = features.to(self.device)
        features = self._normalize(features)

        # Load stats to device
        mu_0 = self.stats["mu_0"].to(self.device)  # (D,)
        prec_0 = self.stats["prec_0"].to(self.device)  # (D, D)
        mu_k = self.stats["mu_k"].to(self.device)  # (K, D)
        prec_cond = self.stats["prec_cond"].to(self.device)  # (D, D)

        # ---------------------------------------------------------
        # 1. Background Score: (x - mu_0)^T P0 (x - mu_0)
        # ---------------------------------------------------------
        delta_0 = features - mu_0.unsqueeze(0)
        # (N, D) @ (D, D) -> (N, D)
        term_0 = torch.matmul(delta_0, prec_0)
        # Row-wise dot product: sum( (N, D) * (N, D), dim=1 )
        md_0 = torch.sum(term_0 * delta_0, dim=1)

        # ---------------------------------------------------------
        # 2. Conditional Score: Min_k [ (x - mu_k)^T P (x - mu_k) ]
        # ---------------------------------------------------------
        # Using expansion: dist = xPx + muPmu - 2*xPmu
        # This allows computing the distance to ALL K classes via matrix mult.

        # Term A: x^T P x (Shape: N, 1)
        # This represents the "energy" of the test sample itself
        x_P = torch.matmul(features, prec_cond)  # (N, D)
        term_x = torch.sum(x_P * features, dim=1).unsqueeze(1)

        # Term B: mu_k^T P mu_k (Shape: 1, K)
        # This represents the "energy" of each class center
        mu_P = torch.matmul(mu_k, prec_cond)  # (K, D)
        term_mu = torch.sum(mu_P * mu_k, dim=1).unsqueeze(0)

        # Term C: x^T P mu_k (Shape: N, K)
        # The interaction term. Since P is symmetric, x^T P mu = (x^T P) @ mu^T
        # We already have (x^T P) as `x_P` from Term A.
        term_cross = torch.matmul(x_P, mu_k.T)

        # Combine: (N, 1) + (1, K) - 2(N, K) -> (N, K)
        all_dists = term_x + term_mu - 2 * term_cross

        # Find the distance to the closest class
        md_min, _ = all_dists.min(dim=1)

        # ---------------------------------------------------------
        # 3. Relative Score
        # ---------------------------------------------------------
        return md_min - md_0
