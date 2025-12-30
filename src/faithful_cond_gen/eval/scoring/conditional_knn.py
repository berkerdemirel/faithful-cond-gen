import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import ScoreFunction

log = logging.getLogger(__name__)


class ConditionalKNNScore(ScoreFunction):
    """
    Computes K-Nearest Neighbor (KNN) Distance score conditioned on metadata.

    Unlike standard KNN which searches across all training data, this variant:
    - Groups training features by condition during fit()
    - Searches for neighbors only within the same condition pool
    - Falls back to global pool for unseen conditions

    This reduces contamination from unrelated conditions and provides
    condition-specific faithfulness scores.

    Score(x) = Average Euclidean distance to k-th nearest neighbors
               within the same condition pool.

    Args:
        k: Number of neighbors to consider.
        chunk_size: Number of test samples to process at once.
        min_pool_size: Minimum samples per condition to use conditional search.
                       If a condition has fewer samples, fall back to global pool.
        fallback_to_global: If True, use global pool for unseen conditions.
                           If False, return NaN for unseen conditions.
    """

    def __init__(
        self,
        device: str = "cuda",
        k: int = 5,
        chunk_size: int = 1000,
        min_pool_size: int = 10,
        fallback_to_global: bool = True,
    ):
        super().__init__(device)
        self.k = k
        self.chunk_size = chunk_size
        self.min_pool_size = min_pool_size
        self.fallback_to_global = fallback_to_global

        self.cond_pools = {}  # condition_hash -> features tensor
        self.global_pool = None  # Fallback for unseen conditions

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """L2 normalize features for cosine-based distance."""
        return F.normalize(x, p=2, dim=1)

    def fit(self, features: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        Groups training features by condition.

        Args:
            features: (N, D) tensor of training features
            metadata: Dictionary with condition attributes (e.g., {'cell_type': [...], 'sirna': [...]})
        """
        features = features.to(self.device)
        features = self._normalize(features)

        N = features.shape[0]
        log.info(f"Grouping {N} training features by condition...")

        # Extract condition keys (sorted for consistency)
        cond_keys = sorted(metadata.keys())

        # Build condition hash for each sample
        condition_hashes = []
        for i in range(N):
            cond_dict = {}
            for key in cond_keys:
                vals = metadata[key]
                if isinstance(vals, torch.Tensor):
                    cond_dict[key] = vals[i].item()
                else:
                    cond_dict[key] = vals[i]
            cond_hash = self._hash_condition(cond_dict)
            condition_hashes.append(cond_hash)

        # Group features by condition
        unique_conditions = set(condition_hashes)
        log.info(f"Found {len(unique_conditions)} unique conditions")

        small_pool_count = 0
        for cond_hash in unique_conditions:
            # Collect indices for this condition
            indices = [i for i, h in enumerate(condition_hashes) if h == cond_hash]
            cond_feats = features[indices]

            if len(cond_feats) >= self.min_pool_size:
                # Store on CPU to save VRAM
                self.cond_pools[cond_hash] = cond_feats.cpu()
            else:
                small_pool_count += 1

        log.info(
            f"Created {len(self.cond_pools)} conditional pools "
            f"({small_pool_count} conditions below min_pool_size={self.min_pool_size})"
        )

        # Store global pool for fallback
        if self.fallback_to_global:
            self.global_pool = features.cpu()
            log.info(f"Stored global pool with {len(self.global_pool)} samples for fallback")

        log.info("Conditional KNN Fit Complete.")

    def score(
        self, features: torch.Tensor, metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Computes KNN distance score within condition-specific pools.

        Args:
            features: (M, D) tensor of test features
            metadata: Conditioning info for test samples

        Returns:
            (M,) tensor of KNN distance scores
        """
        if len(self.cond_pools) == 0 and self.global_pool is None:
            raise RuntimeError("Conditional KNN scorer not fitted!")

        features = features.to(self.device)
        features = self._normalize(features)

        N = features.shape[0]
        cond_keys = sorted(metadata.keys())

        # Build condition hashes for test samples
        test_conditions = []
        for i in range(N):
            cond_dict = {}
            for key in cond_keys:
                vals = metadata[key]
                if isinstance(vals, torch.Tensor):
                    cond_dict[key] = vals[i].item()
                else:
                    cond_dict[key] = vals[i]
            test_conditions.append(self._hash_condition(cond_dict))

        scores = torch.zeros(N, device=self.device)

        log.info(f"Computing Conditional KNN (k={self.k}) for {N} samples...")

        # Group test samples by condition for efficient batching
        unique_test_conds = set(test_conditions)

        for cond_hash in tqdm(unique_test_conds, desc="Conditional KNN"):
            # Get indices of test samples with this condition
            test_indices = [i for i, h in enumerate(test_conditions) if h == cond_hash]
            test_batch = features[test_indices]

            # Select appropriate pool
            if cond_hash in self.cond_pools:
                # Use condition-specific pool
                pool = self.cond_pools[cond_hash].to(self.device)
            elif self.fallback_to_global and self.global_pool is not None:
                # Fall back to global pool
                pool = self.global_pool.to(self.device)
            else:
                # No pool available - return NaN
                scores[test_indices] = float('nan')
                continue

            # Ensure we have enough neighbors
            effective_k = min(self.k, len(pool))
            if effective_k == 0:
                scores[test_indices] = float('nan')
                continue

            # Compute KNN within this pool
            # (test_batch, D) @ (pool, D).T -> (test_batch, pool_size)
            sim_matrix = torch.matmul(test_batch, pool.T)

            # Find top-k nearest neighbors
            topk_sims, _ = torch.topk(sim_matrix, k=effective_k, dim=1)

            # Convert cosine similarity to Euclidean distance
            # dist = sqrt(2 * (1 - sim))
            topk_dists = torch.sqrt(torch.clamp(2 * (1 - topk_sims), min=0.0))

            # Average distance to k neighbors
            batch_scores = topk_dists.mean(dim=1)

            scores[test_indices] = batch_scores

        return scores

    def save_stats(self, path: str):
        """Save conditional pools and global pool."""
        self.stats = {
            "cond_pools": self.cond_pools,
            "global_pool": self.global_pool,
            "min_pool_size": self.min_pool_size,
            "fallback_to_global": self.fallback_to_global,
        }
        super().save_stats(path)

    def load_stats(self, path: str):
        """Load conditional pools and global pool."""
        super().load_stats(path)
        if "cond_pools" in self.stats:
            self.cond_pools = self.stats["cond_pools"]
            self.global_pool = self.stats.get("global_pool", None)
            self.min_pool_size = self.stats.get("min_pool_size", 10)
            self.fallback_to_global = self.stats.get("fallback_to_global", True)
