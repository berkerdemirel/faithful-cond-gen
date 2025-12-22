import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import ScoreFunction

log = logging.getLogger(__name__)


class KNNScore(ScoreFunction):
    """
    Computes K-Nearest Neighbor (KNN) Distance score using pure PyTorch.

    Score(x) = Average Euclidean distance to the k-th nearest neighbors in the training set.

    Args:
        k: Number of neighbors to consider.
        chunk_size: Number of test samples to process at once (to save VRAM).
    """

    def __init__(
        self,
        device: str = "cuda",
        k: int = 5,
        chunk_size: int = 1000,
    ):
        super().__init__(device)
        self.k = k
        self.chunk_size = chunk_size
        self.train_features = None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # We enforce normalization for cosine-based KNN
        return F.normalize(x, p=2, dim=1)

    def fit(self, features: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        Stores training features for retrieval.
        """
        # Store on CPU to save VRAM, move to GPU only during scoring batch-by-batch
        # OR: if dataset fits in VRAM, keep it there.
        # For safety (RxRx1 ~125k samples), let's keep it on CPU or main RAM.
        log.info(f"Storing {features.shape[0]} training features for KNN...")

        # We pre-normalize here so we don't do it every time in the loop
        self.train_features = self._normalize(features.to(self.device)).cpu()

        # We don't really use metadata for standard KNN, but we keep the signature valid
        log.info("KNN Fit Complete.")

    def score(
        self, features: torch.Tensor, metadata: Dict[str, Any] = None
    ) -> torch.Tensor:
        """
        Computes KNN distance score.
        """
        if self.train_features is None:
            # Try to load from stats if available (e.g. reloading from disk)
            if "train_features" in self.stats:
                log.info("Loading training features from saved stats...")
                self.train_features = self.stats["train_features"]
            else:
                raise RuntimeError("KNN scorer not fitted!")

        features = features.to(self.device)
        features = self._normalize(features)

        N = features.shape[0]
        M = self.train_features.shape[0]

        scores = torch.zeros(N, device=self.device)

        train_bank = self.train_features.to(self.device)

        log.info(f"Computing KNN (k={self.k}) for {N} samples...")

        # Process Test samples in chunks to avoid creating massive (N_test, N_train) matrix
        for i in tqdm(range(0, N, self.chunk_size), desc="KNN Search"):
            end_idx = min(i + self.chunk_size, N)
            batch = features[i:end_idx]  # (Batch, D)

            # 1. Compute Cosine Similarity (Dot Product)
            # (Batch, D) @ (Train, D).T -> (Batch, Train)
            sim_matrix = torch.matmul(batch, train_bank.T)

            # 2. Find Top-K Nearest (Highest Similarity)
            # We don't need to sort all, just topk
            topk_sims, _ = torch.topk(sim_matrix, k=self.k, dim=1)

            # 3. Convert Cosine Similarity to Euclidean Distance
            # dist = sqrt(2 * (1 - sim))
            # Clamp to avoid negative sqrt due to float errors
            topk_dists = torch.sqrt(torch.clamp(2 * (1 - topk_sims), min=0.0))

            # 4. Average distance to k neighbors
            batch_scores = topk_dists.mean(dim=1)

            scores[i:end_idx] = batch_scores

        return scores

    # Override save/load to handle the raw feature tensor efficiently
    def save_stats(self, path: str):
        # We just save the tensor inside the stats dict
        self.stats = {"train_features": self.train_features}
        super().save_stats(path)

    def load_stats(self, path: str):
        super().load_stats(path)
        if "train_features" in self.stats:
            self.train_features = self.stats["train_features"]
