import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import ScoreFunction

log = logging.getLogger(__name__)


class CosineScore(ScoreFunction):
    """
    Computes Maximum Cosine Similarity (MCM) or Relative Cosine (RCos).

    Modes:
    1. Standard (MCM): Score = 1 - Max_k( CosineSim(x, mu_k) )
       Pure angular distance to the nearest prototype.

    2. Relative (RCos): Score = 1 - Max_k( Softmax( CosineSim(x, mu_k) / T ) )
       Probability-based score that considers competing classes.
       (Also known as "Cosine Softmax" or "Relative Concept Matching").

    Args:
        use_softmax: If True, applies Softmax scaling (RCos). If False, uses raw Cosine (MCM).
        temperature: Scaling factor T for Softmax. (Default 1.0, lower makes distribution sharper).
    """

    def __init__(
        self, device: str = "cuda", use_softmax: bool = False, temperature: float = 1.0
    ):
        super().__init__(device)
        self.use_softmax = use_softmax
        self.temperature = temperature
        self.prototypes = None  # (K, D) tensor of normalized class means

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)

    def fit(self, features: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        Computes the Normalized Mean (Prototype) for every unique condition.
        """
        features = features.to(self.device)
        features = self._normalize(features)
        N, D = features.shape

        # 1. Group indices by condition
        keys = list(metadata.keys())
        cond_map = {}

        log.info("Grouping features by condition for Cosine Prototypes...")
        for i in range(N):
            sample_cond = {k: metadata[k][i] for k in keys}
            key = self._hash_condition(sample_cond)
            if key not in cond_map:
                cond_map[key] = []
            cond_map[key].append(i)

        log.info(f"Computing prototypes for {len(cond_map)} unique conditions...")

        prototypes_list = []

        # 2. Compute Mean per condition
        for _, indices in tqdm(cond_map.items(), desc="Fitting Prototypes"):
            indices_tensor = torch.tensor(indices, device=self.device)
            feats_sub = features[indices_tensor]

            # Mean vector
            mu = feats_sub.mean(dim=0)

            # Normalize the prototype immediately
            # u_c = mu / ||mu||
            mu_norm = F.normalize(mu.unsqueeze(0), p=2, dim=1).squeeze(0)

            prototypes_list.append(mu_norm.cpu())

        # 3. Stack into a single matrix (K, D)
        self.prototypes = torch.stack(prototypes_list)
        log.info(f"Fit Complete. Stored {self.prototypes.shape[0]} prototypes.")

    def score(
        self, features: torch.Tensor, metadata: Dict[str, Any] = None
    ) -> torch.Tensor:
        """
        Computes Anomaly Score.
        """
        if self.prototypes is None:
            if "prototypes" in self.stats:
                self.prototypes = self.stats["prototypes"]
            else:
                raise RuntimeError("CosineScore not fitted!")

        features = features.to(self.device)
        features = self._normalize(features)

        # (K, D)
        prototypes = self.prototypes.to(self.device)

        # 1. Compute Cosine Similarity to ALL classes
        # (N, D) @ (K, D).T -> (N, K)
        sim_matrix = torch.matmul(features, prototypes.T)

        # 2. Compute Score based on Mode
        if self.use_softmax:
            # --- Relative Cosine (RCos) ---
            # Scale logits
            logits = sim_matrix / self.temperature

            # Apply Softmax to get probabilities P(y=k | x)
            probs = F.softmax(logits, dim=1)

            # Confidence = Max Probability
            conf, _ = probs.max(dim=1)

            # Anomaly Score = 1 - Confidence (Low prob = High Anomaly)
            return 1.0 - conf

        else:
            # --- Standard Maximum Cosine (MCM) ---
            # Confidence = Max Cosine Similarity
            max_sim, _ = sim_matrix.max(dim=1)

            # Anomaly Score = 1 - Confidence
            return 1.0 - max_sim

    def save_stats(self, path: str):
        self.stats = {"prototypes": self.prototypes}
        super().save_stats(path)

    def load_stats(self, path: str):
        super().load_stats(path)
        if "prototypes" in self.stats:
            self.prototypes = self.stats["prototypes"]
