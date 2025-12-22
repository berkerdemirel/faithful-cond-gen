import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import ScoreFunction

log = logging.getLogger(__name__)


class MarginalCosineScore(ScoreFunction):
    """
    Computes Cosine Distance based on Marginal Prototypes.

    Modes:
    1. Standard: Score = Mean( 1 - CosineSim(x, mu_target) )
    2. Relative: Score = Mean( 1 - Softmax(Sim(x, all_mus))[target_idx] )
       (Applies Softmax competition within each attribute domain).

    Args:
        use_softmax: If True, applies Softmax scaling (Relative Cosine).
        temperature: Scaling factor T for Softmax.
    """

    def __init__(
        self, device: str = "cuda", use_softmax: bool = False, temperature: float = 1.0
    ):
        super().__init__(device)
        self.use_softmax = use_softmax
        self.temperature = temperature

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)

    def fit(self, features: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        Fits prototypes for each attribute key independently.
        """
        features = features.to(self.device)
        features = self._normalize(features)

        self.stats["attributes"] = {}
        keys = sorted(list(metadata.keys()))

        log.info("Fitting MARGINAL Cosine Prototypes...")

        for attr_name in keys:
            self.stats["attributes"][attr_name] = {}

            # 1. Group by this attribute's values
            values = metadata[attr_name]
            if isinstance(values, torch.Tensor):
                values = values.tolist()

            val_map = {}
            for i, v in enumerate(values):
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if v not in val_map:
                    val_map[v] = []
                val_map[v].append(i)

            # 2. Fit Prototype for each value
            for val, indices in tqdm(val_map.items(), desc=f"Values of {attr_name}"):
                indices_tensor = torch.tensor(indices, device=self.device)
                feats_sub = features[indices_tensor]

                mu = feats_sub.mean(dim=0)
                mu_norm = F.normalize(mu.unsqueeze(0), p=2, dim=1).squeeze(0)

                self.stats["attributes"][attr_name][val] = mu_norm.cpu()

        log.info("Marginal Cosine Fit Complete.")

    def score(self, features: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Scores samples by Averaging the Cosine Distance (or Relative Confidence)
        for each attribute.
        """
        features = features.to(self.device)
        features = self._normalize(features)
        N = features.shape[0]

        total_scores = torch.zeros(N, device=self.device)
        valid_attr_counts = torch.zeros(N, device=self.device)

        keys = sorted(list(metadata.keys()))

        for attr_name in keys:
            if attr_name not in self.stats["attributes"]:
                continue

            attr_protos = self.stats["attributes"][attr_name]

            # Prepare stack of ALL known prototypes for this attribute
            # Sorted to ensure deterministic indexing for Softmax
            known_vals = sorted(list(attr_protos.keys()))

            if not known_vals:
                continue

            # Map value -> Index in the stack (needed for Relative Score)
            val_to_idx = {v: i for i, v in enumerate(known_vals)}

            # Stack: (K, D)
            known_vecs_list = [attr_protos[k] for k in known_vals]
            known_vecs_stack = torch.stack(known_vecs_list).to(self.device)

            # Get target values for batch
            values = metadata[attr_name]
            if isinstance(values, torch.Tensor):
                values = values.tolist()

            # Group indices by target value
            val_map = {}
            for i, v in enumerate(values):
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if v not in val_map:
                    val_map[v] = []
                val_map[v].append(i)

            attr_scores = torch.zeros(N, device=self.device)

            # --- PRE-COMPUTE SIMILARITIES ---
            # For Relative Score, we need similarities to ALL prototypes anyway.
            # (N, D) @ (K, D).T -> (N, K)
            all_sims = torch.matmul(features, known_vecs_stack.T)

            if self.use_softmax:
                # Apply Softmax over the 'K' attribute options
                probs = F.softmax(all_sims / self.temperature, dim=1)

                # Default score for unseen/unknown (Max Confidence)
                max_conf, _ = probs.max(dim=1)
                base_scores = 1.0 - max_conf  # (N,)
            else:
                # Default score for unseen (Max Sim)
                max_sim, _ = all_sims.max(dim=1)
                base_scores = 1.0 - max_sim  # (N,)

            # Assign scores
            for val, indices in val_map.items():
                indices_tensor = torch.tensor(indices, device=self.device)

                if val in val_to_idx:
                    # --- CASE A: Known Target ---
                    idx = val_to_idx[val]

                    if self.use_softmax:
                        # Score = 1 - P(target_class)
                        # We extract the specific probability for the target class
                        target_probs = probs[indices_tensor, idx]
                        dists = 1.0 - target_probs
                    else:
                        # Score = 1 - Sim(target_class)
                        target_sims = all_sims[indices_tensor, idx]
                        dists = 1.0 - target_sims
                else:
                    # --- CASE B: Unseen Target (Benefit of Doubt) ---
                    # We use the "Best Match" score computed earlier
                    dists = base_scores[indices_tensor]

                attr_scores[indices] = dists

            total_scores += attr_scores
            valid_attr_counts += 1

        return total_scores / (valid_attr_counts + 1e-8)
