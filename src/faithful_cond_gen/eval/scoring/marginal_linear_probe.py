import logging
import pickle
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .base import ScoreFunction

log = logging.getLogger(__name__)


class MarginalLinearProbeScore(ScoreFunction):
    """
    Computes faithfulness using Linear Probes (Classifiers).

    Score(x) = Mean( 1 - P(target_attribute | x) )

    Soft Mode (soft_mode=True): If an attribute value (class) is encountered in scoring
    that was not seen during training, uses prediction entropy as the score instead of
    terminating. This allows graceful handling of unseen combinations.

    Strict Mode (soft_mode=False): Terminates if unseen labels are encountered.
    """

    def __init__(
        self,
        device: str = "cuda",
        normalize_feats: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        soft_mode: bool = False,
    ):
        super().__init__(device)
        self.normalize_feats = normalize_feats
        self.C = C
        self.max_iter = max_iter
        self.soft_mode = soft_mode

        self.models = {}  # Attr -> LogisticRegression
        self.encoders = {}  # Attr -> LabelEncoder

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_feats:
            return F.normalize(x, p=2, dim=1)
        return x

    def fit(self, features: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        Trains a Logistic Regression classifier for every attribute.
        """
        features = features.to(self.device)
        features = self._normalize(features)

        X = features.cpu().numpy()

        keys = sorted(list(metadata.keys()))
        log.info(f"Training Marginal Linear Probes for attributes: {keys}")

        for attr_name in keys:
            log.info(f"Fitting Classifier for: {attr_name}...")

            # Prepare Labels
            values = metadata[attr_name]
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            else:
                values = np.array(values)

            le = LabelEncoder()
            y = le.fit_transform(values)

            # Train
            clf = LogisticRegression(
                C=self.C,
                solver="lbfgs",
                max_iter=self.max_iter,
                verbose=0,
            )

            clf.fit(X, y)

            train_acc = clf.score(X, y)
            log.info(f"  -> {attr_name} Train Acc: {train_acc:.4f}")

            self.models[attr_name] = clf
            self.encoders[attr_name] = le

        log.info("Linear Probe Fit Complete.")

    def score(self, features: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Scores samples based on classifier confidence.

        In soft_mode=True: Uses entropy for unseen labels.
        In soft_mode=False: TERMINATES if unseen labels are found.
        """
        features = features.to(self.device)
        features = self._normalize(features)

        X = features.cpu().numpy()
        N = features.shape[0]

        total_scores = np.zeros(N, dtype=np.float32)
        valid_attr_counts = np.zeros(N, dtype=np.float32)

        keys = sorted(list(metadata.keys()))

        for attr_name in keys:
            if attr_name not in self.models:
                continue

            clf = self.models[attr_name]
            le = self.encoders[attr_name]

            # 1. Get Target Labels
            targets = metadata[attr_name]
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            else:
                targets = np.array(targets)

            # 2. Check for Unseen Labels
            known_classes = set(le.classes_)

            # 3. Predict probabilities for all samples
            probs = clf.predict_proba(X)  # (N, K)

            # 4. Score each sample
            current_scores = np.zeros(N, dtype=np.float32)

            for i in range(N):
                target_val = targets[i]

                if target_val in known_classes:
                    # Seen value: Use normal scoring (1 - P(target))
                    target_idx = le.transform([target_val])[0]
                    target_prob = probs[i, target_idx]
                    current_scores[i] = 1.0 - target_prob
                else:
                    # Unseen value
                    if self.soft_mode:
                        # Soft mode: Use entropy of prediction
                        # High entropy = model is uncertain (higher score)
                        # Normalize by log(num_classes) to get [0, 1] range
                        ent = entropy(probs[i])
                        max_entropy = np.log(len(le.classes_))
                        current_scores[i] = ent / max_entropy if max_entropy > 0 else 1.0
                    else:
                        # Strict mode: Raise error
                        msg = (
                            f"CRITICAL: Unseen label '{target_val}' encountered for attribute '{attr_name}'. "
                            f"The Linear Probe cannot score classes it hasn't seen.\n"
                            f"Known classes: {list(le.classes_)[:10]}..."
                        )
                        log.error(msg)
                        raise ValueError(msg)

            total_scores += current_scores
            valid_attr_counts += 1

        # Avoid div by zero (though if loops ran, counts > 0)
        valid_attr_counts[valid_attr_counts == 0] = 1.0
        final_scores = total_scores / valid_attr_counts

        return torch.from_numpy(final_scores).to(self.device)

    def save_stats(self, path: str):
        """
        Custom save: Pickle the sklearn models to bytes, then save via torch.
        """
        log.info("Pickling sklearn models for saving...")
        # Create a portable blob
        blob = {"models": self.models, "encoders": self.encoders}
        # Dump to bytes using standard pickle
        serialized_blob = pickle.dumps(blob)

        # Store in stats dict
        self.stats = {"sklearn_blob": serialized_blob}

        super().save_stats(path)

    def load_stats(self, path: str):
        """
        Custom load: Load dict via torch, then unpickle the sklearn blob.
        """
        super().load_stats(path)

        if "sklearn_blob" in self.stats:
            log.info("Unpickling sklearn models...")
            blob = pickle.loads(self.stats["sklearn_blob"])
            self.models = blob["models"]
            self.encoders = blob["encoders"]
        else:
            log.warning("No sklearn blob found in loaded stats!")
