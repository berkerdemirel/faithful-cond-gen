import warnings

import numpy as np
import scipy.linalg
import torch
from faithful_cond_gen.data.rxrx1 import to_rgb
from scipy import linalg
from torchmetrics.image.fid import FrechetInceptionDistance

# Add this to imports
warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)


class ConditionalFidelityMetrics:
    def __init__(self, device):
        self.device = device
        # We initialize the metric ONLY to get the Inception backbone.
        # normalize=True means we will pass [0,1] images, but we must handle
        # the scaling manually since we are bypassing .update()
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(
            device
        )
        self.inception = self.fid_metric.inception
        self.inception.eval()

    def _ensure_rgb(self, images: torch.Tensor) -> torch.Tensor:
        """Converts (B, 6, H, W) -> (B, 3, H, W) if needed (GPU-friendly)."""
        if images.shape[1] == 6:
            return to_rgb(images)
        elif images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    def _get_activations_and_stats(self, images: torch.Tensor):
        """
        Manually extracts features and computes mu/sigma.
        Matches torchmetrics behavior: inputs [0,1] float -> [0,255] uint8 -> Inception
        """
        images = self._ensure_rgb(images)

        # TorchMetrics logic: If normalize=True (which we assume for generator outputs),
        # it expects [0,1] floats and converts them to [0,255] uint8 before the network.
        # We must replicate this to get valid features.
        if images.is_floating_point():
            images = images.mul(255).add(0.5).clamp(0, 255).to(torch.uint8)

        # Extract features in batches to save VRAM
        features_list = []
        batch_size = 50

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                feat = self.inception(batch)
                features_list.append(feat)

        # Concatenate and cast to float64 for precision
        features = torch.cat(features_list, dim=0).double()

        # Calculate Statistics
        mu = torch.mean(features, dim=0).cpu().numpy()
        sigma = torch.cov(features.T).cpu().numpy()

        return mu, sigma

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """
        Numpy implementation of the Fréchet Distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        # Stabilize covariances
        sigma1 = (sigma1 + sigma1.T) / 2.0
        sigma2 = (sigma2 + sigma2.T) / 2.0
        eps = 1e-6
        sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
        sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

        assert mu1.shape == mu2.shape
        assert sigma1.shape == sigma2.shape

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            print(
                "fid calculation produces singular product; adding epsilon to diagonal of cov"
            )
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
                m = np.max(np.abs(covmean.imag))
                warnings.warn(
                    f"sqrtm returned large imaginary component {m:.3e}; taking real part anyway"
                )
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def compute_rfid(self, real_samples: torch.Tensor, gen_samples: torch.Tensor):
        """
        Computes rFID using the split-half method on real samples.
        Returns: (rfid_ratio, fid_gen, fid_baseline)
        """
        n = len(real_samples)
        if n < 4:  # Need at least a few samples to split and compute covariance
            return 0.0, 0.0, 0.0

        # 1. Split Data
        # We split Real data into Target (A) and Baseline (B)
        perm = torch.randperm(n, device=real_samples.device)
        real_samples = real_samples[perm]
        gen_samples = gen_samples[
            perm
        ]  # Keep aligned if needed, though usually gen is random

        mid = n // 2
        real_A = real_samples[:mid]
        real_B = real_samples[mid:]
        gen_A = gen_samples[:mid]

        # 2. Extract Features & Stats
        mu_real_A, sigma_real_A = self._get_activations_and_stats(real_A)
        mu_real_B, sigma_real_B = self._get_activations_and_stats(real_B)
        mu_gen_A, sigma_gen_A = self._get_activations_and_stats(gen_A)

        # 3. Compute Distances
        # FID(Gen, Real_A)
        fid_gen = self._calculate_frechet_distance(
            mu_real_A, sigma_real_A, mu_gen_A, sigma_gen_A
        )

        # FID(Real_B, Real_A) -> Baseline Noise Floor
        fid_baseline = self._calculate_frechet_distance(
            mu_real_A, sigma_real_A, mu_real_B, sigma_real_B
        )

        # 4. Compute Ratio
        rfid = fid_gen / (fid_baseline + 1e-6)

        return rfid, fid_gen, fid_baseline
