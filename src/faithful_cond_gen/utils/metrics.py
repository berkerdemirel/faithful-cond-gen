import torch
from faithful_cond_gen.data.rxrx1 import to_rgb
from torchmetrics.image.fid import FrechetInceptionDistance


class ConditionalFidelityMetrics:
    def __init__(self, device):
        # 2048 feature dim is standard for FID
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.device = device
        self.fid.sync_on_compute = False
        self.fid.dist_sync_on_step = False

    def _ensure_rgb(self, images: torch.Tensor) -> torch.Tensor:
        """Converts (B, 6, H, W) -> (B, 3, H, W) if needed."""
        if images.shape[1] == 6:
            device = images.device
            return torch.stack(
                [to_rgb(img.cpu()[None]).squeeze(0) for img in images]
            ).to(device)
        elif images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    def compute_rfid(self, real_samples: torch.Tensor, gen_samples: torch.Tensor):
        """
        Computes rFID using the split-half method on real samples.
        Returns: (rfid_ratio, fid_gen, fid_baseline)
        """
        n = len(real_samples)
        if n < 2:  # Need at least a few samples to split
            return 0.0, 0.0, 0.0

        # Ensure RGB format for Inception
        real_samples = self._ensure_rgb(real_samples)
        gen_samples = self._ensure_rgb(gen_samples)

        # 1. Split Real Data
        perm = torch.randperm(n, device=real_samples.device)
        real_samples = real_samples[perm]
        gen_samples = gen_samples[perm]  # optional; for fid_gen fairness
        mid = n // 2
        real_A = real_samples[:mid]  # Target distribution
        real_B = real_samples[mid:]  # Baseline distribution

        # Ensure gen matches Real_A size
        gen_A = gen_samples[:mid]

        # 2. Compute FID(Gen_A, Real_A)
        self.fid.reset()
        self.fid.update(real_A, real=True)
        self.fid.update(gen_A, real=False)
        fid_gen = self.fid.compute().item()

        # 3. Compute FID(Real_B, Real_A) -> Baseline Noise Floor
        self.fid.reset()
        self.fid.update(real_A, real=True)
        self.fid.update(real_B, real=False)
        fid_baseline = self.fid.compute().item()

        # 4. Compute Ratio
        # Add small epsilon to prevent division by zero in perfect (unlikely) cases
        rfid = fid_gen / (fid_baseline + 1e-6)

        return rfid, fid_gen, fid_baseline
