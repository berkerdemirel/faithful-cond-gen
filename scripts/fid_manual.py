import numpy as np
import torch
from scipy import linalg
from torchmetrics.image.fid import FrechetInceptionDistance


def manual_compute_fid(mu1, sigma1, mu2, sigma2):
    """
    Numpy implementation of the Frechet Distance.
    The Fréchet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Numerical stability check
    if not np.isfinite(covmean).all():
        print(
            "fid calculation produces singular product; adding epsilon to diagonal of cov"
        )
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_activations_and_stats(images, model):
    """
    Extracts features using the model and computes mean/cov.
    """
    model.eval()
    with torch.no_grad():
        # TorchMetrics Inception expects inputs in range [0, 255] dtype uint8
        # but the module itself handles the normalization.
        pred = model(images)

    # FID requires calculations in float64 for precision
    pred_np = pred.cpu().numpy().astype(np.float64)

    mu = np.mean(pred_np, axis=0)
    sigma = np.cov(pred_np, rowvar=False)

    return mu, sigma


def main():
    # 1. Setup
    # Using a small feature dim (64) to make the script run fast,
    # but the logic holds for 2048.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    # Initialize TorchMetrics FID
    # feature=64 uses one of the earlier layers, much faster for testing
    tm_fid = FrechetInceptionDistance(feature=2048, reset_real_features=True).to(device)

    # 2. Create Dummy Data (Uint8, imitating real images)
    # Shape: (Batch, Channels, Height, Width)
    # 100 samples
    real_imgs = torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8).to(device)
    fake_imgs = torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8).to(device)

    print("Data generated.")

    # ---------------------------------------------------------
    # Method A: TorchMetrics Calculation
    # ---------------------------------------------------------
    print("Computing TorchMetrics FID...")
    tm_fid.update(real_imgs, real=True)
    tm_fid.update(fake_imgs, real=False)
    fid_score_tm = tm_fid.compute()

    # ---------------------------------------------------------
    # Method B: Manual Calculation
    # ---------------------------------------------------------
    print("Computing Manual FID...")

    # CRITICAL: We use the EXACT same feature extractor instance
    # contained inside the metric object to avoid model weight mismatches.
    feature_extractor = tm_fid.inception

    mu_real, sigma_real = get_activations_and_stats(real_imgs, feature_extractor)
    mu_fake, sigma_fake = get_activations_and_stats(fake_imgs, feature_extractor)

    fid_score_manual = manual_compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------
    print("\n" + "=" * 30)
    print(f"TorchMetrics FID: {fid_score_tm.item():.6f}")
    print(f"Manual Numpy FID: {fid_score_manual:.6f}")

    diff = abs(fid_score_tm.item() - fid_score_manual)
    print(f"Difference:       {diff:.6f}")

    if diff < 1e-3:
        print("✅ SUCCESS: Implementations match!")
    else:
        print("❌ FAILURE: Results diverge.")
    print("=" * 30)


if __name__ == "__main__":
    main()
