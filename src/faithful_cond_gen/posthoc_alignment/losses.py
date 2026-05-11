"""Loss functions for posthoc alignment mapper training."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKCovLoss(nn.Module):
    """Cosine + top-k projected covariance match, all in L2-normalized space.

        L = beta * (1 - cos(p_n, y_n)).mean()
          + gamma_eff * ||C_p_k - C_y_k||_F^2 / ||C_y_k||_F^2

    where C_{p,y}_k are batch covariances of the *projection* of centered
    normalized pred / target onto the top-k right singular vectors V_k of
    the centered normalized real target pool (precomputed once).

    gamma_eff = gamma * gamma_scale, where gamma_scale is externally scheduled
    so the covariance term can be warmed up without fighting alignment at
    step 0. Pass gamma_scale through forward.
    """

    def __init__(self, beta: float, gamma: float, V_k: torch.Tensor):
        super().__init__()
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.register_buffer("V_k", V_k.contiguous())  # (D, k)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gamma_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        p_n = F.normalize(pred, dim=-1)
        y_n = F.normalize(target, dim=-1)

        cos_sim = F.cosine_similarity(p_n, y_n, dim=-1)
        L_cos = (1.0 - cos_sim).mean()

        p_c = p_n - p_n.mean(dim=0)
        y_c = y_n - y_n.mean(dim=0)
        pp = p_c @ self.V_k   # (B, k)
        yy = y_c @ self.V_k   # (B, k)
        B = pp.shape[0]
        C_p = (pp.T @ pp) / max(B - 1, 1)
        C_y = (yy.T @ yy) / max(B - 1, 1)
        num = ((C_p - C_y) ** 2).sum()
        den = (C_y ** 2).sum().clamp_min(1e-12)
        L_cov = num / den

        gamma_eff = self.gamma * float(gamma_scale)
        total = self.beta * L_cos + gamma_eff * L_cov

        # Monitoring: compare spectrum magnitudes (diag of projected cov) and
        # scale-free trace ratios so we can see the rank fill-in happen live.
        with torch.no_grad():
            diag_p = C_p.diagonal()
            diag_y = C_y.diagonal()
            trace_ratio = (diag_p.sum() / diag_y.sum().clamp_min(1e-12)).item()
            top5_ratio = (
                diag_p[:5].mean() / diag_y[:5].mean().clamp_min(1e-12)
            ).item()

        diagnostics = {
            "loss_total": total.item(),
            "loss_cos": L_cos.item(),
            "loss_cov_topk": L_cov.item(),
            "cos_sim_mean": cos_sim.mean().item(),
            "gamma_eff": gamma_eff,
            "cov_topk_trace_ratio": trace_ratio,   # pred_var_sum / target_var_sum in top-k
            "cov_topk_top5_ratio": top5_ratio,     # pred/target on the 5 biggest target dirs
            # fields the existing val logger expects (kept so compute_val_metrics works):
            "loss_mse": 0.0,
            "loss_whit": 0.0,
        }
        return total, diagnostics


def compute_topk_diagnostics(
    pred: torch.Tensor, V_k_target: torch.Tensor, top_q: int = 64
) -> Dict[str, float]:
    """Measure rank/subspace health of pred in normalized space.

    Returns eff_rank of pred_norm and top-k subspace alignment between
    pred_norm's own top-k basis and the precomputed target V_k.
    """
    with torch.no_grad():
        p_n = F.normalize(pred.float(), dim=-1)
        pc = p_n - p_n.mean(dim=0)
        q = min(top_q, min(pc.shape) - 1)
        _, s, V_p = torch.svd_lowrank(pc, q=q)
        p = s / s.sum()
        pp = p[p > 1e-10]
        eff_rank = math.exp(-(pp * pp.log()).sum().item())
        # Subspace alignment of pred's top-k basis with target's V_k
        k = V_k_target.shape[1]
        V_p_k = V_p[:, :k]
        M = V_k_target.to(V_p_k.dtype).T @ V_p_k
        align = float((M * M).sum().item() / k)
    return {
        "pred_eff_rank_norm": float(eff_rank),
        "topk_subspace_align": align,
    }


class NormalizedAlignmentLoss(nn.Module):
    """Pointwise + global-moment + class-mean alignment, all in L2-normalized space.

        L = λ_pair (1 − cos(p_n, y_n)).mean()
          + λ_mu   ||μ(p_n) − μ_t||² / (||μ_t||² + ε)
          + λ_cov  ||Cov(p_n) − C_t||²_F / (||C_t||²_F + ε)
          + λ_cls  Σ_c ||μ_c(p_n) − μ_c^t||² / (||μ_c^t||² + ε)   (optional)

    All teacher-side constants (μ_t, C_t, per-class μ_c^t, the three scale
    denominators, and the per-class sample weights) are precomputed once from
    the normalized teacher pool and passed in as buffers — the loss does NOT
    look at this-batch's target for moments, only for the pairwise cosine.
    That way the relative-loss denominators stay fixed across training.

    Pointwise is the main driver; moments are (scale-free) regularizers.

    Args:
        lambda_pair, lambda_mu, lambda_cov, lambda_cls: loss weights.
        mu_t: (D,) teacher global mean of normalized features.
        cov_t: (D, D) teacher global cov of normalized features.
        cls_mu_t: (C, D) per-class teacher means or None to disable L_cls.
        eps: stability floor for the relative-norm denominators.
    """

    def __init__(
        self,
        lambda_pair: float,
        lambda_mu: float,
        lambda_cov: float,
        lambda_cls: float,
        mu_t: torch.Tensor,
        cov_t: torch.Tensor,
        cls_mu_t: Optional[torch.Tensor] = None,
        eta_whit: float = 0.0,
        whit_matrix: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_pair = float(lambda_pair)
        self.lambda_mu = float(lambda_mu)
        self.lambda_cov = float(lambda_cov)
        self.lambda_cls = float(lambda_cls)
        self.eta_whit = float(eta_whit)
        self.eps = float(eps)

        self.register_buffer("mu_t", mu_t.contiguous())
        self.register_buffer("cov_t", cov_t.contiguous())
        if eta_whit > 0:
            if whit_matrix is None:
                raise ValueError("eta_whit > 0 requires whit_matrix")
            self.register_buffer("whit_matrix", whit_matrix.contiguous())
        else:
            self.whit_matrix = None
        self.register_buffer(
            "mu_t_sq_denom",
            torch.tensor(float((mu_t * mu_t).sum().item()) + eps),
        )
        self.register_buffer(
            "cov_t_fro_denom",
            torch.tensor(float((cov_t * cov_t).sum().item()) + eps),
        )

        if cls_mu_t is not None and self.lambda_cls > 0:
            self.register_buffer("cls_mu_t", cls_mu_t.contiguous())
            per_class_sq = (cls_mu_t * cls_mu_t).sum(dim=-1) + eps  # (C,)
            self.register_buffer("cls_mu_t_sq_denom", per_class_sq)
            self.num_classes = int(cls_mu_t.shape[0])
        else:
            self.cls_mu_t = None
            self.num_classes = 0

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        p_n = F.normalize(pred, dim=-1)
        y_n = F.normalize(target, dim=-1)

        # Pointwise: cosine + optional weak whitened-MSE in the teacher
        # Mahalanobis metric (errors along small-spread teacher directions
        # are weighted more).
        cos_sim = F.cosine_similarity(p_n, y_n, dim=-1)
        L_cos = (1.0 - cos_sim).mean()
        if self.whit_matrix is not None and self.eta_whit > 0:
            diff = p_n - y_n
            whitened = diff @ self.whit_matrix
            L_whit = (whitened * whitened).sum(dim=-1).mean()
        else:
            L_whit = pred.new_zeros(())
        L_pair = L_cos + self.eta_whit * L_whit

        # Global mean (relative).
        mu_p = p_n.mean(dim=0)
        dmu = mu_p - self.mu_t
        L_mu = (dmu * dmu).sum() / self.mu_t_sq_denom

        # Global covariance (relative Frobenius).
        p_c = p_n - mu_p
        B = p_c.shape[0]
        C_p = (p_c.T @ p_c) / max(B - 1, 1)
        dC = C_p - self.cov_t
        L_cov = (dC * dC).sum() / self.cov_t_fro_denom

        # Class-conditional means (relative), means-only to stay stable at small N.
        if self.cls_mu_t is not None and labels is not None and self.lambda_cls > 0:
            # Scatter-mean: sum per class / count per class.
            cls_sum = torch.zeros_like(self.cls_mu_t)  # (C, D)
            cls_sum.index_add_(0, labels, p_n)
            counts = torch.zeros(self.num_classes, device=p_n.device, dtype=p_n.dtype)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=p_n.dtype))
            present = counts > 0
            mu_cls_p = cls_sum[present] / counts[present].unsqueeze(-1)
            mu_cls_t = self.cls_mu_t[present]
            denom = self.cls_mu_t_sq_denom[present]
            dmc = mu_cls_p - mu_cls_t
            per_class = (dmc * dmc).sum(dim=-1) / denom
            L_cls = per_class.mean()
        else:
            L_cls = pred.new_zeros(())

        total = (
            self.lambda_pair * L_pair
            + self.lambda_mu * L_mu
            + self.lambda_cov * L_cov
            + self.lambda_cls * L_cls
        )

        diagnostics = {
            "loss_total": total.item(),
            "loss_pair": L_pair.item(),
            "loss_cos": L_cos.item(),
            "loss_whit": L_whit.item(),
            "loss_mu": L_mu.item(),
            "loss_cov": L_cov.item(),
            "loss_cls": L_cls.item(),
            "cos_sim_mean": cos_sim.mean().item(),
            # field the legacy val logger expects:
            "loss_mse": 0.0,
        }
        return total, diagnostics


class WhitGeomLoss(nn.Module):
    """L = λ_whit * L_whit + λ_geom * L_geom, all in L2-normalized space.

    L_whit: tempered whitened MSE  ||W(p_n - y_n)||²
    L_geom: relative mean + relative covariance matching against teacher pool.
    """

    def __init__(
        self,
        whit_matrix: torch.Tensor,
        mu_t: torch.Tensor,
        cov_t: torch.Tensor,
        lambda_whit: float = 1.0,
        lambda_geom: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_whit = float(lambda_whit)
        self.lambda_geom = float(lambda_geom)
        self.register_buffer("whit_matrix", whit_matrix.contiguous())
        self.register_buffer("mu_t", mu_t.contiguous())
        self.register_buffer("cov_t", cov_t.contiguous())
        self.register_buffer(
            "mu_t_sq_denom",
            torch.tensor(float((mu_t * mu_t).sum().item()) + eps),
        )
        self.register_buffer(
            "cov_t_fro_denom",
            torch.tensor(float((cov_t * cov_t).sum().item()) + eps),
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        p_n = F.normalize(pred, dim=-1)
        y_n = F.normalize(target, dim=-1)

        cos_sim = F.cosine_similarity(p_n, y_n, dim=-1)

        diff = p_n - y_n
        whitened = diff @ self.whit_matrix
        L_whit = (whitened * whitened).sum(dim=-1).mean()

        mu_p = p_n.mean(dim=0)
        dmu = mu_p - self.mu_t
        L_mu = (dmu * dmu).sum() / self.mu_t_sq_denom

        p_c = p_n - mu_p
        B = p_c.shape[0]
        C_p = (p_c.T @ p_c) / max(B - 1, 1)
        dC = C_p - self.cov_t
        L_cov = (dC * dC).sum() / self.cov_t_fro_denom

        L_geom = L_mu + L_cov

        total = self.lambda_whit * L_whit + self.lambda_geom * L_geom

        diagnostics = {
            "loss_total": total.item(),
            "loss_whit": L_whit.item(),
            "loss_mu": L_mu.item(),
            "loss_cov": L_cov.item(),
            "loss_geom": L_geom.item(),
            "cos_sim_mean": cos_sim.mean().item(),
            # fields the legacy val logger expects:
            "loss_cos": 0.0,
            "loss_mse": 0.0,
        }
        return total, diagnostics


class WhitGeomNbrLoss(WhitGeomLoss):
    """WhitGeomLoss + neighbor distance matching in whitened space.

    L = λ_whit * L_whit + λ_geom * L_geom + λ_nbr * L_nbr

    L_nbr: for each sample i, precomputed teacher neighbors j_1..j_k have
    whitened distances d_t. The loss penalizes distortion of these distances
    by the mapped prediction:
        ratio_l = ||W(ẑ_i - z_{j_l})||² / (d_t_l + ε)
        L_nbr = mean (ratio - 1)²
    """

    def __init__(
        self,
        whit_matrix: torch.Tensor,
        mu_t: torch.Tensor,
        cov_t: torch.Tensor,
        teacher_pool: torch.Tensor,
        nn_idx: torch.Tensor,
        nn_wdist: torch.Tensor,
        lambda_whit: float = 1.0,
        lambda_geom: float = 0.1,
        lambda_nbr: float = 0.01,
        eps: float = 1e-6,
    ):
        super().__init__(whit_matrix, mu_t, cov_t, lambda_whit, lambda_geom, eps)
        self.lambda_nbr = float(lambda_nbr)
        self.register_buffer("teacher_pool", teacher_pool.contiguous())
        self.register_buffer("nn_idx", nn_idx.contiguous())
        self.register_buffer("nn_wdist", nn_wdist.contiguous())

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, img_idx: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        base_loss, diag = super().forward(pred, target)

        if img_idx is None or self.lambda_nbr == 0.0:
            diag["loss_nbr"] = 0.0
            return base_loss, diag

        p_n = F.normalize(pred, dim=-1)
        nbr_ids = self.nn_idx[img_idx]                      # (B, k)
        nbr_feats = self.teacher_pool[nbr_ids]              # (B, k, D)
        d_target = self.nn_wdist[img_idx]                   # (B, k)

        diff = p_n.unsqueeze(1) - nbr_feats                 # (B, k, D)
        whitened = diff @ self.whit_matrix                   # (B, k, D)
        d_pred = (whitened * whitened).sum(dim=-1)           # (B, k)

        log_ratio = torch.log(d_pred + 1e-6) - torch.log(d_target + 1e-6)
        L_nbr = (log_ratio ** 2).mean()

        total = base_loss + self.lambda_nbr * L_nbr
        diag["loss_total"] = total.item()
        diag["loss_nbr"] = L_nbr.item()
        return total, diag


class WhitGeomAnchorLoss(WhitGeomLoss):
    """WhitGeomLoss + anchor-relation distillation (global relational).

    For a fixed anchor bank A=(M, D) of L2-normalized teacher exemplars
    (e.g. k-means centroids on the real teacher pool), penalize mismatch
    between pred-to-anchor and target-to-anchor cosine distances.

    Variants (picked via `variant`):
      - "mse":  L_rel = mean_{i,j} (d_p[i,j] - d_y[i,j])^2
      - "kl":   L_rel = KL(softmax(s_y/tau) || softmax(s_p/tau))  per row, mean over batch
                where s = cos(sample, anchor).
    """

    def __init__(
        self,
        whit_matrix: torch.Tensor,
        mu_t: torch.Tensor,
        cov_t: torch.Tensor,
        anchors: torch.Tensor,
        lambda_whit: float = 1.0,
        lambda_geom: float = 0.1,
        lambda_rel: float = 1.0,
        variant: str = "mse",
        tau: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__(whit_matrix, mu_t, cov_t, lambda_whit, lambda_geom, eps)
        if variant not in ("mse", "kl"):
            raise ValueError(f"unknown anchor variant: {variant}")
        self.lambda_rel = float(lambda_rel)
        self.variant = variant
        self.tau = float(tau)
        self.register_buffer(
            "anchors", F.normalize(anchors, dim=-1).contiguous()
        )  # (M, D)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        base_loss, diag = super().forward(pred, target)

        if self.lambda_rel == 0.0:
            diag["loss_rel"] = 0.0
            return base_loss, diag

        p_n = F.normalize(pred, dim=-1)
        y_n = F.normalize(target, dim=-1)
        s_p = p_n @ self.anchors.T  # (B, M)
        s_y = y_n @ self.anchors.T  # (B, M)

        if self.variant == "mse":
            d_p = 1.0 - s_p
            d_y = 1.0 - s_y
            L_rel = ((d_p - d_y) ** 2).mean()
        else:  # kl
            logp = F.log_softmax(s_p / self.tau, dim=-1)
            logq = F.log_softmax(s_y / self.tau, dim=-1)
            q = logq.exp()
            L_rel = (q * (logq - logp)).sum(dim=-1).mean()

        total = base_loss + self.lambda_rel * L_rel
        diag["loss_total"] = total.item()
        diag["loss_rel"] = L_rel.item()
        return total, diag


class PosthocAlignmentLoss(nn.Module):
    """Cosine + MSE + VICReg-style regularization.

    Components:
        - Cosine similarity loss on L2-normalized pred vs target
        - MSE on L2-normalized pred vs target
        - VICReg variance: hinge on per-dim std of pred (prevents collapse)
        - VICReg covariance: off-diagonal covariance penalty (decorrelation)
        - Whitened MSE (optional, default OFF): ||W·(pred_n - target_n)||^2
          where W = (Σ_target + αI)^(-1/2) is precomputed from the SigLIP
          target pool. Aligns the training metric with the downstream
          Mahalanobis scoring metric so capacity is spent on the same
          directions the score will weight.
    """

    def __init__(
        self,
        lambda_mse: float = 0.1,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        var_thresh: float = 1.0,
        lambda_whit: float = 0.0,
        whit_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.var_thresh = var_thresh
        self.lambda_whit = lambda_whit
        if lambda_whit > 0 and whit_matrix is None:
            raise ValueError("lambda_whit > 0 requires whit_matrix")
        if whit_matrix is not None:
            self.register_buffer("whit_matrix", whit_matrix)
        else:
            self.whit_matrix = None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            pred: (B, D) mapper output (unnormalized)
            target: (B, D) SigLIP target features

        Returns:
            (total_loss, diagnostics_dict)
        """
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
        cos_loss = (1 - cos_sim).mean()

        # MSE on normalized outputs
        mse_loss = F.mse_loss(pred_norm, target_norm)

        # VICReg variance: hinge loss on per-dimension std
        pred_std = pred.std(dim=0)
        var_loss = F.relu(self.var_thresh - pred_std).mean()

        # VICReg covariance: penalize off-diagonal elements
        B, D = pred.shape
        pred_centered = pred - pred.mean(dim=0)
        cov = (pred_centered.T @ pred_centered) / max(B - 1, 1)
        # Zero out diagonal, sum squares of off-diagonal
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag**2).sum() / D

        # Whitened MSE in target Mahalanobis metric (optional)
        if self.lambda_whit > 0:
            diff = pred_norm - target_norm
            whitened = diff @ self.whit_matrix
            whit_loss = (whitened * whitened).sum(dim=-1).mean()
        else:
            whit_loss = pred.new_zeros(())

        total = (
            cos_loss
            + self.lambda_mse * mse_loss
            + self.lambda_var * var_loss
            + self.lambda_cov * cov_loss
            + self.lambda_whit * whit_loss
        )

        diagnostics = {
            "loss_total": total.item(),
            "loss_cos": cos_loss.item(),
            "loss_mse": mse_loss.item(),
            "loss_var": var_loss.item(),
            "loss_cov": cov_loss.item(),
            "loss_whit": whit_loss.item(),
            "cos_sim_mean": cos_sim.mean().item(),
            "pred_std_mean": pred_std.mean().item(),
            "pred_std_min": pred_std.min().item(),
            "pred_norm_mean": pred.norm(dim=-1).mean().item(),
        }

        return total, diagnostics
