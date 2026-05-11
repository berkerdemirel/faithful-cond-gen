"""Train a residual MLP mapper from SiT raw hidden states to SigLIP space.

Trains one mapper per model using data from all timesteps pooled together.
No timestep conditioning on the mapper.

Usage:
    # Vanilla
    PYTHONPATH=src uv run python scripts/posthoc_alignment/train_mapper.py \
        model_key=celeba_vanilla_marginal_v1

    # REPA SigLIP
    PYTHONPATH=src uv run python scripts/posthoc_alignment/train_mapper.py \
        model_key=celeba_repa_siglip_marginal_v1 \
        hidden_dir=outputs/posthoc_alignment/raw_hidden/celeba_repa_siglip_marginal_v1
"""

import json
import logging
import math
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from faithful_cond_gen.posthoc_alignment.dataset import RawHiddenSigLIPDataset
from faithful_cond_gen.posthoc_alignment.losses import (
    NormalizedAlignmentLoss,
    PosthocAlignmentLoss,
    TopKCovLoss,
    WhitGeomAnchorLoss,
    WhitGeomLoss,
    WhitGeomNbrLoss,
    compute_topk_diagnostics,
)
from faithful_cond_gen.posthoc_alignment.mapper import ResidualAlignmentMapper

log = logging.getLogger(__name__)


def effective_rank(features: torch.Tensor) -> float:
    """Compute effective rank = exp(entropy(normalized singular values))."""
    _, s, _ = torch.svd_lowrank(features - features.mean(dim=0), q=min(50, features.shape[1]))
    p = s / s.sum()
    p = p[p > 1e-10]
    entropy = -(p * p.log()).sum().item()
    return math.exp(entropy)


def compute_val_metrics(
    mapper: ResidualAlignmentMapper,
    val_loader: DataLoader,
    criterion: PosthocAlignmentLoss,
    device: torch.device,
    src_mean: torch.Tensor = None,
    tgt_mean: torch.Tensor = None,
) -> dict:
    """Run validation and compute metrics."""
    mapper.eval()
    all_preds = []
    total_loss = 0.0
    total_cos_sim = 0.0
    total_whit = 0.0
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for hidden, target, _, _ in val_loader:
            hidden, target = hidden.to(device), target.to(device)
            if src_mean is not None:
                hidden = F.normalize(hidden - src_mean, dim=-1)
                target = F.normalize(target - tgt_mean, dim=-1)
            pred = mapper(hidden)
            loss, diag = criterion(pred, target)
            total_loss += loss.item()
            total_cos_sim += diag["cos_sim_mean"]
            total_whit += diag["loss_whit"]
            total_mse += diag["loss_mse"]
            all_preds.append(pred.cpu())
            n_batches += 1

    all_preds = torch.cat(all_preds, dim=0)
    pred_std = all_preds.std(dim=0)

    metrics = {
        "val_loss": total_loss / max(n_batches, 1),
        "val_cos_sim": total_cos_sim / max(n_batches, 1),
        "val_mse_norm": total_mse / max(n_batches, 1),
        "val_whit_mse": total_whit / max(n_batches, 1),
        "val_pred_std_mean": pred_std.mean().item(),
        "val_pred_std_min": pred_std.min().item(),
        "val_pred_norm_mean": all_preds.norm(dim=-1).mean().item(),
    }

    # Effective rank (on a subset for speed)
    subset = all_preds[:5000] if len(all_preds) > 5000 else all_preds
    metrics["val_effective_rank"] = effective_rank(subset)

    # Top-50 singular values
    centered = subset - subset.mean(dim=0)
    _, s, _ = torch.svd_lowrank(centered, q=50)
    metrics["val_sv_top50"] = s.tolist()

    mapper.train()
    return metrics


@hydra.main(
    config_path="../../configs/posthoc_alignment",
    config_name="train_mapper",
    version_base="1.3",
)
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    model_key = cfg.model_key
    output_dir = Path(cfg.output_dir) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load datasets
    condition_keys = cfg.get("condition_keys", None)
    if condition_keys is not None:
        condition_keys = list(condition_keys)

    seen_combos = cfg.get("seen_combos", None)
    if seen_combos is not None:
        seen_combos = [tuple(int(x) for x in c) for c in seen_combos]
        log.info(f"Restricting to {len(seen_combos)} seen combos: {seen_combos}")

    timesteps = cfg.get("timesteps", None)
    if timesteps is not None:
        timesteps = [float(t) for t in timesteps]
        log.info(f"Restricting to timesteps: {timesteps}")

    log.info("Loading training dataset...")
    train_ds = RawHiddenSigLIPDataset(
        hidden_dir=cfg.hidden_dir,
        siglip_path=cfg.siglip_path,
        timesteps=timesteps,
        val_fraction=cfg.training.val_fraction,
        split="train",
        seed=seed,
        condition_keys=condition_keys,
        seen_combos=seen_combos,
    )

    log.info("Loading validation dataset...")
    val_ds = RawHiddenSigLIPDataset(
        hidden_dir=cfg.hidden_dir,
        siglip_path=cfg.siglip_path,
        timesteps=timesteps,
        val_fraction=cfg.training.val_fraction,
        split="val",
        seed=seed,
        condition_keys=condition_keys,
        seen_combos=seen_combos,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    log.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    log.info(f"Steps per epoch: {len(train_loader)}")

    # Optional center + L2-norm preprocessing (applied per-batch).
    center_norm = cfg.get("center_norm", False)
    src_mean = None
    tgt_mean = None
    if center_norm:
        all_train_hidden = torch.cat([h[train_ds.image_indices] for h in train_ds.hiddens])
        src_mean = all_train_hidden.mean(dim=0).to(device)
        tgt_mean = train_ds.siglip_targets[train_ds.image_indices].mean(dim=0).to(device)
        del all_train_hidden
        log.info(
            f"Center+norm ON: src_mean norm={src_mean.norm():.4f}, "
            f"tgt_mean norm={tgt_mean.norm():.4f}"
        )
        torch.save(
            {"src_mean": src_mean.cpu(), "tgt_mean": tgt_mean.cpu()},
            output_dir / "preprocessing_stats.pt",
        )

    loss_kind = cfg.loss.get("kind", "legacy")

    # --- norm_align loss branch: pointwise cosine + relative global moments
    #     + relative class-conditional means (all in L2-normalized space,
    #     teacher moments precomputed once from the full target pool) ---
    if loss_kind == "norm_align":
        siglip_data = torch.load(cfg.siglip_path, map_location="cpu", weights_only=False)
        target_feats = siglip_data["features"].float()
        target_meta = siglip_data.get("metadata", {}) or {}

        target_norm = F.normalize(target_feats, dim=-1)
        mu_t = target_norm.mean(dim=0)
        centred = target_norm - mu_t
        cov_t = (centred.T @ centred) / max(len(centred) - 1, 1)

        cls_key = cfg.loss.get("class_label", None)
        cls_mu_t = None
        labels_by_imgidx = None
        if cls_key is not None and cfg.loss.get("lambda_cls", 0.0) > 0:
            if cls_key not in target_meta:
                raise ValueError(f"class_label={cls_key} not in target metadata")
            lbl = target_meta[cls_key].to(torch.long)
            labels_by_imgidx = lbl.clone()
            num_classes = int(lbl.max().item()) + 1
            cls_mu_t = torch.zeros(num_classes, target_norm.shape[1])
            counts = torch.zeros(num_classes)
            cls_mu_t.index_add_(0, lbl, target_norm)
            counts.index_add_(0, lbl, torch.ones_like(lbl, dtype=target_norm.dtype))
            cls_mu_t = cls_mu_t / counts.clamp_min(1).unsqueeze(-1)
            log.info(
                f"norm_align: class_label={cls_key}  num_classes={num_classes}  "
                f"per-class counts min/median/max="
                f"{int(counts.min())}/{int(counts.median())}/{int(counts.max())}"
            )

        lam_pair = float(cfg.loss.get("lambda_pair", 1.0))
        lam_mu = float(cfg.loss.get("lambda_mu", 0.1))
        lam_cov = float(cfg.loss.get("lambda_cov", 0.1))
        lam_cls = float(cfg.loss.get("lambda_cls", 0.1))
        eta_whit = float(cfg.loss.get("eta_whit", 0.0))
        whit_matrix = None
        if eta_whit > 0:
            kappa = float(cfg.loss.get("whit_cond", 1000.0))
            evals, evecs = torch.linalg.eigh(cov_t.double())
            evals = torch.clamp(evals, min=0.0)
            lam_max = float(evals.max().item())
            alpha = lam_max / kappa
            inv_sqrt = 1.0 / torch.sqrt(evals + alpha)
            whit_matrix = ((evecs * inv_sqrt) @ evecs.T).float()
            log.info(
                f"norm_align: whitening ON  eta_whit={eta_whit}  kappa={kappa}  "
                f"alpha={alpha:.3e}  lam_max={lam_max:.3e}  "
                f"W eig in [{1.0/(evals.max()+alpha).sqrt():.3f}, "
                f"{1.0/(evals.min()+alpha).sqrt():.3f}]"
            )

        criterion = NormalizedAlignmentLoss(
            lambda_pair=lam_pair,
            lambda_mu=lam_mu,
            lambda_cov=lam_cov,
            lambda_cls=lam_cls,
            mu_t=mu_t,
            cov_t=cov_t,
            cls_mu_t=cls_mu_t,
            eta_whit=eta_whit,
            whit_matrix=whit_matrix,
        ).to(device)
        log.info(
            f"Loss: norm_align  λ_pair={lam_pair} λ_mu={lam_mu} λ_cov={lam_cov} "
            f"λ_cls={lam_cls} η_whit={eta_whit}  mu_denom={criterion.mu_t_sq_denom.item():.3e}  "
            f"cov_denom={criterion.cov_t_fro_denom.item():.3e}"
        )

        # Precompute V_k too (for eval-only subspace diagnostic; not part of loss).
        _, _, Vfull = torch.svd_lowrank(centred, q=64)
        V_k_eval = Vfull[:, :40].contiguous()

        if labels_by_imgidx is not None:
            labels_by_imgidx = labels_by_imgidx.to(device)

        mapper = ResidualAlignmentMapper(
            in_dim=cfg.mapper.in_dim,
            out_dim=cfg.mapper.out_dim,
            hidden_dim=cfg.mapper.hidden_dim,
        ).to(device)
        log.info(f"Mapper params: {sum(p.numel() for p in mapper.parameters()):,}")

        optimizer = torch.optim.AdamW(
            mapper.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        total_steps = len(train_loader) * cfg.training.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        best_cos = -1.0
        all_metrics = []
        for epoch in range(cfg.training.epochs):
            mapper.train()
            agg = {"cos": 0.0, "whit": 0.0, "mu": 0.0, "cov": 0.0, "cls": 0.0}
            n_batches = 0
            for hidden, target, _, img_idx in train_loader:
                hidden = hidden.to(device)
                target = target.to(device)
                pred = mapper(hidden)
                if labels_by_imgidx is not None:
                    labels = labels_by_imgidx[img_idx.to(device)]
                else:
                    labels = None
                loss, diag = criterion(pred, target, labels=labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                agg["cos"] += diag["loss_cos"]
                agg["whit"] += diag["loss_whit"]
                agg["mu"] += diag["loss_mu"]
                agg["cov"] += diag["loss_cov"]
                agg["cls"] += diag["loss_cls"]
                n_batches += 1
            for k in agg:
                agg[k] /= max(n_batches, 1)

            # Validation: cosine + subspace alignment + eff-rank in normalized space
            mapper.eval()
            with torch.no_grad():
                val_preds = []
                val_cos_vals = []
                for hidden, target, _, _ in val_loader:
                    hidden, target = hidden.to(device), target.to(device)
                    pred = mapper(hidden)
                    p_n = F.normalize(pred, dim=-1)
                    y_n = F.normalize(target, dim=-1)
                    val_cos_vals.append(F.cosine_similarity(p_n, y_n, dim=-1).cpu())
                    val_preds.append(pred.cpu())
                val_preds = torch.cat(val_preds, dim=0)
                val_cos_mean = torch.cat(val_cos_vals).mean().item()
                diags = compute_topk_diagnostics(val_preds, V_k_eval.cpu(), top_q=64)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss_cos": agg["cos"],
                "train_loss_whit": agg["whit"],
                "train_loss_mu": agg["mu"],
                "train_loss_cov": agg["cov"],
                "train_loss_cls": agg["cls"],
                "val_cos_sim": val_cos_mean,
                "val_pred_eff_rank_norm": diags["pred_eff_rank_norm"],
                "val_topk_subspace_align": diags["topk_subspace_align"],
                "lr": scheduler.get_last_lr()[0],
            }
            all_metrics.append(epoch_metrics)
            log.info(
                f"Epoch {epoch:3d} | "
                f"cos={agg['cos']:.4f} whit={agg['whit']:.3f} "
                f"mu={agg['mu']:.4f} cov={agg['cov']:.4f} cls={agg['cls']:.4f} | "
                f"val_cos={val_cos_mean:.4f} "
                f"eff_rank={diags['pred_eff_rank_norm']:.1f} "
                f"topk_align={diags['topk_subspace_align']:.4f}"
            )
            # Checkpoint on val cos (pointwise is the main driver here).
            if val_cos_mean > best_cos:
                best_cos = val_cos_mean
                torch.save(mapper.state_dict(), output_dir / "best_mapper.pt")
                log.info(f"  New best val_cos: {best_cos:.4f}")

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        with open(output_dir / "training_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        log.info(f"Training complete. Best val_cos: {best_cos:.4f}")
        log.info(f"Outputs saved to {output_dir}")
        return

    # --- topk_cov loss branch: precompute V_k from normalized target pool ---
    if loss_kind == "topk_cov":
        siglip_data = torch.load(cfg.siglip_path, map_location="cpu", weights_only=False)
        target_feats = siglip_data["features"].float()
        target_norm = F.normalize(target_feats, dim=-1)
        target_norm_c = target_norm - target_norm.mean(dim=0)
        top_k = int(cfg.loss.get("top_k", 40))
        _, _, Vfull = torch.svd_lowrank(target_norm_c, q=top_k + 16)
        V_k = Vfull[:, :top_k].contiguous()  # (D, k)
        log.info(
            f"TopKCovLoss: V_k from normalized target pool, shape={tuple(V_k.shape)}, "
            f"target eff-rank estimator top{top_k+16}"
        )
        beta = float(cfg.loss.get("beta", 1.0))
        gamma = float(cfg.loss.get("gamma", 0.5))
        gamma_warmup_frac = float(cfg.loss.get("gamma_warmup_frac", 0.33))

        criterion = TopKCovLoss(beta=beta, gamma=gamma, V_k=V_k).to(device)
        log.info(
            f"Loss: topk_cov  beta={beta}  gamma={gamma}  top_k={top_k}  "
            f"warmup_frac={gamma_warmup_frac}"
        )
        var_thresh = None
        whit_info = {}
        # Jump straight to the optimizer/model setup.
        mapper = ResidualAlignmentMapper(
            in_dim=cfg.mapper.in_dim,
            out_dim=cfg.mapper.out_dim,
            hidden_dim=cfg.mapper.hidden_dim,
        ).to(device)
        log.info(f"Mapper params: {sum(p.numel() for p in mapper.parameters()):,}")

        optimizer = torch.optim.AdamW(
            mapper.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        total_steps = len(train_loader) * cfg.training.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        warmup_steps = max(1, int(total_steps * gamma_warmup_frac))

        best_align = -1.0
        all_metrics = []
        global_step = 0
        for epoch in range(cfg.training.epochs):
            mapper.train()
            epoch_cos = 0.0
            epoch_cov = 0.0
            n_batches = 0
            for hidden, target, _, _ in train_loader:
                hidden, target = hidden.to(device), target.to(device)
                pred = mapper(hidden)
                gamma_scale = min(1.0, global_step / warmup_steps)
                loss, diag = criterion(pred, target, gamma_scale=gamma_scale)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_cos += diag["loss_cos"]
                epoch_cov += diag["loss_cov_topk"]
                n_batches += 1
                global_step += 1
            epoch_cos /= max(n_batches, 1)
            epoch_cov /= max(n_batches, 1)

            # Validation: cosine + subspace alignment + eff-rank in normalized space
            mapper.eval()
            with torch.no_grad():
                val_preds = []
                val_cos_vals = []
                for hidden, target, _, _ in val_loader:
                    hidden, target = hidden.to(device), target.to(device)
                    pred = mapper(hidden)
                    p_n = F.normalize(pred, dim=-1)
                    y_n = F.normalize(target, dim=-1)
                    val_cos_vals.append(F.cosine_similarity(p_n, y_n, dim=-1).cpu())
                    val_preds.append(pred.cpu())
                val_preds = torch.cat(val_preds, dim=0)
                val_cos_mean = torch.cat(val_cos_vals).mean().item()
                diags = compute_topk_diagnostics(val_preds, V_k.cpu(), top_q=64)
            epoch_metrics = {
                "epoch": epoch,
                "train_loss_cos": epoch_cos,
                "train_loss_cov_topk": epoch_cov,
                "val_cos_sim": val_cos_mean,
                "val_pred_eff_rank_norm": diags["pred_eff_rank_norm"],
                "val_topk_subspace_align": diags["topk_subspace_align"],
                "gamma_scale_end_of_epoch": min(1.0, global_step / warmup_steps),
                "lr": scheduler.get_last_lr()[0],
            }
            all_metrics.append(epoch_metrics)
            log.info(
                f"Epoch {epoch:3d} | "
                f"cos_loss={epoch_cos:.4f} cov_loss={epoch_cov:.4f} "
                f"γ_scale={epoch_metrics['gamma_scale_end_of_epoch']:.2f} | "
                f"val_cos={val_cos_mean:.4f} "
                f"eff_rank={diags['pred_eff_rank_norm']:.1f} "
                f"topk_align={diags['topk_subspace_align']:.4f}"
            )
            # Check-point best by subspace alignment (our primary signal) — but
            # only once γ has fully ramped, otherwise we may lock in an early
            # alignment from the direction-only phase.
            if (
                epoch_metrics["gamma_scale_end_of_epoch"] >= 0.99
                and diags["topk_subspace_align"] > best_align
            ):
                best_align = diags["topk_subspace_align"]
                torch.save(mapper.state_dict(), output_dir / "best_mapper.pt")
                log.info(f"  New best topk_subspace_align: {best_align:.4f}")

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        with open(output_dir / "training_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        log.info(f"Training complete. Best topk_subspace_align: {best_align:.4f}")
        log.info(f"Outputs saved to {output_dir}")
        return

    # ---- whit_geom: L = λ_whit * L_whit + λ_geom * (L_mu + L_cov) ----
    if loss_kind == "whit_geom":
        siglip_data = torch.load(cfg.siglip_path, map_location="cpu", weights_only=False)
        target_feats = siglip_data["features"].float()
        target_norm = F.normalize(target_feats, dim=-1)
        mu_t = target_norm.mean(dim=0)
        centred = target_norm - mu_t
        cov_t = (centred.T @ centred) / max(len(centred) - 1, 1)

        whit_gamma = float(cfg.loss.get("whit_gamma", 0.75))
        kappa = float(cfg.loss.get("whit_cond", 1000.0))
        evals, evecs = torch.linalg.eigh(cov_t.double())
        evals = torch.clamp(evals, min=0.0)
        lam_max = float(evals.max().item())
        alpha = lam_max / kappa
        scale = (evals + alpha).pow(-whit_gamma / 2.0)
        whit_matrix = ((evecs * scale) @ evecs.T).float()

        lam_whit = float(cfg.loss.get("lambda_whit", 1.0))
        lam_geom = float(cfg.loss.get("lambda_geom", 0.1))

        criterion = WhitGeomLoss(
            whit_matrix=whit_matrix,
            mu_t=mu_t,
            cov_t=cov_t,
            lambda_whit=lam_whit,
            lambda_geom=lam_geom,
        ).to(device)
        log.info(
            f"Loss: whit_geom  λ_whit={lam_whit} λ_geom={lam_geom} "
            f"γ={whit_gamma} κ={kappa} α={alpha:.3e}"
        )
        del siglip_data, target_feats

        mapper = ResidualAlignmentMapper(
            in_dim=cfg.mapper.in_dim,
            out_dim=cfg.mapper.out_dim,
            hidden_dim=cfg.mapper.hidden_dim,
        ).to(device)
        log.info(f"Mapper params: {sum(p.numel() for p in mapper.parameters()):,}")

        optimizer = torch.optim.AdamW(
            mapper.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        total_steps = len(train_loader) * cfg.training.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        best_cos = -1.0
        all_metrics = []
        for epoch in range(cfg.training.epochs):
            mapper.train()
            agg = {"whit": 0.0, "mu": 0.0, "cov": 0.0, "cos": 0.0}
            n_batches = 0
            for hidden, target, _, _ in train_loader:
                hidden, target = hidden.to(device), target.to(device)
                if src_mean is not None:
                    hidden = F.normalize(hidden - src_mean, dim=-1)
                    target = F.normalize(target - tgt_mean, dim=-1)
                pred = mapper(hidden)
                loss, diag = criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                agg["whit"] += diag["loss_whit"]
                agg["mu"] += diag["loss_mu"]
                agg["cov"] += diag["loss_cov"]
                agg["cos"] += diag["cos_sim_mean"]
                n_batches += 1
            for k in agg:
                agg[k] /= max(n_batches, 1)

            mapper.eval()
            with torch.no_grad():
                val_cos_vals = []
                val_whit_vals = []
                for hidden, target, _, _ in val_loader:
                    hidden, target = hidden.to(device), target.to(device)
                    if src_mean is not None:
                        hidden = F.normalize(hidden - src_mean, dim=-1)
                        target = F.normalize(target - tgt_mean, dim=-1)
                    pred = mapper(hidden)
                    _, vdiag = criterion(pred, target)
                    val_cos_vals.append(vdiag["cos_sim_mean"])
                    val_whit_vals.append(vdiag["loss_whit"])
                val_cos_mean = sum(val_cos_vals) / len(val_cos_vals)
                val_whit_mean = sum(val_whit_vals) / len(val_whit_vals)

            epoch_metrics = {
                "epoch": epoch,
                "train_whit": agg["whit"],
                "train_mu": agg["mu"],
                "train_cov": agg["cov"],
                "train_cos": agg["cos"],
                "val_cos_sim": val_cos_mean,
                "val_whit": val_whit_mean,
                "lr": scheduler.get_last_lr()[0],
            }
            all_metrics.append(epoch_metrics)
            log.info(
                f"Epoch {epoch:3d} | whit={agg['whit']:.4f} mu={agg['mu']:.4f} "
                f"cov={agg['cov']:.4f} cos={agg['cos']:.4f} | "
                f"val_cos={val_cos_mean:.4f} val_whit={val_whit_mean:.4f}"
            )
            if val_cos_mean > best_cos:
                best_cos = val_cos_mean
                torch.save(mapper.state_dict(), output_dir / "best_mapper.pt")
                log.info(f"  New best val_cos: {best_cos:.4f}")

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        with open(output_dir / "training_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        log.info(f"Training complete. Best val_cos: {best_cos:.4f}")
        log.info(f"Outputs saved to {output_dir}")
        return

    # ---- whit_geom_nbr: whit_geom + neighbor distance matching ----
    if loss_kind == "whit_geom_nbr":
        siglip_data = torch.load(cfg.siglip_path, map_location="cpu", weights_only=False)
        target_feats = siglip_data["features"].float()
        target_norm = F.normalize(target_feats, dim=-1)
        mu_t = target_norm.mean(dim=0)
        centred = target_norm - mu_t
        cov_t = (centred.T @ centred) / max(len(centred) - 1, 1)

        whit_gamma = float(cfg.loss.get("whit_gamma", 0.75))
        kappa = float(cfg.loss.get("whit_cond", 1000.0))
        evals, evecs = torch.linalg.eigh(cov_t.double())
        evals = torch.clamp(evals, min=0.0)
        lam_max = float(evals.max().item())
        alpha = lam_max / kappa
        scale = (evals + alpha).pow(-whit_gamma / 2.0)
        whit_matrix = ((evecs * scale) @ evecs.T).float()

        nbr_k = int(cfg.loss.get("nbr_k", 16))
        log.info(f"Precomputing {nbr_k} whitened-space neighbors for {len(target_norm)} images...")
        whitened_targets = target_norm @ whit_matrix  # (N, D)
        cdist = torch.cdist(whitened_targets, whitened_targets)  # (N, N)
        cdist.fill_diagonal_(float("inf"))
        nn_wdist_sq, nn_idx = cdist.pow(2).topk(nbr_k, dim=1, largest=False)
        nn_idx = nn_idx.long()
        nn_wdist_sq = nn_wdist_sq.float()
        log.info(f"  Neighbor distances: median={nn_wdist_sq.median():.4f}, max={nn_wdist_sq.max():.4f}")
        del cdist, whitened_targets

        lam_whit = float(cfg.loss.get("lambda_whit", 1.0))
        lam_geom = float(cfg.loss.get("lambda_geom", 0.1))
        lam_nbr = float(cfg.loss.get("lambda_nbr", 0.01))

        criterion = WhitGeomNbrLoss(
            whit_matrix=whit_matrix,
            mu_t=mu_t,
            cov_t=cov_t,
            teacher_pool=target_norm,
            nn_idx=nn_idx,
            nn_wdist=nn_wdist_sq,
            lambda_whit=lam_whit,
            lambda_geom=lam_geom,
            lambda_nbr=lam_nbr,
        ).to(device)
        log.info(
            f"Loss: whit_geom_nbr  λ_whit={lam_whit} λ_geom={lam_geom} λ_nbr={lam_nbr} "
            f"γ={whit_gamma} κ={kappa} α={alpha:.3e} nbr_k={nbr_k}"
        )
        del siglip_data, target_feats

        mapper = ResidualAlignmentMapper(
            in_dim=cfg.mapper.in_dim,
            out_dim=cfg.mapper.out_dim,
            hidden_dim=cfg.mapper.hidden_dim,
        ).to(device)
        log.info(f"Mapper params: {sum(p.numel() for p in mapper.parameters()):,}")

        optimizer = torch.optim.AdamW(
            mapper.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        total_steps = len(train_loader) * cfg.training.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        best_cos = -1.0
        all_metrics = []
        for epoch in range(cfg.training.epochs):
            mapper.train()
            agg = {"whit": 0.0, "mu": 0.0, "cov": 0.0, "nbr": 0.0, "cos": 0.0}
            n_batches = 0
            for hidden, target, _, img_idx in train_loader:
                hidden, target = hidden.to(device), target.to(device)
                img_idx = img_idx.to(device)
                if src_mean is not None:
                    hidden = F.normalize(hidden - src_mean, dim=-1)
                    target = F.normalize(target - tgt_mean, dim=-1)
                pred = mapper(hidden)
                loss, diag = criterion(pred, target, img_idx=img_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                agg["whit"] += diag["loss_whit"]
                agg["mu"] += diag["loss_mu"]
                agg["cov"] += diag["loss_cov"]
                agg["nbr"] += diag["loss_nbr"]
                agg["cos"] += diag["cos_sim_mean"]
                n_batches += 1
            for k in agg:
                agg[k] /= max(n_batches, 1)

            mapper.eval()
            with torch.no_grad():
                val_cos_vals = []
                val_whit_vals = []
                for hidden, target, _, img_idx in val_loader:
                    hidden, target = hidden.to(device), target.to(device)
                    img_idx = img_idx.to(device)
                    if src_mean is not None:
                        hidden = F.normalize(hidden - src_mean, dim=-1)
                        target = F.normalize(target - tgt_mean, dim=-1)
                    pred = mapper(hidden)
                    _, vdiag = criterion(pred, target, img_idx=img_idx)
                    val_cos_vals.append(vdiag["cos_sim_mean"])
                    val_whit_vals.append(vdiag["loss_whit"])
                val_cos_mean = sum(val_cos_vals) / len(val_cos_vals)
                val_whit_mean = sum(val_whit_vals) / len(val_whit_vals)

            epoch_metrics = {
                "epoch": epoch,
                "train_whit": agg["whit"],
                "train_mu": agg["mu"],
                "train_cov": agg["cov"],
                "train_nbr": agg["nbr"],
                "train_cos": agg["cos"],
                "val_cos_sim": val_cos_mean,
                "val_whit": val_whit_mean,
                "lr": scheduler.get_last_lr()[0],
            }
            all_metrics.append(epoch_metrics)
            log.info(
                f"Epoch {epoch:3d} | whit={agg['whit']:.4f} mu={agg['mu']:.4f} "
                f"cov={agg['cov']:.4f} nbr={agg['nbr']:.4f} cos={agg['cos']:.4f} | "
                f"val_cos={val_cos_mean:.4f} val_whit={val_whit_mean:.4f}"
            )
            if val_cos_mean > best_cos:
                best_cos = val_cos_mean
                torch.save(mapper.state_dict(), output_dir / "best_mapper.pt")
                log.info(f"  New best val_cos: {best_cos:.4f}")

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        with open(output_dir / "training_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        log.info(f"Training complete. Best val_cos: {best_cos:.4f}")
        log.info(f"Outputs saved to {output_dir}")
        return

    # ---- whit_geom_anchor: whit_geom + anchor-relation distillation ----
    if loss_kind == "whit_geom_anchor":
        from sklearn.cluster import MiniBatchKMeans

        siglip_data = torch.load(cfg.siglip_path, map_location="cpu", weights_only=False)
        target_feats = siglip_data["features"].float()
        target_norm = F.normalize(target_feats, dim=-1)
        mu_t = target_norm.mean(dim=0)
        centred = target_norm - mu_t
        cov_t = (centred.T @ centred) / max(len(centred) - 1, 1)

        whit_gamma = float(cfg.loss.get("whit_gamma", 0.75))
        kappa = float(cfg.loss.get("whit_cond", 1000.0))
        evals, evecs = torch.linalg.eigh(cov_t.double())
        evals = torch.clamp(evals, min=0.0)
        lam_max = float(evals.max().item())
        alpha = lam_max / kappa
        scale = (evals + alpha).pow(-whit_gamma / 2.0)
        whit_matrix = ((evecs * scale) @ evecs.T).float()

        n_anchors = int(cfg.loss.get("n_anchors", 256))
        anchor_seed = int(cfg.loss.get("anchor_seed", 0))
        log.info(f"Running MiniBatchKMeans: M={n_anchors} anchors on {len(target_norm)} normalized teacher feats...")
        km = MiniBatchKMeans(
            n_clusters=n_anchors,
            random_state=anchor_seed,
            batch_size=4096,
            n_init=3,
            max_iter=100,
        ).fit(target_norm.numpy())
        anchors = torch.from_numpy(km.cluster_centers_).float()  # (M, D)
        anchors = F.normalize(anchors, dim=-1)
        log.info(f"  Anchor bank shape={tuple(anchors.shape)}; mean cluster size ~ {len(target_norm) // n_anchors}")

        lam_whit = float(cfg.loss.get("lambda_whit", 1.0))
        lam_geom = float(cfg.loss.get("lambda_geom", 0.01))
        lam_rel = float(cfg.loss.get("lambda_rel", 1.0))
        variant = str(cfg.loss.get("anchor_variant", "mse"))
        tau = float(cfg.loss.get("anchor_tau", 0.1))

        criterion = WhitGeomAnchorLoss(
            whit_matrix=whit_matrix,
            mu_t=mu_t,
            cov_t=cov_t,
            anchors=anchors,
            lambda_whit=lam_whit,
            lambda_geom=lam_geom,
            lambda_rel=lam_rel,
            variant=variant,
            tau=tau,
        ).to(device)
        log.info(
            f"Loss: whit_geom_anchor  λ_whit={lam_whit} λ_geom={lam_geom} "
            f"λ_rel={lam_rel} variant={variant} τ={tau} M={n_anchors} "
            f"γ={whit_gamma} κ={kappa} α={alpha:.3e}"
        )
        del siglip_data, target_feats, target_norm, centred

        mapper = ResidualAlignmentMapper(
            in_dim=cfg.mapper.in_dim,
            out_dim=cfg.mapper.out_dim,
            hidden_dim=cfg.mapper.hidden_dim,
        ).to(device)
        log.info(f"Mapper params: {sum(p.numel() for p in mapper.parameters()):,}")

        optimizer = torch.optim.AdamW(
            mapper.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        total_steps = len(train_loader) * cfg.training.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        best_cos = -1.0
        all_metrics = []
        for epoch in range(cfg.training.epochs):
            mapper.train()
            agg = {"whit": 0.0, "mu": 0.0, "cov": 0.0, "rel": 0.0, "cos": 0.0}
            n_batches = 0
            for hidden, target, _, _ in train_loader:
                hidden, target = hidden.to(device), target.to(device)
                if src_mean is not None:
                    hidden = F.normalize(hidden - src_mean, dim=-1)
                    target = F.normalize(target - tgt_mean, dim=-1)
                pred = mapper(hidden)
                loss, diag = criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                agg["whit"] += diag["loss_whit"]
                agg["mu"] += diag["loss_mu"]
                agg["cov"] += diag["loss_cov"]
                agg["rel"] += diag["loss_rel"]
                agg["cos"] += diag["cos_sim_mean"]
                n_batches += 1
            for k in agg:
                agg[k] /= max(n_batches, 1)

            mapper.eval()
            with torch.no_grad():
                val_cos_vals = []
                val_whit_vals = []
                val_rel_vals = []
                for hidden, target, _, _ in val_loader:
                    hidden, target = hidden.to(device), target.to(device)
                    if src_mean is not None:
                        hidden = F.normalize(hidden - src_mean, dim=-1)
                        target = F.normalize(target - tgt_mean, dim=-1)
                    pred = mapper(hidden)
                    _, vdiag = criterion(pred, target)
                    val_cos_vals.append(vdiag["cos_sim_mean"])
                    val_whit_vals.append(vdiag["loss_whit"])
                    val_rel_vals.append(vdiag["loss_rel"])
                val_cos_mean = sum(val_cos_vals) / len(val_cos_vals)
                val_whit_mean = sum(val_whit_vals) / len(val_whit_vals)
                val_rel_mean = sum(val_rel_vals) / len(val_rel_vals)

            epoch_metrics = {
                "epoch": epoch,
                "train_whit": agg["whit"],
                "train_mu": agg["mu"],
                "train_cov": agg["cov"],
                "train_rel": agg["rel"],
                "train_cos": agg["cos"],
                "val_cos_sim": val_cos_mean,
                "val_whit": val_whit_mean,
                "val_rel": val_rel_mean,
                "lr": scheduler.get_last_lr()[0],
            }
            all_metrics.append(epoch_metrics)
            log.info(
                f"Epoch {epoch:3d} | whit={agg['whit']:.4f} mu={agg['mu']:.4f} "
                f"cov={agg['cov']:.4f} rel={agg['rel']:.4f} cos={agg['cos']:.4f} | "
                f"val_cos={val_cos_mean:.4f} val_whit={val_whit_mean:.4f} val_rel={val_rel_mean:.4f}"
            )
            if val_cos_mean > best_cos:
                best_cos = val_cos_mean
                torch.save(mapper.state_dict(), output_dir / "best_mapper.pt")
                log.info(f"  New best val_cos: {best_cos:.4f}")

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        with open(output_dir / "training_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        log.info(f"Training complete. Best val_cos: {best_cos:.4f}")
        log.info(f"Outputs saved to {output_dir}")
        return

    # ---- legacy path below (PosthocAlignmentLoss, unchanged) ----
    var_thresh = cfg.loss.var_thresh
    lambda_whit = float(cfg.loss.get("lambda_whit", 0.0))
    whit_matrix = None
    whit_info = {}
    needs_target_stats = (var_thresh == "auto") or (lambda_whit > 0)
    if needs_target_stats:
        siglip_data = torch.load(cfg.siglip_path, map_location="cpu", weights_only=False)
        target_feats = siglip_data["features"]

        if var_thresh == "auto":
            target_std = target_feats.std(dim=0)
            var_thresh = float(target_std.mean().item() * 0.5)
            log.info(
                f"Auto-calibrated var_thresh: {var_thresh:.4f} "
                f"(50% of target mean std={target_std.mean():.4f})"
            )

        if lambda_whit > 0:
            kappa = float(cfg.loss.get("whit_cond", 1000.0))
            whit_gamma = float(cfg.loss.get("whit_gamma", 1.0))
            target_norm = F.normalize(target_feats.double(), dim=-1)
            mean = target_norm.mean(dim=0)
            centred = target_norm - mean
            cov = (centred.T @ centred) / max(len(centred) - 1, 1)
            evals, evecs = torch.linalg.eigh(cov)
            evals = torch.clamp(evals, min=0.0)
            lam_max = float(evals.max().item())
            alpha = lam_max / kappa
            scale = (evals + alpha).pow(-whit_gamma / 2.0)
            whit_matrix = (evecs * scale) @ evecs.T  # (D, D), symmetric
            whit_matrix = whit_matrix.float()
            whit_info = {
                "lambda_max": lam_max,
                "lambda_min": float(evals.min().item()),
                "alpha": alpha,
                "kappa": kappa,
                "whit_gamma": whit_gamma,
                "whit_eig_max": float((evals.min() + alpha).pow(-whit_gamma / 2.0).item()),
                "whit_eig_min": float((evals.max() + alpha).pow(-whit_gamma / 2.0).item()),
            }
            log.info(
                f"Whitening op: lambda_max={lam_max:.4e}  alpha={alpha:.4e} "
                f"(kappa={kappa:.0f}) gamma={whit_gamma} "
                f"W eigvals in [{whit_info['whit_eig_min']:.3f}, "
                f"{whit_info['whit_eig_max']:.3f}]"
            )
        del siglip_data, target_feats

    # Model, loss, optimizer
    mapper = ResidualAlignmentMapper(
        in_dim=cfg.mapper.in_dim,
        out_dim=cfg.mapper.out_dim,
        hidden_dim=cfg.mapper.hidden_dim,
    ).to(device)
    log.info(f"Mapper params: {sum(p.numel() for p in mapper.parameters()):,}")

    criterion = PosthocAlignmentLoss(
        lambda_mse=cfg.loss.lambda_mse,
        lambda_var=cfg.loss.lambda_var,
        lambda_cov=cfg.loss.lambda_cov,
        var_thresh=var_thresh,
        lambda_whit=lambda_whit,
        whit_matrix=whit_matrix,
    ).to(device)
    log.info(
        f"Loss config: lambda_mse={cfg.loss.lambda_mse} lambda_var={cfg.loss.lambda_var} "
        f"lambda_cov={cfg.loss.lambda_cov} lambda_whit={lambda_whit} "
        f"(whitening {'ON' if lambda_whit > 0 else 'OFF'})"
    )

    optimizer = torch.optim.AdamW(
        mapper.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = len(train_loader) * cfg.training.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # Training loop
    best_val_cos = -1.0
    all_metrics = []

    for epoch in range(cfg.training.epochs):
        mapper.train()
        epoch_loss = 0.0
        epoch_cos = 0.0
        n_batches = 0

        for hidden, target, _, _ in train_loader:
            hidden, target = hidden.to(device), target.to(device)
            if src_mean is not None:
                hidden = F.normalize(hidden - src_mean, dim=-1)
                target = F.normalize(target - tgt_mean, dim=-1)

            pred = mapper(hidden)
            loss, diag = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += diag["loss_total"]
            epoch_cos += diag["cos_sim_mean"]
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        epoch_cos /= max(n_batches, 1)

        # Validation
        val_metrics = compute_val_metrics(mapper, val_loader, criterion, device, src_mean, tgt_mean)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_cos_sim": epoch_cos,
            "lr": scheduler.get_last_lr()[0],
            **val_metrics,
        }
        all_metrics.append(epoch_metrics)

        log.info(
            f"Epoch {epoch:3d} | "
            f"train_loss={epoch_loss:.4f} train_cos={epoch_cos:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} val_cos={val_metrics['val_cos_sim']:.4f} "
            f"val_mse={val_metrics['val_mse_norm']:.4f} "
            f"val_whit={val_metrics['val_whit_mse']:.4f} | "
            f"eff_rank={val_metrics['val_effective_rank']:.1f} "
            f"std_min={val_metrics['val_pred_std_min']:.4f}"
        )

        # Checkpoint best
        if val_metrics["val_cos_sim"] > best_val_cos:
            best_val_cos = val_metrics["val_cos_sim"]
            torch.save(mapper.state_dict(), output_dir / "best_mapper.pt")
            log.info(f"  New best val cos_sim: {best_val_cos:.4f}")

    # Save final config and metrics
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict["loss"]["var_thresh"] = var_thresh  # resolved value
    if whit_info:
        config_dict["loss"]["whit_info"] = whit_info
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    with open(output_dir / "training_metrics.json", "w") as f:
        # Convert sv_top50 lists for JSON compatibility
        json.dump(all_metrics, f, indent=2, default=str)

    log.info(f"Training complete. Best val cos_sim: {best_val_cos:.4f}")
    log.info(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
