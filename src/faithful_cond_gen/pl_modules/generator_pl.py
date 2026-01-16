from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from faithful_cond_gen.model.ema import EMA
from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.utils.metrics import ConditionalFidelityMetrics
from torch.optim import AdamW
from transformers import get_scheduler


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def dist_on() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    return torch.distributed.get_rank() if dist_on() else 0


def world() -> int:
    return torch.distributed.get_world_size() if dist_on() else 1


def dprint(*args, **kwargs):
    # avoids buffering so you actually see it during hangs
    prefix = f"[rank {rank()}/{world()} pid={os.getpid()}] "
    print(prefix + " ".join(map(str, args)), flush=True, **kwargs)
    sys.stdout.flush()


@dataclass
class GeneratorPLConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0  # REPA uses warmup
    val_target_conditions: Optional[List[Dict[str, int]]] = None
    cond_dropout_prob: float = (
        0.0  # Probability of dropping individual attributes (compositional generalization)
    )
    # Additivity loss for compositional generalization (Proposal 1)
    additivity_loss_weight: float = (
        0.0  # Weight for additivity regularizer (0 = disabled)
    )
    additivity_loss_prob: float = (
        0.1  # Probability of computing additivity loss per batch
    )
    additivity_t_threshold: float = (
        0.5  # Only apply at high noise levels (t > threshold)
    )


class GeneratorPL(pl.LightningModule):
    def __init__(
        self, generator: GeneratorWrapper, cfg: Optional[GeneratorPLConfig] = None
    ):
        super().__init__()
        self.generator = generator
        self.cfg = cfg or GeneratorPLConfig()

        self.save_hyperparameters(
            {"lr": self.cfg.lr, "weight_decay": self.cfg.weight_decay},
            ignore=["generator", "cfg"],
        )
        # Will be populated in on_fit_start
        self.val_buffer: Dict[Tuple, Any] = {}
        self.target_keys: List[Tuple] = []

    def configure_optimizers(self):
        params = [p for p in self.generator.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        if self.cfg.warmup_steps > 0:
            scheduler = get_scheduler(
                "constant_with_warmup",
                optimizer=optimizer,
                num_warmup_steps=self.cfg.warmup_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return optimizer

    @staticmethod
    def linear_interpolant(
        x0: torch.Tensor,  # (B,C,h,w)
        t: torch.Tensor,  # (B,) float
        eps: torch.Tensor,  # (B,C,h,w)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_t = (1-t)x0 + t eps; v_tgt = d/dt x_t = -x0 + eps."""
        b = x0.shape[0]
        t_b = t.view(b, 1, 1, 1)
        x_t = (1.0 - t_b) * x0 + t_b * eps
        v_tgt = -x0 + eps
        return x_t, v_tgt

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        images, conditioning = batch

        # Agnostic Unpacking:
        # Extract conditioning dict and use canonical key ordering
        cond_dict = conditioning.get("cond")
        if cond_dict is None:
            raise ValueError("Batch missing 'cond' dict in conditioning")

        # Get canonical key order (stored at setup time)
        if not hasattr(self, "_cond_keys"):
            # First batch: infer and store canonical order (sorted for consistency)
            self._cond_keys = sorted(cond_dict.keys())

        # Stack conditioning values in canonical order
        cond_tensors = [cond_dict[k] for k in self._cond_keys]
        cond_ids = torch.stack(cond_tensors, dim=1)

        return images, cond_ids

    def apply_condition_dropout(self, cond_ids: torch.Tensor) -> torch.Tensor:
        """Apply per-attribute dropout for compositional generalization.

        Randomly masks individual attributes (sets to unconditional token),
        forcing the model to learn from partial attribute information.

        This is DIFFERENT from class_dropout (classifier-free guidance):
        - class_dropout: Drops ALL attributes → (UNCOND, UNCOND)
        - cond_dropout: Drops INDIVIDUAL attributes → (attr1, UNCOND) or (UNCOND, attr2)

        Args:
            cond_ids: (B, K) tensor where K is number of attributes

        Returns:
            (B, K) tensor with some attributes masked to unconditional token
        """
        p = self.cfg.cond_dropout_prob
        if p <= 0.0:
            return cond_ids  # Fast path: no dropout

        b, k = cond_ids.shape

        # For each (sample, attribute) pair, decide whether to drop it
        # Shape: (B, K) - each attribute can be independently dropped
        drop_mask = torch.rand(b, k, device=cond_ids.device) < p

        # Get unconditional token ID for each attribute
        # Each embedder in the SiT has num_classes as its unconditional token
        attr_num_classes = (
            self.generator.diffusion_backbone.attr_num_classes
        )  # length k
        uncond = torch.tensor(
            attr_num_classes, device=cond_ids.device, dtype=cond_ids.dtype
        )  # (k,)
        uncond_mat = uncond.unsqueeze(0).expand(b, k)  # (b, k)

        # Vectorized replacement: use unconditional token where mask is True
        return torch.where(drop_mask, uncond_mat, cond_ids)

    def compute_additivity_loss(
        self, x_t: torch.Tensor, t: torch.Tensor, cond_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute inclusion-exclusion additivity loss for compositional generalization.

        Enforces: v(c_full) ≈ v(c_drop_i) + v(c_drop_j) - v(c_drop_ij)

        This formulation works for any categorical attributes (CelebA binary or RxRx1 sirna).
        We "drop" attributes by setting them to the unconditional token, rather than
        constructing new conditions with arbitrary class IDs.

        Only applied at high noise levels (t > threshold) where conditioning dominates.

        Args:
            x_t: (B, C, H, W) noisy latents
            t: (B,) timesteps
            cond_ids: (B, K) original condition IDs (pre-dropout)

        Returns:
            Scalar additivity loss (0 if no high-t samples or disabled)
        """
        high_t_mask = t > self.cfg.additivity_t_threshold
        if not high_t_mask.any():
            return x_t.new_tensor(0.0)

        x = x_t[high_t_mask]
        tt = t[high_t_mask]
        c_full = cond_ids[high_t_mask].clone()
        b, K = c_full.shape

        if K < 2:
            return x_t.new_tensor(0.0)

        # Pick two random attribute indices
        perm = torch.randperm(K, device=x.device)
        i, j = perm[0].item(), perm[1].item()

        # Get unconditional tokens for each attribute
        attr_num_classes = self.generator.diffusion_backbone.attr_num_classes
        uncond_tokens = torch.tensor(
            attr_num_classes, device=x.device, dtype=c_full.dtype
        )

        # Build dropped conditions using inclusion-exclusion
        c_drop_i = c_full.clone()
        c_drop_j = c_full.clone()
        c_drop_ij = c_full.clone()
        c_drop_i[:, i] = uncond_tokens[i]
        c_drop_j[:, j] = uncond_tokens[j]
        c_drop_ij[:, i] = uncond_tokens[i]
        c_drop_ij[:, j] = uncond_tokens[j]

        no_drop = torch.zeros(b, device=x.device, dtype=torch.long)

        # Teacher terms: no grad (save memory, treat as fixed target)
        with torch.no_grad():
            v_drop_i = self.generator.velocity_prediction(
                x, tt, c_drop_i, force_drop_ids=no_drop
            )
            v_drop_j = self.generator.velocity_prediction(
                x, tt, c_drop_j, force_drop_ids=no_drop
            )
            v_drop_ij = self.generator.velocity_prediction(
                x, tt, c_drop_ij, force_drop_ids=no_drop
            )
            target = v_drop_i + v_drop_j - v_drop_ij

        # Student: full condition prediction (gradients flow here)
        v_full = self.generator.velocity_prediction(
            x, tt, c_full, force_drop_ids=no_drop
        )

        return F.mse_loss(v_full, target)

    def training_step(self, batch, batch_idx: int):
        images, cond_ids_raw = self._unpack_batch(batch)

        # --- apply condition dropout (compositional generalization) ---
        cond_ids = self.apply_condition_dropout(cond_ids_raw)

        # --- encode to latents ---
        with torch.no_grad():
            x0 = self.generator.encode(images)  # (B,4,h,w) if VAE frozen

        b = x0.shape[0]

        # --- forward/noising process ---
        t = torch.rand(b, device=x0.device, dtype=torch.float32)
        eps = torch.randn_like(x0)
        x_t, v_tgt = self.linear_interpolant(x0, t, eps)

        # --- REPA: extract encoder features from raw images ---
        zs = None
        if self.repa_encoder is not None:
            with torch.no_grad():
                # zs: (B, num_patches, embed_dim) - patch tokens from frozen encoder
                zs = self.repa_encoder(images)

        # --- velocity prediction ---
        # Note: class_dropout is applied inside velocity_prediction via LabelEmbedder
        # So cond_ids here may have individual attributes masked (cond_dropout)
        # AND the whole condition may be dropped (class_dropout) - they compose!
        use_repa = self.repa_encoder is not None
        if use_repa:
            v_hat, zs_tilde = self.generator.velocity_prediction(
                x_t=x_t, t=t, cond_ids=cond_ids, return_projected=True
            )
        else:
            v_hat = self.generator.velocity_prediction(x_t=x_t, t=t, cond_ids=cond_ids)
            zs_tilde = None

        # --- main velocity loss ---
        denoising_loss = F.mse_loss(v_hat, v_tgt)

        # --- REPA projection loss ---
        proj_loss = torch.tensor(0.0, device=denoising_loss.device)
        breakpoint()
        if use_repa and zs_tilde is not None and zs is not None:
            proj_loss = self._compute_repa_loss(zs, zs_tilde)

        # --- total loss ---
        loss = denoising_loss
        if use_repa:
            proj_coeff = self.generator.cfg.repa_proj_coeff
            loss = loss + proj_coeff * proj_loss

        # --- additivity loss (compositional generalization) ---
        add_loss = torch.tensor(0.0, device=loss.device)
        if self.cfg.additivity_loss_weight > 0:
            # Stochastically apply additivity loss
            if torch.rand(1).item() < self.cfg.additivity_loss_prob:
                # Use pre-dropout conditions for additivity constraint
                add_loss = self.compute_additivity_loss(x_t, t, cond_ids_raw)
                loss = loss + self.cfg.additivity_loss_weight * add_loss

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/denoising_loss", denoising_loss, prog_bar=False)
        if use_repa:
            self.log("train/proj_loss", proj_loss, prog_bar=True)
        if self.cfg.additivity_loss_weight > 0:
            self.log("train/add_loss", add_loss, prog_bar=False)
        return loss

    def _compute_repa_loss(
        self,
        zs: torch.Tensor,
        zs_tilde: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute REPA projection loss (negative cosine similarity).

        Following the original REPA paper: direct per-patch alignment between
        SiT intermediate features (projected) and encoder patch tokens.

        The paper assumes patch counts match:
        - SiT: latent 32x32 with patch_size=2 → 16x16 = 256 patches
        - DINOv2: 224x224 input with patch_size=14 → 16x16 = 256 patches

        Args:
            zs: (B, num_patches, embed_dim) encoder features from real images
            zs_tilde: List of (B, num_patches, z_dim) projected features from SiT

        Returns:
            Scalar projection loss
        """
        if zs_tilde is None or len(zs_tilde) == 0:
            return torch.tensor(0.0, device=zs.device)

        proj_loss = 0.0
        bsz = zs.shape[0]

        for z_tilde in zs_tilde:
            # z_tilde: (B, T_sit, z_dim) from SiT projector
            # zs: (B, T_enc, embed_dim) from encoder

            num_patches_sit = z_tilde.shape[1]
            num_patches_enc = zs.shape[1]
            if num_patches_sit != num_patches_enc:
                # Warn once about mismatch (not ideal, paper assumes matching)
                if not hasattr(self, "_repa_patch_mismatch_warned"):
                    print(
                        f"⚠️ [REPA] Patch count mismatch: SiT={num_patches_sit}, Encoder={num_patches_enc}. "
                        f"Using interpolation (deviates from original paper)."
                    )
                    self._repa_patch_mismatch_warned = True

                # Fallback: interpolate SiT features to encoder resolution
                h_sit = int(num_patches_sit**0.5)
                h_enc = int(num_patches_enc**0.5)
                z_tilde_spatial = z_tilde.permute(0, 2, 1).reshape(
                    bsz, -1, h_sit, h_sit
                )
                z_tilde_interp = F.interpolate(
                    z_tilde_spatial,
                    size=(h_enc, h_enc),
                    mode="bilinear",
                    align_corners=False,
                )
                z_tilde = z_tilde_interp.reshape(bsz, -1, h_enc * h_enc).permute(
                    0, 2, 1
                )

            # Per-patch cosine similarity loss (following REPA paper)
            # For each batch element, for each patch: compute -cos_sim
            z_tilde_norm = F.normalize(z_tilde, dim=-1)  # (B, T, D)
            z_norm = F.normalize(zs, dim=-1)  # (B, T, D)

            # Loss = -mean(cos_sim) across all patches and batch
            cos_sim = (z_norm * z_tilde_norm).sum(dim=-1)  # (B, T)
            proj_loss = proj_loss + (-cos_sim).mean()

        proj_loss = proj_loss / len(zs_tilde)
        return proj_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        images, cond_ids = self._unpack_batch(batch)

        x0 = self.generator.encode(images)
        b = x0.shape[0]

        t = torch.rand(b, device=x0.device, dtype=torch.float32)
        eps = torch.randn_like(x0)
        x_t, v_tgt = self.linear_interpolant(x0, t, eps)

        v_hat = self.generator.velocity_prediction(x_t=x_t, t=t, cond_ids=cond_ids)
        loss = F.mse_loss(v_hat, v_tgt)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int):
        images, cond_ids = self._unpack_batch(batch)

        x0 = self.generator.encode(images)
        b = x0.shape[0]

        t = torch.rand(b, device=x0.device, dtype=torch.float32)
        eps = torch.randn_like(x0)
        x_t, v_tgt = self.linear_interpolant(x0, t, eps)

        v_hat = self.generator.velocity_prediction(x_t=x_t, t=t, cond_ids=cond_ids)
        loss = F.mse_loss(v_hat, v_tgt)

        self.log("test/loss", loss, prog_bar=True)
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad(set_to_none=True)
        if hasattr(self, "ema"):
            self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "ema"):
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            # make sure self.ema exists before calling load (e.g. create in on_fit_start or here)
            if not hasattr(self, "ema"):
                self.ema = EMA(self.generator, decay=checkpoint["ema"]["decay"])
            self.ema.load_state_dict(checkpoint["ema"])

    def on_train_batch_start(self, batch, batch_idx):
        if self.trainer.sanity_checking or self.trainer.fast_dev_run:
            return

        # single GPU: normal
        if not (self.trainer.world_size > 1 and torch.distributed.is_initialized()):
            if self.global_step % 5000 == 0:
                self._log_images_and_metrics()
            return

        # DDP: rank0 decides, then broadcasts decision
        do_log = torch.zeros(1, device=self.device, dtype=torch.int32)
        if self.global_rank == 0:
            do_log.fill_(1 if (self.global_step % 5000 == 0) else 0)

        torch.distributed.broadcast(do_log, src=0)

        if do_log.item() == 1:
            # align ranks before/after logging so collectives happen in lockstep
            self.trainer.strategy.barrier()
            self._log_images_and_metrics()
            self.trainer.strategy.barrier()

    def on_fit_start(self):
        if self.trainer.sanity_checking:
            return
        print(f"[GeneratorPL] Rank {self.global_rank} initializing metrics...")
        if not hasattr(self, "ema"):
            self.ema = EMA(self.generator, decay=0.9999)
        self._last_avg_rfid = torch.tensor(float("inf"), device=self.device)
        self.fidelity_metrics = ConditionalFidelityMetrics(self.device)
        self.val_buffer = {}
        self.target_keys = []

        # --- REPA encoder loading ---
        # Use rank-based loading to avoid race conditions with HuggingFace downloads
        self.repa_encoder = None
        if self.generator.cfg.use_repa:
            from faithful_cond_gen.model.repa_encoder import load_repa_encoder

            is_distributed = (
                self.trainer.world_size > 1 and torch.distributed.is_initialized()
            )

            if is_distributed:
                # Rank 0 downloads first, others wait
                if self.global_rank == 0:
                    print(
                        f"[GeneratorPL] Rank 0 loading REPA encoder: {self.generator.cfg.repa_encoder}"
                    )
                    self.repa_encoder = load_repa_encoder(
                        encoder_name=self.generator.cfg.repa_encoder,
                        resolution=self.generator.cfg.image_size,
                        in_channels=self.generator.cfg.in_channels,
                        device=self.device,
                    )
                    print(
                        f"[GeneratorPL] Rank 0 REPA encoder loaded (embed_dim={self.repa_encoder.embed_dim})"
                    )

                # Wait for rank 0 to finish downloading
                torch.distributed.barrier()

                # Now other ranks can load from cache
                if self.global_rank != 0:
                    print(
                        f"[GeneratorPL] Rank {self.global_rank} loading REPA encoder from cache"
                    )
                    self.repa_encoder = load_repa_encoder(
                        encoder_name=self.generator.cfg.repa_encoder,
                        resolution=self.generator.cfg.image_size,
                        in_channels=self.generator.cfg.in_channels,
                        device=self.device,
                    )
                    print(f"[GeneratorPL] Rank {self.global_rank} REPA encoder loaded")

                # Final barrier to ensure all ranks are ready
                torch.distributed.barrier()
            else:
                # Single GPU - just load directly
                print(
                    f"[GeneratorPL] Loading REPA encoder: {self.generator.cfg.repa_encoder}"
                )
                self.repa_encoder = load_repa_encoder(
                    encoder_name=self.generator.cfg.repa_encoder,
                    resolution=self.generator.cfg.image_size,
                    in_channels=self.generator.cfg.in_channels,
                    device=self.device,
                )
                print(
                    f"[GeneratorPL] REPA encoder loaded (embed_dim={self.repa_encoder.embed_dim})"
                )
        if self.trainer.fast_dev_run:
            return
        dm = self.trainer.datamodule

        # Initialize _cond_keys before any batch processing
        # Get canonical key order from a sample from the training set
        if not hasattr(self, "_cond_keys"):
            train_ds = dm.train_dataloader().dataset
            if len(train_ds) > 0:
                _, first_sample_cond = train_ds[0]
                self._cond_keys = sorted(first_sample_cond["cond"].keys())
                print(f"[GeneratorPL] Initialized condition keys: {self._cond_keys}")
            else:
                # Fallback: try to infer from datamodule config
                print(
                    "⚠️ [GeneratorPL] Warning: Could not get sample from train dataset"
                )

        is_distributed = (
            self.trainer.world_size > 1 and torch.distributed.is_initialized()
        )
        world_size = self.trainer.world_size
        rank = self.global_rank

        # --- build targets (rank0 decides if distributed) ---
        targets = []
        if (self.cfg.val_target_conditions is not None) and len(
            self.cfg.val_target_conditions
        ) > 0:
            targets = self.cfg.val_target_conditions
        else:
            df_conds = dm.available_conditions("val")
            drop_cols = ["count", "comp_category"]
            attr_cols = [c for c in df_conds.columns if c not in drop_cols]

            MIN_SAMPLES = 4
            df_filtered = df_conds[df_conds["count"] >= MIN_SAMPLES].copy()
            df_filtered = df_filtered.sort_values("count", ascending=False)

            # n_samples = min(8, len(df_filtered))
            desired_n = self.trainer.world_size if is_distributed else 8
            n_samples = min(desired_n, len(df_filtered))
            sample_df = df_filtered.head(n_samples)
            targets = sample_df[attr_cols].to_dict(orient="records")
            if n_samples == 0:
                print(
                    f"⚠️ Rank {rank}: No conditions with >= {MIN_SAMPLES} samples found!"
                )
                return

            # sample_df = df_filtered.head(n_samples)
            # targets = sample_df[attr_cols].to_dict(orient="records")

        if is_distributed:
            obj = [targets] if rank == 0 else [None]
            torch.distributed.broadcast_object_list(obj, src=0)
            targets = obj[0]

        MAX_SAMPLES = 128

        for i, cond_dict in enumerate(targets):
            dict_key = tuple(sorted(cond_dict.items()))
            self.target_keys.append(dict_key)  # EXACTLY ONCE ON ALL RANKS

            # only the assigned rank loads images
            if is_distributed and i != rank:
                continue

            print(f"[Rank {rank}] Loading buffer for target {i}: {cond_dict}")

            ds_subset = dm.get_matching_dataset(
                "val", cond_dict, max_samples=MAX_SAMPLES
            )
            if len(ds_subset) < 4:
                print(
                    f"⚠️ Rank {rank}: Condition {cond_dict} has too few samples. Skipping."
                )
                continue

            imgs = torch.stack([ds_subset[j][0] for j in range(len(ds_subset))]).to(
                self.device
            )

            _, first_sample_cond = ds_subset[0]
            # Use canonical key order (same as training)
            cond_vals = [first_sample_cond["cond"][k] for k in self._cond_keys]
            cond_tensor = torch.stack(cond_vals).to(self.device)

            self.val_buffer[dict_key] = {
                "real_images": imgs,
                "cond_tensor": cond_tensor,
                "cond_dict": cond_dict,
            }

    @torch.no_grad()
    def _log_images_and_metrics(self):
        """
        Computes rFID and generates visualization images.
        """
        print(
            f"[GeneratorPL] Rank {self.global_rank} entering logging step {self.global_step}..."
        )
        self.generator.eval()
        if hasattr(self, "ema"):
            self.ema.apply()

        # We want Rank N to compute FID *only* on its own data, not wait for others.
        if hasattr(self.fidelity_metrics, "fid"):
            self.fidelity_metrics.fid.sync_on_compute = False
            self.fidelity_metrics.fid.dist_sync_on_step = False

        # --- Helper: Core Generation Logic ---
        def process_single_target(c_key):
            """Generates images and computes metrics for a specific target key."""
            # If we are in DDP, we only have data for our assigned target.
            # If we are Single GPU, we have data for all.
            data = self.val_buffer.get(c_key)
            if data is None:
                return None

            real_imgs = data["real_images"].to(self.device)
            cond_tensor = data["cond_tensor"].to(self.device)

            # 1. Visualization
            vis_real = real_imgs[:8]
            latents = self.generator.encode(vis_real)
            rec_imgs = self.generator.decode(latents)

            cond_batch_vis = cond_tensor.unsqueeze(0).repeat(8, 1)
            gen_vis = self.generator.sample(
                cond_batch_vis, num_inference_steps=50, t_cutoff=0.04
            )

            # Stack for transport/logging: (24, C, H, W) -> [Real|Rec|Gen]
            vis_imgs = torch.cat([vis_real, rec_imgs, gen_vis], dim=0)
            # inside process_single_target, after vis_imgs is created:
            if vis_imgs.shape[1] == 6:
                vis_imgs = self.fidelity_metrics._ensure_rgb(
                    vis_imgs
                )  # now (24,3,H,W) float
            vis_imgs = vis_imgs.clamp(0, 1)  # assuming ToTensor -> [0,1]

            # 2. Metrics (rFID)
            n_gen = len(real_imgs)
            cond_batch_metric = cond_tensor.unsqueeze(0).repeat(n_gen, 1)

            gen_metric_list = []
            batch_size = 16
            for j in range(0, n_gen, batch_size):
                curr_batch = cond_batch_metric[j : j + batch_size]
                gen_metric_list.append(
                    self.generator.sample(
                        curr_batch, num_inference_steps=25, t_cutoff=0.04
                    )
                )
            gen_metric_imgs = torch.cat(gen_metric_list, dim=0)

            # Because sync_on_compute=False, this now returns LOCAL FID without waiting
            self.fidelity_metrics.fid.reset()

            rfid, fid_gen, fid_base = self.fidelity_metrics.compute_rfid(
                real_imgs, gen_metric_imgs
            )
            self.fidelity_metrics.fid.reset()

            return (rfid, fid_gen, fid_base), vis_imgs

        # --- Helper: Logging Logic ---
        def log_target_results(c_key, metrics, imgs_stack):
            """Logs metrics and images to WandB/Console for a single target."""
            rfid, fid_gen, fid_base = metrics

            # Unpack images
            vis_real = imgs_stack[0:8]
            rec_imgs = imgs_stack[8:16]
            gen_vis = imgs_stack[16:24]

            # check for channel count and convert to rgb if needed
            # if vis_real.shape[1] == 6:
            #     vis_real = self.fidelity_metrics._ensure_rgb(vis_real)
            #     rec_imgs = self.fidelity_metrics._ensure_rgb(rec_imgs)
            #     gen_vis = self.fidelity_metrics._ensure_rgb(gen_vis)

            cond_str = "_".join([f"{v}" for k, v in c_key])  # c_key is tuple of items
            cond_tag = f"cond_{cond_str}"

            # Log Metrics
            # sync_dist=False is safer here since keys are unique to Rank 0 logging pass
            self.log_dict(
                {
                    f"val/{cond_tag}/rFID": rfid,
                    f"val/{cond_tag}/FID_gen": fid_gen,
                    f"val/{cond_tag}/FID_base": fid_base,
                },
                logger=True,
                rank_zero_only=True,
                sync_dist=False,
            )

            # Log Images
            if hasattr(self.logger, "experiment") and hasattr(
                self.logger.experiment, "log"
            ):
                import wandb
                from torchvision.utils import make_grid

                def grid_to_wandb(tensor):
                    grid = make_grid(tensor, nrow=8, normalize=False)
                    return wandb.Image(grid.cpu().permute(1, 2, 0).numpy())

                # breakpoint()
                self.logger.experiment.log(
                    {
                        f"vis/{cond_tag}/fixed_reconstruction": grid_to_wandb(
                            torch.cat([vis_real, rec_imgs], dim=0)
                        ),
                        f"vis/{cond_tag}/random_generation": grid_to_wandb(gen_vis),
                    },
                    step=self.global_step,
                )
            return rfid

        # --- Strategy Selection ---
        is_distributed = (
            self.trainer.world_size > 1
        ) and torch.distributed.is_initialized()

        avg_rfid = 0.0
        valid_targets = 0

        if is_distributed:
            # === Parallelized Mode (DDP) ===
            # Rank N processes Target N
            target_idx = self.global_rank

            # Initialize placeholders for gathering
            local_metrics = torch.zeros(3, device=self.device)
            C, H, W = (
                3,  # self.generator.cfg.in_channels,
                self.generator.cfg.image_size,
                self.generator.cfg.image_size,
            )
            local_vis_imgs = torch.zeros((24, C, H, W), device=self.device)
            did_work = torch.tensor(0.0, device=self.device)

            if target_idx < len(self.target_keys):

                c_key = self.target_keys[target_idx]
                result = process_single_target(c_key)

                if result is not None:
                    (m_rfid, m_fg, m_fb), m_imgs = result
                    local_metrics[0], local_metrics[1], local_metrics[2] = (
                        m_rfid,
                        m_fg,
                        m_fb,
                    )
                    local_vis_imgs.copy_(m_imgs)
                    did_work.fill_(1.0)
                    print(
                        f"[Rank {self.global_rank}] Completed target {target_idx} (rFID: {m_rfid:.4f})"
                    )
            # Gather from all ranks
            gathered_metrics = [
                torch.zeros_like(local_metrics) for _ in range(self.trainer.world_size)
            ]
            torch.distributed.all_gather(gathered_metrics, local_metrics)

            local_vis_u8 = (local_vis_imgs.clamp(0, 1) * 255).to(torch.uint8)
            world_size = self.trainer.world_size
            if self.global_rank == 0:
                gathered_vis_u8 = [
                    torch.empty_like(local_vis_u8) for _ in range(world_size)
                ]
                torch.distributed.gather(
                    local_vis_u8, gather_list=gathered_vis_u8, dst=0
                )
            else:
                torch.distributed.gather(local_vis_u8, dst=0)
            gathered_work_flags = [
                torch.zeros_like(did_work) for _ in range(self.trainer.world_size)
            ]
            torch.distributed.all_gather(gathered_work_flags, did_work)

            # Log gathered results on Rank 0
            if self.global_rank == 0:
                for i in range(self.trainer.world_size):
                    if gathered_work_flags[i].item() < 0.5:
                        continue
                    if i >= len(self.target_keys):
                        continue

                    c_key = self.target_keys[i]
                    metrics_tuple = (
                        gathered_metrics[i][0].item(),
                        gathered_metrics[i][1].item(),
                        gathered_metrics[i][2].item(),
                    )
                    imgs_stack = gathered_vis_u8[i]

                    val = log_target_results(c_key, metrics_tuple, imgs_stack)
                    avg_rfid += val
                    valid_targets += 1

        else:
            # === Serial Mode (Single GPU) ===
            # Loop through all targets
            for c_key in self.target_keys:
                result = process_single_target(c_key)
                if result is None:
                    continue

                metrics_tuple, imgs_stack = result
                val = log_target_results(c_key, metrics_tuple, imgs_stack)
                avg_rfid += val
                valid_targets += 1

        avg = torch.tensor(float("inf"), device=self.device)
        if self.global_rank == 0 and valid_targets > 0:
            avg.fill_(avg_rfid / valid_targets)

        if dist_on():
            torch.distributed.broadcast(avg, src=0)
        self._last_avg_rfid = avg.detach()
        # Final average log
        self.log(
            "val/avg_rFID",
            avg,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        print(f"[GeneratorPL] Logging complete. Processed {valid_targets} targets.")

        self.generator.train()
        if hasattr(self, "ema"):
            self.ema.restore()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or self.trainer.fast_dev_run:
            return

        avg = self._last_avg_rfid
        if dist_on():
            # ensure all ranks have the same scalar
            torch.distributed.broadcast(avg, src=0)

        # make it visible to ModelCheckpoint
        self.log(
            "val/avg_rFID",
            avg,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
            rank_zero_only=False,
        )
        self.log(
            "val_avg_rFID",
            avg,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
