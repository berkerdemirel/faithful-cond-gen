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
            # First batch: infer and store canonical order from dict keys
            self._cond_keys = list(cond_dict.keys())
        
        # Stack conditioning values in canonical order
        cond_tensors = [cond_dict[k] for k in self._cond_keys]
        cond_ids = torch.stack(cond_tensors, dim=1)

        return images, cond_ids

    def training_step(self, batch, batch_idx: int):
        images, cond_ids = self._unpack_batch(batch)
        # --- encode to latents ---
        with torch.no_grad():
            x0 = self.generator.encode(images)  # (B,4,h,w) if VAE frozen

        b = x0.shape[0]

        # --- forward/noising process ---
        t = torch.rand(b, device=x0.device, dtype=torch.float32)
        eps = torch.randn_like(x0)
        x_t, v_tgt = self.linear_interpolant(x0, t, eps)

        # --- velocity prediction ---
        v_hat = self.generator.velocity_prediction(x_t=x_t, t=t, cond_ids=cond_ids)

        # --- loss ---
        loss = F.mse_loss(v_hat, v_tgt)

        self.log("train/loss", loss, prog_bar=True)
        return loss

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
        if self.trainer.sanity_checking:
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
        print(f"[GeneratorPL] Rank {self.global_rank} initializing metrics...")
        if not hasattr(self, "ema"):
            self.ema = EMA(self.generator, decay=0.9999)
        self._last_avg_rfid = torch.tensor(float("inf"), device=self.device)
        self.fidelity_metrics = ConditionalFidelityMetrics(self.device)
        self.val_buffer = {}
        self.target_keys = []

        dm = self.trainer.datamodule

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
                    self.generator.sample(curr_batch, num_inference_steps=25, t_cutoff=0.04)
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
        if self.trainer.sanity_checking:
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
