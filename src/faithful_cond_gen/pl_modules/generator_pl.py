from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.utils.metrics import ConditionalFidelityMetrics
from torch.optim import AdamW


@dataclass
class GeneratorPLConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
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
        return AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

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
        # We assume the dataset __getitem__ inserted keys in the correct order.
        # e.g. RxRx1: {'cell_type': ..., 'sirna': ...} -> [cell_type, sirna]
        # e.g. CelebA: {'Male': ..., 'Smiling': ...} -> [Male, Smiling]
        cond_dict = conditioning.get("cond")
        if cond_dict is None:
            raise ValueError("Batch missing 'cond' dict in conditioning")

        # dict.values() preserves insertion order in Python 3.7+
        cond_tensors = list(cond_dict.values())

        # Stack (B,) tensors into (B, K)
        cond_ids = torch.stack(cond_tensors, dim=1)

        return images, cond_ids

    def training_step(self, batch, batch_idx: int):
        images, cond_ids = self._unpack_batch(batch)

        # --- encode to latents ---
        with torch.no_grad():
            x0 = self.generator.encode(images)  # (B,4,h,w) if VAE frozen

        b = x0.shape[0]

        # --- forward/noising process ---
        t = torch.rand(b, device=x0.device, dtype=x0.dtype)
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

        t = torch.rand(b, device=x0.device, dtype=x0.dtype)
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

        t = torch.rand(b, device=x0.device, dtype=x0.dtype)
        eps = torch.randn_like(x0)
        x_t, v_tgt = self.linear_interpolant(x0, t, eps)

        v_hat = self.generator.velocity_prediction(x_t=x_t, t=t, cond_ids=cond_ids)
        loss = F.mse_loss(v_hat, v_tgt)

        self.log("test/loss", loss, prog_bar=True)
        return loss

    def on_fit_start(self):
        """
        Selects target conditions (from config or random sampling)
        and buffers their real validation images using get_matching_dataset.
        """
        if self.global_rank == 0:
            print("[GeneratorPL] Initializing validation targets...")
            self.val_buffer = {}  # Store {cond_tuple: tensor_images}
            self.fidelity_metrics = ConditionalFidelityMetrics(self.device)

            dm = self.trainer.datamodule
            targets = []

            # 1. Determine Target Conditions
            if (
                self.cfg.val_target_conditions is not None
                and len(self.cfg.val_target_conditions) > 0
            ):
                targets = self.cfg.val_target_conditions
            else:
                # Randomly sample from available validation conditions
                print(
                    "No preset conditions found. Sampling 8 random combos from validation set."
                )
                df_conds = dm.available_conditions("val")
                # Exclude count/category cols to get pure attribute keys
                drop_cols = ["count", "comp_category"]
                attr_cols = [c for c in df_conds.columns if c not in drop_cols]

                # Sample 8 random rows
                sample_df = df_conds.sample(n=8, random_state=42)
                targets = sample_df[attr_cols].to_dict(orient="records")

            # 2. Buffer Images for each Target
            # We assume we need ~128 images for stable FID
            MAX_SAMPLES = 128

            for cond_dict in targets:
                # Use the new API we added to DataModules
                ds_subset = dm.get_matching_dataset(
                    "val", cond_dict, max_samples=MAX_SAMPLES
                )

                if len(ds_subset) < 4:
                    print(
                        f"⚠️ Condition {cond_dict} has too few samples ({len(ds_subset)}). Skipping."
                    )
                    continue

                # Load all images into a tensor
                # Note: We access [0] which is the image tensor from __getitem__
                imgs = torch.stack([ds_subset[i][0] for i in range(len(ds_subset))])
                imgs = imgs.to(self.device)

                # Create a tuple key for storage (values sorted by key for determinism)
                # e.g. tuple(1, 1138)
                # We also need the tensor version of this condition for the model
                # We grab it from the first sample's 'cond' dict
                _, first_sample_cond = ds_subset[0]
                cond_vals = list(first_sample_cond["cond"].values())
                cond_tensor = torch.stack(cond_vals).to(self.device)  # (K,)

                # Store
                dict_key = tuple(
                    sorted(cond_dict.items())
                )  # Hashable dict representation
                self.val_buffer[dict_key] = {
                    "real_images": imgs,
                    "cond_tensor": cond_tensor,
                    "cond_dict": cond_dict,
                }
                self.target_keys.append(dict_key)
                print(f"  ✅ Buffered {len(imgs)} images for {cond_dict}")

    def on_train_batch_start(self, batch, batch_idx):
        """
        Trigger logging every 5000 global steps.
        """
        if self.global_step % 5000 == 0:
            self._log_images_and_metrics()

    # @torch.no_grad()
    # def _log_images_and_metrics(self):
    #     """
    #     Parallelized Logging:
    #     Each GPU (Rank N) processes the Nth target condition.
    #     Results are gathered and logged by Rank 0.
    #     """
    #     print(
    #         f"[GeneratorPL] Rank {self.global_rank} entering logging step {self.global_step}..."
    #     )
    #     self.generator.eval()

    #     # 1. Identify the target for this rank
    #     target_idx = self.global_rank
    #     num_targets = len(self.target_keys)

    #     # Tensors to store results (initialize placeholders)
    #     # Metrics: [rFID, FID_gen, FID_base]
    #     local_metrics = torch.zeros(3, device=self.device)
    #     # Images: Stack of [Real(8) | Rec(8) | Gen(8)] -> (24, C, H, W)
    #     # We need a reference shape for initialization; use config or deduce
    #     C, H, W = (
    #         self.generator.cfg.in_channels,
    #         self.generator.cfg.image_size,
    #         self.generator.cfg.image_size,
    #     )
    #     local_vis_imgs = torch.zeros((24, C, H, W), device=self.device)

    #     # Flag to track if this rank actually did work (to avoid logging zeros from idle ranks)
    #     did_work = torch.tensor(0.0, device=self.device)
    #     # 2. Process specific target if within bounds
    #     if target_idx < num_targets:
    #         c_key = self.target_keys[target_idx]
    #         data = self.val_buffer.get(c_key)

    #         real_imgs = data["real_images"]  # (N, C, H, W)
    #         cond_tensor = data["cond_tensor"]  # (K,)

    #         # --- A. Visualization (Fixed Reconstruction & Sampling) ---
    #         # Take 8 samples for visualization
    #         vis_real = real_imgs[:8].to(self.device)

    #         # Reconstruct
    #         latents = self.generator.encode(vis_real)
    #         rec_imgs = self.generator.decode(latents)

    #         # Generate Random (8 samples)
    #         cond_batch_vis = cond_tensor.unsqueeze(0).repeat(8, 1).to(self.device)
    #         gen_vis = self.generator.sample(cond_batch_vis, num_inference_steps=50)

    #         # Pack images: (8+8+8, C, H, W)
    #         local_vis_imgs = torch.cat([vis_real, rec_imgs, gen_vis], dim=0)

    #         # --- B. Metric Computation (rFID) ---
    #         # Generate N samples (matching Real count) for split-half calculation
    #         n_gen = len(real_imgs)
    #         cond_batch_metric = (
    #             cond_tensor.unsqueeze(0).repeat(n_gen, 1).to(self.device)
    #         )

    #         # Batch generation loop
    #         gen_metric_list = []
    #         batch_size = 16
    #         for j in range(0, n_gen, batch_size):
    #             curr_batch = cond_batch_metric[j : j + batch_size]
    #             gen_metric_list.append(
    #                 self.generator.sample(curr_batch, num_inference_steps=25)
    #             )
    #         gen_metric_imgs = torch.cat(gen_metric_list, dim=0)

    #         # Compute rFID
    #         rfid, fid_gen, fid_base = self.fidelity_metrics.compute_rfid(
    #             real_imgs, gen_metric_imgs
    #         )

    #         local_metrics[0] = rfid
    #         local_metrics[1] = fid_gen
    #         local_metrics[2] = fid_base
    #         did_work.fill_(1.0)

    #         print(
    #             f"[Rank {self.global_rank}] Completed target {target_idx} (rFID: {rfid:.4f})"
    #         )

    #     # 3. Gather results from all ranks to Rank 0
    #     # gathered_metrics: (World_Size, 3)
    #     gathered_metrics = [
    #         torch.zeros_like(local_metrics) for _ in range(self.trainer.world_size)
    #     ]
    #     torch.distributed.all_gather(gathered_metrics, local_metrics)

    #     # gathered_vis_imgs: (World_Size, 24, C, H, W)
    #     # Note: all_gather requires tensors to be same size across ranks.
    #     # Our initialization guarantees this (zeros if idle).
    #     gathered_vis_imgs = [
    #         torch.zeros_like(local_vis_imgs) for _ in range(self.trainer.world_size)
    #     ]
    #     torch.distributed.all_gather(gathered_vis_imgs, local_vis_imgs)

    #     # gathered_work_flags: (World_Size,)
    #     gathered_work_flags = [
    #         torch.zeros_like(did_work) for _ in range(self.trainer.world_size)
    #     ]
    #     torch.distributed.all_gather(gathered_work_flags, did_work)

    #     # 4. Log everything on Rank 0
    #     if self.global_rank == 0:
    #         avg_rfid = 0.0
    #         valid_targets = 0

    #         for i in range(self.trainer.world_size):
    #             # Skip if this rank didn't process a valid target
    #             if gathered_work_flags[i].item() < 0.5:
    #                 continue

    #             # Match index i to the target key
    #             # (Assumes world_size >= num_targets, and ranks map 1:1 to indices)
    #             if i >= len(self.target_keys):
    #                 continue

    #             c_key = self.target_keys[i]
    #             # We need the dictionary for the tag string; grab from local buffer or re-derive
    #             # Since Rank 0 has the full buffer (from on_fit_start), we can look it up
    #             if c_key not in self.val_buffer:
    #                 continue
    #             cond_dict = self.val_buffer[c_key]["cond_dict"]

    #             # Extract metrics
    #             m = gathered_metrics[i]
    #             rfid, fid_gen, fid_base = m[0].item(), m[1].item(), m[2].item()

    #             # Extract images
    #             # Shape (24, C, H, W) -> Split into Real(8), Rec(8), Gen(8)
    #             imgs_stack = gathered_vis_imgs[i]
    #             vis_real = imgs_stack[0:8]
    #             rec_imgs = imgs_stack[8:16]
    #             gen_vis = imgs_stack[16:24]

    #             # Logging Keys
    #             cond_str = "_".join([f"{v}" for k, v in cond_dict.items()])
    #             cond_tag = f"cond_{cond_str}"

    #             # 1. Log Metrics
    #             self.log_dict(
    #                 {
    #                     f"val/{cond_tag}/rFID": rfid,
    #                     f"val/{cond_tag}/FID_gen": fid_gen,
    #                     f"val/{cond_tag}/FID_base": fid_base,
    #                 },
    #                 logger=True,
    #                 rank_zero_only=True,  # redundant but safe
    #             )

    #             avg_rfid += rfid
    #             valid_targets += 1

    #             # 2. Log Images

    #             import wandb
    #             from torchvision.utils import make_grid

    #             def grid_to_wandb(tensor):
    #                 grid = make_grid(tensor, nrow=8, normalize=False)
    #                 return wandb.Image(grid.cpu().permute(1, 2, 0).numpy())

    #             self.logger.experiment.log(
    #                 {
    #                     f"vis/{cond_tag}/fixed_reconstruction": grid_to_wandb(
    #                         torch.cat([vis_real, rec_imgs], dim=0)
    #                     ),
    #                     f"vis/{cond_tag}/random_generation": grid_to_wandb(gen_vis),
    #                 },
    #                 step=self.global_step,
    #             )

    #         if valid_targets > 0:
    #             self.log(
    #                 "val/avg_rFID",
    #                 avg_rfid / valid_targets,
    #                 logger=True,
    #                 rank_zero_only=True,
    #             )

    #         print(f"[GeneratorPL] Logging complete. Processed {valid_targets} targets.")

    #     self.generator.train()

    @torch.no_grad()
    def _log_images_and_metrics(self):
        """
        Computes rFID and generates visualization images.
        - Single GPU: Processes all targets sequentially.
        - DDP: Distributes targets across ranks (Rank N processes Target N), then gathers results.
        """
        print(
            f"[GeneratorPL] Rank {self.global_rank} entering logging step {self.global_step}..."
        )
        self.generator.eval()

        # --- Helper: Core Generation Logic ---
        def process_single_target(c_key):
            """Generates images and computes metrics for a specific target key."""
            data = self.val_buffer.get(c_key)
            if data is None:
                return None

            real_imgs = data["real_images"]
            cond_tensor = data["cond_tensor"]  # (K,)

            # 1. Visualization
            vis_real = real_imgs[:8].to(self.device)
            latents = self.generator.encode(vis_real)
            rec_imgs = self.generator.decode(latents)

            cond_batch_vis = cond_tensor.unsqueeze(0).repeat(8, 1).to(self.device)
            gen_vis = self.generator.sample(cond_batch_vis, num_inference_steps=50)

            # Stack for transport/logging: (24, C, H, W) -> [Real|Rec|Gen]
            vis_imgs = torch.cat([vis_real, rec_imgs, gen_vis], dim=0)

            # 2. Metrics (rFID)
            n_gen = len(real_imgs)
            cond_batch_metric = (
                cond_tensor.unsqueeze(0).repeat(n_gen, 1).to(self.device)
            )

            gen_metric_list = []
            batch_size = 16
            for j in range(0, n_gen, batch_size):
                curr_batch = cond_batch_metric[j : j + batch_size]
                gen_metric_list.append(
                    self.generator.sample(curr_batch, num_inference_steps=25)
                )
            gen_metric_imgs = torch.cat(gen_metric_list, dim=0)

            rfid, fid_gen, fid_base = self.fidelity_metrics.compute_rfid(
                real_imgs, gen_metric_imgs
            )

            return (rfid, fid_gen, fid_base), vis_imgs

        # --- Helper: Logging Logic ---
        def log_target_results(c_key, metrics, imgs_stack):
            """Logs metrics and images to WandB/Console for a single target."""
            rfid, fid_gen, fid_base = metrics

            # Unpack images
            vis_real = imgs_stack[0:8]
            rec_imgs = imgs_stack[8:16]
            gen_vis = imgs_stack[16:24]

            cond_dict = self.val_buffer[c_key]["cond_dict"]
            cond_str = "_".join([f"{v}" for k, v in cond_dict.items()])
            cond_tag = f"cond_{cond_str}"

            # Log Metrics
            self.log_dict(
                {
                    f"val/{cond_tag}/rFID": rfid,
                    f"val/{cond_tag}/FID_gen": fid_gen,
                    f"val/{cond_tag}/FID_base": fid_base,
                },
                logger=True,
                rank_zero_only=True,
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
                self.generator.cfg.in_channels,
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

            gathered_vis_imgs = [
                torch.zeros_like(local_vis_imgs) for _ in range(self.trainer.world_size)
            ]
            torch.distributed.all_gather(gathered_vis_imgs, local_vis_imgs)

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
                    imgs_stack = gathered_vis_imgs[i]

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

        # Final average log
        if valid_targets > 0:
            self.log(
                "val/avg_rFID",
                avg_rfid / valid_targets,
                logger=True,
                rank_zero_only=True,
            )
            print(f"[GeneratorPL] Logging complete. Processed {valid_targets} targets.")

        self.generator.train()
