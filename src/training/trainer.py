"""Modular training loop for semantic segmentation.

Provides a clean, configurable training loop with:
- Multi-loss support (CrossEntropy, Dice)
- Optimizer and LR scheduler configuration
- TensorBoard logging
- Checkpoint save/resume
- Early stopping
- Gradient clipping
"""

import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.evaluation.metrics import mean_iou, pixel_accuracy
from src.utils.logger import get_logger


logger = get_logger("trainer", file=False)


class DiceLoss(nn.Module):
    """Soft Dice Loss for segmentation."""

    def __init__(self, smooth: float = 1.0, num_classes: int = 21):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred_soft)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class Trainer:
    """Semantic segmentation trainer with full training pipeline.

    Args:
        model: Segmentation model.
        config: Training configuration dictionary.
        device: Torch device for training.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)

        # Loss function
        loss_name = train_cfg.get("loss", "cross_entropy")
        num_classes = config.get("model", {}).get("num_classes", 21)
        if loss_name == "dice":
            self.criterion = DiceLoss(num_classes=num_classes)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # Optimizer
        opt_name = train_cfg.get("optimizer", "adam")
        lr = train_cfg.get("learning_rate", 0.001)
        wd = train_cfg.get("weight_decay", 0.0001)
        if opt_name == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=wd
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=wd
            )

        # LR Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        sched_name = sched_cfg.get("name", "cosine")
        if sched_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sched_cfg.get("step_size", 10),
                gamma=sched_cfg.get("gamma", 0.1),
            )
        elif sched_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5,
            )
        else:  # cosine
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg.get("T_max", self.epochs),
                eta_min=sched_cfg.get("eta_min", 1e-5),
            )

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = es_cfg.get("enabled", True)
        self.patience = es_cfg.get("patience", 10)
        self.min_delta = es_cfg.get("min_delta", 0.001)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Checkpointing
        ckpt_cfg = config.get("checkpoint", {})
        self.save_dir = ckpt_cfg.get("save_dir", "experiments/checkpoints")
        self.save_best = ckpt_cfg.get("save_best", True)
        self.save_every = ckpt_cfg.get("save_every", 5)
        os.makedirs(self.save_dir, exist_ok=True)

        # Logging
        log_cfg = config.get("logging", {})
        self.log_every = log_cfg.get("log_every", 10)
        self.writer = None
        if log_cfg.get("tensorboard", True):
            log_dir = log_cfg.get("log_dir", "experiments/logs")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        # Resume from checkpoint
        resume_path = ckpt_cfg.get("resume")
        if resume_path and os.path.exists(resume_path):
            self._load_checkpoint(resume_path)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mixup_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            mixup_fn: Optional Mixup augmentation function.

        Returns:
            Dictionary of training history.
        """
        logger.info(
            f"Starting training: {self.epochs} epochs, "
            f"device={self.device}, "
            f"train_batches={len(train_loader)}, "
            f"val_batches={len(val_loader)}"
        )

        num_classes = self.config.get("model", {}).get("num_classes", 21)

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_miou, train_acc = self._train_epoch(
                train_loader, epoch, num_classes, mixup_fn
            )
            self.train_losses.append(train_loss)
            self.train_metrics.append({"miou": train_miou, "accuracy": train_acc})

            # Validation epoch
            val_loss, val_miou, val_acc = self._validate_epoch(
                val_loader, epoch, num_classes
            )
            self.val_losses.append(val_loss)
            self.val_metrics.append({"miou": val_miou, "accuracy": val_acc})

            # Log epoch summary
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train mIoU: {train_miou:.4f} | Val mIoU: {val_miou:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalars(
                    "Loss", {"train": train_loss, "val": val_loss}, epoch
                )
                self.writer.add_scalars(
                    "mIoU", {"train": train_miou, "val": val_miou}, epoch
                )
                self.writer.add_scalars(
                    "Accuracy", {"train": train_acc, "val": val_acc}, epoch
                )
                self.writer.add_scalar(
                    "LR", self.optimizer.param_groups[0]["lr"], epoch
                )

            # LR scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Checkpointing
            if self.save_best and val_loss < self.best_val_loss:
                self._save_checkpoint(epoch, "best_model.pth")
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, f"checkpoint_epoch_{epoch+1}.pth")

            # Early stopping
            if self.early_stopping_enabled:
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch+1}"
                        )
                        break

        if self.writer:
            self.writer.close()

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        num_classes: int,
        mixup_fn: Optional[Any] = None,
    ) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_miou = 0.0
        total_acc = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Optional Mixup
            if mixup_fn is not None:
                images, masks_a, masks_b, lam = mixup_fn(images, masks)

            # Forward pass
            outputs = self.model(images)
            if mixup_fn is not None:
                loss = lam * self.criterion(outputs, masks_a) + \
                       (1 - lam) * self.criterion(outputs, masks_b)
            else:
                loss = self.criterion(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            self.optimizer.step()

            # Metrics
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                batch_miou = mean_iou(preds, masks, num_classes)
                batch_acc = pixel_accuracy(preds, masks)

            total_loss += loss.item()
            total_miou += batch_miou
            total_acc += batch_acc
            num_batches += 1
            self.global_step += 1

            # Progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mIoU": f"{batch_miou:.4f}",
            })

            # Batch-level TensorBoard logging
            if self.writer and batch_idx % self.log_every == 0:
                self.writer.add_scalar(
                    "Batch/train_loss", loss.item(), self.global_step
                )

        avg_loss = total_loss / max(num_batches, 1)
        avg_miou = total_miou / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)
        return avg_loss, avg_miou, avg_acc

    @torch.no_grad()
    def _validate_epoch(
        self, loader: DataLoader, epoch: int, num_classes: int
    ) -> Tuple[float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_miou = 0.0
        total_acc = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            preds = outputs.argmax(dim=1)
            batch_miou = mean_iou(preds, masks, num_classes)
            batch_acc = pixel_accuracy(preds, masks)

            total_loss += loss.item()
            total_miou += batch_miou
            total_acc += batch_acc
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_miou = total_miou / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)
        return avg_loss, avg_miou, avg_acc

    def _save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.global_step = checkpoint.get("global_step", 0)
        logger.info(f"Resumed from checkpoint: {path} (epoch {self.current_epoch})")
