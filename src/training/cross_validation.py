"""K-Fold cross-validation runner for segmentation models.

Splits the dataset into K folds, trains a model per fold, and
reports aggregated metrics across all folds.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from src.models import get_model
from src.training.trainer import Trainer
from src.utils.logger import get_logger


logger = get_logger("cross_validation", file=False)


class CrossValidator:
    """K-Fold cross-validation for segmentation models.

    Args:
        config: Full configuration dictionary.
        dataset: Full training dataset.
        k_folds: Number of folds.
        device: Torch device.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dataset: Any,
        k_folds: int = 5,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.dataset = dataset
        self.k_folds = k_folds
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def run(self) -> Dict[str, Any]:
        """Run K-fold cross-validation.

        Returns:
            Dictionary with per-fold and aggregated results.
        """
        kfold = KFold(
            n_splits=self.k_folds, shuffle=True, random_state=42
        )

        fold_results: List[Dict[str, Any]] = []
        indices = list(range(len(self.dataset)))

        logger.info(f"Starting {self.k_folds}-fold cross-validation")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold+1}/{self.k_folds}")
            logger.info(f"{'='*60}")
            logger.info(
                f"Train samples: {len(train_idx)}, "
                f"Val samples: {len(val_idx)}"
            )

            # Create data loaders for this fold
            train_subset = Subset(self.dataset, train_idx.tolist())
            val_subset = Subset(self.dataset, val_idx.tolist())

            batch_size = self.config.get("training", {}).get("batch_size", 8)
            num_workers = self.config.get("dataset", {}).get("num_workers", 4)

            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            # Create a fresh model for each fold
            model_cfg = self.config.get("model", {})
            model = get_model(
                name=model_cfg.get("name", "unet"),
                num_classes=model_cfg.get("num_classes", 21),
                pretrained=model_cfg.get("pretrained", True),
            )

            # Configure checkpoint dir per fold
            fold_config = dict(self.config)
            fold_config["checkpoint"] = dict(self.config.get("checkpoint", {}))
            fold_config["checkpoint"]["save_dir"] = os.path.join(
                self.config.get("checkpoint", {}).get(
                    "save_dir", "experiments/checkpoints"
                ),
                f"fold_{fold+1}",
            )
            fold_config["logging"] = dict(self.config.get("logging", {}))
            fold_config["logging"]["log_dir"] = os.path.join(
                self.config.get("logging", {}).get(
                    "log_dir", "experiments/logs"
                ),
                f"fold_{fold+1}",
            )

            # Train
            trainer = Trainer(model, fold_config, self.device)
            history = trainer.train(train_loader, val_loader)

            # Collect best results from this fold
            best_val_idx = int(np.argmin(history["val_losses"]))
            fold_result = {
                "fold": fold + 1,
                "best_epoch": best_val_idx + 1,
                "best_val_loss": history["val_losses"][best_val_idx],
                "best_val_miou": history["val_metrics"][best_val_idx]["miou"],
                "best_val_accuracy": history["val_metrics"][best_val_idx]["accuracy"],
                "final_train_loss": history["train_losses"][-1],
                "final_val_loss": history["val_losses"][-1],
            }
            fold_results.append(fold_result)

            logger.info(
                f"Fold {fold+1} best: "
                f"val_loss={fold_result['best_val_loss']:.4f}, "
                f"val_mIoU={fold_result['best_val_miou']:.4f}, "
                f"val_acc={fold_result['best_val_accuracy']:.4f}"
            )

        # Aggregate results
        aggregated = self._aggregate_results(fold_results)

        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Results Summary")
        logger.info(f"{'='*60}")
        logger.info(
            f"Mean Val Loss: {aggregated['mean_val_loss']:.4f} "
            f"± {aggregated['std_val_loss']:.4f}"
        )
        logger.info(
            f"Mean Val mIoU: {aggregated['mean_val_miou']:.4f} "
            f"± {aggregated['std_val_miou']:.4f}"
        )
        logger.info(
            f"Mean Val Acc:  {aggregated['mean_val_accuracy']:.4f} "
            f"± {aggregated['std_val_accuracy']:.4f}"
        )

        return {
            "fold_results": fold_results,
            "aggregated": aggregated,
        }

    def _aggregate_results(
        self, fold_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute aggregated statistics across folds."""
        val_losses = [r["best_val_loss"] for r in fold_results]
        val_mious = [r["best_val_miou"] for r in fold_results]
        val_accs = [r["best_val_accuracy"] for r in fold_results]

        return {
            "mean_val_loss": float(np.mean(val_losses)),
            "std_val_loss": float(np.std(val_losses)),
            "mean_val_miou": float(np.mean(val_mious)),
            "std_val_miou": float(np.std(val_mious)),
            "mean_val_accuracy": float(np.mean(val_accs)),
            "std_val_accuracy": float(np.std(val_accs)),
        }
