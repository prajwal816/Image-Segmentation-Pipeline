"""Ablation study runner for comparing models and augmentations.

Iterates over experiment configurations, trains each variant,
collects metrics, and produces comparison tables and plots.
"""

import os
import json
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.models import get_model
from src.datasets.segmentation_dataset import SyntheticSegmentationDataset, create_dataloaders
from src.augmentation.transforms import get_augmentation_pipeline
from src.training.trainer import Trainer
from src.utils.config import load_config, merge_configs
from src.utils.logger import get_logger


logger = get_logger("ablation", file=False)


class AblationRunner:
    """Run ablation studies comparing models or augmentation strategies.

    Args:
        ablation_config_path: Path to ablation config YAML.
        device: Torch device.
    """

    def __init__(
        self,
        ablation_config_path: str,
        device: Optional[str] = None,
    ):
        import torch
        self.ablation_config = load_config(ablation_config_path)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.ablation_cfg = self.ablation_config.get("ablation", {})
        self.output_dir = self.ablation_cfg.get(
            "output_dir", "experiments/ablation"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Execute all ablation experiments.

        Returns:
            Dictionary with all experiment results.
        """
        experiments = self.ablation_cfg.get("experiments", [])
        ablation_type = self.ablation_cfg.get("type", "model")

        logger.info(f"Starting {ablation_type} ablation study "
                     f"with {len(experiments)} experiments")

        results = []
        for i, exp_cfg in enumerate(experiments):
            exp_name = exp_cfg.get("name", f"experiment_{i+1}")
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {i+1}/{len(experiments)}: {exp_name}")
            logger.info(f"{'='*60}")

            result = self._run_experiment(exp_name, exp_cfg)
            results.append(result)

        # Generate comparison outputs
        self._generate_comparison_table(results)
        self._generate_comparison_plots(results)

        # Save results
        results_path = os.path.join(self.output_dir, "ablation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")

        return {"experiments": results}

    def _run_experiment(
        self, name: str, exp_overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single ablation experiment.

        Args:
            name: Experiment name.
            exp_overrides: Config overrides for this experiment.

        Returns:
            Experiment results dictionary.
        """
        import torch

        # Build config for this experiment
        base_path = self.ablation_cfg.get("base_config", "configs/default.yaml")
        if os.path.exists(base_path):
            base_config = load_config(base_path)
        else:
            base_config = {}

        # Apply training overrides from ablation config
        train_overrides = self.ablation_cfg.get("training", {})
        config = merge_configs(base_config, {"training": train_overrides})

        # Apply experiment-specific overrides
        config = merge_configs(config, exp_overrides)

        # Setup checkpoint/log dirs for this experiment
        config["checkpoint"] = config.get("checkpoint", {})
        config["checkpoint"]["save_dir"] = os.path.join(
            self.output_dir, name, "checkpoints"
        )
        config["logging"] = config.get("logging", {})
        config["logging"]["log_dir"] = os.path.join(
            self.output_dir, name, "logs"
        )

        # Create model
        model_cfg = config.get("model", {})
        model = get_model(
            name=model_cfg.get("name", "unet"),
            num_classes=model_cfg.get("num_classes", 21),
            pretrained=model_cfg.get("pretrained", True),
        )

        # Create data
        aug_pipeline = config.get("augmentation", {}).get("pipeline", "full")
        transform = get_augmentation_pipeline(
            aug_pipeline, config.get("augmentation", {}),
            config.get("dataset", {}).get("image_size", 256),
        )
        train_loader, val_loader = create_dataloaders(config, transform)

        # Train
        trainer = Trainer(model, config, self.device)
        history = trainer.train(train_loader, val_loader)

        # Collect best results
        best_idx = int(np.argmin(history["val_losses"]))
        result = {
            "name": name,
            "model": model_cfg.get("name", "unet"),
            "augmentation": aug_pipeline,
            "pretrained": model_cfg.get("pretrained", True),
            "best_epoch": best_idx + 1,
            "best_val_loss": float(history["val_losses"][best_idx]),
            "best_val_miou": float(history["val_metrics"][best_idx]["miou"]),
            "best_val_accuracy": float(history["val_metrics"][best_idx]["accuracy"]),
            "final_train_loss": float(history["train_losses"][-1]),
            "total_epochs": len(history["train_losses"]),
        }

        logger.info(
            f"Experiment '{name}' complete: "
            f"mIoU={result['best_val_miou']:.4f}, "
            f"acc={result['best_val_accuracy']:.4f}"
        )

        return result

    def _generate_comparison_table(self, results: List[Dict]) -> None:
        """Generate and print comparison table."""
        header = f"{'Experiment':<30} {'Model':<12} {'Aug':<10} "
        header += f"{'mIoU':<8} {'Accuracy':<10} {'Val Loss':<10} {'Epochs':<8}"
        separator = "-" * len(header)

        lines = [separator, header, separator]
        for r in results:
            line = (
                f"{r['name']:<30} {r['model']:<12} {r['augmentation']:<10} "
                f"{r['best_val_miou']:<8.4f} {r['best_val_accuracy']:<10.4f} "
                f"{r['best_val_loss']:<10.4f} {r['total_epochs']:<8}"
            )
            lines.append(line)
        lines.append(separator)

        table_str = "\n".join(lines)
        logger.info(f"\nAblation Results:\n{table_str}")

        # Save to file
        table_path = os.path.join(self.output_dir, "comparison_table.txt")
        with open(table_path, "w") as f:
            f.write(table_str)

    def _generate_comparison_plots(self, results: List[Dict]) -> None:
        """Generate comparison bar charts."""
        if not results:
            return

        names = [r["name"] for r in results]
        mious = [r["best_val_miou"] for r in results]
        accs = [r["best_val_accuracy"] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # mIoU comparison
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars1 = axes[0].bar(names, mious, color=colors, edgecolor="black")
        axes[0].set_ylabel("Mean IoU", fontsize=12)
        axes[0].set_title("mIoU Comparison", fontsize=14)
        axes[0].set_ylim(0, 1)
        for bar, val in zip(bars1, mious):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9,
            )
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha="right")

        # Accuracy comparison
        bars2 = axes[1].bar(names, accs, color=colors, edgecolor="black")
        axes[1].set_ylabel("Pixel Accuracy", fontsize=12)
        axes[1].set_title("Accuracy Comparison", fontsize=14)
        axes[1].set_ylim(0, 1)
        for bar, val in zip(bars2, accs):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9,
            )
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "ablation_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Comparison plots saved to {save_path}")
