"""Python post-processing for segmentation masks.

Provides mask refinement operations including small region removal,
morphological filtering, and mask smoothing using OpenCV.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class PostProcessor:
    """Post-processing pipeline for segmentation masks.

    Args:
        config: Post-processing configuration dictionary.
    """

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        pp_cfg = config.get("postprocessing", config)
        self.small_region_threshold = pp_cfg.get("small_region_threshold", 100)
        morph_cfg = pp_cfg.get("morphology", {})
        self.kernel_size = morph_cfg.get("kernel_size", 5)
        self.operations = morph_cfg.get("operations", ["opening", "closing"])

    def process(self, mask: np.ndarray) -> np.ndarray:
        """Apply full post-processing pipeline to a segmentation mask.

        Args:
            mask: Input mask of shape (H, W) with integer class labels.

        Returns:
            Post-processed mask.
        """
        result = mask.copy()

        # Apply morphological operations per class
        result = self.morphological_filter(result)

        # Remove small regions
        result = self.remove_small_regions(result)

        return result

    def morphological_filter(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to each class in the mask.

        Args:
            mask: Segmentation mask (H, W).

        Returns:
            Morphologically filtered mask.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )

        unique_classes = np.unique(mask)
        result = np.zeros_like(mask)

        for cls in unique_classes:
            if cls == 0:  # Skip background
                continue

            binary_mask = (mask == cls).astype(np.uint8) * 255

            for op_name in self.operations:
                if op_name == "opening":
                    binary_mask = cv2.morphologyEx(
                        binary_mask, cv2.MORPH_OPEN, kernel
                    )
                elif op_name == "closing":
                    binary_mask = cv2.morphologyEx(
                        binary_mask, cv2.MORPH_CLOSE, kernel
                    )
                elif op_name == "erosion":
                    binary_mask = cv2.erode(binary_mask, kernel)
                elif op_name == "dilation":
                    binary_mask = cv2.dilate(binary_mask, kernel)

            result[binary_mask > 127] = cls

        # Fill unassigned pixels with background
        # (background where no class was assigned)
        return result

    def remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove small connected regions from each class.

        Args:
            mask: Segmentation mask (H, W).

        Returns:
            Mask with small regions removed.
        """
        unique_classes = np.unique(mask)
        result = np.zeros_like(mask)

        for cls in unique_classes:
            if cls == 0:
                continue

            binary_mask = (mask == cls).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )

            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area >= self.small_region_threshold:
                    result[labels == label_id] = cls

        return result

    def extract_contours(
        self, mask: np.ndarray, class_id: Optional[int] = None
    ) -> List[np.ndarray]:
        """Extract contours from segmentation mask.

        Args:
            mask: Segmentation mask (H, W).
            class_id: Specific class to extract. If None, extract all.

        Returns:
            List of contour arrays.
        """
        all_contours = []

        if class_id is not None:
            classes = [class_id]
        else:
            classes = [c for c in np.unique(mask) if c != 0]

        for cls in classes:
            binary_mask = (mask == cls).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            all_contours.extend(contours)

        return all_contours

    def smooth_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """Smooth mask boundaries using Gaussian blur and re-threshold.

        Args:
            mask: Segmentation mask (H, W).
            kernel_size: Gaussian kernel size.

        Returns:
            Smoothed mask.
        """
        unique_classes = np.unique(mask)
        result = np.zeros_like(mask)

        for cls in unique_classes:
            if cls == 0:
                continue

            binary = (mask == cls).astype(np.float32)
            smoothed = cv2.GaussianBlur(
                binary, (kernel_size, kernel_size), 0
            )
            result[smoothed > 0.5] = cls

        return result

    @staticmethod
    def visualize_contours(
        image: np.ndarray,
        contours: List[np.ndarray],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Draw contours on an image.

        Args:
            image: RGB image (H, W, 3).
            contours: List of contour arrays.
            save_path: Optional path to save visualization.

        Returns:
            Image with contours drawn.
        """
        vis = image.copy()
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        return vis
