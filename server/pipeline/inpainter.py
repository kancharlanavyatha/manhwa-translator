"""
Text Erasure & Background Inpainting Module.

Reconstructs the underlying artwork and speech balloon backgrounds after removing
original Korean text. Uses adaptive bubble color restoration for pristine speech balloons
and Navier-Stokes / Telea inpainting for textured art.
"""

import logging
from typing import List
import cv2
import numpy as np

from server.schemas import BoundingBox

logger = logging.getLogger(__name__)


class BubbleInpainter:
    """High-quality text erasure and background restoration engine."""

    def __init__(self, dilation_kernel_size: int = 3, inpaint_radius: int = 3):
        self.dilation_kernel_size = dilation_kernel_size
        self.inpaint_radius = inpaint_radius

    def inpaint(self, image: np.ndarray, boxes: List[BoundingBox]) -> np.ndarray:
        """
        Erase Korean text inside each bounding box while preserving the background.

        Args:
            image: Original panel image (BGR format).
            boxes: Bounding boxes of text / speech bubbles to clean.

        Returns:
            Cleaned image with text seamlessly erased.
        """
        if image is None or not boxes:
            return image

        result = image.copy()
        height, width = image.shape[:2]

        for box in boxes:
            y1 = max(0, box.y)
            y2 = min(height, box.y2)
            x1 = max(0, box.x)
            x2 = min(width, box.x2)

            if (y2 - y1) <= 6 or (x2 - x1) <= 6:
                continue

            roi = result[y1:y2, x1:x2]
            cleaned_roi = self._clean_roi(roi)
            result[y1:y2, x1:x2] = cleaned_roi

        return result

    def _clean_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Clean text out of an individual region of interest (ROI).
        Accurately separates speech balloon fill from dark text strokes.
        """
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # In manhwa, dialogue balloons have white/bright interior backgrounds (>160 luminosity)
        # Sample the interior 75% of the ROI to avoid dark panel backgrounds outside oval corners
        inner_y1, inner_y2 = int(h * 0.15), int(h * 0.85)
        inner_x1, inner_x2 = int(w * 0.15), int(w * 0.85)

        inner_roi = roi[inner_y1:inner_y2, inner_x1:inner_x2]
        inner_gray = gray[inner_y1:inner_y2, inner_x1:inner_x2]

        bright_pixels = inner_roi[inner_gray > 160]

        if len(bright_pixels) > 50:
            # Enclosed bright speech balloon
            median_bg = np.median(bright_pixels, axis=0).astype(np.uint8)

            # Detect dark text pixels inside the balloon
            text_threshold = int(np.mean(median_bg) * 0.72)
            _, dark_mask = cv2.threshold(gray, text_threshold, 255, cv2.THRESH_BINARY_INV)

            # Ensure we only erase text inside the bright balloon region, not the exterior dark borders/art
            # Balloon region has brightness close to median_bg
            color_diff = cv2.absdiff(roi, median_bg)
            diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
            # Text deviates strongly from median_bg, but must be within the balloon perimeter
            _, text_candidate = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)
            text_mask = cv2.bitwise_and(dark_mask, text_candidate)

            # Dilate slightly to cover font anti-aliasing edges
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.dilation_kernel_size, self.dilation_kernel_size)
            )
            dilated_mask = cv2.dilate(text_mask, kernel, iterations=1)

            # Seamlessly fill text pixels with the pristine balloon background color
            cleaned = roi.copy()
            cleaned[dilated_mask > 0] = median_bg
            return cleaned
        else:
            # Complex art / textured background: use OpenCV Telea inpainting
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
            _, edge_mask = cv2.threshold(grad, 40, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.dilation_kernel_size, self.dilation_kernel_size)
            )
            dilated = cv2.dilate(edge_mask, kernel, iterations=1)
            try:
                return cv2.inpaint(roi, dilated, self.inpaint_radius, cv2.INPAINT_TELEA)
            except Exception as exc:
                logger.warning("Inpainting fallback: %s", exc)
                return roi
