"""
Webtoon Vertical Strip Slicer and Stitcher.

Manages ultra-tall continuous vertical manhwa strips. Handles slicing into
overlapping chunks for GPU memory efficiency, resolves speech bubbles bisected
across slice boundaries, and stitches translated strips back with seamless seams.
"""

import logging
from typing import List, Tuple
import cv2
import numpy as np

from server.schemas import BoundingBox

logger = logging.getLogger(__name__)


class WebtoonStripManager:
    """Handles splitting and re-stitching long vertical webtoon strips."""

    def __init__(self, slice_height: int = 1600, overlap: int = 200):
        self.slice_height = slice_height
        self.overlap = overlap

    def slice_strip(self, strip: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        Split a tall strip into overlapping horizontal slices.

        Args:
            strip: Full vertical strip array (H, W, 3).

        Returns:
            List of (slice_image, y_offset) tuples.
        """
        h, w = strip.shape[:2]
        if h <= self.slice_height:
            return [(strip, 0)]

        slices: List[Tuple[np.ndarray, int]] = []
        current_y = 0

        while current_y < h:
            end_y = min(h, current_y + self.slice_height)
            slice_crop = strip[current_y:end_y, :]
            slices.append((slice_crop, current_y))

            if end_y >= h:
                break

            current_y += (self.slice_height - self.overlap)

        return slices

    def merge_slice_boxes(
        self, slice_boxes: List[Tuple[List[BoundingBox], int]], total_height: int
    ) -> List[BoundingBox]:
        """
        Map bounding boxes detected on individual slices back to full strip coordinates,
        merging overlapping duplicates at slice seams.
        """
        global_boxes: List[BoundingBox] = []

        for boxes, y_offset in slice_boxes:
            for b in boxes:
                global_b = BoundingBox(
                    x=b.x,
                    y=b.y + y_offset,
                    w=b.w,
                    h=b.h,
                    confidence=b.confidence,
                )
                global_boxes.append(global_b)

        # De-duplicate boxes that appeared in the overlapping seam
        return self._deduplicate_seam_boxes(global_boxes)

    def _deduplicate_seam_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Merge boxes with high vertical and horizontal overlap."""
        if len(boxes) <= 1:
            return boxes

        # Sort by vertical center
        sorted_boxes = sorted(boxes, key=lambda b: (b.y, b.x))
        merged: List[BoundingBox] = []
        used = set()

        for i, box_a in enumerate(sorted_boxes):
            if i in used:
                continue

            current_box = box_a
            for j in range(i + 1, len(sorted_boxes)):
                if j in used:
                    continue
                box_b = sorted_boxes[j]

                # If box_b is too far down, stop checking
                if box_b.y > current_box.y2:
                    break

                # Calculate IoU
                inter_x1 = max(current_box.x, box_b.x)
                inter_y1 = max(current_box.y, box_b.y)
                inter_x2 = min(current_box.x2, box_b.x2)
                inter_y2 = min(current_box.y2, box_b.y2)

                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                union_area = current_box.area + box_b.area - inter_area
                iou = inter_area / float(union_area) if union_area > 0 else 0

                if iou > 0.4:
                    # Merge into a union bounding box
                    nx = min(current_box.x, box_b.x)
                    ny = min(current_box.y, box_b.y)
                    nw = max(current_box.x2, box_b.x2) - nx
                    nh = max(current_box.y2, box_b.y2) - ny
                    current_box = BoundingBox(
                        x=nx,
                        y=ny,
                        w=nw,
                        h=nh,
                        confidence=max(current_box.confidence, box_b.confidence),
                    )
                    used.add(j)

            merged.append(current_box)
            used.add(i)

        return merged
