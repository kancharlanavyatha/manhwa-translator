"""
Speech Bubble and Text Region Detection Module.

Identifies speech balloons, thought clouds, narration boxes, and free text regions
within webtoon panels using computer vision contour analysis, morphological operators,
and non-maximum suppression.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Tuple
import cv2
import numpy as np

from server.schemas import BoundingBox

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract interface for comic text and speech bubble detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detect speech bubbles and text regions in a BGR or RGB image.

        Args:
            image: Image array of shape (H, W, 3).

        Returns:
            List of BoundingBox objects sorted by natural reading flow.
        """
        pass


class HeuristicBubbleDetector(BaseDetector):
    """
    Robust, CPU-friendly speech bubble detector.

    Combines:
    1. High-luminance speech balloon contour extraction (thresholding + morphology).
    2. Dark edge / gradient text density verification inside candidates.
    3. Non-Maximum Suppression (NMS) to eliminate duplicate/nested bounding boxes.
    4. Webtoon reading-order sorting (top-to-bottom, left-to-right).
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area_ratio: float = 0.20,
        min_aspect_ratio: float = 0.25,
        max_aspect_ratio: float = 4.0,
        iou_threshold: float = 0.35,
    ):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.iou_threshold = iou_threshold

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect bubbles and return them sorted in reading order."""
        if image is None or image.size == 0:
            return []

        height, width = image.shape[:2]
        max_area = int(height * width * self.max_area_ratio)

        # Convert to Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        candidate_boxes: List[Tuple[int, int, int, int, float]] = []

        # -------------------------------------------------------------
        # Strategy A: Speech Bubble Extraction (Bright solid balloons)
        # -------------------------------------------------------------
        # In manhwa, >85% of dialogue balloons have white/bright fills (>215 luminosity)
        _, bright_mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)

        # Morphological Closing to merge small text gaps inside the bubble
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed_bubbles = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed_bubbles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Speech bubbles never cover full comic strip frames or exceed panel boundaries
            if w > width * 0.70 or h > height * 0.35:
                continue

            area = cv2.contourArea(cnt)
            if area < self.min_area or area > max_area:
                continue

            aspect = w / float(h)
            if not (self.min_aspect_ratio <= aspect <= self.max_aspect_ratio):
                continue

            # Verify that the interior contains contrast (text) and bright balloon fill
            roi_gray = gray[y : y + h, x : x + w]
            std_dev = np.std(roi_gray)
            bright_ratio = np.mean(roi_gray > 190)

            # Text stroke verification: speech balloons contain multiple small letter strokes
            # Characters/animals (like white cats/dogs) have smooth white fur without text strokes
            text_thresh = min(int(np.mean(roi_gray) * 0.75), 180)
            _, text_bin = cv2.threshold(roi_gray, text_thresh, 255, cv2.THRESH_BINARY_INV)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(text_bin)
            text_strokes = sum(
                1 for i in range(1, num_labels)
                if 8 <= stats[i, cv2.CC_STAT_AREA] <= (w * h * 0.15)
                and 3 <= stats[i, cv2.CC_STAT_WIDTH] <= (w * 0.60)
                and 4 <= stats[i, cv2.CC_STAT_HEIGHT] <= (h * 0.60)
            )

            if std_dev > 5.0 and bright_ratio >= 0.45 and text_strokes >= 4:
                confidence = min(1.0, 0.6 + (std_dev / 40.0))
                candidate_boxes.append((x, y, w, h, confidence))

        # -------------------------------------------------------------
        # Strategy B: Dark text regions on lighter art / narration boxes
        # -------------------------------------------------------------
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        _, text_edges = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
        # Connect adjacent text characters into words and sentences
        text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
        connected_text = cv2.morphologyEx(text_edges, cv2.MORPH_CLOSE, text_kernel)

        text_contours, _ = cv2.findContours(
            connected_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in text_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Strictly reject full panel frames and gutter borders
            if w > width * 0.65 or h > height * 0.30:
                continue
            if x <= 5 and (x + w) >= (width - 5):
                continue

            area = cv2.contourArea(cnt)
            if area < (self.min_area // 2) or area > max_area:
                continue

            aspect = w / float(h)
            if 0.3 <= aspect <= 5.0:
                pad_x = int(w * 0.08)
                pad_y = int(h * 0.08)
                nx = max(0, x - pad_x)
                ny = max(0, y - pad_y)
                nw = min(width - nx, w + 2 * pad_x)
                nh = min(height - ny, h + 2 * pad_y)

                # Avoid adding if this text region is already inside an identified speech bubble
                already_covered = any(
                    (bx <= nx and by <= ny and (bx + bw) >= (nx + nw) and (by + bh) >= (ny + nh))
                    for bx, by, bw, bh, _ in candidate_boxes
                )
                if not already_covered:
                    candidate_boxes.append((nx, ny, nw, nh, 0.70))

        # Apply Non-Maximum Suppression to remove duplicates/overlapping boxes
        filtered = self._apply_nms(candidate_boxes, self.iou_threshold)

        # Eliminate any giant box that encloses another valid box
        pruned = []
        for i, b1 in enumerate(filtered):
            x1, y1, w1, h1, _ = b1
            encloses_smaller = False
            for j, b2 in enumerate(filtered):
                if i != j:
                    x2, y2, w2, h2, _ = b2
                    if x1 <= x2 and y1 <= y2 and (x1 + w1) >= (x2 + w2) and (y1 + h1) >= (y2 + h2):
                        if (w1 * h1) > 2.0 * (w2 * h2):
                            encloses_smaller = True
                            break
            if not encloses_smaller:
                pruned.append(b1)

        # Convert to BoundingBox schemas
        boxes = [
            BoundingBox(x=x, y=y, w=w, h=h, confidence=round(conf, 2))
            for x, y, w, h, conf in pruned
        ]

        # Sort in natural Korean reading order
        return self._sort_reading_order(boxes)

    def _apply_nms(
        self, boxes: List[Tuple[int, int, int, int, float]], iou_thresh: float
    ) -> List[Tuple[int, int, int, int, float]]:
        """Non-Maximum Suppression to eliminate overlapping boxes."""
        if not boxes:
            return []

        # Convert to [x1, y1, x2, y2, score]
        box_arr = []
        for x, y, w, h, score in boxes:
            box_arr.append([x, y, x + w, y + h, score])
        box_arr = np.array(box_arr, dtype=np.float32)

        x1 = box_arr[:, 0]
        y1 = box_arr[:, 1]
        x2 = box_arr[:, 2]
        y2 = box_arr[:, 3]
        scores = box_arr[:, 4]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            # Intersection over Union
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        result = []
        for idx in keep:
            r = box_arr[idx]
            rx = int(r[0])
            ry = int(r[1])
            rw = int(r[2] - r[0])
            rh = int(r[3] - r[1])
            result.append((rx, ry, rw, rh, float(r[4])))

        return result

    def _sort_reading_order(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Sort boxes by Korean webtoon reading order.
        Webtoons are vertical scrollers read top-to-bottom, left-to-right.
        Clusters bubbles that appear on roughly the same horizontal level.
        """
        if len(boxes) <= 1:
            return boxes

        # Sort primarily by vertical Y centroid with a line threshold band
        # Boxes within 80px vertically are considered on the same "row", ordered left to right.
        def reading_key(box: BoundingBox) -> Tuple[int, int]:
            row_bucket = int(box.center_y // 80)
            col_x = int(box.center_x)
            return (row_bucket, col_x)

        return sorted(boxes, key=reading_key)
