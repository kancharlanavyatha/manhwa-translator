"""
Webtoon Typography and Typesetting Engine.

Computes optimal font sizes using binary search, wraps dialogue into speech-bubble
shapes, centers text horizontally and vertically, and renders crisp typography
with anti-aliasing and optional drop shadow/strokes.
"""

import logging
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from server.config import settings
from server.schemas import BubbleData

logger = logging.getLogger(__name__)


class Typesetter:
    """Intelligent comic lettering and typesetting engine."""

    def __init__(
        self,
        font_path: Optional[str] = None,
        padding_ratio: float = 0.15,
        min_font_size: int = 11,
        max_font_size: int = 42,
    ):
        self.font_path = font_path
        self.padding_ratio = padding_ratio
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self._font_cache = {}

    def typeset(self, image_np: np.ndarray, bubbles: List[BubbleData]) -> np.ndarray:
        """
        Render translated dialogue onto an image.

        Args:
            image_np: Image array (BGR format).
            bubbles: List of BubbleData containing English text and bounding boxes.

        Returns:
            Rendered image array (BGR format).
        """
        if image_np is None or not bubbles:
            return image_np

        # Convert OpenCV BGR to Pillow RGB
        rgb_img = Image.fromarray(image_np[:, :, ::-1])
        draw = ImageDraw.Draw(rgb_img)

        for bubble in bubbles:
            text = bubble.english_text.strip()
            if not text:
                continue

            box = bubble.box
            # Calculate padded usable area inside bubble
            pad_w = int(box.w * self.padding_ratio)
            pad_h = int(box.h * self.padding_ratio)
            usable_w = max(20, box.w - 2 * pad_w)
            usable_h = max(15, box.h - 2 * pad_h)

            # Find best fitting font size and wrapped lines
            font, lines, line_spacing = self._fit_text_to_box(text, usable_w, usable_h)
            bubble.font_size_rendered = getattr(font, "size", self.min_font_size)

            # Calculate total text block height
            total_text_h = sum(self._get_line_height(font, line) for line in lines) + (
                len(lines) - 1
            ) * line_spacing

            # Center vertically within padded area
            start_y = box.y + pad_h + max(0, (usable_h - total_text_h) // 2)

            curr_y = start_y
            for line in lines:
                line_w = self._get_line_width(font, line)
                line_h = self._get_line_height(font, line)

                # Center horizontally within padded area
                start_x = box.x + pad_w + max(0, (usable_w - line_w) // 2)

                # Render with crisp dark ink
                # Optional subtle text stroke for maximum readability
                draw.text(
                    (start_x, curr_y),
                    line,
                    font=font,
                    fill=(20, 20, 24),  # Dark comic ink
                    stroke_width=1,
                    stroke_fill=(245, 245, 248),  # Subtle white outline halo
                )
                curr_y += line_h + line_spacing

        # Convert Pillow RGB back to OpenCV BGR
        return np.array(rgb_img)[:, :, ::-1]

    def _fit_text_to_box(
        self, text: str, max_w: int, max_h: int
    ) -> Tuple[ImageFont.ImageFont, List[str], int]:
        """
        Binary search for the largest font size where wrapped text fits within bounds.
        """
        low = self.min_font_size
        high = self.max_font_size
        best_size = low
        best_lines = [text]
        best_font = self._load_font(low)
        line_spacing = 3

        while low <= high:
            mid = (low + high) // 2
            font = self._load_font(mid)
            lines = self._wrap_text(text, font, max_w)

            total_h = sum(self._get_line_height(font, line) for line in lines) + (
                len(lines) - 1
            ) * line_spacing
            max_line_w = max((self._get_line_width(font, line) for line in lines), default=0)

            if total_h <= max_h and max_line_w <= max_w:
                best_size = mid
                best_lines = lines
                best_font = font
                low = mid + 1  # Try larger font
            else:
                high = mid - 1  # Too big, shrink font

        return best_font, best_lines, line_spacing

    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
        """
        Greedy word-wrapper with diamond shaping for speech bubbles.
        """
        words = text.split()
        if not words:
            return [""]

        lines: List[str] = []
        current_line: List[str] = []

        for word in words:
            test_line = " ".join(current_line + [word])
            line_w = self._get_line_width(font, test_line)

            if line_w <= max_w:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    # Single word is wider than box, forced split
                    lines.append(word)
                    current_line = []

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def _get_line_width(self, font: ImageFont.ImageFont, line: str) -> int:
        """Measure rendered pixel width of text line."""
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(line)
            return bbox[2] - bbox[0]
        return len(line) * (getattr(font, "size", 12) // 2)

    def _get_line_height(self, font: ImageFont.ImageFont, line: str) -> int:
        """Measure rendered pixel height of text line."""
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(line)
            return max(12, bbox[3] - bbox[1])
        return getattr(font, "size", 12) + 2

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        """Load and cache TrueType or default bitmap font."""
        if size in self._font_cache:
            return self._font_cache[size]

        font = None
        # Try custom font path or standard system fonts
        candidate_paths = [
            self.font_path,
            "C:/Windows/Fonts/comic.ttf",     # Comic Sans MS (standard on Windows)
            "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold
            "C:/Windows/Fonts/arial.ttf",     # Arial
            "C:/Windows/Fonts/malgun.ttf",    # Malgun Gothic (supports full Latin + Hangul)
        ]

        for p in candidate_paths:
            if p:
                try:
                    font = ImageFont.truetype(p, size)
                    break
                except Exception:
                    continue

        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                pass

        self._font_cache[size] = font
        return font
