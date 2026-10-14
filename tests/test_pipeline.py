"""
Comprehensive Test Suite for Webtoon Translator Pipeline.

Validates detection accuracy, reading-order sorting, OCR text cleaning,
contextual dialogue translation, inpainting, and typesetting.
"""

import cv2
import numpy as np
import pytest
from pathlib import Path

from server.pipeline.detector import HeuristicBubbleDetector
from server.pipeline.ocr import HangulOCR
from server.pipeline.translator import ContextualTranslator
from server.pipeline.inpainter import BubbleInpainter
from server.pipeline.typesetter import Typesetter
from server.pipeline.stitcher import WebtoonStripManager
from server.schemas import BoundingBox, BubbleData
from tests.create_sample_panel import create_manhwa_strip


@pytest.fixture(scope="session")
def sample_image_path(tmp_path_factory) -> Path:
    """Fixture providing a generated webtoon panel image."""
    img_dir = tmp_path_factory.mktemp("fixtures")
    img_path = img_dir / "test_panel.jpg"
    create_manhwa_strip(img_path)
    return img_path


@pytest.fixture
def sample_image(sample_image_path: Path) -> np.ndarray:
    """Load sample image as BGR numpy array."""
    img = cv2.imread(str(sample_image_path))
    assert img is not None, "Failed to load test panel image"
    return img


def test_detector_finds_speech_bubbles(sample_image: np.ndarray):
    """Verify that HeuristicBubbleDetector finds the speech bubbles on the test strip."""
    detector = HeuristicBubbleDetector(min_area=400)
    boxes = detector.detect(sample_image)

    # The test strip has 2 prominent speech bubbles
    assert len(boxes) >= 2, f"Expected at least 2 bubbles, found {len(boxes)}"

    # Check reading order: top bubble (panel 1) must come before bottom bubble (panel 2)
    top_bubble = boxes[0]
    bottom_bubble = boxes[1]
    assert top_bubble.center_y < bottom_bubble.center_y, "Bubbles should be sorted in reading order"

    # Verify coordinates are reasonable
    for b in boxes:
        assert b.w > 50
        assert b.h > 40
        assert 0.0 <= b.confidence <= 1.0


def test_ocr_text_extraction(sample_image: np.ndarray):
    """Verify HangulOCR cleans and extracts text from regions."""
    ocr = HangulOCR()
    test_boxes = [
        BoundingBox(x=120, y=100, w=360, h=140),
        BoundingBox(x=320, y=630, w=380, h=150),
    ]

    results = ocr.extract_text(sample_image, test_boxes)
    assert len(results) == 2

    # Verify Korean normalization
    raw_str = "안 녕 하 세 요 ... "
    cleaned = HangulOCR.clean_hangul(raw_str)
    assert cleaned == "안 녕 하 세 요 …"


@pytest.mark.asyncio
async def test_contextual_translator_with_glossary():
    """Verify translation preserves context, applies glossaries, and generates vocab."""
    translator = ContextualTranslator(provider="heuristic")

    test_input = [
        (BoundingBox(x=120, y=100, w=360, h=140), "이게... 대체 무슨 일이지? 내가 다시 살아난 건가?!"),
        (BoundingBox(x=320, y=630, w=380, h=150), "형, 조심해! 뒤쪽에 괴물이 나타났어!"),
    ]

    # Custom glossary mapping: 'Hyung' -> 'Brother'
    custom_glossary = {"Hyung": "Brother"}

    bubbles = await translator.translate_chapter_context(
        test_input, target_lang="en", glossary=custom_glossary
    )

    assert len(bubbles) == 2
    assert "come back to life" in bubbles[0].english_text.lower()
    # Glossary substitution check
    assert "brother" in bubbles[1].english_text.lower()

    # Verify romanization and language learner vocab
    for b in bubbles:
        assert len(b.romanization) > 0
        assert len(b.vocab_breakdown) > 0
        assert b.vocab_breakdown[0].korean != ""


def test_inpainter_erases_text(sample_image: np.ndarray):
    """Verify inpainter erases dark text inside bubbles without destroying background."""
    inpainter = BubbleInpainter()
    bubble_box = BoundingBox(x=120, y=100, w=360, h=140)

    # Sample a pixel known to be text inside the bubble before inpainting
    original_roi = sample_image[bubble_box.y : bubble_box.y2, bubble_box.x : bubble_box.x2]

    cleaned_image = inpainter.inpaint(sample_image, [bubble_box])
    cleaned_roi = cleaned_image[bubble_box.y : bubble_box.y2, bubble_box.x : bubble_box.x2]

    # The cleaned bubble should have lower variance (more uniform white fill) than original with text
    assert np.var(cleaned_roi) <= np.var(original_roi)
    # The bubble shape dimensions should be preserved
    assert cleaned_image.shape == sample_image.shape


def test_typesetter_renders_dialogue(sample_image: np.ndarray):
    """Verify typesetter fits text within bounds and renders onto the image."""
    typesetter = Typesetter()
    test_bubbles = [
        BubbleData(
            id="b1",
            box=BoundingBox(x=120, y=100, w=360, h=140),
            korean_text="이게... 대체 무슨 일이지?",
            english_text="What on earth happened? Did I come back to life?!",
            reading_order=0,
        ),
        BubbleData(
            id="b2",
            box=BoundingBox(x=320, y=630, w=380, h=150),
            korean_text="형, 조심해!",
            english_text="Brother, watch out! A monster appeared behind you!",
            reading_order=1,
        ),
    ]

    rendered = typesetter.typeset(sample_image, test_bubbles)
    assert rendered is not None
    assert rendered.shape == sample_image.shape
    # Ensure rendered image has changed from original (ink was drawn)
    diff = np.abs(rendered.astype(np.int32) - sample_image.astype(np.int32))
    assert np.sum(diff) > 0


def test_strip_manager_slicing_and_stitching():
    """Verify WebtoonStripManager slices continuous strips and handles overlaps."""
    manager = WebtoonStripManager(slice_height=700, overlap=100)
    mock_strip = np.zeros((1200, 800, 3), dtype=np.uint8)

    slices = manager.slice_strip(mock_strip)
    assert len(slices) == 2
    assert slices[0][1] == 0       # First slice offset
    assert slices[1][1] == 600     # Second slice offset (700 - 100)

    # Test de-duplication across overlapping boundary
    box1 = BoundingBox(x=100, y=620, w=200, h=100)
    box2 = BoundingBox(x=105, y=625, w=195, h=95)  # duplicate in slice overlap
    deduped = manager._deduplicate_seam_boxes([box1, box2])
    assert len(deduped) == 1
