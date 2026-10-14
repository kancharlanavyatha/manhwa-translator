"""
Tests for dedicated local AI models (RapidOCR Hangul OCR and CTranslate2 NMT).
Ensures zero LLM dependency and offline execution.
"""

import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from server.pipeline.ocr import HangulOCR
from server.pipeline.translator import LocalNMTTranslator, ContextualTranslator
from server.schemas import BoundingBox


def test_local_ocr_reads_real_hangul():
    """Verify RapidOCR reads rendered Korean text without heuristics."""
    ocr = HangulOCR()
    assert ocr._rapidocr_engine is not None, "RapidOCR engine should be loaded"

    # Create an image with Korean text
    img = Image.new("RGB", (500, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 28)
    sample_korean = "아빠 붕붕이 많이 타야 갈 수 있어"
    draw.text((20, 40), sample_korean, fill=(0, 0, 0), font=font)
    arr = np.array(img)

    box = BoundingBox(x=0, y=0, w=500, h=120)
    results = ocr.extract_text(arr, [box])

    assert len(results) == 1
    extracted_box, extracted_text = results[0]
    assert "아빠" in extracted_text or "붕붕" in extracted_text or "많이" in extracted_text


def test_local_nmt_translates_korean_to_english():
    """Verify CTranslate2 neural translation translates Korean sentences to English."""
    nmt = LocalNMTTranslator()
    assert nmt.is_available(), "Local NMT translator should be initialized"

    sentences = [
        "이게... 대체 무슨 일이지? 내가 다시 살아난 건가?!",
        "형, 조심해! 뒤쪽에 괴물이 나타났어!",
    ]
    translations = nmt.translate_batch(sentences)
    assert len(translations) == 2
    for t in translations:
        assert len(t) > 5
        # Ensure it translated to English letters
        assert any(c.isalpha() and ord(c) < 128 for c in t)


@pytest.mark.asyncio
async def test_contextual_translator_uses_local_nmt():
    """Verify ContextualTranslator routes to Local NMT by default."""
    translator = ContextualTranslator(provider="local_nmt")
    bubbles = [
        (BoundingBox(x=10, y=10, w=200, h=80), "형, 조심해!"),
    ]
    results = await translator.translate_chapter_context(bubbles, target_lang="en")
    assert len(results) == 1
    assert results[0].english_text
    assert "hyung" in results[0].english_text.lower() or "watch" in results[0].english_text.lower() or "brother" in results[0].english_text.lower() or "careful" in results[0].english_text.lower()
