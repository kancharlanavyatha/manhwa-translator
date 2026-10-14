"""
Main Web Server for Webtoon Translator.

Provides REST API endpoints for real-time comic image translation,
speech bubble detection, glossary management, and health checks.
Built with Flask and Flask-CORS for zero native DLL dependencies and rock-solid cross-platform stability.
"""

import asyncio
import base64
import logging
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

from server.config import settings
from server.cache.storage import panel_cache
from server.schemas import (
    TranslationRequest,
    TranslationResponse,
    HealthResponse,
    GlossaryItem,
    BubbleData,
)
from server.pipeline.detector import HeuristicBubbleDetector
from server.pipeline.ocr import HangulOCR
from server.pipeline.translator import ContextualTranslator
from server.pipeline.inpainter import BubbleInpainter
from server.pipeline.typesetter import Typesetter
from server.services.glossary import glossary_manager

# Configure structured logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webtoon_translator")

# Initialize Flask application
app = Flask(__name__)
CORS(app, origins=settings.cors_origins)

# Initialize pipeline components
detector = HeuristicBubbleDetector()
ocr = HangulOCR()
translator = ContextualTranslator()
inpainter = BubbleInpainter()
typesetter = Typesetter(
    padding_ratio=settings.bubble_padding_ratio,
    min_font_size=settings.min_font_size,
    max_font_size=settings.max_font_size,
)


def decode_image_base64(data_str: str) -> Tuple[np.ndarray, bytes]:
    """Decode a base64 string (with or without data URI prefix) into a BGR numpy array."""
    if not data_str:
        raise ValueError("Empty image data provided.")

    if "," in data_str:
        # Strip data:image/...;base64, prefix
        data_str = data_str.split(",", 1)[1]

    raw_bytes = base64.b64decode(data_str)
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image from base64 string.")

    return image, raw_bytes


def fetch_image_from_url(url: str) -> Tuple[np.ndarray, bytes]:
    """Fetch image bytes directly with appropriate Referer header and decode to BGR numpy array."""
    import urllib.request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    if "pstatic.net" in url or "naver.com" in url:
        headers["Referer"] = "https://comic.naver.com/"
    elif "kakao.com" in url:
        headers["Referer"] = "https://webtoon.kakao.com/"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=12) as resp:
        raw_bytes = resp.read()

    np_arr = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image downloaded from {url}")

    return image, raw_bytes


def encode_image_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """Encode an OpenCV BGR image into a base64 JPEG/PNG string."""
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == ".jpg" else []
    success, buffer = cv2.imencode(ext, image, params)
    if not success:
        raise RuntimeError("Failed to encode image to output format.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64_str}"


@app.route("/api/health", methods=["GET"])
def health_check():
    """Service health and diagnostic status."""
    resp = HealthResponse(
        status="ok",
        version="1.0.0",
        translation_provider=settings.translation_provider,
        cache_items=panel_cache.count(),
        device="cpu",
    )
    return jsonify(resp.to_dict())


@app.route("/api/translate", methods=["POST"])
def translate_panel():
    """
    Full end-to-end translation pipeline:
    1. Decode base64 image
    2. Check SHA-256 panel cache
    3. Detect bubbles -> OCR Hangul -> Contextual Translation -> Inpaint -> Typeset
    4. Cache and return rendered image & bubble metadata
    """
    start_time = time.perf_counter()
    data = request.get_json(force=True, silent=True)

    if not data or ("image_base64" not in data and "image_url" not in data):
        return jsonify({"error": "Missing image_base64 or image_url in request payload"}), 400

    image_b64 = data.get("image_base64")
    image_url = data.get("image_url")
    chapter_id = data.get("chapter_id", "default_chapter")
    panel_index = int(data.get("panel_index", 0))
    target_lang = data.get("target_lang", "en")
    mode = data.get("mode", "both")
    custom_glossary = data.get("glossary", {})

    try:
        if image_b64:
            image, raw_bytes = decode_image_base64(image_b64)
        elif image_url:
            image, raw_bytes = fetch_image_from_url(image_url)
        else:
            raise ValueError("No image data provided.")
    except Exception as exc:
        logger.warning("Invalid image upload: %s", exc)
        return jsonify({"error": f"Invalid image format: {exc}"}), 400

    # Combine custom glossary with series-stored glossary
    series_glossary = glossary_manager.get_series_glossary(chapter_id)
    if custom_glossary:
        series_glossary.update(custom_glossary)

    # Check content-addressed cache
    cache_key = panel_cache.compute_hash(raw_bytes, target_lang, series_glossary)
    cached_entry = panel_cache.get(cache_key)

    if cached_entry is not None:
        cached_img_bytes, metadata = cached_entry
        cached_b64 = f"data:image/jpeg;base64,{base64.b64encode(cached_img_bytes).decode('utf-8')}"
        bubbles = [BubbleData.from_dict(b) for b in metadata.get("bubbles", [])]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        resp = TranslationResponse(
            translated_image_base64=cached_b64 if mode in ("both", "image") else None,
            bubbles=bubbles,
            cached=True,
            processing_time_ms=round(elapsed_ms, 2),
            panel_index=panel_index,
        )
        return jsonify(resp.to_dict())

    api_key = data.get("api_key") or settings.gemini_api_key

    # 1. Detect candidate speech bubbles
    boxes = detector.detect(image)
    logger.info("Panel %d: detected %d candidate text regions", panel_index, len(boxes))

    # 2. Extract Korean text (processes candidate bubbles + full-image neural scan for borderless dialogue)
    ocr_results = ocr.extract_text(image, boxes, api_key=api_key)
    logger.info("Panel %d: OCR extracted text from %d regions", panel_index, len(ocr_results))

    if not ocr_results:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        output_b64 = encode_image_base64(image) if mode in ("both", "image") else None
        resp = TranslationResponse(
            translated_image_base64=output_b64,
            bubbles=[],
            cached=False,
            processing_time_ms=round(elapsed_ms, 2),
            panel_index=panel_index,
        )
        return jsonify(resp.to_dict())

    # 3. Translate dialogue with narrative context
    bubbles_data = asyncio.run(
        translator.translate_chapter_context(
            bubbles=ocr_results,
            target_lang=target_lang,
            glossary=series_glossary,
        )
    )

    # 4. Inpaint original Korean text
    cleaned_image = inpainter.inpaint(image, [b.box for b in bubbles_data])

    # 5. Typeset English dialogue inside speech balloons
    rendered_image = typesetter.typeset(cleaned_image, bubbles_data)

    # Encode result
    output_b64 = encode_image_base64(rendered_image)

    # Save to content cache
    _, encoded_bytes = cv2.imencode(".jpg", rendered_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    panel_cache.put(
        key=cache_key,
        image_bytes=encoded_bytes.tobytes(),
        metadata={"bubbles": [b.to_dict() for b in bubbles_data]},
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info("Panel %d processed in %.2f ms", panel_index, elapsed_ms)

    resp = TranslationResponse(
        translated_image_base64=output_b64 if mode in ("both", "image") else None,
        bubbles=bubbles_data,
        cached=False,
        processing_time_ms=round(elapsed_ms, 2),
        panel_index=panel_index,
    )
    return jsonify(resp.to_dict())


@app.route("/api/glossary", methods=["GET"])
def list_glossary():
    """Retrieve all glossary terms for a series."""
    series_id = request.args.get("series_id", "default")
    items = glossary_manager.list_items(series_id)
    return jsonify([i.to_dict() for i in items])


@app.route("/api/glossary", methods=["POST"])
def add_glossary_item():
    """Add or update a glossary translation."""
    data = request.get_json(force=True)
    item = GlossaryItem.from_dict(data)
    glossary_manager.upsert_item(item)
    return jsonify(item.to_dict())


@app.route("/api/glossary", methods=["DELETE"])
def delete_glossary_item():
    """Delete a term from a glossary."""
    series_id = request.args.get("series_id")
    source_term = request.args.get("source_term")

    if not series_id or not source_term:
        return jsonify({"error": "Missing series_id or source_term"}), 400

    success = glossary_manager.delete_item(series_id, source_term)
    if not success:
        return jsonify({"error": "Term not found in series glossary"}), 404

    return jsonify({"deleted": True})


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Flush the panel translation cache."""
    panel_cache.clear()
    return jsonify({"status": "cache cleared"})


@app.route("/demo", methods=["GET"])
@app.route("/demo/<path:filename>", methods=["GET"])
def serve_demo(filename="index.html"):
    """Serve the interactive webtoon demo viewer directly over HTTP."""
    from pathlib import Path
    demo_dir = Path(__file__).resolve().parent.parent / "demo"
    return send_from_directory(demo_dir, filename)


if __name__ == "__main__":
    logger.info("Starting Webtoon Translator server on %s:%d...", settings.host, settings.port)
    app.run(host=settings.host, port=settings.port, debug=False)
