"""
Integration Tests for Webtoon Translator REST Endpoints.

Tests full end-to-end request/response flows using Flask test_client.
"""

import base64
from pathlib import Path
import pytest
from flask.testing import FlaskClient

from server.main import app
from tests.create_sample_panel import create_manhwa_strip


@pytest.fixture(scope="session")
def client() -> FlaskClient:
    """Session-scoped Flask test client."""
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(scope="session")
def sample_b64(tmp_path_factory) -> str:
    """Generate and encode a sample webtoon strip for API tests."""
    temp_dir = tmp_path_factory.mktemp("api_fixtures")
    img_path = temp_dir / "panel.jpg"
    create_manhwa_strip(img_path)
    raw_bytes = img_path.read_bytes()
    return f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"


def test_health_endpoint(client: FlaskClient):
    """GET /api/health should return ok status and system info."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "translation_provider" in data


def test_translate_panel_end_to_end(client: FlaskClient, sample_b64: str):
    """POST /api/translate should process panel, detect bubbles, and return translated image."""
    payload = {
        "image_base64": sample_b64,
        "chapter_id": "test_ch_1",
        "panel_index": 0,
        "target_lang": "en",
        "mode": "both",
        "glossary": {"Hyung": "Brother"},
    }

    resp = client.post("/api/translate", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["cached"] is False
    assert data["translated_image_base64"] is not None
    assert data["translated_image_base64"].startswith("data:image/jpeg;base64,")
    assert len(data["bubbles"]) >= 2
    assert data["processing_time_ms"] > 0

    # Test cache hit on immediate second call
    resp_cached = client.post("/api/translate", json=payload)
    assert resp_cached.status_code == 200
    cached_data = resp_cached.get_json()
    assert cached_data["cached"] is True
    assert len(cached_data["bubbles"]) == len(data["bubbles"])


def test_glossary_crud_lifecycle(client: FlaskClient):
    """Verify listing, creating, and deleting glossary entries."""
    series = "solo_leveling_test"

    # Add item
    item = {
        "series_id": series,
        "source_term": "성진우",
        "target_term": "Sung Jin-woo",
        "category": "character",
        "notes": "Main protagonist",
    }
    create_resp = client.post("/api/glossary", json=item)
    assert create_resp.status_code == 200
    assert create_resp.get_json()["target_term"] == "Sung Jin-woo"

    # List items for this series
    list_resp = client.get(f"/api/glossary?series_id={series}")
    assert list_resp.status_code == 200
    items = list_resp.get_json()
    assert any(i["source_term"] == "성진우" for i in items)

    # Delete item
    del_resp = client.delete(f"/api/glossary?series_id={series}&source_term=성진우")
    assert del_resp.status_code == 200
    assert del_resp.get_json()["deleted"] is True


def test_clear_cache_endpoint(client: FlaskClient):
    """POST /api/cache/clear should empty the panel cache."""
    resp = client.post("/api/cache/clear")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "cache cleared"
