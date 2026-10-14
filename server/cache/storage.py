"""
Content-Addressed Storage & Caching Layer.

Uses SHA-256 digests of image binaries combined with target language and glossary
hash to guarantee that identical panels are never processed twice.
Includes both an in-memory fast tier and persistent disk storage.
"""

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from server.config import settings

logger = logging.getLogger(__name__)


class PanelCache:
    """Two-tier cache (In-Memory LRU + Persistent File System)."""

    def __init__(self, cache_dir: Path | None = None, max_memory_items: int = 250):
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_items = max_memory_items
        self._memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    @staticmethod
    def compute_hash(image_bytes: bytes, target_lang: str, glossary: dict | None = None) -> str:
        """Compute SHA-256 key from image bytes, language, and glossary."""
        hasher = hashlib.sha256()
        hasher.update(image_bytes)
        hasher.update(target_lang.encode("utf-8"))
        if glossary:
            # Deterministic serialization of glossary keys and values
            glossary_str = json.dumps(glossary, sort_keys=True)
            hasher.update(glossary_str.encode("utf-8"))
        return hasher.hexdigest()

    def get(self, key: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """
        Retrieve cached translated image bytes and bubble metadata.
        Returns (image_bytes, metadata_dict) or None if cache miss.
        """
        if not settings.cache_enabled:
            return None

        # Check in-memory LRU
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            entry = self._memory_cache[key]
            logger.debug("Memory cache hit for panel key: %s", key[:12])
            return entry["image_bytes"], entry["metadata"]

        # Check disk cache
        img_file = self.cache_dir / f"{key}.img"
        meta_file = self.cache_dir / f"{key}.json"

        if img_file.exists() and meta_file.exists():
            try:
                img_bytes = img_file.read_bytes()
                metadata = json.loads(meta_file.read_text(encoding="utf-8"))

                # Populate memory tier
                self._put_memory(key, img_bytes, metadata)
                logger.debug("Disk cache hit for panel key: %s", key[:12])
                return img_bytes, metadata
            except Exception as exc:
                logger.warning("Failed to read disk cache for %s: %s", key, exc)
                return None

        return None

    def put(self, key: str, image_bytes: bytes, metadata: Dict[str, Any]) -> None:
        """Persist translated image bytes and bubble metadata."""
        if not settings.cache_enabled:
            return

        self._put_memory(key, image_bytes, metadata)

        # Write to disk asynchronously/safely
        try:
            img_file = self.cache_dir / f"{key}.img"
            meta_file = self.cache_dir / f"{key}.json"

            img_file.write_bytes(image_bytes)
            meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write to disk cache for %s: %s", key, exc)

    def _put_memory(self, key: str, image_bytes: bytes, metadata: Dict[str, Any]) -> None:
        """Helper to maintain in-memory LRU bounds."""
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        self._memory_cache[key] = {
            "image_bytes": image_bytes,
            "metadata": metadata,
        }
        if len(self._memory_cache) > self.max_memory_items:
            self._memory_cache.popitem(last=False)

    def count(self) -> int:
        """Count cached panels on disk."""
        try:
            return len(list(self.cache_dir.glob("*.json")))
        except Exception:
            return 0

    def clear(self) -> None:
        """Clear both memory and disk caches."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.*"):
            try:
                f.unlink()
            except Exception:
                pass


# Global cache instance
panel_cache = PanelCache()
