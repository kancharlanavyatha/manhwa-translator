"""
Series Glossary Manager.

Stores, organizes, and applies customized translations for character names,
titles, skills, locations, and honorifics per manhwa series.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from server.schemas import GlossaryItem

logger = logging.getLogger(__name__)


class GlossaryManager:
    """Manages persistent glossaries keyed by series_id."""

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path("data/glossaries.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict[str, GlossaryItem]] = {}
        self._load()

    def _load(self) -> None:
        """Load stored glossaries from disk."""
        if not self.storage_path.exists():
            # Seed with common webtoon presets
            self._seed_default_presets()
            self._save()
            return

        try:
            content = self.storage_path.read_text(encoding="utf-8")
            raw_data = json.loads(content)
            for series_id, items in raw_data.items():
                self._data[series_id] = {}
                for source_term, item_dict in items.items():
                    self._data[series_id][source_term] = GlossaryItem(**item_dict)
        except Exception as exc:
            logger.error("Failed to load glossary file: %s", exc)
            self._data = {}

    def _save(self) -> None:
        """Persist glossaries to disk."""
        try:
            serializable: Dict[str, Dict[str, dict]] = {}
            for series_id, items in self._data.items():
                serializable[series_id] = {k: v.to_dict() for k, v in items.items()}
            self.storage_path.write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to save glossary file: %s", exc)

    def _seed_default_presets(self) -> None:
        """Seed presets for popular manhwa tropes & honorifics."""
        self._data["default"] = {
            "형": GlossaryItem(
                series_id="default",
                source_term="형",
                target_term="Hyung",
                category="honorific",
                notes="Older brother / respected male elder",
            ),
            "오빠": GlossaryItem(
                series_id="default",
                source_term="오빠",
                target_term="Oppa",
                category="honorific",
                notes="Female to older male",
            ),
            "누나": GlossaryItem(
                series_id="default",
                source_term="누나",
                target_term="Noona",
                category="honorific",
                notes="Male to older female",
            ),
            "선배": GlossaryItem(
                series_id="default",
                source_term="선배",
                target_term="Sunbae",
                category="honorific",
                notes="Senior colleague/student",
            ),
        }

    def get_series_glossary(self, series_id: str = "default") -> Dict[str, str]:
        """Return a simple dictionary mapping {source_korean: target_english}."""
        result: Dict[str, str] = {}
        # Apply defaults first
        if "default" in self._data:
            for k, item in self._data["default"].items():
                result[k] = item.target_term
        # Override with series-specific items
        if series_id in self._data and series_id != "default":
            for k, item in self._data[series_id].items():
                result[k] = item.target_term
        return result

    def list_items(self, series_id: str) -> List[GlossaryItem]:
        """List all full glossary items for a series."""
        if series_id not in self._data:
            return []
        return list(self._data[series_id].values())

    def upsert_item(self, item: GlossaryItem) -> None:
        """Add or update a term in a series glossary."""
        if item.series_id not in self._data:
            self._data[item.series_id] = {}
        self._data[item.series_id][item.source_term] = item
        self._save()

    def delete_item(self, series_id: str, source_term: str) -> bool:
        """Remove a term from a series glossary."""
        if series_id in self._data and source_term in self._data[series_id]:
            del self._data[series_id][source_term]
            self._save()
            return True
        return False

    def apply_glossary_to_text(self, text: str, glossary: Dict[str, str]) -> str:
        """Apply glossary substitutions to translated English text."""
        result = text
        for source_korean, target_english in glossary.items():
            # If the translated text mistakenly retained the Korean name or needs exact word replacement
            result = result.replace(source_korean, target_english)
        return result


# Global glossary manager singleton
glossary_manager = GlossaryManager()
