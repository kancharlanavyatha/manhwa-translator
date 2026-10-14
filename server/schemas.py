"""
Data Schemas for Webtoon Translator.

Strict, type-checked dataclass models representing bounding boxes,
speech bubbles, translation requests/responses, and glossaries.
Compatible with standard library serialization and immune to native DLL restrictions.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class BoundingBox:
    """Pixel bounding box for speech bubbles and text regions."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    @property
    def x2(self) -> int:
        """Right boundary X coordinate."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Bottom boundary Y coordinate."""
        return self.y + self.h

    @property
    def center_x(self) -> float:
        """Horizontal centroid."""
        return self.x + self.w / 2.0

    @property
    def center_y(self) -> float:
        """Vertical centroid."""
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        """Bounding box area."""
        return self.w * self.h

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "confidence": round(self.confidence, 2),
            "x2": self.x2,
            "y2": self.y2,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "area": self.area,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class VocabWord:
    """Vocabulary breakdown item for Korean language learners."""

    korean: str
    romanization: str
    english: str
    part_of_speech: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BubbleData:
    """Structured data for a detected speech bubble or text region."""

    id: str
    box: BoundingBox
    korean_text: str
    english_text: str
    reading_order: int
    romanization: str = ""
    vocab_breakdown: List[VocabWord] = field(default_factory=list)
    is_speech_bubble: bool = True
    font_size_rendered: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "box": self.box.to_dict(),
            "korean_text": self.korean_text,
            "english_text": self.english_text,
            "reading_order": self.reading_order,
            "romanization": self.romanization,
            "vocab_breakdown": [v.to_dict() for v in self.vocab_breakdown],
            "is_speech_bubble": self.is_speech_bubble,
            "font_size_rendered": self.font_size_rendered,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BubbleData":
        box_data = data["box"]
        box = BoundingBox.from_dict(box_data) if isinstance(box_data, dict) else box_data
        vocab = [
            VocabWord(**v) if isinstance(v, dict) else v
            for v in data.get("vocab_breakdown", [])
        ]
        return cls(
            id=data["id"],
            box=box,
            korean_text=data["korean_text"],
            english_text=data["english_text"],
            reading_order=int(data["reading_order"]),
            romanization=data.get("romanization", ""),
            vocab_breakdown=vocab,
            is_speech_bubble=bool(data.get("is_speech_bubble", True)),
            font_size_rendered=data.get("font_size_rendered"),
        )


@dataclass
class TranslationRequest:
    """Request payload sent from browser extension or client."""

    image_base64: str
    chapter_id: str = "default_chapter"
    panel_index: int = 0
    target_lang: str = "en"
    mode: str = "both"
    glossary: Dict[str, str] = field(default_factory=dict)


@dataclass
class TranslationResponse:
    """Response returned to browser extension."""

    translated_image_base64: Optional[str]
    bubbles: List[BubbleData]
    cached: bool
    processing_time_ms: float
    panel_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "translated_image_base64": self.translated_image_base64,
            "bubbles": [b.to_dict() for b in self.bubbles],
            "cached": self.cached,
            "processing_time_ms": self.processing_time_ms,
            "panel_index": self.panel_index,
        }


@dataclass
class GlossaryItem:
    """Custom term mapping for a webtoon series."""

    series_id: str
    source_term: str
    target_term: str
    category: str = "character"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlossaryItem":
        return cls(**data)


@dataclass
class HealthResponse:
    """Service health and diagnostic status."""

    status: str = "ok"
    version: str = "1.0.0"
    translation_provider: str = "heuristic"
    cache_items: int = 0
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
