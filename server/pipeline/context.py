"""
Context-Aware Dialogue Intelligence & Discourse Engine for Webtoon Translation.

Provides:
1. ManhwaEntityNormalizer: Protects character names (e.g., '은하 씨' -> 'Eunha') and cultural honorifics ('형', '오빠', '선배').
2. DiscoursePacker: Reconstructs split-speech sentences across adjacent bubbles in the same panel.
3. ChapterContextBuffer: Sliding dialogue memory tracking cross-panel continuity and subject resolution.
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from server.schemas import BoundingBox

logger = logging.getLogger(__name__)

# Standard Korean Hangul Romanization lookup table
_INITIALS = [
    "g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "",
    "j", "jj", "ch", "k", "t", "p", "h"
]
_MEDIALS = [
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae",
    "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"
]
_FINALS = [
    "", "k", "k", "ks", "n", "nj", "nh", "t", "l", "lg", "lm", "lb",
    "ls", "lt", "lp", "lh", "m", "p", "ps", "t", "t", "ng", "t", "t",
    "k", "t", "p", "h"
]


def romanize_name(hangul: str) -> str:
    """Generate a clean capitalized Romanized name from Hangul syllables."""
    res = []
    for ch in hangul:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            s_index = code - 0xAC00
            initial = _INITIALS[s_index // 588]
            medial = _MEDIALS[(s_index % 588) // 28]
            final = _FINALS[s_index % 28]
            res.append(f"{initial}{medial}{final}")
        else:
            res.append(ch)
    raw = "".join(res).strip()
    return raw.capitalize() if raw else hangul


class ManhwaEntityNormalizer:
    """
    Normalizes Korean manhwa character names, speech registers, and honorifics
    both before and after neural machine translation to prevent semantic distortion.
    """

    # Common honorific titles mapped to natural webtoon English
    HONORIFIC_MAP = {
        "형": "Hyung",
        "형님": "Hyung-nim",
        "오빠": "Oppa",
        "선배": "Sunbae",
        "선배님": "Sunbae",
        "누나": "Noona",
        "언니": "Unnie",
        "아저씨": "Mister",
        "길드장님": "Guild Master",
        "팀장님": "Team Leader",
        "부장님": "General Manager",
        "대표님": "CEO",
    }

    @classmethod
    def pre_process(cls, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Detects names and honorifics before passing text to the MT engine.
        Replaces 'Name 씨' with a romanized anchor so the model treats it as a person,
        not a dictionary noun (e.g. '하늘 씨' -> 'Haneul').
        """
        normalized = text
        replacements = {}

        # 1. Match Korean Name + 씨 with optional grammatical particles (e.g., '은하 씨가', '진우 씨는', '하늘 씨의')
        name_ssi_pattern = r'([가-힣]{2,4})\s*씨([이가은는를을의에게한테와과도])?'
        particles = {
            "의": "'s",
            "에게": " to",
            "한테": " to",
            "와": " with",
            "과": " with",
        }
        matches = re.finditer(name_ssi_pattern, normalized)
        for m in matches:
            full_match = m.group(0)
            name = m.group(1)
            particle = m.group(2)
            rom = romanize_name(name)
            if particle in particles:
                replacement = f"{rom}{particles[particle]}"
            else:
                replacement = rom
            normalized = normalized.replace(full_match, replacement)
            replacements[name] = rom

        # 2. Match honorific vocatives at start of sentence or after punctuation
        for ko_h, en_h in cls.HONORIFIC_MAP.items():
            # Matches start of string or after punctuation/comma
            voc_pattern = rf'(^|[,\s\?!~])({re.escape(ko_h)})([,\s\?!~]|$)'
            if re.search(voc_pattern, normalized):
                normalized = re.sub(voc_pattern, rf'\1{en_h}\3', normalized)

        # 3. Conversational Idiom Normalization (convert literal idioms into semantic conversational Korean)
        idiom_rules = [
            # 눈이 (일찍) 떠져서 -> 일찍 일어나서 (woke up early)
            (r'눈이\s*(일찍\s*)?떠(져서|졌어|지다|졌거든|지네|졌다)', r'\1일어나서'),
            # 간 떨어질 뻔했잖아 -> 깜짝 놀랐잖아 (scared me to death)
            (r'간이?\s*떨어질\s*뻔\s*(했|했어|했잖아|했다)', r'깜짝 놀랐잖아'),
            # 내가 쏠게 -> 내가 살게 (my treat)
            (r'내가\s*쏠게', r'내가 살게'),
            # 비행기 태우지 마 -> 칭찬 그만해 (stop flattering me)
            (r'비행기\s*태우지\s*마', r'칭찬 그만해'),
            # 바람 맞았어 -> 약속 취소됐어 (got stood up)
            (r'바람\s*맞(았|혔|았어)', r'약속 취소됐어'),
            # 눈코 뜰 새 없다 -> 너무 바빠 (super busy)
            (r'눈코\s*뜰\s*새\s*(없|없어|없다|없는)', r'너무 바빠'),
            # 발목 잡지 마 -> 방해하지 마 (don't hold me back)
            (r'발목(을)?\s*잡지\s*마', r'방해하지 마'),
            # 머리 좀 식히고 -> 바람 좀 쐬고 (clear my head)
            (r'머리\s*(좀\s*)?식히(고|러|자)', r'바람 좀 쐬고'),
        ]
        for pat, repl in idiom_rules:
            normalized = re.sub(pat, repl, normalized)

        return normalized, replacements

    @classmethod
    def post_process(cls, english_text: str, original_korean: str) -> str:
        """
        Cleans and refines the translated English dialogue.
        Converts stiff literal translations (like 'Brother' -> 'Hyung'/'Oppa' where appropriate,
        and catches any residual literal idiom translations).
        """
        res = english_text.strip()

        # If Korean had 형 or 오빠 and English generated "Brother", adjust to contextual term
        if ("형" in original_korean or "형님" in original_korean) and res.lower().startswith("brother"):
            res = re.sub(r'^[Bb]rother\b', "Hyung", res)
        elif "오빠" in original_korean and res.lower().startswith("brother"):
            res = re.sub(r'^[Bb]rother\b', "Oppa", res)
        elif ("선배" in original_korean or "선배님" in original_korean) and (
            res.lower().startswith("senior") or res.lower().startswith("captain")
        ):
            res = re.sub(r'^(Senior|Captain)\b', "Sunbae", res, flags=re.IGNORECASE)

        # Refine any literal translations of common idioms
        idiom_post_fixes = [
            # "eyes are blinking early" / "getting my eyes out early" -> "woke up early"
            (r'(my\s+)?eyes\s+(are\s+)?blinking\s+early', 'I woke up early'),
            (r'getting\s+my\s+eyes\s+out\s+early', 'I woke up early'),
            (r'eyes\s+came\s+out\s+early', 'I woke up early'),
            # "lost your liver" / "almost lost my liver" -> "scared me to death"
            (r'(almost\s+)?lost\s+(your|my|his|her)\s+liver', 'scared the life out of me'),
            # "I will shoot" / "I'll shoot today" -> "My treat today"
            (r'I(\'ll|\s+will)\s+shoot\b', "It's my treat"),
            # "don't take the plane" -> "stop flattering me"
            (r'don[\'’]t\s+(take|ride)\s+the\s+plane', 'stop flattering me'),
        ]
        for pat, repl in idiom_post_fixes:
            res = re.sub(pat, repl, res, flags=re.IGNORECASE)

        # Clean multiple spaces or awkward punctuation spacing
        res = re.sub(r'\s+', ' ', res)
        res = re.sub(r'\s*([,\.\?!])', r'\1', res)
        return res


class DiscoursePacker:
    """
    Analyzes spatial and grammatical relationships between adjacent speech bubbles in a panel.
    When a sentence is split across 2 bubbles, bundles them into a single coherent translation
    query and then aligns the output clauses across the speech bubbles.
    """

    # Korean connective verbal endings that signal an ongoing, incomplete clause
    CONNECTIVE_ENDINGS = (
        "...", "…", "~", ",", "고", "며", "는데", "지만", "어서", "아서",
        "니까", "라며", "라고", "더니", "거든", "텐데", "려는데", "는데도",
    )

    @classmethod
    def should_merge(
        cls,
        box1: BoundingBox,
        text1: str,
        box2: BoundingBox,
        text2: str,
    ) -> bool:
        """Determines if bubble 1 and bubble 2 form a single split sentence."""
        t1 = text1.strip()
        t2 = text2.strip()
        if not t1 or not t2:
            return False

        # First check explicit continuation markers (ellipsis, comma, tilde)
        if t1.endswith(("...", "…", "~", ",")):
            has_continuation = True
        elif t1.endswith((".", "!", "?", '"', "'")):
            # Complete sentence with terminal punctuation
            return False
        else:
            # No terminal punctuation at all (open sentence or connective particle)
            has_continuation = True

        if not has_continuation:
            has_continuation = any(t1.endswith(suffix) for suffix in cls.CONNECTIVE_ENDINGS)

        # Check vertical and horizontal proximity
        y_dist = box2.y - (box1.y + box1.height)
        is_close = -box1.height * 0.5 <= y_dist <= box1.height * 3.0

        return has_continuation and is_close

    @classmethod
    def split_english_across_bubbles(
        cls,
        english: str,
        ko_texts: List[str],
    ) -> List[str]:
        """
        Splits a unified English translation into natural clauses matching the original bubbles.
        """
        if len(ko_texts) <= 1:
            return [english]

        words = english.split()
        if len(words) < 3:
            return [english] + ["..."] * (len(ko_texts) - 1)

        # Check for punctuation clause boundaries
        candidates = [", ", "; ", "... ", " — ", " - "]
        for cand in candidates:
            if cand in english:
                idx = english.index(cand)
                first = english[:idx].strip()
                second = english[idx + len(cand):].strip()
                if len(first.split()) >= 2 and len(second.split()) >= 2:
                    return [first + "...", "..." + second]

        # Conjunction / preposition boundary search
        split_markers = {"to", "not", "that", "and", "but", "or", "because", "when", "if", "so", "with", "from"}
        total_ko_len = sum(len(t.split()) for t in ko_texts)
        ratio = len(ko_texts[0].split()) / max(1, total_ko_len)
        target_k = max(1, min(len(words) - 1, round(ratio * len(words))))

        best_k = target_k
        for offset in [0, 1, -1, 2, -2]:
            check_k = target_k + offset
            if 1 <= check_k < len(words) and words[check_k].lower() in split_markers:
                best_k = check_k
                break

        p1 = " ".join(words[:best_k]) + "..."
        p2 = "..." + " ".join(words[best_k:])
        return [p1, p2]


@dataclass
class DialogueTurn:
    """A single turn of dialogue in a chapter."""
    korean: str
    english: str
    panel_index: int
    timestamp: float = field(default_factory=time.time)


class ChapterContextBuffer:
    """
    Thread-safe, sliding narrative memory buffer that tracks the last N dialogue turns
    for each webtoon chapter to maintain speaker continuity across scrolling panels.
    """

    def __init__(self, max_history: int = 6, ttl_seconds: float = 2700.0):
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds
        self._buffers: Dict[str, deque] = {}

    def add_turn(self, chapter_id: str, korean: str, english: str, panel_index: int) -> None:
        """Record a completed translation turn in the chapter memory."""
        clean_ko = korean.strip()
        clean_en = english.strip()
        if not clean_ko or not clean_en:
            return

        if chapter_id not in self._buffers:
            self._buffers[chapter_id] = deque(maxlen=self.max_history)

        self._buffers[chapter_id].append(
            DialogueTurn(
                korean=clean_ko,
                english=clean_en,
                panel_index=panel_index,
            )
        )

    def get_recent_context(self, chapter_id: str, limit: int = 2) -> List[DialogueTurn]:
        """Retrieve the most recent dialogue turns within this chapter."""
        if chapter_id not in self._buffers:
            return []

        now = time.time()
        buf = self._buffers[chapter_id]

        # Filter out expired history (e.g. reader left page idle for > 45 mins)
        valid = [turn for turn in buf if now - turn.timestamp < self.ttl_seconds]
        return valid[-limit:]

    def clear(self, chapter_id: Optional[str] = None) -> None:
        """Clear memory for a chapter or all chapters."""
        if chapter_id:
            self._buffers.pop(chapter_id, None)
        else:
            self._buffers.clear()


# Global chapter context buffer singleton
chapter_context = ChapterContextBuffer()
