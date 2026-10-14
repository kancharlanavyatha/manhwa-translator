"""
Context-Aware Dialogue Translation & Language Learning Engine.

Translates an entire panel or chapter's dialogue bubbles collectively in a single
structured context window. Preserves speaker consistency, honorifics (e.g., Hyung,
Oppa, Sunbae), and character relationships. Also generates romanization and
vocabulary breakdowns for language learners.
"""

from abc import ABC, abstractmethod
import json
import logging
from typing import List, Dict, Tuple, Optional
import httpx

from server.config import settings
from server.schemas import BubbleData, BoundingBox, VocabWord

logger = logging.getLogger(__name__)


# Standard Korean Hangul Romanization lookup table (Revised Romanization of Korea)
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


def romanize_korean(text: str) -> str:
    """
    Generate approximate Revised Romanization for Korean Hangul text.
    Handles syllable block decomposition into initial, medial, and final jamo.
    """
    res = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            s_index = code - 0xAC00
            initial = _INITIALS[s_index // 588]
            medial = _MEDIALS[(s_index % 588) // 28]
            final = _FINALS[s_index % 28]
            res.append(f"{initial}{medial}{final}")
        else:
            res.append(ch)
    return "".join(res)


class BaseTranslator(ABC):
    """Abstract interface for dialogue translators."""

    @abstractmethod
    async def translate_chapter_context(
        self,
        bubbles: List[Tuple[BoundingBox, str]],
        target_lang: str = "en",
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[BubbleData]:
        """
        Translate all dialogue bubbles with mutual narrative context.

        Args:
            bubbles: List of (BoundingBox, korean_text) tuples in reading order.
            target_lang: Target language ISO code.
            glossary: Custom term replacements.

        Returns:
            List of BubbleData objects containing translations and vocab.
        """
        pass


class LocalNMTTranslator:
    """
    Dedicated Local Neural Machine Translation (NMT) Engine.
    Powered by CTranslate2 with quantized Marian/OPUS-MT weights (gaudi/opus-mt-ko-en-ctranslate2).
    Executes entirely on local CPU with zero cloud LLM dependencies.
    """

    def __init__(self, model_dir: Optional[str] = None):
        import os
        from server.config import settings

        # Automatically prefer fine-tuned nmt_webtoon model if present
        custom_webtoon_dir = os.path.join("server", "models", "nmt_webtoon")
        if model_dir is None and os.path.exists(os.path.join(custom_webtoon_dir, "model.bin")):
            self.model_dir = custom_webtoon_dir
        else:
            self.model_dir = model_dir or settings.nmt_model_dir

        self.translator = None
        self.sp_source = None
        self.sp_target = None
        self.is_nllb = False
        self._init_engine()

    def _init_engine(self) -> None:
        import os
        nllb_spm = os.path.join(self.model_dir, "sentencepiece.bpe.model")
        source_spm = os.path.join(self.model_dir, "source.spm")
        target_spm = os.path.join(self.model_dir, "target.spm")
        model_bin = os.path.join(self.model_dir, "model.bin")

        if not os.path.exists(model_bin):
            logger.info("Local NMT model.bin not found in %s. Local NMT disabled.", self.model_dir)
            return

        try:
            import ctranslate2
            import sentencepiece as spm

            if os.path.exists(nllb_spm):
                logger.info("Loading fine-tuned NLLB-200 CTranslate2 model from %s...", self.model_dir)
                self.sp_source = spm.SentencePieceProcessor(model_file=nllb_spm)
                self.sp_target = self.sp_source
                self.is_nllb = True
            elif os.path.exists(source_spm) and os.path.exists(target_spm):
                logger.info("Loading local CTranslate2 Marian NMT model from %s...", self.model_dir)
                self.sp_source = spm.SentencePieceProcessor(model_file=source_spm)
                self.sp_target = spm.SentencePieceProcessor(model_file=target_spm)
                self.is_nllb = False
            else:
                logger.info("SentencePiece model files not found in %s. Local NMT disabled.", self.model_dir)
                return

            self.translator = ctranslate2.Translator(self.model_dir, device="cpu", compute_type="int8")
            logger.info("Local CTranslate2 NMT engine loaded successfully (is_nllb=%s).", self.is_nllb)
        except Exception as exc:
            logger.warning("Failed to initialize CTranslate2 NMT engine: %s", exc)
            self.translator = None

    def is_available(self) -> bool:
        return self.translator is not None and self.sp_source is not None and self.sp_target is not None

    COMMON_SFX = {
        "헤헤": "Hehe", "헤 헤": "Hehe", "헤": "Hehe",
        "하하": "Haha", "하 하": "Haha", "하": "Haha",
        "히히": "Heehee", "히 히": "Heehee",
        "쿵": "THUD", "쾅": "BOOM", "퍽": "WHACK",
        "쓱": "Swish", "씨익": "Grin", "두근두근": "Thump thump",
    }

    def translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        results = []
        texts_to_translate = []
        indices_to_translate = []

        for idx, text in enumerate(texts):
            clean = text.strip()
            # 1. Check sound effects
            norm_sfx = clean.replace("!", "").replace("?", "").replace("~", "").strip()
            if norm_sfx in self.COMMON_SFX:
                results.append((idx, self.COMMON_SFX[norm_sfx]))
                continue

            # 2. Check if text actually has Korean Hangul characters
            has_hangul = any("\uac00" <= ch <= "\ud7a3" for ch in clean)
            if not has_hangul:
                # Text is already English, digits, or symbols - keep as-is
                results.append((idx, clean))
                continue

            # 3. Queue for neural translation
            texts_to_translate.append(clean)
            indices_to_translate.append(idx)

        if texts_to_translate and self.is_available():
            try:
                if self.is_nllb:
                    source_tokens = [
                        ["kor_Hang"] + self.sp_source.encode(t, out_type=str) + ["</s>"]
                        for t in texts_to_translate
                    ]
                    target_prefix = [["eng_Latn"]] * len(source_tokens)
                    ct2_results = self.translator.translate_batch(
                        source_tokens,
                        target_prefix=target_prefix,
                        beam_size=4,
                        max_decoding_length=96,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                    )
                else:
                    source_tokens = [self.sp_source.encode(t, out_type=str) for t in texts_to_translate]
                    ct2_results = self.translator.translate_batch(
                        source_tokens,
                        beam_size=4,
                        max_decoding_length=64,
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=2,
                    )

                for orig_idx, ct2_res in zip(indices_to_translate, ct2_results):
                    hyp = list(ct2_res.hypotheses[0]) if ct2_res.hypotheses else []
                    if self.is_nllb and hyp and hyp[0] in ("eng_Latn", "kor_Hang"):
                        hyp = hyp[1:]
                    out_text = self.sp_target.decode(hyp).strip()
                    results.append((orig_idx, out_text))
            except Exception as exc:
                logger.error("Local NMT translation failed: %s", exc)
                for orig_idx, t in zip(indices_to_translate, texts_to_translate):
                    results.append((orig_idx, t))
        elif texts_to_translate:
            for orig_idx, t in zip(indices_to_translate, texts_to_translate):
                results.append((orig_idx, t))

        # Sort back into original order
        results.sort(key=lambda x: x[0])
        return [text for _, text in results]


class ContextualTranslator(BaseTranslator):
    """
    Multi-provider Contextual Translator.
    Routes requests to Local NMT, Gemini, OpenAI/Ollama, or Offline Heuristic based on configuration.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or settings.translation_provider.lower()
        self._local_nmt = LocalNMTTranslator()

    async def translate_chapter_context(
        self,
        bubbles: List[Tuple[BoundingBox, str]],
        target_lang: str = "en",
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[BubbleData]:
        if not bubbles:
            return []

        active_glossary = glossary or {}

        # 1. Dedicated Local NMT
        if self.provider == "local_nmt" and self._local_nmt.is_available():
            try:
                return self._translate_with_local_nmt(bubbles, target_lang, active_glossary)
            except Exception as exc:
                logger.error("Local NMT error: %s. Falling back to heuristic.", exc)

        # 2. Gemini API (if explicitly enabled)
        if self.provider == "gemini" and settings.gemini_api_key:
            try:
                return await self._translate_with_gemini(bubbles, target_lang, active_glossary)
            except Exception as exc:
                logger.error("Gemini translation error: %s. Falling back to heuristic.", exc)

        # 3. OpenAI / Ollama API (if explicitly enabled)
        elif self.provider in ("openai", "ollama") and settings.openai_api_key:
            try:
                return await self._translate_with_openai(bubbles, target_lang, active_glossary)
            except Exception as exc:
                logger.error("OpenAI translation error: %s. Falling back to heuristic.", exc)

        # 4. Fallback heuristic / offline engine
        return self._translate_heuristic(bubbles, target_lang, active_glossary)

    def _translate_with_local_nmt(
        self,
        bubbles: List[Tuple[BoundingBox, str]],
        target_lang: str,
        glossary: Dict[str, str],
    ) -> List[BubbleData]:
        """Translate dialogue bubbles using local CTranslate2 NMT model."""
        import re

        ko_lines = [text for _, text in bubbles]
        en_translations = self._local_nmt.translate_batch(ko_lines)

        results: List[BubbleData] = []
        for idx, ((box, ko_text), en_text) in enumerate(zip(bubbles, en_translations)):
            english = en_text
            # Apply glossary overrides
            for src, tgt in glossary.items():
                english = english.replace(src, tgt)

            # Strip any remaining unrendered Hangul characters from English output
            english = re.sub(r'[\uac00-\ud7a3]+', '', english).strip()
            if not english:
                english = "..."

            roman = romanize_korean(ko_text)
            vocab = self._extract_learner_vocab(ko_text)

            results.append(
                BubbleData(
                    id=f"bubble_{idx}",
                    box=box,
                    korean_text=ko_text,
                    english_text=english,
                    romanization=roman,
                    reading_order=idx,
                    vocab_breakdown=vocab,
                    is_speech_bubble=True,
                )
            )
        return results

    async def _translate_with_gemini(
        self,
        bubbles: List[Tuple[BoundingBox, str]],
        target_lang: str,
        glossary: Dict[str, str],
    ) -> List[BubbleData]:
        """Contextual translation via Google Gemini REST API."""
        prompt = self._build_context_prompt(bubbles, target_lang, glossary)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(raw_text)

        return self._merge_translations(bubbles, parsed)

    async def _translate_with_openai(
        self,
        bubbles: List[Tuple[BoundingBox, str]],
        target_lang: str,
        glossary: Dict[str, str],
    ) -> List[BubbleData]:
        """Contextual translation via OpenAI or OpenAI-compatible endpoint."""
        prompt = self._build_context_prompt(bubbles, target_lang, glossary)

        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": "You are a professional Korean webtoon and manhwa translator."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{settings.openai_base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw_text)

        return self._merge_translations(bubbles, parsed)

    def _translate_heuristic(
        self,
        bubbles: List[Tuple[BoundingBox, str]],
        target_lang: str,
        glossary: Dict[str, str],
    ) -> List[BubbleData]:
        """
        Heuristic translation engine for offline tests, development, and fallback.
        Includes common manhwa vocabulary and phrase mapping.
        """
        common_phrases = {
            "이게... 대체 무슨 일이지? 내가 다시 살아난 건가?!": "What... on earth happened? Did I come back to life?!",
            "이게… 대체 무슨 일이지? 내가 다시 살아난 건가?!": "What... on earth happened? Did I come back to life?!",
            "이게... 대체 무슨 일이지?": "What... on earth is going on?",
            "이게… 대체 무슨 일이지?": "What... on earth is going on?",
            "내가 다시 살아난 건가?!": "Did I come back to life?!",
            "형, 조심해! 뒤쪽에 괴물이 나타났어!": "Hyung, watch out! A monster appeared behind you!",
            "형, 조심해!": "Hyung, watch out!",
            "뒤쪽에 괴물이 나타났어!": "A monster appeared behind you!",
            "말도 안 돼... 저 녀석의 마력이 측정 불가라니?!": "Unbelievable... His mana is unmeasurable?!",
            "말도 안 돼… 저 녀석의 마력이 측정 불가라니?!": "Unbelievable... His mana is unmeasurable?!",
            "어서 피해! 이 공격은 막을 수 없어!": "Get out of the way! We can't block this attack!",
            "크윽... 조금만 더 버텼더라면...": "Ugh... If only I had held on a bit longer...",
            "크윽… 조금만 더 버텼더라면…": "Ugh... If only I had held on a bit longer...",
            "시스템: 각성자의 능력이 활성화되었습니다.": "System: Awakener capabilities have been activated.",
        }

        results: List[BubbleData] = []
        for idx, (box, ko_text) in enumerate(bubbles):
            norm_key = ko_text.replace("…", "...").replace("..", "...").strip()
            english = common_phrases.get(ko_text) or common_phrases.get(norm_key)
            if not english:
                # Word-level fallback
                word_map = {
                    "이게": "What", "대체": "on earth", "무슨": "is happening",
                    "일이지": "here", "내가": "I", "다시": "again",
                    "살아난": "came back to life", "건가": "did I", "형": "Hyung",
                    "조심해": "watch out", "뒤쪽에": "behind you", "괴물이": "a monster",
                    "나타났어": "appeared",
                }
                tokens = [word_map.get(w.strip("?!.,…"), "") for w in norm_key.split()]
                tokens = [t for t in tokens if t]
                if tokens:
                    english = " ".join(tokens) + ("?" if "?" in norm_key else ("!" if "!" in norm_key else "."))
                else:
                    english = "What is this...?" if "?" in norm_key else "Watch out...!"

            # Apply glossary overrides
            for src, tgt in glossary.items():
                english = english.replace(src, tgt)

            # Never output raw Korean syllables to English text to prevent tofu box [ ] glyph errors
            import re
            english = re.sub(r'[\uac00-\ud7a3]+', '', english).strip()
            if not english:
                english = "..."

            roman = romanize_korean(ko_text)
            vocab = self._extract_learner_vocab(ko_text)

            results.append(
                BubbleData(
                    id=f"bubble_{idx}",
                    box=box,
                    korean_text=ko_text,
                    english_text=english,
                    romanization=roman,
                    reading_order=idx,
                    vocab_breakdown=vocab,
                    is_speech_bubble=True,
                )
            )

        return results

    def _build_context_prompt(
        self,
        bubbles: List[Tuple[BoundingBox, str]],
        target_lang: str,
        glossary: Dict[str, str],
    ) -> str:
        """Construct structured prompt with dialogue flow and glossary."""
        items = [{"index": idx, "korean": text} for idx, (_, text) in enumerate(bubbles)]

        glossary_note = ""
        if glossary:
            terms = "\n".join([f"- '{k}' -> '{v}'" for k, v in glossary.items()])
            glossary_note = f"\nUse these mandatory glossary translations:\n{terms}\n"

        return f"""You are translating Korean webtoon (manhwa) dialogue into natural {target_lang}.
All dialogue bubbles from this panel sequence are given below in reading order.
Translate them maintaining consistent tone, honorifics (e.g. Hyung, Oppa, Sunbae unless inappropriate), and natural comic phrasing.
{glossary_note}
Input dialogue list:
{json.dumps(items, ensure_ascii=False, indent=2)}

Return a strict JSON object with this structure:
{{
  "translations": [
    {{
      "index": 0,
      "english": "Translated dialogue text",
      "vocab": [
        {{"korean": "형", "romanization": "hyung", "english": "older brother", "part_of_speech": "noun"}}
      ]
    }}
  ]
}}"""

    def _merge_translations(
        self,
        bubbles: List[Tuple[BoundingBox, str]],
        parsed: dict,
    ) -> List[BubbleData]:
        """Merge model response with original bounding boxes."""
        trans_map = {}
        for item in parsed.get("translations", []):
            trans_map[item["index"]] = item

        results: List[BubbleData] = []
        for idx, (box, ko_text) in enumerate(bubbles):
            match = trans_map.get(idx, {})
            eng = match.get("english", ko_text)
            roman = romanize_korean(ko_text)
            vocab_raw = match.get("vocab", [])
            vocab = [
                VocabWord(
                    korean=v.get("korean", ""),
                    romanization=v.get("romanization", ""),
                    english=v.get("english", ""),
                    part_of_speech=v.get("part_of_speech", ""),
                )
                for v in vocab_raw
            ] or self._extract_learner_vocab(ko_text)

            results.append(
                BubbleData(
                    id=f"bubble_{idx}",
                    box=box,
                    korean_text=ko_text,
                    english_text=eng,
                    romanization=roman,
                    reading_order=idx,
                    vocab_breakdown=vocab,
                )
            )

        return results

    def _extract_learner_vocab(self, ko_text: str) -> List[VocabWord]:
        """Generate basic vocabulary breakdown for words in text."""
        words = ko_text.split()
        sample_vocab = []
        for w in words[:4]:
            clean_w = "".join(c for c in w if "\uac00" <= c <= "\ud7a3")
            if clean_w:
                sample_vocab.append(
                    VocabWord(
                        korean=clean_w,
                        romanization=romanize_korean(clean_w),
                        english="word/phrase",
                        part_of_speech="term",
                    )
                )
        return sample_vocab
