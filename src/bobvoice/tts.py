from __future__ import annotations

import asyncio
import io
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_VOICES_DIR = Path.home() / ".openclaw" / "bobvoice-voices"

_VOICE_PROFILES: dict[str, dict[str, Any]] = {
    "en": {
        "instruct": "male, australian accent, young adult",
        "ref_text": (
            "G'day! I'm Bob, your friendly voice assistant. "
            "I'm here to help you learn and practice all sorts of new things."
        ),
        "speed": 1.0,
    },
    "fr": {
        "instruct": "female, young adult",
        "ref_text": (
            "Bonjour! Je suis votre professeur de français. "
            "Je suis ravie de vous aider à apprendre cette belle langue."
        ),
        "speed": 1.0,
    },
    "pt": {
        "instruct": "female, portuguese accent, young adult",
        "ref_text": (
            "Olá! Eu sou sua professora de português brasileiro. "
            "É um prazer ajudar você a aprender esta língua maravilhosa."
        ),
        "speed": 1.0,
    },
}

_DEFAULT_VOICE = "en"

_ABBREVIATIONS = frozenset({"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie", "approx"})


def _split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.strip().split())
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    result: list[str] = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        if result and (s[0].islower() or _is_abbreviation(result[-1])):
            result[-1] = result[-1] + " " + s
        else:
            result.append(s)
    return result if result else [cleaned]


def _is_abbreviation(text: str) -> bool:
    word = text.rstrip().split()[-1].rstrip(".!?").lower()
    return word in _ABBREVIATIONS


def samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def generate_silence_wav(duration: float = 0.3, sample_rate: int = 16000) -> bytes:
    num_samples = int(sample_rate * duration)
    samples = np.zeros(num_samples, dtype=np.float32)
    return _samples_to_wav(samples, sample_rate)


def generate_tone_wav(
    frequency: float = 440.0,
    duration: float = 0.3,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> bytes:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return _samples_to_wav(samples, sample_rate)


def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


class TTSEngine:
    def __init__(self, num_steps: int = 32) -> None:
        self._num_steps = num_steps
        self._model: Any = None
        self._voice_prompts: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from omnivoice import OmniVoice

        logger.info("Loading OmniVoice model (first call)...")
        self._model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0",
            dtype="float16",
        )
        logger.info("OmniVoice model loaded")

    def _ensure_ref_audio(self, lang: str) -> Path:
        """Generate reference audio for a voice if it doesn't already exist."""
        wav_path = _VOICES_DIR / f"{lang}.wav"
        if wav_path.exists():
            return wav_path

        profile = _VOICE_PROFILES[lang]
        logger.info("Generating reference audio for %s voice: %s", lang, profile["instruct"])

        from omnivoice import OmniVoiceGenerationConfig

        audio_arrays = self._model.generate(
            text=profile["ref_text"],
            language=lang,
            instruct=profile["instruct"],
            generation_config=OmniVoiceGenerationConfig(num_step=self._num_steps),
        )
        audio = audio_arrays[0]

        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_path), audio, 24000, subtype="PCM_16", format="WAV")
        duration = len(audio) / 24000
        logger.info("Saved %s reference audio: %.1fs -> %s", lang, duration, wav_path)
        return wav_path

    def preload(self) -> None:
        self._ensure_model()

        for lang in _VOICE_PROFILES:
            wav_path = self._ensure_ref_audio(lang)
            logger.info("Creating voice clone prompt for %s from %s", lang, wav_path)
            self._voice_prompts[lang] = self._model.create_voice_clone_prompt(str(wav_path))

        logger.info("All voices ready: %s", list(self._voice_prompts.keys()))

    def generate(self, text: str, language: str) -> tuple[np.ndarray, int]:
        """Generate speech for full text. OmniVoice handles internal chunking."""
        from omnivoice import OmniVoiceGenerationConfig

        self._ensure_model()

        cleaned = " ".join(text.strip().split())

        voice_lang = language if language in self._voice_prompts else _DEFAULT_VOICE
        voice_prompt = self._voice_prompts.get(voice_lang)
        profile = _VOICE_PROFILES.get(voice_lang, _VOICE_PROFILES[_DEFAULT_VOICE])

        audio_arrays = self._model.generate(
            text=cleaned,
            language=language,
            voice_clone_prompt=voice_prompt,
            speed=profile["speed"],
            generation_config=OmniVoiceGenerationConfig(num_step=self._num_steps),
        )
        audio = audio_arrays[0]

        if audio.size == 0:
            logger.warning("OmniVoice returned empty audio for %r (lang=%s), substituting silence", text[:40], language)
            audio = np.zeros(int(24000 * 0.3), dtype=np.float32)

        return audio, 24000
