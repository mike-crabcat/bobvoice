from __future__ import annotations

import asyncio
import io
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_REF_AUDIO_DEFAULT = (
    Path.home()
    / ".openclaw/workspace/projects/bob-cast-skill-setup/tasks/fb7538ce/bob_reference_voice_recovered.wav"
)


_ABBREVIATIONS = frozenset({"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie", "approx"})

_VOICE_INSTRUCT_BOB = "male, australian accent, young adult"
_VOICE_INSTRUCT: dict[str, str] = {
    "pt": "female, portuguese accent, young adult",
    "fr": "female, young adult",
}


def _split_sentences(text: str) -> list[str]:
    # Normalize whitespace but don't split on bare newlines
    cleaned = " ".join(text.strip().split())
    # Split after sentence-ending punctuation followed by space,
    # but not after common abbreviations (Dr. Mr. etc.)
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    result: list[str] = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        # Re-attach fragments that start with lowercase or follow an abbreviation
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
    def __init__(self, ref_audio_path: str | None = None, num_steps: int = 32) -> None:
        self._ref_audio_path = ref_audio_path or os.getenv(
            "BOBVOICE_TTS_REF_AUDIO", str(_REF_AUDIO_DEFAULT)
        )
        self._num_steps = num_steps
        self._model: Any = None
        self._voice_prompt: Any = None
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
        if Path(str(self._ref_audio_path)).exists():
            logger.info("Creating voice clone prompt from %s", self._ref_audio_path)
            self._voice_prompt = self._model.create_voice_clone_prompt(self._ref_audio_path)
        else:
            logger.warning("Reference audio not found: %s — using default voice", self._ref_audio_path)
        logger.info("OmniVoice ready")

    def preload(self) -> None:
        self._ensure_model()

    def generate(self, text: str, language: str) -> tuple[np.ndarray, int]:
        """Generate speech for full text. OmniVoice handles internal chunking."""
        from omnivoice import OmniVoiceGenerationConfig

        self._ensure_model()
        cleaned = " ".join(text.strip().split())
        if language in ("en", ""):
            voice_prompt = self._voice_prompt
            instruct = None
            speed = 1.2
        else:
            voice_prompt = None
            instruct = _VOICE_INSTRUCT.get(language, _VOICE_INSTRUCT_BOB)
            speed = 1.0
        audio_arrays = self._model.generate(
            text=cleaned,
            language=language,
            voice_clone_prompt=voice_prompt,
            instruct=instruct,
            speed=speed,
            generation_config=OmniVoiceGenerationConfig(num_step=self._num_steps),
        )
        return audio_arrays[0], 24000
