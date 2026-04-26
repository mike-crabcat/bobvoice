from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import av
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


def _convert_to_wav(audio_data: bytes) -> bytes:
    """Convert arbitrary audio bytes to 16kHz mono WAV using PyAV."""
    inp = av.open(io.BytesIO(audio_data))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    out_buf = io.BytesIO()
    out = av.open(out_buf, mode="w", format="wav")

    out_stream = out.add_stream("pcm_s16le", rate=16000, layout="mono")

    for frame in inp.decode(audio=0):  # type: ignore[union-attr]
        for rf in resampler.resample(frame):
            for packet in out_stream.encode(rf):
                out.mux(packet)

    for packet in out_stream.encode(None):
        out.mux(packet)

    out.close()
    inp.close()
    return out_buf.getvalue()


class STTEngine:
    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cuda", compute_type: str = "int8") -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: WhisperModel | None = None

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def preload(self) -> None:
        logger.info("Preloading STT model (%s, %s)...", self._model_size, self._device)
        self._ensure_model()
        logger.info("STT model ready")

    def transcribe(self, audio_data: bytes, language: str | None = None) -> tuple[str, str]:
        model = self._ensure_model()

        try:
            wav_bytes = _convert_to_wav(audio_data)
        except Exception:
            logger.error("Audio conversion failed", exc_info=True)
            return "", language or "en"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        try:
            segments, info = model.transcribe(wav_path, language=language)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text, info.language
        finally:
            Path(wav_path).unlink(missing_ok=True)
