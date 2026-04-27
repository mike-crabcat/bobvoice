from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from bobvoice.stt import STTEngine
from bobvoice.tts import (
    TTSEngine,
    _split_sentences,
    generate_silence_wav,
    generate_tone_wav,
    samples_to_wav,
)


class TestTTS:
    def test_generate_silence_wav(self) -> None:
        wav = generate_silence_wav(duration=0.1, sample_rate=16000)
        assert isinstance(wav, bytes)
        assert wav[:4] == b"RIFF"
        assert b"WAVE" in wav

    def test_generate_tone_wav(self) -> None:
        wav = generate_tone_wav(frequency=440.0, duration=0.1, sample_rate=16000)
        assert isinstance(wav, bytes)
        assert wav[:4] == b"RIFF"

    def test_samples_to_wav(self) -> None:
        samples = np.zeros(100, dtype=np.float32)
        wav = samples_to_wav(samples, 24000)
        assert isinstance(wav, bytes)
        assert wav[:4] == b"RIFF"

    def test_wav_valid_audio(self) -> None:
        import io

        import soundfile as sf

        wav = generate_tone_wav(frequency=440.0, duration=0.5, sample_rate=16000)
        data, sr = sf.read(io.BytesIO(wav))
        assert sr == 16000
        assert len(data) == int(16000 * 0.5)
        assert data.dtype == np.float64


class TestSplitSentences:
    def test_basic_split(self) -> None:
        result = _split_sentences("Hello world. How are you? I am fine.")
        assert result == ["Hello world.", "How are you?", "I am fine."]

    def test_single_sentence(self) -> None:
        assert _split_sentences("Just one sentence.") == ["Just one sentence."]

    def test_newlines_collapsed(self) -> None:
        # Newlines should be collapsed to spaces, not create separate chunks
        result = _split_sentences("Hello,\nworld. How are you?")
        assert result == ["Hello, world.", "How are you?"]

    def test_whitespace_handling(self) -> None:
        assert _split_sentences("  Hello.  ") == ["Hello."]

    def test_fallback_to_full_text(self) -> None:
        assert _split_sentences("no punctuation") == ["no punctuation"]

    def test_abbreviation_not_split(self) -> None:
        result = _split_sentences("Hello Dr. Smith. How are you?")
        assert result == ["Hello Dr. Smith.", "How are you?"]

    def test_lowercase_continuation(self) -> None:
        # Lowercase after period = not a real sentence break
        result = _split_sentences("e.g. this thing. That is all.")
        assert result == ["e.g. this thing.", "That is all."]


class TestTTSEngine:
    def test_generate_with_mock(self) -> None:
        audio = np.random.randn(24000).astype(np.float32)
        with patch("bobvoice.tts.TTSEngine._ensure_model"):
            engine = TTSEngine()
            engine._model = MagicMock()
            engine._voice_prompts = {"en": MagicMock(), "fr": MagicMock(), "pt": MagicMock()}
            engine._model.generate.return_value = [audio]
            result_audio, sr = engine.generate("Hello world.", "en")
        assert sr == 24000
        assert len(result_audio) == 24000


class TestSTTUnit:
    @patch("bobvoice.stt._convert_to_wav", return_value=b"RIFF\x00\x00\x00\x00WAVE")
    @patch("bobvoice.stt.WhisperModel")
    def test_transcribe_returns_text_and_language(self, mock_whisper_cls: MagicMock, _mock_conv: MagicMock) -> None:
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = " hello world "
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = STTEngine()
        text, lang = engine.transcribe(b"fake audio data", language="en")

        assert text == "hello world"
        assert lang == "en"

    @patch("bobvoice.stt._convert_to_wav", return_value=b"RIFF\x00\x00\x00\x00WAVE")
    @patch("bobvoice.stt.WhisperModel")
    def test_transcribe_no_language_auto_detect(self, mock_whisper_cls: MagicMock, _mock_conv: MagicMock) -> None:
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = " olá mundo "
        mock_info = MagicMock()
        mock_info.language = "pt"
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = STTEngine()
        text, lang = engine.transcribe(b"fake audio data", language=None)

        assert text == "olá mundo"
        assert lang == "pt"

    @patch("bobvoice.stt._convert_to_wav", return_value=b"RIFF\x00\x00\x00\x00WAVE")
    @patch("bobvoice.stt.WhisperModel")
    def test_transcribe_empty_result(self, mock_whisper_cls: MagicMock, _mock_conv: MagicMock) -> None:
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = STTEngine()
        text, lang = engine.transcribe(b"silence")

        assert text == ""
        assert lang == "en"

    @patch("bobvoice.stt.WhisperModel")
    def test_transcribe_conversion_failure(self, mock_whisper_cls: MagicMock) -> None:
        mock_whisper_cls.return_value = MagicMock()

        engine = STTEngine()
        # Pass garbage that PyAV can't decode
        text, lang = engine.transcribe(b"not audio at all", language="en")

        assert text == ""
        assert lang == "en"
