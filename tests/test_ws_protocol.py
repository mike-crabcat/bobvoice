from __future__ import annotations

from bobvoice.ws_protocol import (
    CancelMessage,
    ErrorMessage,
    LatencyMessage,
    ResponseTextMessage,
    SetLanguageMessage,
    StartRecordingMessage,
    StatusMessage,
    StopRecordingMessage,
    TranscriptMessage,
    parse_client_message,
)


class TestParseClientMessage:
    def test_start_recording(self) -> None:
        raw = '{"type": "start_recording", "userId": "mike", "sessionKey": "voice-mike", "language": "en"}'
        msg = parse_client_message(raw)
        assert isinstance(msg, StartRecordingMessage)
        assert msg.userId == "mike"
        assert msg.sessionKey == "voice-mike"
        assert msg.language == "en"

    def test_start_recording_no_language(self) -> None:
        raw = '{"type": "start_recording", "userId": "mike", "sessionKey": "voice-mike"}'
        msg = parse_client_message(raw)
        assert isinstance(msg, StartRecordingMessage)
        assert msg.language is None

    def test_stop_recording(self) -> None:
        msg = parse_client_message('{"type": "stop_recording"}')
        assert isinstance(msg, StopRecordingMessage)

    def test_cancel(self) -> None:
        msg = parse_client_message('{"type": "cancel"}')
        assert isinstance(msg, CancelMessage)

    def test_set_language(self) -> None:
        msg = parse_client_message('{"type": "set_language", "language": "pt"}')
        assert isinstance(msg, SetLanguageMessage)
        assert msg.language == "pt"

    def test_invalid_json(self) -> None:
        assert parse_client_message("not json") is None

    def test_unknown_type(self) -> None:
        assert parse_client_message('{"type": "unknown"}') is None

    def test_empty_string(self) -> None:
        assert parse_client_message("") is None


class TestServerMessages:
    def test_status_message_serialization(self) -> None:
        import json

        msg = StatusMessage(state="recording")
        data = json.loads(msg.model_dump_json())
        assert data["type"] == "status"
        assert data["state"] == "recording"

    def test_transcript_message(self) -> None:
        import json

        msg = TranscriptMessage(text="hello", language="en", latency_ms=150)
        data = json.loads(msg.model_dump_json())
        assert data["type"] == "transcript"
        assert data["text"] == "hello"
        assert data["latency_ms"] == 150

    def test_response_text_message(self) -> None:
        msg = ResponseTextMessage(text="Hi there!")
        assert msg.type == "response_text"
        assert msg.text == "Hi there!"

    def test_latency_message_optional_fields(self) -> None:
        msg = LatencyMessage(stt_ms=100, e2e_ms=500)
        assert msg.openclaw_total_ms is None
        assert msg.tts_first_chunk_ms is None

    def test_error_message(self) -> None:
        msg = ErrorMessage(message="Something went wrong")
        assert msg.type == "error"
