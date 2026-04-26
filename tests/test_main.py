from __future__ import annotations

import json

from fastapi.testclient import TestClient


def test_serves_frontend_index() -> None:
    from bobvoice.main import app

    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Bob Voice" in resp.text


def test_websocket_full_flow(mock_stt: None, mock_openclaw: None, mock_tts: None) -> None:
    from bobvoice.main import app

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        # Start recording
        ws.send_json({
            "type": "start_recording",
            "userId": "test",
            "sessionKey": "voice-test",
            "language": "en",
        })
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["state"] == "recording"

        # Send audio chunk
        ws.send_bytes(b"fake audio chunk")

        # Stop recording — triggers full pipeline
        ws.send_json({"type": "stop_recording"})

        # transcribing status
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["state"] == "transcribing"

        # transcript
        msg = ws.receive_json()
        assert msg["type"] == "transcript"
        assert msg["text"] == "hello world"
        assert msg["language"] == "en"

        # thinking status
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["state"] == "thinking"

        # response text
        msg = ws.receive_json()
        assert msg["type"] == "response_text"

        # speaking status
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["state"] == "speaking"

        # binary audio (one chunk per sentence in the response)
        audio_data = ws.receive_bytes()
        assert len(audio_data) > 0

        # Read any additional audio chunks (response may have multiple sentences)
        while True:
            raw = ws.receive()
            if "text" in raw:
                msg = json.loads(raw["text"])
                break

        # After all audio chunks: audio_done, then latency, then idle
        assert msg["type"] == "audio_done"

        # latency
        msg = ws.receive_json()
        assert msg["type"] == "latency"
        assert msg["type"] == "latency"
        assert "stt_ms" in msg
        assert "e2e_ms" in msg

        # idle
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["state"] == "idle"


def test_websocket_cancel_during_recording() -> None:
    from bobvoice.main import app

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "start_recording",
            "userId": "test",
            "sessionKey": "voice-test",
            "language": "en",
        })
        msg = ws.receive_json()
        assert msg["state"] == "recording"

        ws.send_json({"type": "cancel"})
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["state"] == "idle"


def test_websocket_set_language() -> None:
    from bobvoice.main import app

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "set_language", "language": "pt"})
        # No response expected — just shouldn't error


def test_websocket_empty_audio(mock_stt: None, mock_openclaw: None) -> None:
    from bobvoice.main import app

    # Mock STT to return empty text for this test
    mock_stt.transcribe.return_value = ("", "en")

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "start_recording",
            "userId": "test",
            "sessionKey": "voice-test",
            "language": "en",
        })
        ws.receive_json()  # recording status

        ws.send_json({"type": "stop_recording"})

        # transcribing status
        msg = ws.receive_json()
        assert msg["state"] == "transcribing"

        # error message
        msg = ws.receive_json()
        assert msg["type"] == "error"

        # idle
        msg = ws.receive_json()
        assert msg["state"] == "idle"
