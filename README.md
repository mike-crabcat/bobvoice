# Bob Voice

Local voice interface for talking to Bob (OpenClaw AI assistant) from mobile devices over Tailscale.

Push-to-talk web app → WebSocket → FastAPI bridge → STT (faster-whisper) → OpenClaw → TTS → audio back to browser. All inference local on RTX 3060.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the server
uv run python main.py

# Open http://localhost:8421 in a browser
```

## Architecture

```
┌─────────────┐    WebSocket     ┌──────────────┐    ┌─────────────────┐
│  Mobile SPA  │ ◄──────────────► │  FastAPI      │    │  faster-whisper  │
│  (frontend/) │   binary audio   │  bridge       │───►│  (STT)           │
│              │   JSON messages  │  (port 8421)  │    └─────────────────┘
└─────────────┘                   │               │    ┌─────────────────┐
                                  │               │───►│  OpenClaw API    │
                                  │               │    │  (LLM response)  │
                                  │               │    └─────────────────┘
                                  │               │    ┌─────────────────┐
                                  │               │───►│  TTS engine     │
                                  │               │    │  (OmniVoice)    │
                                  └──────────────┘    └─────────────────┘
```

## WebSocket Protocol

### Client → Server (text)

| Type | Fields | Description |
|------|--------|-------------|
| `start_recording` | `userId`, `sessionKey`, `language` (`en`/`pt`) | Begin audio capture |
| `stop_recording` | — | End audio capture, trigger transcription |
| `cancel` | — | Cancel current recording/processing |
| `set_language` | `language` | Switch language mid-session |

### Client → Server (binary)

Raw audio chunks from MediaRecorder — MP4/AAC (Safari) or WebM/Opus (Chrome).

### Server → Client (text)

| Type | Fields | Description |
|------|--------|-------------|
| `status` | `state` (`recording`/`transcribing`/`thinking`/`speaking`/`idle`) | State change |
| `transcript` | `text`, `language`, `latency_ms` | STT result |
| `response_text` | `text` | LLM response text |
| `latency` | `stt_ms`, `openclaw_total_ms`, `e2e_ms` | Performance metrics |
| `audio_done` | — | TTS playback complete |
| `error` | `message` | Error description |

### Server → Client (binary)

WAV audio chunks (TTS output, one per sentence — progressive playback).

## Project Structure

```
bobvoice/
├── main.py              # Entry point (uvicorn runner)
├── pyproject.toml       # Dependencies and tool config
├── ruff.toml            # Linter/formatter config
├── mypy.ini             # Type checker config
├── frontend/            # Static push-to-talk SPA (unchanged)
│   ├── index.html
│   ├── app.js
│   └── style.css
├── src/bobvoice/
│   ├── main.py          # FastAPI app, static files, WebSocket endpoint
│   ├── ws_protocol.py   # Pydantic models for WebSocket messages
│   ├── stt.py           # faster-whisper transcription
│   ├── tts.py           # TTS engine (stub → OmniVoice)
│   └── openclaw.py      # OpenClaw API client (stub → real integration)
└── tests/
    ├── conftest.py
    ├── test_main.py
    ├── test_stt.py
    └── test_ws_protocol.py
```

## Development

```bash
# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/

# Run tests
uv run pytest tests/ -v

# Format
uv run ruff format src/ tests/
```

## Languages

- English (`en`)
- Brazilian Portuguese (`pt`)

## Status

- [x] FastAPI bridge server with WebSocket protocol
- [x] Frontend push-to-talk SPA (Safari + Chrome)
- [x] faster-whisper STT (local, GPU-accelerated)
- [x] Stub TTS (returns silence — awaiting OmniVoice integration)
- [x] Stub OpenClaw client (echo response — awaiting real API integration)
- [ ] Real TTS with Bob's cloned voice (OmniVoice)
- [ ] Real OpenClaw API integration
- [ ] Tailscale serve/funnel for remote access
- [ ] Helen's independent session support
