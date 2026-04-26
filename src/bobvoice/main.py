from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from starlette.requests import Request

from .openclaw import OpenClawClient, OpenClawConfig
from .stt import STTEngine
from .tts import TTSEngine, samples_to_wav
from .ws_protocol import (
    AudioDoneMessage,
    CancelMessage,
    ErrorMessage,
    LatencyMessage,
    PartialResponseMessage,
    ResponseTextMessage,
    SetLanguageMessage,
    StartRecordingMessage,
    StatusMessage,
    StopRecordingMessage,
    TranscriptMessage,
    parse_client_message,
)

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"

app = FastAPI(title="Bob Voice")

stt_engine = STTEngine()
tts_engine = TTSEngine()

_logger = logging.getLogger("bobvoice.main")
openclaw_config = OpenClawConfig.from_env()
openclaw_client = OpenClawClient(config=openclaw_config)

if openclaw_config.enabled:
    _logger.info("OpenClaw gateway configured: %s", openclaw_config.resolved_gateway_url)
else:
    _logger.warning("OpenClaw gateway not configured — running in stub mode")


@app.on_event("startup")
def _preload_models() -> None:
    _logger.info("Preloading models...")
    stt_engine.preload()
    tts_engine.preload()
    _logger.info("All models loaded")


_StatusState = Literal["recording", "transcribing", "thinking", "speaking", "idle"]


async def _send_status(websocket: WebSocket, state: _StatusState) -> None:
    await websocket.send_text(StatusMessage(state=state).model_dump_json())


async def _send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_text(ErrorMessage(message=message).model_dump_json())


_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


async def _process_audio(
    websocket: WebSocket,
    audio_chunks: list[bytes],
    language: str | None,
    user_id: str = "mike",
) -> None:
    _logger.info("Processing audio: %d chunks, %d bytes", len(audio_chunks), sum(len(c) for c in audio_chunks))
    await _send_status(websocket, "transcribing")

    audio_data = b"".join(audio_chunks)
    if not audio_data:
        await _send_error(websocket, "No audio data received")
        await _send_status(websocket, "idle")
        return

    t0 = time.monotonic()

    try:
        text, detected_lang = await asyncio.to_thread(stt_engine.transcribe, audio_data, language)
    except Exception:
        _logger.error("STT transcription failed", exc_info=True)
        await _send_error(websocket, "Transcription failed")
        await _send_status(websocket, "idle")
        return
    stt_ms = int((time.monotonic() - t0) * 1000)

    if not text:
        await _send_error(websocket, "Could not transcribe audio")
        await _send_status(websocket, "idle")
        return

    _logger.info("STT done: %dms, text=%r, lang=%s", stt_ms, text[:100], detected_lang)

    await websocket.send_text(
        TranscriptMessage(text=text, language=detected_lang, latency_ms=stt_ms).model_dump_json()
    )

    await _send_status(websocket, "thinking")
    t1 = time.monotonic()
    _logger.info("Calling OpenClaw...")

    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
    spoken_text = ""
    tts_first_chunk_ms: int | None = None

    async def on_delta(accumulated: str) -> None:
        nonlocal spoken_text
        while len(spoken_text) < len(accumulated):
            unspoken = accumulated[len(spoken_text):]
            match = _SENTENCE_END.search(unspoken)
            if match is None:
                break
            sentence = unspoken[: match.end()]
            spoken_text = accumulated[: len(spoken_text) + match.end()]
            await sentence_queue.put(sentence)
            await websocket.send_text(
                PartialResponseMessage(text=spoken_text).model_dump_json()
            )

    async def tts_consumer() -> None:
        nonlocal tts_first_chunk_ms
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            async with tts_engine._lock:
                audio, sr = await asyncio.to_thread(tts_engine.generate, sentence, detected_lang)
                wav_bytes = samples_to_wav(audio, sr)
            if tts_first_chunk_ms is None:
                tts_first_chunk_ms = int((time.monotonic() - t1) * 1000)
                await _send_status(websocket, "speaking")
            await websocket.send_bytes(wav_bytes)

    tts_task = asyncio.create_task(tts_consumer())

    response = await openclaw_client.respond(text, detected_lang, on_delta=on_delta, user_id=user_id)
    openclaw_ms = int((time.monotonic() - t1) * 1000)
    _logger.info("OpenClaw done: %dms, response=%d chars", openclaw_ms, len(response))

    # Flush any remaining text that didn't end with a sentence boundary
    remaining = response[len(spoken_text):]
    if remaining.strip():
        spoken_text = response
        await sentence_queue.put(remaining)
        await websocket.send_text(
            PartialResponseMessage(text=spoken_text).model_dump_json()
        )

    # Signal TTS consumer to finish
    await sentence_queue.put(None)
    await tts_task

    if tts_first_chunk_ms is None:
        tts_first_chunk_ms = int((time.monotonic() - t1) * 1000)

    await websocket.send_text(
        ResponseTextMessage(text=response).model_dump_json()
    )

    e2e_ms = int((time.monotonic() - t0) * 1000)

    await websocket.send_text(AudioDoneMessage().model_dump_json())
    await websocket.send_text(
        LatencyMessage(
            stt_ms=stt_ms,
            openclaw_total_ms=openclaw_ms,
            tts_first_chunk_ms=tts_first_chunk_ms,
            e2e_ms=e2e_ms,
        ).model_dump_json()
    )
    await _send_status(websocket, "idle")


@app.post("/log")
async def _client_log(request: Any) -> JSONResponse:
    body = await request.json()
    level = body.get("level", "info")
    message = body.get("message", "")
    tag = body.get("tag", "frontend")
    context = body.get("context")
    log_fn = _logger.warning if level == "error" else _logger.info
    extra = f" — {context}" if context else ""
    log_fn("[%s] %s%s", tag, message, extra)
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    client = websocket.client.host if websocket.client else "unknown"
    _logger.info("WS connected from %s (active: checking)", client)

    audio_chunks: list[bytes] = []
    language: str | None = None
    user_id: str = "mike"

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "text" in msg:
                parsed = parse_client_message(msg["text"])
                if parsed is None:
                    continue

                match parsed:
                    case StartRecordingMessage() as m:
                        audio_chunks = []
                        language = m.language
                        user_id = m.userId
                        await _send_status(websocket, "recording")

                    case StopRecordingMessage():
                        await _process_audio(websocket, audio_chunks, language, user_id)

                    case CancelMessage():
                        audio_chunks = []
                        await _send_status(websocket, "idle")

                    case SetLanguageMessage() as m:
                        language = m.language

            elif "bytes" in msg:
                audio_chunks.append(msg["bytes"])

    except WebSocketDisconnect:
        pass


_NO_CACHE_EXTENSIONS = {".js", ".css", ".html"}


@app.middleware("http")
async def _nocache_middleware(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    path = request.url.path
    if any(path.endswith(ext) for ext in _NO_CACHE_EXTENSIONS):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Mount static files LAST — catch-all must come after all routes
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
