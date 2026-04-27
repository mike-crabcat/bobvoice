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
from .session_store import SessionStore
from .stt import STTEngine
from .tts import TTSEngine, generate_tone_wav, samples_to_wav
from .ws_protocol import (
    AudioDoneMessage,
    CancelMessage,
    ClearHistoryMessage,
    ErrorMessage,
    HistoryEntry,
    HistoryMessage,
    LatencyMessage,
    LessonProgressMessage,
    PartialResponseMessage,
    ReplayTtsMessage,
    ResponseTextMessage,
    SessionHistoryMessage,
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
session_store = SessionStore()

_logger = logging.getLogger("bobvoice.main")
openclaw_config = OpenClawConfig.from_env()
openclaw_client = OpenClawClient(config=openclaw_config)

if openclaw_config.enabled:
    _logger.info("OpenClaw gateway configured: %s", openclaw_config.resolved_gateway_url)
else:
    _logger.warning("OpenClaw gateway not configured — running in stub mode")


_FILLER_PHRASES = ["Ok.", "Hurmm.", "I'm thinking.", "Okaay."]


def _generate_filler_sounds() -> list[bytes]:
    """Pre-generate filler audio clips via TTS. Falls back to ping tones if TTS fails."""
    fillers: list[bytes] = []
    for phrase in _FILLER_PHRASES:
        audio, sr = tts_engine.generate(phrase, "en")
        if audio.size > 0:
            wav = samples_to_wav(audio, sr)
            _logger.info("Generated filler %r (%d bytes)", phrase, len(wav))
            fillers.append(wav)
        else:
            _logger.warning("Filler TTS returned empty for %r", phrase)
    if not fillers:
        freq = 520.0
        for _ in _FILLER_PHRASES:
            fillers.append(generate_tone_wav(frequency=freq, duration=0.15, sample_rate=24000, amplitude=0.15))
            freq += 40
        _logger.warning("All filler TTS failed, using ping tones as fallback")
    return fillers


_FILLER_SOUNDS: list[bytes] = []


@app.on_event("startup")
def _preload_models() -> None:
    global _FILLER_SOUNDS
    _logger.info("Preloading models...")
    stt_engine.preload()
    tts_engine.preload()
    _FILLER_SOUNDS = _generate_filler_sounds()
    session_store.delete_old_sessions(max_age_days=30)
    _logger.info("All models loaded (%d filler sounds ready)", len(_FILLER_SOUNDS))


@app.on_event("shutdown")
def _cleanup() -> None:
    session_store.close()


_StatusState = Literal["recording", "transcribing", "thinking", "speaking", "idle"]


def _session_key(user_id: str, session_mode: str) -> str:
    return f"bobvoice:{session_mode}:{user_id}"


async def _send_status(websocket: WebSocket, state: _StatusState) -> None:
    await websocket.send_text(StatusMessage(state=state).model_dump_json())


async def _send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_text(ErrorMessage(message=message).model_dump_json())


_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

_LESSONS_DIR = Path(__file__).parent.parent.parent / "lessons"
_TOTAL_LESSONS = 10

_LANG_TAG_RE = re.compile(r"<lang\s+(\w+)>(.*?)</lang>", re.DOTALL)
_STEP_COMPLETE_RE = re.compile(r'<step_complete\s+lesson="(\d+)"\s+step="(\d+)"\s*/?>')
_LESSON_COMPLETE_RE = re.compile(r'<lesson_complete\s+lesson="(\d+)"\s*/?>')


def _load_lesson(lesson_number: int) -> str:
    path = _LESSONS_DIR / f"beginner_french_{lesson_number:02d}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Lesson file not found: {path}")
    return path.read_text(encoding="utf-8")


def _load_beginner_french_prompt() -> str:
    path = Path(__file__).parent.parent.parent / "prompts" / "beginner_french.txt"
    return path.read_text(encoding="utf-8").strip()


def _build_lesson_context(
    lesson_number: int, lesson_content: str, completed_steps: list[int], user_id: str,
) -> str:
    template = _load_beginner_french_prompt()
    steps_str = ", ".join(str(s) for s in completed_steps) if completed_steps else "none"
    return template.format(
        LESSON_NUMBER=lesson_number,
        LESSON_CONTENT=lesson_content,
        COMPLETED_STEPS_LIST=steps_str,
        USER_ID=user_id,
    )


def _process_language_tags(text: str, default_lang: str) -> list[tuple[str, str]]:
    """Split text at <lang> tag boundaries into (text, lang) TTS fragments."""
    fragments = []
    last_end = 0
    for m in _LANG_TAG_RE.finditer(text):
        before = text[last_end:m.start()]
        if before.strip():
            fragments.append((before.strip(), default_lang))
        tagged_text = m.group(2).strip()
        if tagged_text:
            fragments.append((tagged_text, m.group(1)))
        last_end = m.end()
    remaining = text[last_end:]
    if remaining.strip():
        fragments.append((remaining.strip(), default_lang))
    return fragments if fragments else [(text, default_lang)]


def _extract_instruction_tokens(text: str) -> tuple[str, list[tuple[int, int]], list[int]]:
    """Extract step_complete and lesson_complete tokens. Returns (cleaned_text, steps, lessons)."""
    steps = [(int(m.group(1)), int(m.group(2))) for m in _STEP_COMPLETE_RE.finditer(text)]
    lessons = [int(m.group(1)) for m in _LESSON_COMPLETE_RE.finditer(text)]
    cleaned = _STEP_COMPLETE_RE.sub("", text)
    cleaned = _LESSON_COMPLETE_RE.sub("", cleaned)
    return cleaned, steps, lessons


def _strip_display_tags(text: str) -> str:
    """Strip <lang> tags from text for display in the frontend."""
    return _LANG_TAG_RE.sub(r"\2", text)


def _clean_display_text(text: str) -> str:
    """Strip all non-display markup: <lang> tags, step_complete, lesson_complete."""
    text = _strip_display_tags(text)
    text = _STEP_COMPLETE_RE.sub("", text)
    text = _LESSON_COMPLETE_RE.sub("", text)
    return text


async def _process_audio(
    websocket: WebSocket,
    audio_chunks: list[bytes],
    language: str | None,
    user_id: str = "mike",
    session_mode: str = "chat",
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

    is_beginner = session_mode == "beginner_french"
    default_tts_lang = "en" if is_beginner else detected_lang

    # Build lesson context for beginner mode
    lesson_context: str | None = None
    current_lesson_num = 1
    if is_beginner:
        current_lesson_num = session_store.get_current_lesson(user_id, "beginner_french", _TOTAL_LESSONS)
        try:
            lesson_content = _load_lesson(current_lesson_num)
        except FileNotFoundError:
            current_lesson_num = 1
            lesson_content = _load_lesson(current_lesson_num)
        completed_steps = session_store.get_completed_steps(user_id, "beginner_french", current_lesson_num)
        lesson_context = _build_lesson_context(current_lesson_num, lesson_content, completed_steps, user_id)
        _logger.info("Beginner French: lesson %d, completed steps: %s", current_lesson_num, completed_steps)

    sentence_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
    spoken_text = ""
    tts_first_chunk_ms: int | None = None
    last_accumulated = ""

    # Filler sounds for tool-call feedback (pre-generated at startup)
    filler_last_time: float = 0
    filler_waiting_for_text = False
    filler_index = 0

    async def on_tool_start() -> None:
        nonlocal filler_last_time, filler_waiting_for_text, filler_index
        if filler_waiting_for_text:
            _logger.info("Tool start: skipped (waiting for text)")
            return
        now = time.monotonic()
        elapsed = now - filler_last_time if filler_last_time > 0 else 0
        filler_waiting_for_text = True
        filler_last_time = now
        wav = _FILLER_SOUNDS[filler_index % len(_FILLER_SOUNDS)]
        filler_index += 1
        _logger.info("Tool start: playing filler #%d (elapsed=%.1fs)", filler_index, elapsed)
        try:
            await websocket.send_bytes(wav)
        except Exception:
            pass  # client disconnected

    async def on_delta(accumulated: str) -> None:
        nonlocal spoken_text, filler_waiting_for_text, last_accumulated
        last_accumulated = accumulated
        if filler_waiting_for_text and len(accumulated) > 0:
            filler_waiting_for_text = False
        while len(spoken_text) < len(accumulated):
            unspoken = accumulated[len(spoken_text):]
            # Find a sentence boundary that's not inside an unclosed <lang> tag
            match = None
            search_start = 0
            while True:
                candidate = _SENTENCE_END.search(unspoken, search_start)
                if candidate is None:
                    break
                prefix = unspoken[:candidate.end()]
                if prefix.count("<lang ") == prefix.count("</lang>"):
                    match = candidate
                    break
                search_start = candidate.end()
            if match is None:
                break
            sentence = unspoken[: match.end()]
            spoken_text = accumulated[: len(spoken_text) + match.end()]
            clean_sentence = _STEP_COMPLETE_RE.sub("", sentence)
            clean_sentence = _LESSON_COMPLETE_RE.sub("", clean_sentence)
            fragments = _process_language_tags(clean_sentence, default_tts_lang)
            for frag_text, frag_lang in fragments:
                await sentence_queue.put((frag_text, frag_lang))
            try:
                await websocket.send_text(
                    PartialResponseMessage(text=_clean_display_text(spoken_text)).model_dump_json()
                )
            except Exception:
                pass  # client disconnected

    async def tts_consumer() -> None:
        nonlocal tts_first_chunk_ms
        while True:
            item = await sentence_queue.get()
            if item is None:
                break
            sentence_text, sentence_lang = item
            try:
                async with tts_engine._lock:
                    audio, sr = await asyncio.to_thread(tts_engine.generate, sentence_text, sentence_lang)
                    wav_bytes = samples_to_wav(audio, sr)
                if tts_first_chunk_ms is None:
                    tts_first_chunk_ms = int((time.monotonic() - t1) * 1000)
                    try:
                        await _send_status(websocket, "speaking")
                    except Exception:
                        pass
                try:
                    await websocket.send_bytes(wav_bytes)
                except Exception:
                    pass  # client disconnected
            except Exception:
                _logger.warning("TTS failed for %r (lang=%s)", sentence_text[:80], sentence_lang, exc_info=True)

    tts_task = asyncio.create_task(tts_consumer())

    response = await openclaw_client.respond(
        text, detected_lang,
        on_delta=on_delta, on_tool_start=on_tool_start,
        user_id=user_id, session_mode=session_mode,
        lesson_context=lesson_context,
    )
    openclaw_ms = int((time.monotonic() - t1) * 1000)
    _logger.info("OpenClaw done: %dms, response=%d chars", openclaw_ms, len(response))

    # Use the streamed accumulated text as the authoritative response.
    # The extracted response from gateway payloads may differ (e.g. concatenated
    # payloads can produce duplicated text).
    if last_accumulated:
        response = last_accumulated

    # Extract instruction tokens and clean the response
    clean_response, completed_steps_raw, completed_lessons = _extract_instruction_tokens(response)

    # Align spoken_text with clean_response: spoken_text tracked position in the
    # raw stream which includes instruction tokens, but clean_response has them removed.
    clean_spoken = _STEP_COMPLETE_RE.sub("", spoken_text)
    clean_spoken = _LESSON_COMPLETE_RE.sub("", clean_spoken)

    # Flush any remaining text that didn't end with a sentence boundary
    remaining = clean_response[len(clean_spoken):]
    if remaining.strip():
        clean_spoken = clean_response
        fragments = _process_language_tags(remaining, default_tts_lang)
        for frag_text, frag_lang in fragments:
            await sentence_queue.put((frag_text, frag_lang))
        try:
            await websocket.send_text(
                PartialResponseMessage(text=_clean_display_text(clean_spoken)).model_dump_json()
            )
        except Exception:
            pass  # client disconnected

    # Signal TTS consumer to finish
    await sentence_queue.put(None)
    await tts_task

    # Persist to session store unconditionally — even if client disconnected,
    # the exchange should appear in history on reconnect.
    if is_beginner:
        for lesson_num, step_idx in completed_steps_raw:
            session_store.mark_step_complete(user_id, "beginner_french", lesson_num, step_idx)
            _logger.info("Step completed: lesson %d step %d", lesson_num, step_idx)
        for lesson_num in completed_lessons:
            _logger.info("Lesson completed: %d", lesson_num)
            new_lesson = session_store.advance_lesson(user_id, "beginner_french", _TOTAL_LESSONS)
            session_store.reset_lesson(user_id, "beginner_french", new_lesson)
            _logger.info("Advanced to lesson %d", new_lesson)

    key = _session_key(user_id, session_mode)
    session_store.add_message(key, "user", text, language=detected_lang)
    session_store.add_message(key, "assistant", clean_response)
    _logger.info("Persisted exchange to session store (key=%s)", key)

    # Everything below here is best-effort WebSocket sends — client may have disconnected
    if tts_first_chunk_ms is None:
        tts_first_chunk_ms = int((time.monotonic() - t1) * 1000)

    try:
        await websocket.send_text(
            ResponseTextMessage(text=_clean_display_text(clean_response)).model_dump_json()
        )
    except Exception:
        pass

    if is_beginner:
        updated_lesson = session_store.get_current_lesson(user_id, "beginner_french", _TOTAL_LESSONS)
        try:
            lesson_md = _load_lesson(updated_lesson)
        except FileNotFoundError:
            updated_lesson = 1
            lesson_md = _load_lesson(updated_lesson)
        total_steps = lesson_md.count("- [ ] Step") + lesson_md.count("- [ ] Q")
        updated_completed = session_store.get_completed_steps(user_id, "beginner_french", updated_lesson)
        try:
            await websocket.send_text(
                LessonProgressMessage(
                    lessonNumber=updated_lesson,
                    totalLessons=_TOTAL_LESSONS,
                    completedSteps=updated_completed,
                    totalSteps=total_steps,
                ).model_dump_json()
            )
        except Exception:
            pass

    e2e_ms = int((time.monotonic() - t0) * 1000)
    try:
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
    except Exception:
        pass


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
    session_mode: str = "chat"

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
                        session_mode = m.sessionMode
                        await _send_status(websocket, "recording")

                    case StopRecordingMessage():
                        await _process_audio(websocket, audio_chunks, language, user_id, session_mode)

                    case CancelMessage():
                        audio_chunks = []
                        await _send_status(websocket, "idle")

                    case SetLanguageMessage() as m:
                        language = m.language

                    case SessionHistoryMessage() as m:
                        key = _session_key(m.userId, m.sessionMode)
                        messages = session_store.get_messages(key)
                        await websocket.send_text(
                            HistoryMessage(
                                messages=[HistoryEntry(role=e["role"], text=e["text"], language=e.get("language")) for e in messages]
                            ).model_dump_json()
                        )
                        if m.sessionMode == "beginner_french":
                            lesson_num = session_store.get_current_lesson(m.userId, "beginner_french", _TOTAL_LESSONS)
                            try:
                                lesson_md = _load_lesson(lesson_num)
                            except FileNotFoundError:
                                lesson_num = 1
                                lesson_md = _load_lesson(lesson_num)
                            total_steps = lesson_md.count("- [ ] Step") + lesson_md.count("- [ ] Q")
                            completed = session_store.get_completed_steps(m.userId, "beginner_french", lesson_num)
                            await websocket.send_text(
                                LessonProgressMessage(
                                    lessonNumber=lesson_num,
                                    totalLessons=_TOTAL_LESSONS,
                                    completedSteps=completed,
                                    totalSteps=total_steps,
                                ).model_dump_json()
                            )

                    case ClearHistoryMessage() as m:
                        key = _session_key(m.userId, m.sessionMode)
                        session_store.delete_session(key)
                        session_store.reset_all_lessons(m.userId, m.sessionMode)

                    case ReplayTtsMessage() as m:
                        default_lang = "en" if m.sessionMode == "beginner_french" else (language or "en")
                        fragments = _process_language_tags(m.text, default_lang)
                        for frag_text, frag_lang in fragments:
                            try:
                                async with tts_engine._lock:
                                    audio, sr = await asyncio.to_thread(tts_engine.generate, frag_text, frag_lang)
                                    wav_bytes = samples_to_wav(audio, sr)
                                await websocket.send_bytes(wav_bytes)
                            except Exception:
                                _logger.warning("Replay TTS failed for %r (lang=%s)", frag_text[:80], frag_lang, exc_info=True)
                        await websocket.send_text(AudioDoneMessage().model_dump_json())

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
