"""OpenClaw gateway client with Ed25519 device authentication."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

# ---------------------------------------------------------------------------
# Dotenv loader (same pattern as cyborg)
# ---------------------------------------------------------------------------

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        key = key.strip()
        if not _ENV_KEY_RE.fullmatch(key):
            continue
        value = raw_value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = bytes(value[1:-1], "utf-8").decode("unicode_escape")
        elif value.startswith("'") and value.endswith("'") and len(value) >= 2:
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        os.environ.setdefault(key, os.path.expandvars(value))


def _load_env_files() -> None:
    _load_env_file(Path.home() / ".config" / "cyborg" / ".env")
    _load_env_file(Path.cwd() / ".env")

# ---------------------------------------------------------------------------
# Device identity helpers (Ed25519 key pair for gateway authentication)
# ---------------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


@dataclass(slots=True)
class DeviceIdentity:
    device_id: str  # SHA-256 hex of raw public key
    public_key_pem: str
    private_key_pem: str


def _derive_device_id(raw_public_key_bytes: bytes) -> str:
    return hashlib.sha256(raw_public_key_bytes).hexdigest()


def _extract_raw_public_key(pem: str) -> bytes:
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
        load_pem_public_key,
    )

    pub = load_pem_public_key(pem.encode())
    return pub.public_bytes(Encoding.Raw, PublicFormat.Raw)


def _extract_raw_private_key(pem: str) -> bytes:
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        load_pem_private_key,
    )

    priv = load_pem_private_key(pem.encode(), password=None)
    return priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())


def load_or_create_identity(identity_path: Path) -> DeviceIdentity:
    if identity_path.exists():
        data = json.loads(identity_path.read_text(encoding="utf-8"))
        if (
            data.get("version") == 1
            and data.get("deviceId")
            and data.get("publicKeyPem")
            and data.get("privateKeyPem")
        ):
            return DeviceIdentity(
                device_id=data["deviceId"],
                public_key_pem=data["publicKeyPem"],
                private_key_pem=data["privateKeyPem"],
            )

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    raw_pub = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    device_id = _derive_device_id(raw_pub)

    public_key_pem = public_key.public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    private_key_pem = private_key.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption(),
    ).decode()

    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(
        json.dumps(
            {
                "version": 1,
                "deviceId": device_id,
                "publicKeyPem": public_key_pem,
                "privateKeyPem": private_key_pem,
                "createdAtMs": int(time.time() * 1000),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    identity_path.chmod(0o600)
    logging.getLogger(__name__).info("Created device identity: %s", identity_path)
    return DeviceIdentity(
        device_id=device_id,
        public_key_pem=public_key_pem,
        private_key_pem=private_key_pem,
    )


def build_device_auth(
    identity: DeviceIdentity,
    *,
    client_id: str,
    client_mode: str,
    role: str,
    scopes: list[str],
    token: str,
    nonce: str,
    platform: str,
) -> dict[str, Any]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    signed_at_ms = int(time.time() * 1000)
    scopes_csv = ",".join(scopes)
    device_family = ""

    payload = (
        f"v3|{identity.device_id}|{client_id}|{client_mode}|{role}"
        f"|{scopes_csv}|{signed_at_ms}|{token}|{nonce}|{platform.lower()}|{device_family}"
    )

    priv = load_pem_private_key(identity.private_key_pem.encode(), password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        raise TypeError("Expected Ed25519 private key")
    signature = priv.sign(payload.encode("utf-8"))

    raw_pub = _extract_raw_public_key(identity.public_key_pem)

    return {
        "id": identity.device_id,
        "publicKey": _b64url(raw_pub),
        "signature": _b64url(signature),
        "signedAt": signed_at_ms,
        "nonce": nonce,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = 3
CLIENT_ID = "gateway-client"
CLIENT_MODE = "backend"
ROLE = "operator"
SCOPES = ["operator.write"]
IDENTITY_FILE_NAME = "device-identity.json"


@dataclass(slots=True)
class OpenClawConfig:
    gateway_url: str = ""
    gateway_token: str = ""
    agent_id: str | None = None
    timeout_seconds: float = 120.0
    identity_dir: Path = Path.home() / ".openclaw"

    @property
    def enabled(self) -> bool:
        return bool(self.resolved_gateway_url and self.resolved_gateway_token)

    @property
    def resolved_gateway_url(self) -> str:
        candidate = self.gateway_url.strip()
        if not candidate:
            return ""
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            scheme = "wss" if parsed.scheme == "https" else "ws"
            parsed = parsed._replace(scheme=scheme)
        elif not parsed.scheme:
            parsed = urlparse(f"ws://{candidate}")
        return urlunparse(parsed._replace(params="", query="", fragment=""))

    @property
    def resolved_gateway_token(self) -> str:
        return self.gateway_token.strip()

    @property
    def identity_path(self) -> Path:
        return self.identity_dir / IDENTITY_FILE_NAME

    @classmethod
    def from_env(cls) -> OpenClawConfig:
        _load_env_files()
        return cls(
            gateway_url=(
                os.getenv("BOBVOICE_OPENCLAW_GATEWAY_URL", "").rstrip("/")
                or os.getenv("BOBVOICE_OPENCLAW_BASE_URL", "").rstrip("/")
                or os.getenv("CYBORG_OPENCLAW_GATEWAY_URL", "").rstrip("/")
                or os.getenv("CYBORG_OPENCLAW_BASE_URL", "").rstrip("/")
            ),
            gateway_token=(
                os.getenv("BOBVOICE_OPENCLAW_GATEWAY_TOKEN", "")
                or os.getenv("BOBVOICE_OPENCLAW_TOKEN", "")
                or os.getenv("CYBORG_OPENCLAW_GATEWAY_TOKEN", "")
                or os.getenv("CYBORG_OPENCLAW_TOKEN", "")
            ),
            agent_id=(
                os.getenv("BOBVOICE_OPENCLAW_AGENT_ID")
                or os.getenv("CYBORG_OPENCLAW_AGENT_ID")
                or None
            ),
            timeout_seconds=float(
                os.getenv("BOBVOICE_OPENCLAW_TIMEOUT_SECONDS", "")
                or os.getenv("CYBORG_OPENCLAW_TIMEOUT_SECONDS", "120"),
            ),
        )


# ---------------------------------------------------------------------------
# Response text extraction
# ---------------------------------------------------------------------------


def _extract_payload_text(payloads: Any) -> str:
    if not isinstance(payloads, list):
        return ""
    return "\n".join(
        str(item.get("text", "")).strip()
        for item in payloads
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    ).strip()


def _extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        result = response.get("result")
        if isinstance(result, dict):
            text = _extract_payload_text(result.get("payloads"))
            if text:
                return text
            for key in ("content", "text", "message"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        text = _extract_payload_text(response.get("payloads"))
        if text:
            return text
        for key in ("content", "text", "message", "summary"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return str(response)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_LANGUAGE_NAMES: dict[str, str] = {
    "pt": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
}


class OpenClawClient:
    def __init__(self, config: OpenClawConfig | None = None) -> None:
        self._config = config or OpenClawConfig.from_env()
        self._identity: DeviceIdentity | None = None
        self._logger = logging.getLogger("bobvoice.openclaw")

    @property
    def identity(self) -> DeviceIdentity:
        if self._identity is None:
            self._identity = load_or_create_identity(self._config.identity_path)
        return self._identity

    async def respond(
        self,
        text: str,
        language: str = "en",
        *,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_start: Callable[[], Awaitable[None]] | None = None,
        user_id: str = "mike",
    ) -> str:
        if not self._config.enabled:
            return "I'm not connected to an AI service yet. Please configure the OpenClaw gateway."

        try:
            return await self._dispatch(text, language=language, on_delta=on_delta, on_tool_start=on_tool_start, user_id=user_id)
        except Exception as exc:
            self._logger.error("OpenClaw gateway error: %s", exc)
            return "Sorry, I couldn't reach the AI service. Please try again."

    async def _dispatch(
        self,
        text: str,
        *,
        language: str = "en",
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_start: Callable[[], Awaitable[None]] | None = None,
        user_id: str = "mike",
    ) -> str:
        import websockets

        gateway_url = self._config.resolved_gateway_url
        timeout = self._config.timeout_seconds
        connect_id = str(uuid4())
        request_id = str(uuid4())

        async with websockets.connect(
            gateway_url,
            open_timeout=timeout,
            close_timeout=timeout,
            max_size=1_048_576,
        ) as websocket:
            nonce = await self._await_challenge(websocket, timeout)

            connect_params = self._build_connect_params(nonce)
            await websocket.send(json.dumps({
                "type": "req",
                "id": connect_id,
                "method": "connect",
                "params": connect_params,
            }))
            await self._await_response(websocket, connect_id, timeout)

            message = text
            prefix = "[You are a voice assistant. Respond in plain spoken language: no emojis, no markdown formatting, no asterisks, no bullet points. Just natural speech.]"
            if language and language != "en":
                lang_name = _LANGUAGE_NAMES.get(language, language)
                prefix += f" [Respond in {lang_name}. Act as a language coach: suggest corrections to the user's grammar and phrasing when they make mistakes.]"
            message = f"{prefix} {text}"

            agent_params: dict[str, Any] = {
                "message": message,
                "deliver": False,
                "sessionKey": f"bobvoice:voice:{user_id}",
                "thinking": "off",
                "timeout": int(timeout * 1000),
                "idempotencyKey": str(uuid4()),
            }
            if self._config.agent_id:
                agent_params["agentId"] = self._config.agent_id

            await websocket.send(json.dumps({
                "type": "req",
                "id": request_id,
                "method": "agent",
                "params": agent_params,
            }))
            response = await self._await_response(
                websocket, request_id, timeout, expect_final=True,
                on_delta=on_delta, on_tool_start=on_tool_start,
            )
            return _extract_response_text(response)

    async def _await_challenge(self, websocket: Any, timeout: float) -> str:
        while True:
            raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            frame = json.loads(raw)
            if frame.get("type") != "event":
                if frame.get("type") == "res" and frame.get("ok") is False:
                    error = frame.get("error")
                    msg = (
                        error.get("message") if isinstance(error, dict) else str(error)
                    )
                    raise RuntimeError(f"Gateway challenge failed: {msg}")
                continue
            if frame.get("event") != "connect.challenge":
                continue
            payload = frame.get("payload")
            nonce = payload.get("nonce") if isinstance(payload, dict) else None
            if not isinstance(nonce, str) or not nonce.strip():
                raise RuntimeError("Gateway challenge missing nonce")
            return nonce

    async def _await_response(
        self,
        websocket: Any,
        expected_id: str,
        timeout: float,
        *,
        expect_final: bool = False,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_start: Callable[[], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        while True:
            raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            frame = json.loads(raw)
            if frame.get("type") == "event":
                if frame.get("event") == "agent":
                    payload = frame.get("payload")
                    if isinstance(payload, dict):
                        stream = payload.get("stream")
                        if on_delta is not None and stream == "assistant":
                            data = payload.get("data")
                            if isinstance(data, dict):
                                text = data.get("text", "")
                                if text:
                                    await on_delta(text)
                        if stream == "item":
                            data = payload.get("data")
                            if isinstance(data, dict):
                                if on_tool_start is not None and data.get("kind") == "tool" and data.get("phase") == "start":
                                    await on_tool_start()
                continue
            if frame.get("type") != "res" or frame.get("id") != expected_id:
                continue
            payload = frame.get("payload")
            if expect_final and isinstance(payload, dict) and payload.get("status") == "accepted":
                continue
            if frame.get("ok") is True:
                return payload if isinstance(payload, dict) else {"payload": payload}
            error = frame.get("error")
            msg = error.get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(f"Gateway error for {expected_id}: {msg}")

    def _build_connect_params(self, nonce: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "minProtocol": PROTOCOL_VERSION,
            "maxProtocol": PROTOCOL_VERSION,
            "client": {
                "id": CLIENT_ID,
                "displayName": "Bob Voice",
                "version": "0.1.0",
                "platform": "python",
                "mode": CLIENT_MODE,
                "instanceId": str(uuid4()),
            },
            "role": ROLE,
            "scopes": SCOPES,
        }
        token = self._config.resolved_gateway_token
        if token:
            params["auth"] = {"token": token}

        try:
            device_auth = build_device_auth(
                self.identity,
                client_id=CLIENT_ID,
                client_mode=CLIENT_MODE,
                role=ROLE,
                scopes=SCOPES,
                token=token,
                nonce=nonce,
                platform="python",
            )
            params["device"] = device_auth
        except Exception:
            self._logger.warning(
                "Failed to build device auth, connecting without device identity",
                exc_info=True,
            )

        return params
