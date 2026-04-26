from __future__ import annotations

import json
import os
from pathlib import Path  # noqa: TC003
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bobvoice.openclaw import (
    OpenClawClient,
    OpenClawConfig,
    _derive_device_id,
    _extract_response_text,
    _load_env_file,
    build_device_auth,
    load_or_create_identity,
)

# ---------------------------------------------------------------------------
# Device identity
# ---------------------------------------------------------------------------


class TestDeviceIdentity:
    def test_load_or_create_identity_creates_file(self, tmp_path: Path) -> None:
        identity_path = tmp_path / "identity.json"
        identity = load_or_create_identity(identity_path)

        assert identity_path.exists()
        data = json.loads(identity_path.read_text())
        assert data["version"] == 1
        assert data["deviceId"] == identity.device_id

    def test_load_or_create_identity_idempotent(self, tmp_path: Path) -> None:
        identity_path = tmp_path / "identity.json"
        first = load_or_create_identity(identity_path)
        second = load_or_create_identity(identity_path)
        assert first.device_id == second.device_id

    def test_load_or_create_identity_rejects_bad_version(self, tmp_path: Path) -> None:
        identity_path = tmp_path / "identity.json"
        identity_path.write_text(json.dumps({"version": 99, "deviceId": "x"}))
        identity = load_or_create_identity(identity_path)
        assert identity.device_id != "x"

    def test_derive_device_id_deterministic(self) -> None:
        raw = b"\x01" * 32
        assert _derive_device_id(raw) == _derive_device_id(raw)

    def test_derive_device_id_differs_for_different_keys(self) -> None:
        assert _derive_device_id(b"\x01" * 32) != _derive_device_id(b"\x02" * 32)


# ---------------------------------------------------------------------------
# build_device_auth
# ---------------------------------------------------------------------------


class TestBuildDeviceAuth:
    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        identity = load_or_create_identity(tmp_path / "id.json")
        auth = build_device_auth(
            identity,
            client_id="test",
            client_mode="test",
            role="operator",
            scopes=["operator.write"],
            token="tok",
            nonce="nonce123",
            platform="python",
        )
        assert set(auth.keys()) == {"id", "publicKey", "signature", "signedAt", "nonce"}
        assert auth["id"] == identity.device_id
        assert auth["nonce"] == "nonce123"


# ---------------------------------------------------------------------------
# OpenClawConfig
# ---------------------------------------------------------------------------


class TestOpenClawConfig:
    def test_from_env_reads_bobvoice_vars(self) -> None:
        with patch("bobvoice.openclaw._load_env_files"), patch.dict(os.environ, {
            "BOBVOICE_OPENCLAW_GATEWAY_URL": "wss://gw.example.com",
            "BOBVOICE_OPENCLAW_GATEWAY_TOKEN": "secret",
            "BOBVOICE_OPENCLAW_AGENT_ID": "bob",
            "BOBVOICE_OPENCLAW_TIMEOUT_SECONDS": "60",
        }, clear=False):
            cfg = OpenClawConfig.from_env()
        assert cfg.resolved_gateway_url == "wss://gw.example.com"
        assert cfg.resolved_gateway_token == "secret"
        assert cfg.agent_id == "bob"
        assert cfg.timeout_seconds == 60.0

    def test_from_env_fallback_to_cyborg_vars(self) -> None:
        env = {
            "CYBORG_OPENCLAW_BASE_URL": "http://127.0.0.1:18789",
            "CYBORG_OPENCLAW_GATEWAY_TOKEN": "cyborg-tok",
            "CYBORG_OPENCLAW_AGENT_ID": "main",
        }
        remove = {
            "BOBVOICE_OPENCLAW_GATEWAY_URL", "BOBVOICE_OPENCLAW_BASE_URL",
            "BOBVOICE_OPENCLAW_GATEWAY_TOKEN", "BOBVOICE_OPENCLAW_TOKEN",
            "BOBVOICE_OPENCLAW_AGENT_ID",
        }
        with patch("bobvoice.openclaw._load_env_files"), patch.dict(os.environ, env, clear=False):
            for k in remove:
                os.environ.pop(k, None)
            cfg = OpenClawConfig.from_env()
        assert "ws://" in cfg.resolved_gateway_url
        assert cfg.resolved_gateway_token == "cyborg-tok"
        assert cfg.agent_id == "main"

    def test_bobvoice_vars_override_cyborg(self) -> None:
        with patch("bobvoice.openclaw._load_env_files"), patch.dict(os.environ, {
            "BOBVOICE_OPENCLAW_GATEWAY_URL": "wss://override.example.com",
            "BOBVOICE_OPENCLAW_GATEWAY_TOKEN": "override-tok",
            "CYBORG_OPENCLAW_BASE_URL": "http://127.0.0.1:18789",
            "CYBORG_OPENCLAW_GATEWAY_TOKEN": "cyborg-tok",
        }, clear=False):
            cfg = OpenClawConfig.from_env()
        assert cfg.resolved_gateway_url == "wss://override.example.com"
        assert cfg.resolved_gateway_token == "override-tok"

    def test_from_env_fallback_vars(self) -> None:
        env = {
            "BOBVOICE_OPENCLAW_BASE_URL": "https://gw.example.com",
            "BOBVOICE_OPENCLAW_TOKEN": "tok",
        }
        remove = {"BOBVOICE_OPENCLAW_GATEWAY_URL", "BOBVOICE_OPENCLAW_GATEWAY_TOKEN"}
        with patch("bobvoice.openclaw._load_env_files"), patch.dict(os.environ, env, clear=False):
            for k in remove:
                os.environ.pop(k, None)
            cfg = OpenClawConfig.from_env()
        assert "wss://" in cfg.resolved_gateway_url
        assert cfg.resolved_gateway_token == "tok"

    def test_url_resolution_https_to_wss(self) -> None:
        cfg = OpenClawConfig(gateway_url="https://gw.example.com/path")
        assert cfg.resolved_gateway_url == "wss://gw.example.com/path"

    def test_url_resolution_http_to_ws(self) -> None:
        cfg = OpenClawConfig(gateway_url="http://gw.example.com")
        assert cfg.resolved_gateway_url == "ws://gw.example.com"

    def test_url_resolution_already_ws(self) -> None:
        cfg = OpenClawConfig(gateway_url="wss://gw.example.com")
        assert cfg.resolved_gateway_url == "wss://gw.example.com"

    def test_enabled_requires_url_and_token(self) -> None:
        assert not OpenClawConfig().enabled
        assert not OpenClawConfig(gateway_url="wss://gw").enabled
        assert not OpenClawConfig(gateway_token="tok").enabled
        assert OpenClawConfig(gateway_url="wss://gw", gateway_token="tok").enabled


# ---------------------------------------------------------------------------
# Response text extraction
# ---------------------------------------------------------------------------


class TestExtractResponseText:
    def test_string_response(self) -> None:
        assert _extract_response_text("hello") == "hello"

    def test_result_payloads(self) -> None:
        resp = {"result": {"payloads": [{"text": "hello"}, {"text": "world"}]}}
        assert _extract_response_text(resp) == "hello\nworld"

    def test_result_text_field(self) -> None:
        resp = {"result": {"text": "hello"}}
        assert _extract_response_text(resp) == "hello"

    def test_top_level_text(self) -> None:
        assert _extract_response_text({"text": "hello"}) == "hello"

    def test_empty_payloads_falls_through(self) -> None:
        resp = {"result": {"payloads": []}, "text": "fallback"}
        assert _extract_response_text(resp) == "fallback"


# ---------------------------------------------------------------------------
# OpenClawClient
# ---------------------------------------------------------------------------


class TestOpenClawClient:
    @pytest.mark.asyncio
    async def test_respond_fallback_when_not_configured(self) -> None:
        cfg = OpenClawConfig()
        client = OpenClawClient(config=cfg)
        result = await client.respond("hello")
        assert "not connected" in result.lower()

    @pytest.mark.asyncio
    async def test_respond_catches_connection_error(self) -> None:
        cfg = OpenClawConfig(gateway_url="wss://gw", gateway_token="tok")
        client = OpenClawClient(config=cfg)
        with patch("websockets.connect", side_effect=ConnectionRefusedError("refused")):
            result = await client.respond("hello")
        assert "couldn't reach" in result.lower()

    @pytest.mark.asyncio
    async def test_respond_success(self) -> None:
        cfg = OpenClawConfig(gateway_url="wss://gw", gateway_token="tok")
        client = OpenClawClient(config=cfg)

        sent_frames: list[dict[str, object]] = []

        async def mock_send(raw: str) -> None:
            sent_frames.append(json.loads(raw))

        recv_queue: list[str] = [
            json.dumps({"type": "event", "event": "connect.challenge", "payload": {"nonce": "n"}}),
        ]

        async def mock_recv() -> str:
            if not recv_queue:
                raise StopAsyncIteration
            return recv_queue.pop(0)

        ws = AsyncMock()
        ws.send = mock_send
        ws.recv = mock_recv

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=ws)
        ctx.__aexit__ = AsyncMock(return_value=False)

        # Intercept send to inject matching response IDs
        original_send = mock_send

        async def intercept_send(raw: str) -> None:
            await original_send(raw)
            frame = json.loads(raw)
            if frame.get("method") == "connect":
                recv_queue.append(json.dumps({
                    "type": "res", "id": frame["id"], "ok": True, "payload": {},
                }))
            elif frame.get("method") == "agent":
                recv_queue.append(json.dumps({
                    "type": "res", "id": frame["id"], "ok": True,
                    "payload": {"result": {"payloads": [{"text": "Bob says hi"}]}},
                }))

        ws.send = intercept_send

        with patch("websockets.connect", return_value=ctx):
            result = await client.respond("hello")
        assert result == "Bob says hi"


# ---------------------------------------------------------------------------
# Dotenv loader
# ---------------------------------------------------------------------------


class TestLoadEnvFile:
    def test_loads_key_value(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("MY_TEST_KEY=hello\n")
        _load_env_file(env_file)
        assert os.environ.get("MY_TEST_KEY") == "hello"
        os.environ.pop("MY_TEST_KEY", None)

    def test_skips_comments_and_blanks(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\n  \nKEY=val\n")
        _load_env_file(env_file)
        assert os.environ.get("KEY") == "val"
        os.environ.pop("KEY", None)

    def test_does_not_override_existing(self, tmp_path: Path) -> None:
        os.environ["EXISTING_KEY"] = "original"
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_KEY=from_file\n")
        _load_env_file(env_file)
        assert os.environ["EXISTING_KEY"] == "original"
        os.environ.pop("EXISTING_KEY", None)

    def test_handles_quoted_values(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text('KEY="quoted value"\n')
        _load_env_file(env_file)
        assert os.environ.get("KEY") == "quoted value"
        os.environ.pop("KEY", None)

    def test_from_env_loads_from_cyborg_env_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "CYBORG_OPENCLAW_BASE_URL=http://localhost:9999\n"
            "CYBORG_OPENCLAW_GATEWAY_TOKEN=file-tok\n"
        )
        remove = {
            "BOBVOICE_OPENCLAW_GATEWAY_URL", "BOBVOICE_OPENCLAW_BASE_URL",
            "BOBVOICE_OPENCLAW_GATEWAY_TOKEN", "BOBVOICE_OPENCLAW_TOKEN",
            "CYBORG_OPENCLAW_GATEWAY_URL", "CYBORG_OPENCLAW_BASE_URL",
            "CYBORG_OPENCLAW_GATEWAY_TOKEN", "CYBORG_OPENCLAW_TOKEN",
        }
        for k in remove:
            os.environ.pop(k, None)
        _load_env_file(env_file)
        cfg = OpenClawConfig.from_env()
        assert cfg.resolved_gateway_token == "file-tok"
        assert cfg.resolved_gateway_url == "ws://localhost:9999"
