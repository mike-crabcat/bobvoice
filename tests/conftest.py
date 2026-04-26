from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_stt():
    with patch("bobvoice.main.stt_engine") as mock:
        mock.transcribe = MagicMock(return_value=("hello world", "en"))
        yield mock


@pytest.fixture
def mock_openclaw():
    with patch("bobvoice.main.openclaw_client") as mock:
        mock.respond = AsyncMock(return_value="I heard you say: hello world. This is a stub response!")
        yield mock


@pytest.fixture
def mock_tts():
    with patch("bobvoice.main.tts_engine") as mock:
        mock._lock = MagicMock()
        mock._lock.__aenter__ = AsyncMock(return_value=None)
        mock._lock.__aexit__ = AsyncMock(return_value=False)
        mock.generate = MagicMock(
            return_value=(np.zeros(24000, dtype=np.float32), 24000)
        )
        yield mock
