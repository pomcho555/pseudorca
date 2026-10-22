from __future__ import annotations

from pathlib import Path

import pytest

from pseudorca.store import Store


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Store:
    """A fresh store in a temporary directory, with a fixed actor."""
    monkeypatch.delenv("PSEUDORCA_STORE", raising=False)
    monkeypatch.setenv("PSEUDORCA_ACTOR", "tester")
    return Store.init(tmp_path / "team-repo", name="test-team", competition="some-competition")
