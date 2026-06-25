"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_network_tools(monkeypatch):
    monkeypatch.setenv("RETRIEVER_ENABLED", "0")
    monkeypatch.setenv("AGENT_ARTIFACTS", "0")
    monkeypatch.setenv("SELF_EVOLVE", "0")
    monkeypatch.setenv("PIPELINE_DEPTH", "minimal")
