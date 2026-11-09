import sys
from pathlib import Path

# Repo root for `agents`, `engine`, ...; this directory for the shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest


@pytest.fixture(autouse=True)
def _inspector_off(monkeypatch):
    """The Inspector agent posts extra chat messages; only tests/test_inspector.py turns it on."""
    from dashboard import inspect_agent
    monkeypatch.setattr(inspect_agent.InspectAgent, "inspect_mode", "off")
