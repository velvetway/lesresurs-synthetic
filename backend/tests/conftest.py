"""Фикстуры: изолированная БД журнала и клиент API."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LESRESURS_DB", str(Path(tempfile.mkdtemp(prefix="lesresurs-test-")) / "journal.db"))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def scn(client):
    """Демо-сценарий модельного субъекта (10 пожаров, 9 станций, дефицит сил)."""
    client.post("/api/region/region_a")
    return client.get("/api/scenario").json()
