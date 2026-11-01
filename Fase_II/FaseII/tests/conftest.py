# VOREKO IA · Fase II · Configuración de pruebas

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
import main


@pytest.fixture(scope="session")
def cliente():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="session")
def metadatos(cliente):
    return cliente.get("/api/modelo").json()


@pytest.fixture(scope="session")
def obras(cliente):
    return cliente.get("/api/obras-ejemplo?completo=true").json()


@pytest.fixture()
def obra(obras):
    o = obras[0]
    return {"id_proyecto": o["id_proyecto"], "variables": dict(o["variables"]), "incluir_factores": True}
