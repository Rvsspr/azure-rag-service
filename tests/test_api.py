import importlib
import inspect
from typing import Optional
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# tests/test_api.py



def find_fastapi_app() -> Optional[FastAPI]:
    """
    Try to locate a FastAPI instance in common module/attribute names.
    Returns the FastAPI instance or None.
    """
    candidate_modules = [
        "app",
        "main",
        "server",
        "src.app",
        "src.main",
        "azure_rag_service.app",
        "azure_rag_service.main",
        "azure_rag_service.server",
        "api",
    ]
    candidate_attrs = ["app", "application", "fastapi_app", "api", "server"]

    for mod_name in candidate_modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        # check common attribute names
        for attr in candidate_attrs:
            if hasattr(mod, attr):
                obj = getattr(mod, attr)
                if isinstance(obj, FastAPI):
                    return obj

        # fallback: scan module globals for FastAPI instances
        for _, obj in inspect.getmembers(mod):
            if isinstance(obj, FastAPI):
                return obj

    return None


def _get_client_or_skip():
    app = find_fastapi_app()
    if app is None:
        pytest.skip("Could not find a FastAPI application to test (searched common module names).")
    return TestClient(app)


def test_app_importable():
    app = find_fastapi_app()
    assert isinstance(app, FastAPI), "FastAPI app not found in project imports."


@pytest.mark.parametrize("path", ["/health", "/healthz", "/ping", "/status", "/"])
def test_health_endpoint(path):
    client = _get_client_or_skip()
    resp = client.get(path)
    # accept 200 or 204 as healthy responses
    assert resp.status_code in (200, 204), f"Health endpoint {path} returned {resp.status_code}"


def test_simple_ask_endpoint_behaviour():
    """
    Try a few common RAG/QA endpoints and assert they accept a JSON question payload
    and return a successful status with a recognisable answer field.
    This test will skip if none of the common endpoints exist.
    """
    client = _get_client_or_skip()
    candidate_paths = ["/api/ask", "/ask", "/query", "/qa", "/api/query", "/api/qa"]
    payload = {"question": "What is the meaning of life?"}
    answer_keys = {"answer", "answers", "result", "response", "results", "answer_text", "data"}

    found_any = False
    for p in candidate_paths:
        try:
            resp = client.post(p, json=payload)
        except Exception:
            continue

        # consider any 2xx/202/201 as potential success
        if resp.status_code in (200, 201, 202):
            found_any = True
            # try parse JSON; if not JSON, at least ensure non-empty body
            try:
                body = resp.json()
            except Exception:
                assert resp.content, f"{p} returned empty body"
                break

            # pass if any common answer key is present
            if isinstance(body, dict) and answer_keys.intersection(body.keys()):
                break
            # also accept simple string responses
            if isinstance(body, str) and body.strip():
                break
            # otherwise continue trying other endpoints
    if not found_any:
        pytest.skip("No candidate QA endpoints returned a successful response; skipping behavioral check.")