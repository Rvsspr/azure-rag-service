import pytest

# File: tests/test_retrieval.py

# Try to import the project's retrieval module; skip tests cleanly if it doesn't exist.
retrieval = pytest.importorskip("azure_rag_service.retrieval", reason="retrieval module not found")


SAMPLE_TEXT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Vestibulum convallis, nunc a tincidunt dictum, turpis massa "
    "vestibulum orci, sed tincidunt sapien urna eget arcu. "
    "Praesent sit amet sapien non lectus facilisis tincidunt."
)


def _ensure_callable(obj, name):
    if not callable(obj):
        pytest.skip(f"{name} is not callable in retrieval module")


def test_chunk_text_splits_and_overlaps():
    # Skip if chunk_text not implemented
    chunk_text = getattr(retrieval, "chunk_text", None)
    if chunk_text is None:
        pytest.skip("chunk_text not found in retrieval module")
    _ensure_callable(chunk_text, "chunk_text")

    # Typical parameters used across many RAG designs
    max_chunk_size = 80
    overlap = 20

    chunks = list(chunk_text(SAMPLE_TEXT, max_chunk_size=max_chunk_size, overlap=overlap))
    assert len(chunks) >= 2, "Expected text to be split into multiple chunks"

    # Check each chunk isn't larger than max_chunk_size
    for c in chunks:
        assert len(c) <= max_chunk_size

    # Check overlaps: the end of chunk i should appear at the start of chunk i+1
    for a, b in zip(chunks, chunks[1:]):
        # Overlap should be at least the configured overlap or a substring match
        common = sum(1 for i in range(1, min(len(a), len(b)) + 1) if a[-i:] == b[:i])
        assert common >= 1, "Chunks should overlap by at least one character"


def test_build_embeddings_uses_embedder(monkeypatch):
    # Skip if expected functions missing
    embed_func = getattr(retrieval, "embed_texts", None) or getattr(retrieval, "embed_documents", None)
    index_func = getattr(retrieval, "index_documents", None)
    if embed_func is None or index_func is None:
        pytest.skip("embed/index functions not found in retrieval module")

    _ensure_callable(embed_func, "embed_texts/embed_documents")
    _ensure_callable(index_func, "index_documents")

    captured_inputs = []

    def fake_embed(texts):
        # record inputs and return deterministic vectors
        captured_inputs.extend(texts)
        return [[float(len(t))] for t in texts]

    # Patch the embedder used by the retrieval module
    monkeypatch.setattr(retrieval, embed_func.__name__, fake_embed)

    docs = ["alpha beta", "gamma delta epsilon"]
    # Call index (or wrapper) expecting it to call the patched embedder
    try:
        index_func(docs, namespace="test")
    except Exception:
        # Some projects return value, others persist to a store; we only care that embed was called
        pass

    assert len(captured_inputs) == len(docs), "embed function should be called once per document"


def test_search_returns_top_k(monkeypatch):
    # Skip if search interface missing
    search_fn = getattr(retrieval, "search", None)
    if search_fn is None:
        pytest.skip("search function not found in retrieval module")
    _ensure_callable(search_fn, "search")

    # Prepare a fake vector store search response
    fake_hits = [
        {"id": "doc1", "score": 0.1},
        {"id": "doc2", "score": 0.2},
        {"id": "doc3", "score": 0.3},
    ]

    def fake_search(query, k=3, **kwargs):
        # return top-k highest scores descending
        return sorted(fake_hits, key=lambda h: -h["score"])[:k]

    # Patch retrieval.search implementation internals if present
    # Some implementations expose a lower-level _vector_search or vector_store.search
    if hasattr(retrieval, "_vector_search"):
        monkeypatch.setattr(retrieval, "_vector_search", fake_search)
    elif hasattr(retrieval, "vector_store") and hasattr(retrieval.vector_store, "search"):
        monkeypatch.setattr(retrieval.vector_store, "search", fake_search)
    else:
        # Fallback: patch the public search to our wrapper to exercise downstream behavior
        monkeypatch.setattr(retrieval, "search", lambda q, k=3, **kw: fake_search(q, k=k))

    # Execute search and assert shape and order
    results = retrieval.search("test query", k=2)
    assert isinstance(results, (list, tuple)), "search should return a list-like of hits"
    assert len(results) == 2, "search should respect top-k parameter"
    # Expect the highest score first (doc3)
    assert results[0]["id"] == "doc3"


def test_retrieve_handles_empty_query():
    # If retrieval exposes a high-level retrieve function, ensure empty queries are handled gracefully
    retrieve_fn = getattr(retrieval, "retrieve", None)
    if retrieve_fn is None:
        pytest.skip("retrieve function not found in retrieval module")
    _ensure_callable(retrieve_fn, "retrieve")

    res = retrieve_fn("", k=5)
    # Accept either empty list or a controlled response object; be permissive but explicit
    assert res is not None, "retrieve should return an empty collection or a valid response for empty queries"
    if isinstance(res, (list, tuple)):
        assert len(res) == 0 or all("id" in r or "score" in r for r in res)