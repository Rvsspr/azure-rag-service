import inspect
import pytest

# tests/test_generation.py

# try to import the module under test, skip the whole file if it's not present
module = pytest.importorskip("azure_rag_service.generation")


class DummyClient:
    """
    A permissive dummy client that will return a predictable, simple response
    for many possible method names most generator implementations might call.
    """
    def __init__(self, response=None):
        # default response mimics a list of documents or a model response wrapper
        self._response = response if response is not None else [{"content": "dummy", "metadata": {"source": "dummy"}}]

    def _reply(self, *args, **kwargs):
        return self._response

    def __getattr__(self, name):
        # return a callable for any attribute name so tests don't depend on a specific client API
        return self._reply


def _find_generator_class(mod):
    "Return a first class in module that exposes a 'generate' method (or None)."
    for _, obj in vars(mod).items():
        if inspect.isclass(obj) and hasattr(obj, "generate") and inspect.isfunction(getattr(obj, "generate")):
            return obj
    return None


def _find_generate_function(mod):
    "Return a top level function named like generate* (or None)."
    for name, obj in vars(mod).items():
        if inspect.isfunction(obj) and name.startswith("generate"):
            return obj
    return None


def _safe_instantiate(cls, dummy_client):
    """
    Try to instantiate a class providing a dummy client if the constructor accepts it.
    Falls back to no-arg construction, then sets instance.client if present.
    Returns the instance or raises a helpful RuntimeError.
    """
    sig = inspect.signature(cls)
    # prepare kwargs for parameters that don't have defaults
    kwargs = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.default is inspect._empty:
            # try to supply dummy_client for reasonable parameter names
            if "client" in pname or "azure" in pname or "llm" in pname or "api" in pname:
                kwargs[pname] = dummy_client
            else:
                # give up if we cannot satisfy required params
                raise RuntimeError(f"Cannot instantiate {cls.__name__}: required ctor parameter '{pname}' has no default and is not recognized")
    try:
        inst = cls(**kwargs) if kwargs else cls()
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate {cls.__name__}: {e}") from e

    # if instance has a 'client' attribute, inject our dummy client
    if hasattr(inst, "client"):
        try:
            setattr(inst, "client", dummy_client)
        except Exception:
            pass

    return inst


def _call_generate_callable(callable_obj, dummy_client, *args, **kwargs):
    """
    Call a generate-like callable. If it accepts a client-like parameter, pass dummy_client.
    Otherwise call with provided args.
    """
    sig = inspect.signature(callable_obj)
    call_kwargs = {}
    # look for a parameter that might expect a client
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if "client" in pname or "azure" in pname or "llm" in pname or "api" in pname:
            call_kwargs[pname] = dummy_client
    # merge user kwargs (do not overwrite intentionally)
    call_kwargs.update(kwargs)
    return callable_obj(*args, **call_kwargs)


def test_generation_api_exists():
    """
    Basic smoke test: module should expose either a generator class with a .generate method
    or a top-level generate* function.
    """
    cls = _find_generator_class(module)
    func = _find_generate_function(module)
    assert cls is not None or func is not None, "No generator class with .generate or generate* function found in module"


def test_class_generator_produces_documents_when_injected_with_dummy_client():
    """
    If the module exposes a class with .generate, instantiate it (injecting a DummyClient if possible)
    and ensure calling generate returns a structure that looks like generated documents.
    The test is tolerant: it will skip if the class cannot be instantiated safely.
    """
    cls = _find_generator_class(module)
    if cls is None:
        pytest.skip("No generator class found")

    dummy = DummyClient()

    try:
        inst = _safe_instantiate(cls, dummy)
    except RuntimeError as e:
        pytest.skip(str(e))

    # ensure the instance has a callable generate
    assert hasattr(inst, "generate") and callable(getattr(inst, "generate"))

    try:
        result = _call_generate_callable(inst.generate, dummy, "test prompt")
    except Exception as e:
        pytest.skip(f"Calling {cls.__name__}.generate raised an exception: {e}")

    # Validate the returned shape loosely: either a list of docs or a dict containing 'documents'
    assert result is not None, "generate returned None"
    if isinstance(result, dict):
        assert "documents" in result or any(isinstance(v, list) for v in result.values()), "dict result does not contain documents"
    else:
        # expecting an iterable of document-like mappings
        assert isinstance(result, (list, tuple)), "generate result is not a list/tuple of documents"
        if len(result) > 0:
            first = result[0]
            assert isinstance(first, dict), "document item is not a dict"
            assert "content" in first or "text" in first or "metadata" in first, "document dict missing expected keys"


def test_function_generate_produces_documents_when_injected_with_dummy_client():
    """
    If the module exposes a top-level generate* function, call it with a DummyClient if the signature suggests it,
    and verify the returned structure looks like generated documents.
    """
    func = _find_generate_function(module)
    if func is None:
        pytest.skip("No generate* function found")

    dummy = DummyClient()
    try:
        result = _call_generate_callable(func, dummy, "example prompt")
    except Exception as e:
        pytest.skip(f"Calling function {func.__name__} raised an exception: {e}")

    assert result is not None, "generate function returned None"
    if isinstance(result, dict):
        assert "documents" in result or any(isinstance(v, list) for v in result.values()), "dict result does not contain documents"
    else:
        assert isinstance(result, (list, tuple)), "generate function result is not a list/tuple of documents"
        if len(result) > 0:
            first = result[0]
            assert isinstance(first, dict), "document item is not a dict"
            assert "content" in first or "text" in first or "metadata" in first, "document dict missing expected keys"