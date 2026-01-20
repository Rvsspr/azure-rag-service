from typing import Optional
import time
# prometheus_client imports are optional and will be attempted in a try/except below
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import FastAPI
from app.monitoring.metrics import init_metrics
from fastapi import APIRouter
from fastapi.responses import Response as FastAPIResponse

"""
app/monitoring/metrics.py

Prometheus-based metrics helpers and FastAPI/Starlette middleware for instrumentation.

Usage:
- Call init_metrics(app) from your FastAPI app startup to register /metrics endpoint
    and add middleware that records request count, latency, in-progress requests and exceptions.
- Use helper functions to record domain-specific events (embedding calls, LLM calls, cache hits, etc).
"""


# Try to import prometheus_client, fallback to no-op shim if not available.
# Try to import prometheus_client, fallback to no-op shim if not available.
try:
        from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
        from prometheus_client import REGISTRY as DEFAULT_REGISTRY
except Exception:
        # Minimal no-op replacements so code using this module doesn't crash when prometheus_client
        # isn't present (useful for local development without the dependency).
        class _Noop:
                def __init__(self, *_, **__): pass
                def labels(self, *_, **__): return self
                def inc(self, *_, **__): pass
                def dec(self, *_, **__): pass
                def observe(self, *_, **__): pass
                def set(self, *_, **__): pass

        Counter = Histogram = Gauge = _Noop  # type: ignore
        DEFAULT_REGISTRY = None
        def generate_latest(_: Optional[object] = None) -> bytes:
                return b""
        CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
# Metric name prefix for this service
_METRIC_PREFIX = "azure_rag_service_"

# Common histogram buckets tuned for typical web request/LLM latencies
_DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

# Core HTTP metrics
REQUEST_COUNT = Counter(
        _METRIC_PREFIX + "http_requests_total",
        "Total HTTP requests processed",
        labelnames=("method", "path", "status_code"),
)
REQUEST_LATENCY = Histogram(
        _METRIC_PREFIX + "http_request_duration_seconds",
        "HTTP request latency in seconds",
        labelnames=("method", "path"),
        buckets=_DEFAULT_BUCKETS,
)
IN_PROGRESS = Gauge(
        _METRIC_PREFIX + "http_inprogress_requests",
        "Number of in-progress HTTP requests",
        labelnames=("method", "path"),
)
REQUEST_EXCEPTIONS = Counter(
        _METRIC_PREFIX + "http_exceptions_total",
        "Total exceptions raised while handling HTTP requests",
        labelnames=("exception_type", "method", "path"),
)

# Domain-specific metrics
EMBEDDING_CALLS = Counter(
        _METRIC_PREFIX + "embedding_calls_total",
        "Total embedding calls",
        labelnames=("model",),
)
EMBEDDING_LATENCY = Histogram(
        _METRIC_PREFIX + "embedding_latency_seconds",
        "Embedding call latency seconds",
        labelnames=("model",),
        buckets=_DEFAULT_BUCKETS,
)

LLM_CALLS = Counter(
        _METRIC_PREFIX + "llm_calls_total",
        "Total LLM calls",
        labelnames=("model",),
)
LLM_LATENCY = Histogram(
        _METRIC_PREFIX + "llm_latency_seconds",
        "LLM call latency seconds",
        labelnames=("model",),
        buckets=_DEFAULT_BUCKETS,
)

VECTOR_SEARCHES = Counter(
        _METRIC_PREFIX + "vector_search_total",
        "Total vector search operations",
        labelnames=("index_name", "namespace"),
)
CACHE_HITS = Counter(
        _METRIC_PREFIX + "cache_hits_total",
        "Total cache hits",
        labelnames=("cache_name",),
)
CACHE_MISSES = Counter(
        _METRIC_PREFIX + "cache_misses_total",
        "Total cache misses",
        labelnames=("cache_name",),
)

DOCUMENT_CHUNKS = Gauge(
        _METRIC_PREFIX + "document_chunks",
        "Number of document chunks indexed or stored",
        labelnames=("index_name",),
)

# Registry used to expose metrics (use default registry from prometheus_client when available)
_REGISTRY = DEFAULT_REGISTRY

# FastAPI/Starlette instrumentation middleware
try:

        class MetricsMiddleware(BaseHTTPMiddleware):
                """
                Records basic HTTP metrics for each request:
                - increments in-progress gauge
                - measures latency histogram
                - increments request counter on completion
                - records exceptions
                """

                async def dispatch(self, request: Request, call_next):
                        method = request.method
                        # Use path without query to reduce cardinality
                        path = request.url.path

                        IN_PROGRESS.labels(method, path).inc()
                        start = time.time()

                        status_code = "500"
                        try:
                                response: Response = await call_next(request)
                                status_code = str(response.status_code)
                                return response
                        except Exception as exc:
                                REQUEST_EXCEPTIONS.labels(type(exc).__name__, method, path).inc()
                                raise
                        finally:
                                elapsed = time.time() - start
                                REQUEST_LATENCY.labels(method, path).observe(elapsed)
                                REQUEST_COUNT.labels(method, path, status_code).inc()
                                IN_PROGRESS.labels(method, path).dec()

except Exception:
        # If starlette is not available, provide a placeholder middleware name for callers.
        MetricsMiddleware = None  # type: ignore

# Helper functions to register instrumentation with a FastAPI app
def init_metrics(app=None, route: str = "/metrics", middleware: bool = True):
        """
        Initialize metrics exporting for a FastAPI app.

        - app: Optional FastAPI app instance. If provided, middleware is added and /metrics route registered.
        - route: path to expose Prometheus metrics.
        - middleware: whether to add HTTP instrumentation middleware.

        Example:
                app = FastAPI()
                init_metrics(app)
        """
        if app is None:
                return

        # Add middleware if available and requested
        if middleware and MetricsMiddleware is not None:
                try:
                        app.add_middleware(MetricsMiddleware)
                except Exception:
                        # Some frameworks or test harnesses may not support adding middleware at this point.
                        pass

        # Register metrics endpoint
        try:

                router = APIRouter()

                @router.get(route, include_in_schema=False)
                async def metrics_endpoint():
                        data = generate_latest(_REGISTRY)
                        return FastAPIResponse(content=data, media_type=CONTENT_TYPE_LATEST)

                app.include_router(router)
        except Exception:
                # If FastAPI isn't available at import time, skip auto-registration.
                pass

# Domain instrumentation helpers
def record_embedding_call(model: str, duration_seconds: Optional[float] = None):
        EMBEDDING_CALLS.labels(model).inc()
        if duration_seconds is not None:
                EMBEDDING_LATENCY.labels(model).observe(duration_seconds)

def record_llm_call(model: str, duration_seconds: Optional[float] = None):
        LLM_CALLS.labels(model).inc()
        if duration_seconds is not None:
                LLM_LATENCY.labels(model).observe(duration_seconds)

def record_vector_search(index_name: str = "default", namespace: str = "default"):
        VECTOR_SEARCHES.labels(index_name, namespace).inc()

def record_cache_hit(cache_name: str = "default"):
        CACHE_HITS.labels(cache_name).inc()

def record_cache_miss(cache_name: str = "default"):
        CACHE_MISSES.labels(cache_name).inc()

def set_document_chunks(index_name: str, count: int):
        DOCUMENT_CHUNKS.labels(index_name).set(count)