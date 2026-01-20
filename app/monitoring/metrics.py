import time
import logging

logger = logging.getLogger("rag")

def log_request(metric: str, value: float):
    logger.info(f"{metric}: {value}")
In /query
start = time.time()
# run RAG
latency = time.time() - start
log_request("latency_ms", latency * 1000)