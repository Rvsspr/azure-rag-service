from fastapi import FastAPI
from app.api import query, ingest, health

app = FastAPI(title="Azure RAG Service")

app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(health.router)