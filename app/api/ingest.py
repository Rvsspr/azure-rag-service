from fastapi import APIRouter
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import embed_and_store

router = APIRouter()

@router.post("/ingest")
def ingest(collection: str):
    docs = load_documents()
    chunks = chunk_documents(docs)
    embed_and_store(chunks, collection)
    return {"status": "ingested", "chunks": len(chunks)}