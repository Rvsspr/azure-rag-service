# Azure RAG Service

Production-grade Retrieval-Augmented Generation (RAG) service built on Azure OpenAI.

## Features
- Document ingestion (PDF, TXT)
- Embeddings + vector search
- LLM answer generation with citations
- Confidence scoring & fallback logic
- Cost-aware token limits
- Monitoring-ready architecture

## Architecture
- Azure OpenAI (GPT + embeddings)
- Azure AI Search (vector search)
- FastAPI backend
- Dockerized deployment

## API
POST /ingest  
POST /query  

## Run Locally
```bash
docker build -t azure-rag .
docker run -p 8000:8000 --env-file .env azure-rag
