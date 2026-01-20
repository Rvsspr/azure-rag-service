from openai import AzureOpenAI
from app.config import *
from app.infra.vector_store import upload_embeddings

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01"
)

def embed_and_store(chunks, collection):
    texts = [c["text"] for c in chunks]
    response = client.embeddings.create(
        model=AZURE_OPENAI_EMBED_DEPLOYMENT,
        input=texts
    )

    vectors = [e.embedding for e in response.data]
    upload_embeddings(vectors, chunks, collection)