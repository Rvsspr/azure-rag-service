from app.infra.vector_store import search
from app.ingestion.embedder import client
from app.config import *

def retrieve_context(question, collection):
    embed = client.embeddings.create(
        model=AZURE_OPENAI_EMBED_DEPLOYMENT,
        input=question
    ).data[0].embedding

    results = search(embed)
    context = []
    citations = []

    for r in results:
        context.append(r["content"])
        citations.append(r["source"])

    return "\n".join(context), list(set(citations))
