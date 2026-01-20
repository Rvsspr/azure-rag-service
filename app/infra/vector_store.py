from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from app.config import *

client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def upload_embeddings(vectors, chunks, collection):
    docs = []
    for i, vec in enumerate(vectors):
        docs.append({
            "id": f"{collection}-{i}",
            "content": chunks[i]["text"],
            "source": chunks[i]["source"],
            "vector": vec
        })
    client.upload_documents(docs)

def search(query_vector):
    results = client.search(
        search_text="",
        vector=query_vector,
        top_k=5,
        vector_fields="vector"
    )
    return results