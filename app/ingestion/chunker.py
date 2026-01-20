def chunk_documents(docs, chunk_size=500):
    chunks = []
    for doc in docs:
        text = doc["text"]
        for i in range(0, len(text), chunk_size):
            chunks.append({
                "text": text[i:i+chunk_size],
                "source": doc["source"]
            })
    return chunks